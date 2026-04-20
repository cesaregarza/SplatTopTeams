from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from shared_lib.db import create_engine, get_schema, resolve_database_url
from shared_lib.team_vector_utils import (
    build_identity_idf_lookup,
    canonicalize_player_ids,
    hash_index,
    hash_sign,
    lineup_key,
    pair_key,
    unordered_player_pairs,
)
from team_api.sql import get_vector_column_info, ensure_search_tables, validate_identifier

logger = logging.getLogger(__name__)


def _is_permission_denied(exc: Exception) -> bool:
    message = str(exc).lower()
    return "insufficientprivilege" in message or "permission denied" in message


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.10g}" for v in vector) + "]"


@dataclass
class TeamPayload:
    team_id: int
    tournament_id: Optional[int]
    team_name: str
    event_time_ms: Optional[int]
    lineup_count: int
    tournament_count: int
    unique_player_count: int
    distinct_lineup_count: int
    top_lineup_share: float
    lineup_entropy: float
    effective_lineups: float
    semantic_vector: np.ndarray
    identity_vector: np.ndarray
    final_vector: np.ndarray
    top_lineup_summary: str
    top_lineup_player_ids: Tuple[int, ...]
    top_lineup_player_names: Tuple[str, ...]
    roster_player_ids: Tuple[int, ...]
    roster_player_names: Tuple[str, ...]
    roster_player_match_counts: Tuple[int, ...]
    player_support: Dict[str, float]
    pair_support: Dict[str, float]
    lineup_variant_counts: Dict[str, int]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _to_ms_days_ago(days: Optional[int]) -> Optional[int]:
    if days is None or days < 0:
        return None
    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    return now_ms - int(days) * 86_400_000


def _sec_expr(column_expr: str) -> str:
    return (
        f"(CASE "
        f"WHEN {column_expr} IS NULL THEN NULL "
        f"WHEN {column_expr} >= 1000000000000 THEN {column_expr} / 1000 "
        f"ELSE {column_expr} END)"
    )


def _to_db_seconds(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    iv = int(v)
    return iv // 1000 if iv > 1_000_000_000_000 else iv


def _fetch_lineups(
    db_url: str,
    schema: str,
    since_ms: Optional[int],
    until_ms: Optional[int],
    min_players: int,
    max_players: int,
) -> List[Dict[str, object]]:
    schema = validate_identifier(schema)
    engine = create_engine(db_url)

    where: List[str] = []
    params: Dict[str, object] = {
        "min_players": int(min_players),
        "max_players": int(max_players),
    }

    time_expr = (
        f"COALESCE({_sec_expr('m.last_game_finished_at_ms')}, "
        f"{_sec_expr('m.created_at_ms')}, "
        f"{_sec_expr('t.start_time_ms')})"
    )
    since_ts = _to_db_seconds(since_ms)
    until_ts = _to_db_seconds(until_ms)
    if since_ts is not None:
        where.append(f"{time_expr} >= :since_ts")
        params["since_ts"] = since_ts
    if until_ts is not None:
        where.append(f"{time_expr} <= :until_ts")
        params["until_ts"] = until_ts

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    sql = f"""
        SELECT
            pat.tournament_id,
            pat.match_id,
            pat.team_id,
            ARRAY_AGG(DISTINCT pat.player_id ORDER BY pat.player_id) AS player_ids
        FROM {schema}.player_appearance_teams pat
        JOIN {schema}.matches m
          ON m.match_id = pat.match_id
         AND m.tournament_id = pat.tournament_id
        JOIN {schema}.tournaments t
          ON t.tournament_id = pat.tournament_id
        {where_clause}
        GROUP BY pat.tournament_id, pat.match_id, pat.team_id
        HAVING COUNT(DISTINCT pat.player_id) BETWEEN :min_players AND :max_players
    """

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(row) for row in rows]


def _fetch_team_metadata(
    db_url: str,
    schema: str,
    team_ids: Sequence[int],
) -> Dict[int, Dict[str, object]]:
    if not team_ids:
        return {}

    schema = validate_identifier(schema)
    engine = create_engine(db_url)
    meta: Dict[int, Dict[str, object]] = {}

    with engine.connect() as conn:
        for i in range(0, len(team_ids), 1000):
            chunk = team_ids[i : i + 1000]
            params = {f"id{j}": int(v) for j, v in enumerate(chunk)}
            clause = ", ".join(f":id{j}" for j in range(len(chunk)))
            sql = f"""
                SELECT
                    tt.team_id,
                    tt.tournament_id,
                    tt.name,
                    t.start_time_ms
                FROM {schema}.tournament_teams tt
                JOIN {schema}.tournaments t
                  ON t.tournament_id = tt.tournament_id
                WHERE tt.team_id IN ({clause})
            """
            rows = conn.execute(text(sql), params).mappings().all()
            for row in rows:
                team_id = int(row["team_id"])
                meta[team_id] = {
                    "tournament_id": (
                        int(row["tournament_id"])
                        if row["tournament_id"] is not None
                        else None
                    ),
                    "team_name": str(row["name"] or team_id),
                    "event_time_ms": (
                        int(row["start_time_ms"])
                        if row["start_time_ms"] is not None
                        else None
                    ),
                }

    return meta


def _fetch_player_names(
    db_url: str,
    schema: str,
    player_ids: Sequence[int],
) -> Dict[int, str]:
    if not player_ids:
        return {}

    schema = validate_identifier(schema)
    engine = create_engine(db_url)
    names: Dict[int, str] = {}

    with engine.connect() as conn:
        for i in range(0, len(player_ids), 1000):
            chunk = player_ids[i : i + 1000]
            params = {f"id{j}": int(v) for j, v in enumerate(chunk)}
            clause = ", ".join(f":id{j}" for j in range(len(chunk)))
            sql = f"""
                SELECT player_id, display_name
                FROM {schema}.players
                WHERE player_id IN ({clause})
            """
            rows = conn.execute(text(sql), params).mappings().all()
            for row in rows:
                player_id = int(row["player_id"])
                raw_name = row["display_name"]
                if raw_name is None:
                    continue
                display_name = str(raw_name).strip()
                if display_name and not display_name.isdigit():
                    names[player_id] = display_name

    return names


def _index_lineups(
    rows: Sequence[Dict[str, object]],
) -> Tuple[
    Dict[int, List[Tuple[int, ...]]],
    Dict[int, Counter[Tuple[int, ...]]],
    Dict[int, Dict[int, int]],
    Dict[int, set[int]],
]:
    team_lineups: Dict[int, List[Tuple[int, ...]]] = defaultdict(list)
    lineup_counts: Dict[int, Counter[Tuple[int, ...]]] = defaultdict(Counter)
    team_player_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    team_tournament_ids: Dict[int, set[int]] = defaultdict(set)

    for row in rows:
        team_id = int(row["team_id"])
        tournament_id = row.get("tournament_id")
        if tournament_id is not None:
            try:
                team_tournament_ids[team_id].add(int(tournament_id))
            except (TypeError, ValueError):
                pass
        player_ids = tuple(int(pid) for pid in (row["player_ids"] or []) if pid is not None)
        if not player_ids:
            continue
        team_lineups[team_id].append(player_ids)
        lineup_counts[team_id][player_ids] += 1
        for player_id in player_ids:
            team_player_counts[team_id][player_id] += 1

    return team_lineups, lineup_counts, team_player_counts, team_tournament_ids


def _build_semantic_vectors(
    team_lineups: Dict[int, List[Tuple[int, ...]]],
    dim: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for team_id, lineups in team_lineups.items():
        if not lineups:
            out[team_id] = np.zeros(dim, dtype=np.float64)
            continue

        lineup_vecs: List[np.ndarray] = []
        for lineup in lineups:
            vec = np.zeros(dim, dtype=np.float64)
            for player_id in lineup:
                idx = hash_index(player_id, dim, "sem_idx")
                sign = hash_sign(player_id, "sem_sign")
                vec[idx] += sign
            lineup_vecs.append(_normalize(vec))

        team_vec = np.mean(np.stack(lineup_vecs, axis=0), axis=0)
        out[team_id] = _normalize(team_vec)

    return out


def _build_identity_vectors(
    team_player_counts: Dict[int, Dict[int, int]],
    dim: int,
    idf_cap: float,
) -> Dict[int, np.ndarray]:
    idf = build_identity_idf_lookup(
        (team_counts.keys() for team_counts in team_player_counts.values()),
        idf_cap=idf_cap,
    )

    out: Dict[int, np.ndarray] = {}
    for team_id, counts in team_player_counts.items():
        vec = np.zeros(dim, dtype=np.float64)
        for player_id, count in counts.items():
            idx = hash_index(player_id, dim, "id_idx")
            sign = hash_sign(player_id, "id_sign")
            vec[idx] += sign * float(count) * idf.get(player_id, 1.0)
        out[team_id] = _normalize(vec)

    return out


def _build_support_maps(
    lineups: Sequence[Tuple[int, ...]],
) -> tuple[Dict[str, float], Dict[str, float]]:
    if not lineups:
        return {}, {}

    total_lineups = max(1, len(lineups))
    player_counts: Counter[int] = Counter()
    pair_counts: Counter[str] = Counter()

    for lineup in lineups:
        player_ids = canonicalize_player_ids(lineup)
        if not player_ids:
            continue
        for player_id in player_ids:
            player_counts[int(player_id)] += 1
        for left, right in unordered_player_pairs(player_ids):
            pair_counts[pair_key(left, right)] += 1

    player_support = {
        str(player_id): float(count) / float(total_lineups)
        for player_id, count in sorted(player_counts.items())
        if int(player_id) > 0 and int(count) > 0
    }
    pair_support = {
        key: float(count) / float(total_lineups)
        for key, count in sorted(pair_counts.items())
        if key and int(count) > 0
    }
    return player_support, pair_support


def _build_lineup_variant_counts(
    lineup_counter: Counter[Tuple[int, ...]],
) -> Dict[str, int]:
    if not lineup_counter:
        return {}
    return {
        lineup_key(lineup): int(count)
        for lineup, count in sorted(lineup_counter.items())
        if lineup and int(count) > 0
    }


def _top_lineup(lineup_counts: Counter[Tuple[int, ...]]) -> tuple[int, Tuple[int, ...]]:
    if not lineup_counts:
        return 0, tuple()
    lineup, count = lineup_counts.most_common(1)[0]
    return int(count), tuple(int(pid) for pid in lineup)


def _top_lineup_summary(top_count: int, top_lineup_names: Sequence[str]) -> str:
    if top_count <= 0 or not top_lineup_names:
        return ""
    players = ",".join(top_lineup_names)
    return f"{top_count}x:{players}"


def _lineup_stats(
    lineup_counter: Counter[Tuple[int, ...]],
    total_lineups: int,
) -> tuple[int, float, float, float]:
    if total_lineups <= 0 or not lineup_counter:
        return 0, 0.0, 0.0, 0.0
    distinct = len(lineup_counter)
    top_share = max(lineup_counter.values()) / float(total_lineups)

    probs = np.asarray(
        [count / float(total_lineups) for count in lineup_counter.values()],
        dtype=np.float64,
    )
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    if distinct > 1:
        entropy /= float(np.log(distinct))
    effective_lineups = float(np.exp(entropy))
    return distinct, top_share, entropy, effective_lineups


def _build_payloads(
    team_ids: Sequence[int],
    metadata: Dict[int, Dict[str, object]],
    player_name_by_id: Dict[int, str],
    team_lineups: Dict[int, List[Tuple[int, ...]]],
    lineup_counts: Dict[int, Counter[Tuple[int, ...]]],
    team_player_counts: Dict[int, Dict[int, int]],
    team_tournament_ids: Dict[int, set[int]],
    semantic_vectors: Dict[int, np.ndarray],
    identity_vectors: Dict[int, np.ndarray],
    identity_beta: float,
) -> List[TeamPayload]:
    payloads: List[TeamPayload] = []
    for team_id in team_ids:
        sem = semantic_vectors.get(team_id)
        ident = identity_vectors.get(team_id)
        if sem is None or ident is None:
            continue
        final = _normalize(np.concatenate([sem, identity_beta * ident]))
        meta = metadata.get(team_id, {})
        team_lineup_count = len(team_lineups.get(team_id, []))
        lineup_counter = lineup_counts.get(team_id, Counter())
        tournament_count = len(team_tournament_ids.get(team_id, set()))
        distinct_lineups, top_share, lineup_entropy, effective_lineups = (
            _lineup_stats(lineup_counter, team_lineup_count)
        )
        top_count, top_lineup = _top_lineup(lineup_counter)
        top_lineup_names = tuple(
            player_name_by_id.get(player_id, "Unknown Player")
            for player_id in top_lineup
        )
        roster_counts = team_player_counts.get(team_id, {})
        unique_player_count = len(roster_counts)
        roster_sorted = sorted(
            (
                (int(player_id), int(count))
                for player_id, count in roster_counts.items()
                if player_id is not None
            ),
            key=lambda item: (-item[1], item[0]),
        )
        roster_player_ids = tuple(player_id for player_id, _ in roster_sorted)
        roster_player_match_counts = tuple(count for _, count in roster_sorted)
        roster_player_names = tuple(
            player_name_by_id.get(player_id, "Unknown Player")
            for player_id in roster_player_ids
        )
        player_support, pair_support = _build_support_maps(
            team_lineups.get(team_id, [])
        )
        lineup_variant_counts = _build_lineup_variant_counts(lineup_counter)
        payloads.append(
            TeamPayload(
                team_id=team_id,
                tournament_id=meta.get("tournament_id"),
                team_name=str(meta.get("team_name") or team_id),
                event_time_ms=meta.get("event_time_ms"),
                lineup_count=team_lineup_count,
                tournament_count=tournament_count,
                unique_player_count=unique_player_count,
                distinct_lineup_count=distinct_lineups,
                top_lineup_share=float(top_share),
                lineup_entropy=float(lineup_entropy),
                effective_lineups=float(effective_lineups),
                semantic_vector=sem,
                identity_vector=ident,
                final_vector=final,
                top_lineup_summary=_top_lineup_summary(top_count, top_lineup_names),
                top_lineup_player_ids=top_lineup,
                top_lineup_player_names=top_lineup_names,
                roster_player_ids=roster_player_ids,
                roster_player_names=roster_player_names,
                roster_player_match_counts=roster_player_match_counts,
                player_support=player_support,
                pair_support=pair_support,
                lineup_variant_counts=lineup_variant_counts,
            )
        )
    return payloads


def _greedy_clusters(
    vectors: np.ndarray,
    team_ids: Sequence[int],
    lineup_counts: np.ndarray,
    threshold: float,
    min_cluster_size: int,
    max_teams: int,
) -> Dict[int, Dict[str, int]]:
    if vectors.size == 0:
        return {}

    order_all = np.argsort(-lineup_counts)
    selected = order_all[: max(1, min(max_teams, len(order_all)))]
    assigned = np.zeros(len(selected), dtype=bool)

    cluster_map: Dict[int, Dict[str, int]] = {}
    next_cluster_id = 0

    for pos, idx in enumerate(selected):
        if assigned[pos]:
            continue

        sims = vectors[selected] @ vectors[idx]
        member_positions = np.where((sims >= threshold) & (~assigned))[0]

        if len(member_positions) < min_cluster_size:
            assigned[pos] = True
            continue

        member_team_ids = [int(team_ids[selected[m]]) for m in member_positions]
        cluster_size = len(member_team_ids)
        for team_id in member_team_ids:
            cluster_map[team_id] = {
                "cluster_id": next_cluster_id,
                "cluster_size": cluster_size,
            }
        assigned[member_positions] = True
        next_cluster_id += 1

    return cluster_map


def _start_run(
    db_url: str,
    schema: str,
    *,
    days_window: int,
    semantic_dim: int,
    identity_dim: int,
    identity_beta: float,
    identity_idf_cap: float,
    since_ms: Optional[int],
    until_ms: Optional[int],
) -> int:
    engine = create_engine(db_url)
    sql = f"""
        INSERT INTO {schema}.team_search_refresh_runs (
            status,
            days_window,
            semantic_dim,
            identity_dim,
            identity_beta,
            identity_idf_cap,
            source_since_ms,
            source_until_ms
        ) VALUES (
            'running',
            :days_window,
            :semantic_dim,
            :identity_dim,
            :identity_beta,
            :identity_idf_cap,
            :source_since_ms,
            :source_until_ms
        )
        RETURNING run_id
    """
    with engine.begin() as conn:
        run_id = conn.execute(
            text(sql),
            {
                "days_window": int(days_window),
                "semantic_dim": int(semantic_dim),
                "identity_dim": int(identity_dim),
                "identity_beta": float(identity_beta),
                "identity_idf_cap": float(identity_idf_cap),
                "source_since_ms": since_ms,
                "source_until_ms": until_ms,
            },
        ).scalar_one()
    return int(run_id)


def _mark_run(
    db_url: str,
    schema: str,
    run_id: int,
    *,
    status: str,
    message: str,
    teams_indexed: Optional[int] = None,
    clusters_strict: Optional[int] = None,
    clusters_explore: Optional[int] = None,
) -> None:
    engine = create_engine(db_url)
    sql = f"""
        UPDATE {schema}.team_search_refresh_runs
        SET
            status = :status,
            finished_at = NOW(),
            message = :message,
            teams_indexed = :teams_indexed,
            clusters_strict = :clusters_strict,
            clusters_explore = :clusters_explore
        WHERE run_id = :run_id
    """
    with engine.begin() as conn:
        conn.execute(
            text(sql),
            {
                "run_id": int(run_id),
                "status": status,
                "message": message,
                "teams_indexed": teams_indexed,
                "clusters_strict": clusters_strict,
                "clusters_explore": clusters_explore,
            },
        )


def _persist_snapshot(
    db_url: str,
    schema: str,
    run_id: int,
    identity_beta: float,
    payloads: Sequence[TeamPayload],
    strict_clusters: Dict[int, Dict[str, int]],
    explore_clusters: Dict[int, Dict[str, int]],
    final_vector_dim: int,
) -> None:
    engine = create_engine(db_url)
    vector_columns = get_vector_column_info(engine, schema)
    use_vector_storage = bool(
        vector_columns.get("has_final_vector_vec")
        and vector_columns.get("extension_enabled")
        and vector_columns.get("final_vector_vec_dim") == int(final_vector_dim)
    )

    insert_embeddings_sql = f"""
        INSERT INTO {schema}.team_search_embeddings (
            snapshot_id,
            team_id,
            tournament_id,
            team_name,
            event_time_ms,
            lineup_count,
            tournament_count,
            unique_player_count,
            distinct_lineup_count,
            top_lineup_share,
            lineup_entropy,
            effective_lineups,
            semantic_vector,
            identity_vector,
            final_vector,
            top_lineup_summary,
            top_lineup_player_ids,
            top_lineup_player_names,
            roster_player_ids,
            roster_player_names,
            roster_player_match_counts,
            player_support,
            pair_support,
            lineup_variant_counts,
            semantic_dim,
            identity_dim,
            identity_beta
        ) VALUES (
            :snapshot_id,
            :team_id,
            :tournament_id,
            :team_name,
            :event_time_ms,
            :lineup_count,
            :tournament_count,
            :unique_player_count,
            :distinct_lineup_count,
            :top_lineup_share,
            :lineup_entropy,
            :effective_lineups,
            :semantic_vector,
            :identity_vector,
            :final_vector,
            :top_lineup_summary,
            :top_lineup_player_ids,
            :top_lineup_player_names,
            :roster_player_ids,
            :roster_player_names,
            :roster_player_match_counts,
            CAST(:player_support AS JSONB),
            CAST(:pair_support AS JSONB),
            CAST(:lineup_variant_counts AS JSONB),
            :semantic_dim,
            :identity_dim,
            :identity_beta
        )
    """

    insert_embeddings_sql_vec = f"""
        INSERT INTO {schema}.team_search_embeddings (
            snapshot_id,
            team_id,
            tournament_id,
            team_name,
            event_time_ms,
            lineup_count,
            tournament_count,
            unique_player_count,
            distinct_lineup_count,
            top_lineup_share,
            lineup_entropy,
            effective_lineups,
            semantic_vector,
            identity_vector,
            final_vector,
            top_lineup_summary,
            top_lineup_player_ids,
            top_lineup_player_names,
            roster_player_ids,
            roster_player_names,
            roster_player_match_counts,
            player_support,
            pair_support,
            lineup_variant_counts,
            semantic_dim,
            identity_dim,
            identity_beta,
            final_vector_vec
        ) VALUES (
            :snapshot_id,
            :team_id,
            :tournament_id,
            :team_name,
            :event_time_ms,
            :lineup_count,
            :tournament_count,
            :unique_player_count,
            :distinct_lineup_count,
            :top_lineup_share,
            :lineup_entropy,
            :effective_lineups,
            :semantic_vector,
            :identity_vector,
            :final_vector,
            :top_lineup_summary,
            :top_lineup_player_ids,
            :top_lineup_player_names,
            :roster_player_ids,
            :roster_player_names,
            :roster_player_match_counts,
            CAST(:player_support AS JSONB),
            CAST(:pair_support AS JSONB),
            CAST(:lineup_variant_counts AS JSONB),
            :semantic_dim,
            :identity_dim,
            :identity_beta,
            CAST(:final_vector_vec AS vector({final_vector_dim}))
        )
    """

    embed_rows = []
    for payload in payloads:
        embed_rows.append(
            {
                "snapshot_id": int(run_id),
                "team_id": payload.team_id,
                "tournament_id": payload.tournament_id,
                "team_name": payload.team_name,
                "event_time_ms": payload.event_time_ms,
                "lineup_count": payload.lineup_count,
                "tournament_count": payload.tournament_count,
                "unique_player_count": payload.unique_player_count,
                "distinct_lineup_count": payload.distinct_lineup_count,
                "top_lineup_share": payload.top_lineup_share,
                "lineup_entropy": payload.lineup_entropy,
                "effective_lineups": payload.effective_lineups,
                "semantic_vector": payload.semantic_vector.tolist(),
                "identity_vector": payload.identity_vector.tolist(),
                "final_vector": payload.final_vector.tolist(),
                "top_lineup_summary": payload.top_lineup_summary,
                "top_lineup_player_ids": list(payload.top_lineup_player_ids),
                "top_lineup_player_names": list(payload.top_lineup_player_names),
                "roster_player_ids": list(payload.roster_player_ids),
                "roster_player_names": list(payload.roster_player_names),
                "roster_player_match_counts": [
                    int(value) for value in payload.roster_player_match_counts
                ],
                "player_support": json.dumps(payload.player_support, sort_keys=True),
                "pair_support": json.dumps(payload.pair_support, sort_keys=True),
                "lineup_variant_counts": json.dumps(
                    payload.lineup_variant_counts,
                    sort_keys=True,
                ),
                "semantic_dim": int(payload.semantic_vector.shape[0]),
                "identity_dim": int(payload.identity_vector.shape[0]),
                "identity_beta": float(identity_beta),
            }
        )
        if use_vector_storage:
            embed_rows[-1]["final_vector_vec"] = _vector_literal(
                payload.final_vector.tolist()
            )

    def _cluster_rows(
        profile: str,
        cluster_map: Dict[int, Dict[str, int]],
        payload_lookup: Dict[int, TeamPayload],
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []

        rep_name_by_cluster: Dict[int, str] = {}
        for team_id, info in cluster_map.items():
            cid = int(info["cluster_id"])
            current = rep_name_by_cluster.get(cid)
            candidate = payload_lookup[team_id].team_name
            if current is None:
                rep_name_by_cluster[cid] = candidate

        for team_id, info in cluster_map.items():
            cid = int(info["cluster_id"])
            rows.append(
                {
                    "snapshot_id": int(run_id),
                    "profile": profile,
                    "cluster_id": cid,
                    "team_id": int(team_id),
                    "cluster_size": int(info["cluster_size"]),
                    "representative_team_name": rep_name_by_cluster.get(cid),
                }
            )
        return rows

    payload_lookup = {payload.team_id: payload for payload in payloads}
    cluster_rows = _cluster_rows("strict", strict_clusters, payload_lookup)
    cluster_rows.extend(_cluster_rows("explore", explore_clusters, payload_lookup))

    insert_clusters_sql = f"""
        INSERT INTO {schema}.team_search_clusters (
            snapshot_id,
            profile,
            cluster_id,
            team_id,
            cluster_size,
            representative_team_name
        ) VALUES (
            :snapshot_id,
            :profile,
            :cluster_id,
            :team_id,
            :cluster_size,
            :representative_team_name
        )
    """

    with engine.begin() as conn:
        if embed_rows:
            if use_vector_storage:
                try:
                    conn.execute(text(insert_embeddings_sql_vec), embed_rows)
                except SQLAlchemyError:
                    logger.warning(
                        "Vector-backed insert failed in %s for run %s; falling back to array insert.",
                        schema,
                        run_id,
                    )
                    conn.execute(text(insert_embeddings_sql), embed_rows)
            else:
                conn.execute(text(insert_embeddings_sql), embed_rows)
        if cluster_rows:
            conn.execute(text(insert_clusters_sql), cluster_rows)


def _cleanup_old_runs(db_url: str, schema: str, keep_runs: int) -> None:
    keep_runs = max(1, keep_runs)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        sql = f"""
            SELECT run_id
            FROM {schema}.team_search_refresh_runs
            WHERE status = 'completed'
            ORDER BY finished_at DESC NULLS LAST, run_id DESC
            OFFSET :offset
        """
        old_rows = conn.execute(text(sql), {"offset": keep_runs}).fetchall()
        old_ids = [int(r[0]) for r in old_rows]
        if not old_ids:
            return
        delete_sql = f"""
            DELETE FROM {schema}.team_search_refresh_runs
            WHERE run_id = ANY(:run_ids)
        """
        conn.execute(text(delete_sql), {"run_ids": old_ids})


def run_refresh(
    *,
    source_db_url: str,
    target_db_url: str,
    schema: str,
    days: int,
    until_days: Optional[int],
    min_players: int,
    max_players: int,
    semantic_dim: int,
    identity_dim: int,
    identity_beta: float,
    identity_idf_cap: float,
    max_cluster_teams: int,
    strict_threshold: float,
    explore_threshold: float,
    keep_runs: int,
) -> int:
    schema = validate_identifier(schema)
    engine = create_engine(target_db_url)
    final_vector_dim = int(semantic_dim) + int(identity_dim)
    try:
        ensure_search_tables(engine, schema, final_vector_dim=final_vector_dim)
    except SQLAlchemyError as exc:
        if not _is_permission_denied(exc):
            raise
        logger.warning(
            "Skipping schema bootstrap for %s during refresh due to insufficient privileges.",
            schema,
        )

    since_ms = _to_ms_days_ago(days)
    until_ms = _to_ms_days_ago(until_days)

    run_id = _start_run(
        target_db_url,
        schema,
        days_window=days,
        semantic_dim=semantic_dim,
        identity_dim=identity_dim,
        identity_beta=identity_beta,
        identity_idf_cap=identity_idf_cap,
        since_ms=since_ms,
        until_ms=until_ms,
    )

    try:
        lineup_rows = _fetch_lineups(
            source_db_url,
            schema,
            since_ms,
            until_ms,
            min_players,
            max_players,
        )
        if not lineup_rows:
            _mark_run(
                target_db_url,
                schema,
                run_id,
                status="completed",
                message="No lineups found in requested window.",
                teams_indexed=0,
                clusters_strict=0,
                clusters_explore=0,
            )
            return run_id

        (
            team_lineups,
            lineup_counts,
            team_player_counts,
            team_tournament_ids,
        ) = _index_lineups(lineup_rows)
        team_ids = sorted(team_lineups.keys())

        semantic_vectors = _build_semantic_vectors(team_lineups, semantic_dim)
        identity_vectors = _build_identity_vectors(
            team_player_counts,
            identity_dim,
            identity_idf_cap,
        )

        metadata = _fetch_team_metadata(source_db_url, schema, team_ids)
        all_player_ids = sorted(
            {
                int(player_id)
                for player_counts in team_player_counts.values()
                for player_id in player_counts.keys()
            }
        )
        player_name_by_id = _fetch_player_names(source_db_url, schema, all_player_ids)

        payloads = _build_payloads(
            team_ids,
            metadata,
            player_name_by_id,
            team_lineups,
            lineup_counts,
            team_player_counts,
            team_tournament_ids,
            semantic_vectors,
            identity_vectors,
            identity_beta,
        )
        if not payloads:
            _mark_run(
                target_db_url,
                schema,
                run_id,
                status="completed",
                message="No team payloads generated from lineups.",
                teams_indexed=0,
                clusters_strict=0,
                clusters_explore=0,
            )
            return run_id

        finals = np.stack([p.final_vector for p in payloads], axis=0)
        lineup_count_arr = np.asarray([p.lineup_count for p in payloads], dtype=np.int64)
        payload_team_ids = [p.team_id for p in payloads]

        strict_clusters = _greedy_clusters(
            vectors=finals,
            team_ids=payload_team_ids,
            lineup_counts=lineup_count_arr,
            threshold=strict_threshold,
            min_cluster_size=3,
            max_teams=max_cluster_teams,
        )
        explore_clusters = _greedy_clusters(
            vectors=finals,
            team_ids=payload_team_ids,
            lineup_counts=lineup_count_arr,
            threshold=explore_threshold,
            min_cluster_size=2,
            max_teams=max_cluster_teams,
        )

        _persist_snapshot(
            target_db_url,
            schema,
            run_id,
            identity_beta,
            payloads,
            strict_clusters,
            explore_clusters,
            final_vector_dim=final_vector_dim,
        )

        _mark_run(
            target_db_url,
            schema,
            run_id,
            status="completed",
            message="Refresh completed successfully.",
            teams_indexed=len(payloads),
            clusters_strict=len({v["cluster_id"] for v in strict_clusters.values()}),
            clusters_explore=len({v["cluster_id"] for v in explore_clusters.values()}),
        )

        _cleanup_old_runs(target_db_url, schema, keep_runs)
        return run_id

    except Exception as exc:
        _mark_run(
            target_db_url,
            schema,
            run_id,
            status="failed",
            message=str(exc),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh team-search embeddings and clusters from Postgres."
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Legacy single DB URL used for both source reads and target writes.",
    )
    parser.add_argument(
        "--source-db-url",
        type=str,
        default=None,
        help="DB URL used to read source tournament tables.",
    )
    parser.add_argument(
        "--target-db-url",
        type=str,
        default=None,
        help="DB URL used to write team-search snapshot tables.",
    )
    parser.add_argument("--schema", type=str, default=None)
    parser.add_argument("--days", type=int, default=540)
    parser.add_argument("--until-days", type=int, default=None)
    parser.add_argument("--min-players", type=int, default=4)
    parser.add_argument("--max-players", type=int, default=4)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--identity-dim", type=int, default=256)
    parser.add_argument("--identity-beta", type=float, default=3.0)
    parser.add_argument("--identity-idf-cap", type=float, default=3.0)
    parser.add_argument("--max-cluster-teams", type=int, default=2500)
    parser.add_argument("--strict-threshold", type=float, default=0.94)
    parser.add_argument("--explore-threshold", type=float, default=0.90)
    parser.add_argument("--keep-runs", type=int, default=5)
    return parser


def _resolve_refresh_db_urls(args: argparse.Namespace) -> tuple[str, str]:
    source_explicit = args.source_db_url or os.getenv("SOURCE_DATABASE_URL")
    target_explicit = args.target_db_url or os.getenv("TARGET_DATABASE_URL")

    source_db_url = (
        resolve_database_url(source_explicit) if source_explicit else None
    )
    target_db_url = (
        resolve_database_url(target_explicit) if target_explicit else None
    )

    if source_db_url is None or target_db_url is None:
        shared_db_url = resolve_database_url(args.db_url)
        source_db_url = source_db_url or shared_db_url
        target_db_url = target_db_url or shared_db_url

    return source_db_url, target_db_url


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source_db_url, target_db_url = _resolve_refresh_db_urls(args)
    schema = args.schema or get_schema()

    run_id = run_refresh(
        source_db_url=source_db_url,
        target_db_url=target_db_url,
        schema=schema,
        days=int(args.days),
        until_days=args.until_days,
        min_players=int(args.min_players),
        max_players=int(args.max_players),
        semantic_dim=int(args.semantic_dim),
        identity_dim=int(args.identity_dim),
        identity_beta=float(args.identity_beta),
        identity_idf_cap=float(args.identity_idf_cap),
        max_cluster_teams=int(args.max_cluster_teams),
        strict_threshold=float(args.strict_threshold),
        explore_threshold=float(args.explore_threshold),
        keep_runs=int(args.keep_runs),
    )
    print(f"Refresh run complete. run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
