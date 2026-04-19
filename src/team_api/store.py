from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import json
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from shared_lib.team_vector_utils import (
    build_identity_idf_lookup,
    parse_lineup_key,
    parse_pair_key,
)
from team_api.sql import get_vector_column_info, validate_identifier

logger = logging.getLogger(__name__)


def _is_missing_relation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "undefinedtable" in message
        or "does not exist" in message
        or "no such table" in message
    )


def _is_missing_column_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "undefinedcolumn" in message or "no such column" in message or (
        "column" in message and "does not exist" in message
    )


def _is_access_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "permission denied" in message
        or "insufficientprivilege" in message
        or "access denied" in message
    )


_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")
_SENDOU_ROUTE_KEY_TOURNAMENT = "features/tournament/routes/to.$id"
_SENDOU_TURBO_SPECIAL_VALUES: Dict[int, Any] = {
    -1: None,
    -2: True,
    -3: False,
    -4: float("nan"),
    -5: None,
    -6: float("inf"),
    -7: None,
    -8: float("-inf"),
    -9: -0.0,
}
_TOURNAMENT_SCORE_TIERS: tuple[tuple[float, str, str], ...] = (
    (5.0, "X", "x"),
    (10.0, "S+", "s_plus"),
    (20.0, "S", "s"),
    (40.0, "A+", "a_plus"),
    (80.0, "A", "a"),
    (160.0, "A-", "a_minus"),
)


def _normalize_query(value: str) -> str:
    return _NON_ALPHANUMERIC_RE.sub("", str(value or "").strip().lower())


def _tokenized_query(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _normalize_id_sequence(values: Any) -> List[int]:
    if values is None:
        return []

    if isinstance(values, (int, float, str)):
        values = [values]

    out: List[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, list):
            nested = _normalize_id_sequence(value)
            for team_id in nested:
                if team_id not in seen:
                    seen.add(team_id)
                    out.append(team_id)
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed <= 0:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
    return out


def _decode_turbo_stream_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, list) or not payload:
        return {}

    cache: Dict[int, Any] = {}

    def _decode_value(index: int) -> Any:
        if index in cache:
            return cache[index]
        if index < 0:
            return _SENDOU_TURBO_SPECIAL_VALUES.get(index)
        if index >= len(payload):
            return None

        raw = payload[index]
        if isinstance(raw, dict):
            decoded_object: Dict[str, Any] = {}
            for encoded_key, encoded_value_index in raw.items():
                if not isinstance(encoded_key, str) or not encoded_key.startswith("_"):
                    continue
                try:
                    key_index = int(encoded_key[1:])
                    value_index = int(encoded_value_index)
                except (TypeError, ValueError):
                    continue
                real_key = _decode_value(key_index)
                if real_key is None:
                    continue
                decoded_object[str(real_key)] = _decode_value(value_index)
            cache[index] = decoded_object
            return decoded_object

        if isinstance(raw, list):
            decoded_list: List[Any] = []
            for value in raw:
                if isinstance(value, int):
                    decoded_list.append(_decode_value(value))
                else:
                    decoded_list.append(value)
            cache[index] = decoded_list
            return decoded_list

        return raw

    header = payload[0]
    if not isinstance(header, dict):
        return {}

    decoded: Dict[str, Any] = {}
    for encoded_key, encoded_value_index in header.items():
        if not isinstance(encoded_key, str) or not encoded_key.startswith("_"):
            continue
        try:
            key_index = int(encoded_key[1:])
            value_index = int(encoded_value_index)
        except (TypeError, ValueError):
            continue
        key = _decode_value(key_index)
        if key is None:
            continue
        decoded[str(key)] = _decode_value(value_index)

    return decoded


def _extract_sendou_teams_from_turbo_payload(payload: Any) -> List[Dict[str, Any]]:
    decoded = _decode_turbo_stream_payload(payload)
    route_payload = decoded.get(_SENDOU_ROUTE_KEY_TOURNAMENT)
    if not isinstance(route_payload, dict):
        return []

    route_data = route_payload.get("data")
    if isinstance(route_data, str):
        try:
            route_data = json.loads(route_data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []

    if not isinstance(route_data, dict):
        return []

    teams = (
        route_data.get("tournament", {})
        .get("ctx", {})
        .get("teams", [])
    )
    if not isinstance(teams, list):
        return []

    out: List[Dict[str, Any]] = []
    for team in teams:
        if not isinstance(team, dict):
            continue
        team_id = team.get("id")
        team_name = str(team.get("name") or "").strip()
        member_ids: List[int] = []
        member_names: List[str] = []
        seen_member_ids: set[int] = set()
        seen_member_names: set[str] = set()
        for member in team.get("members", []) or []:
            if not isinstance(member, dict):
                continue

            member_user_id = member.get("userId")
            if isinstance(member_user_id, (int, float)):
                parsed_member_id = int(member_user_id)
                if parsed_member_id > 0 and parsed_member_id not in seen_member_ids:
                    seen_member_ids.add(parsed_member_id)
                    member_ids.append(parsed_member_id)

            for name_key in ("username", "inGameName"):
                raw_member_name = str(member.get(name_key) or "").strip()
                if not raw_member_name:
                    continue
                dedupe_member_name = raw_member_name.lower()
                if dedupe_member_name in seen_member_names:
                    continue
                seen_member_names.add(dedupe_member_name)
                member_names.append(raw_member_name)

        if team_name:
            display_name = team_name
        elif member_names:
            preview = " / ".join(member_names[:3])
            if len(member_names) > 3:
                preview += f" +{len(member_names) - 3}"
            display_name = f"[No team name] {preview}"
        elif isinstance(team_id, (int, float)):
            display_name = f"Untitled team {int(team_id)}"
        else:
            display_name = "Untitled team"

        normalized: Dict[str, Any] = {
            "team_id": int(team_id) if isinstance(team_id, (int, float)) else None,
            "team_name": team_name,
            "display_name": display_name,
            "member_user_ids": member_ids,
            "member_names": member_names,
        }
        out.append(normalized)
    return out


def _sendou_team_matches_query(
    team: Dict[str, Any],
    *,
    normalized_query: str,
    token_query: set[str],
) -> bool:
    if not normalized_query:
        return True

    candidates = []
    for key in ("team_name", "display_name"):
        raw = str(team.get(key) or "").strip()
        if raw:
            candidates.append(raw)
    for raw_member_name in team.get("member_names", []) or []:
        raw = str(raw_member_name or "").strip()
        if raw:
            candidates.append(raw)

    for candidate in candidates:
        normalized_candidate = _normalize_query(candidate)
        token_candidate = set(_tokenized_query(candidate))
        if normalized_query == normalized_candidate:
            return True
        if normalized_query and normalized_query in normalized_candidate:
            return True
        if len(normalized_query) <= 4 and normalized_query in token_candidate:
            return True
        if token_query and token_query.issubset(token_candidate):
            return True

    return False


def _tournament_tier(value: Any) -> dict[str, str]:
    parsed = None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = None

    if parsed is None or not np.isfinite(parsed):
        return {
            "tier_id": "unscored",
            "tier_label": "Unscored",
        }

    # Smaller rank indicates stronger tournaments because rank 1 is the best
    # player. Lower values are better tiers.
    for threshold, label, tier_id in _TOURNAMENT_SCORE_TIERS:
        if parsed <= threshold:
            return {
                "tier_id": tier_id,
                "tier_label": label,
            }

    return {
        "tier_id": "a_minus",
        "tier_label": "A-",
    }


@dataclass
class EmbeddingRow:
    team_id: int
    tournament_id: Optional[int]
    team_name: str
    event_time_ms: Optional[int]
    lineup_count: int
    semantic_vector: np.ndarray
    identity_vector: np.ndarray
    final_vector: np.ndarray
    top_lineup_summary: Optional[str] = None
    unique_player_count: int = 0
    distinct_lineup_count: int = 0
    top_lineup_share: float = 0.0
    lineup_entropy: float = 0.0
    effective_lineups: float = 0.0
    top_lineup_player_ids: tuple[int, ...] = ()
    top_lineup_player_names: tuple[str, ...] = ()
    roster_player_ids: tuple[int, ...] = ()
    roster_player_names: tuple[str, ...] = ()
    roster_player_match_counts: tuple[int, ...] = ()
    player_support: dict[int, float] = field(default_factory=dict)
    pair_support: dict[tuple[int, int], float] = field(default_factory=dict)
    lineup_variant_counts: dict[tuple[int, ...], int] = field(default_factory=dict)
    tournament_count: int = 0


@dataclass
class EmbeddingSnapshotCacheEntry:
    snapshot_id: int
    rows: List[EmbeddingRow]
    finals: np.ndarray
    semantics: np.ndarray
    identities: np.ndarray
    lineup_counts: np.ndarray
    index_by_team_id: Dict[int, int]
    team_ids_by_player_id: Dict[int, tuple[int, ...]]
    idf_lookup: Dict[int, float]
    semantic_dim: int
    identity_dim: int
    identity_beta: float
    identity_idf_cap: float
    built_at_ms: int


class TeamSearchStore:
    _MATCH_SCORE_CANDIDATES = (
        ("team1_score", "team2_score"),
        ("team_1_score", "team_2_score"),
        ("score1", "score2"),
        ("score_1", "score_2"),
        ("team1_points", "team2_points"),
        ("team1_goals", "team2_goals"),
    )

    def __init__(self, engine: Engine, schema: str):
        self.engine = engine
        self.schema = validate_identifier(schema)
        self._vector_capability: Optional[Dict[str, Any]] = None
        self._match_columns: Optional[set[str]] = None
        self._table_exists_cache: Dict[str, bool] = {}
        self._table_columns_cache: Dict[str, set[str]] = {}
        self._match_score_columns_cache: Optional[tuple[str, str] | None] = None
        self._player_rankings_rank_column: Optional[str] = None
        self._player_rankings_rank_column_checked: bool = False
        self._cache_lock = threading.RLock()
        self._embedding_snapshot_cache: Optional[EmbeddingSnapshotCacheEntry] = None
        self._cluster_map_cache: Dict[tuple[int, str], Dict[int, Dict[str, Any]]] = {}
        self._tournament_count_cache: Dict[tuple[int, int], int] = {}
        self._sendou_tournament_team_cache: Dict[int, List[Dict[str, Any]]] = {}
        self._has_trgm: Optional[bool] = None

    def _detect_trgm_support(self) -> bool:
        if self._has_trgm is not None:
            return self._has_trgm
        try:
            with self.engine.connect() as conn:
                self._has_trgm = bool(
                    conn.execute(
                        text(
                            "SELECT EXISTS("
                            "SELECT 1 FROM pg_extension WHERE extname='pg_trgm'"
                            ")"
                        )
                    ).scalar_one()
                )
        except SQLAlchemyError:
            self._has_trgm = False
        return self._has_trgm

    @staticmethod
    def _rank_similar_teams(
        embeddings: Sequence[EmbeddingRow],
        target_team_ids: Sequence[int],
        cluster_map: Dict[int, Dict[str, object]],
        top_n: int,
        min_relevance: float,
        candidate_top_n: Optional[int] = None,
        cache_entry: Optional[EmbeddingSnapshotCacheEntry] = None,
        recency_weight: float = 0.0,
        query_team_weights: Optional[Dict[int, float]] = None,
        query_profile: Optional[Dict[str, object]] = None,
        rerank_candidate_limit: int = 200,
        rerank_weight_embed: float = 0.70,
        rerank_weight_player: float = 0.15,
        rerank_weight_pair: float = 0.15,
        use_pair_rerank: bool = True,
    ) -> Dict[str, object]:
        from team_api.search_logic import rank_similar_teams

        use_cached_arrays = cache_entry is not None and embeddings is cache_entry.rows
        return rank_similar_teams(
            embeddings=embeddings,
            target_team_ids=target_team_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            candidate_top_n=candidate_top_n,
            min_relevance=min_relevance,
            recency_weight=recency_weight,
            query_team_weights=query_team_weights,
            query_profile=query_profile,
            rerank_candidate_limit=rerank_candidate_limit,
            rerank_weight_embed=rerank_weight_embed,
            rerank_weight_player=rerank_weight_player,
            rerank_weight_pair=rerank_weight_pair,
            use_pair_rerank=use_pair_rerank,
            precomputed_finals=cache_entry.finals if use_cached_arrays else None,
            precomputed_semantics=cache_entry.semantics if use_cached_arrays else None,
            precomputed_identities=cache_entry.identities if use_cached_arrays else None,
            precomputed_index=cache_entry.index_by_team_id if use_cached_arrays else None,
        )

    @staticmethod
    def _consolidation_candidate_top_n(
        top_n: int,
        rerank_candidate_limit: int,
    ) -> int:
        requested = max(1, int(top_n))
        limit = max(requested, int(rerank_candidate_limit))
        return min(limit, max(requested + 10, requested * 5))

    @staticmethod
    def _finalize_ranked_results(
        ranked: Dict[str, Any],
        *,
        top_n: int,
    ) -> Dict[str, Any]:
        trimmed = list(ranked.get("results") or [])[: max(1, int(top_n))]
        for rank, result in enumerate(trimmed, start=1):
            result["rank"] = rank
        ranked["results"] = trimmed
        return ranked

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return vec / norm

    @staticmethod
    def _weighted_centroid(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return np.array([], dtype=np.float64)
        if weights.size == 0 or float(weights.sum()) <= 0:
            return TeamSearchStore._normalize_vector(vectors.mean(axis=0))
        centroid = (vectors * weights[:, None]).sum(axis=0) / float(weights.sum())
        return TeamSearchStore._normalize_vector(centroid)

    def _match_table_columns(self) -> set[str]:
        if self._match_columns is None:
            try:
                with self.engine.connect() as conn:
                    self._match_columns = {
                        row["column_name"].lower()
                        for row in conn.execute(
                            text(
                                """
                                SELECT column_name
                                FROM information_schema.columns
                                WHERE table_schema = :schema
                                  AND table_name = 'matches'
                                """
                            ),
                            {"schema": self.schema},
                        ).mappings().all()
                    }
            except SQLAlchemyError:
                self._match_columns = set()
        return self._match_columns

    def _table_exists(self, table_name: str) -> bool:
        table = table_name.lower()
        if table in self._table_exists_cache:
            return self._table_exists_cache[table]

        try:
            with self.engine.connect() as conn:
                exists = bool(
                    conn.execute(
                        text(
                            """
                            SELECT EXISTS(
                                SELECT 1
                                FROM information_schema.tables
                                WHERE table_schema = :schema
                                  AND table_name = :table_name
                            )
                            """
                        ),
                        {"schema": self.schema, "table_name": table},
                    ).scalar_one()
                )
        except SQLAlchemyError:
            exists = False
        self._table_exists_cache[table] = exists
        return exists

    def _table_columns(self, table_name: str) -> set[str]:
        table = table_name.lower()
        if table in self._table_columns_cache:
            return self._table_columns_cache[table]

        if not self._table_exists(table):
            self._table_columns_cache[table] = set()
            return self._table_columns_cache[table]

        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                          AND table_name = :table_name
                        """
                    ),
                    {"schema": self.schema, "table_name": table},
                ).scalars().all()
            columns = {str(row).lower() for row in rows}
        except SQLAlchemyError:
            columns = set()
        self._table_columns_cache[table] = columns
        return columns

    @staticmethod
    def _pick_first(columns: set[str], *candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    @staticmethod
    def _quote(identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _resolve_match_score_columns(self) -> Optional[tuple[str, str]]:
        if self._match_score_columns_cache is None:
            columns = self._match_table_columns()
            self._match_score_columns_cache = None
            for col_a, col_b in self._MATCH_SCORE_CANDIDATES:
                if col_a in columns and col_b in columns:
                    self._match_score_columns_cache = (col_a, col_b)
                    break
        return self._match_score_columns_cache

    def _fetch_player_names(self, player_ids: Sequence[int]) -> Dict[int, str]:
        if not player_ids:
            return {}

        table = validate_identifier(self.schema)
        names: Dict[int, str] = {}
        with self.engine.connect() as conn:
            for index in range(0, len(player_ids), 1000):
                chunk = player_ids[index : index + 1000]
                placeholders = ", ".join(f":player_id_{idx}" for idx in range(len(chunk)))
                params = {f"player_id_{idx}": int(player_id) for idx, player_id in enumerate(chunk)}
                sql = f"""
                    SELECT player_id, display_name
                    FROM {table}.players
                    WHERE player_id IN ({placeholders})
                """
                rows = conn.execute(text(sql), params).mappings().all()
                for row in rows:
                    names[int(row["player_id"])] = str(row["display_name"] or str(row["player_id"]))
        return names

    def _fetch_match_rosters(
        self,
        rows: Sequence[dict],
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
    ) -> Dict[tuple[int, int], Dict[str, Dict[str, object]]]:
        if not rows:
            return {}

        match_ids: list[int] = []
        match_lookup: dict[tuple[int, int], dict] = {}
        team_a_set = set(int(team_id) for team_id in team_a_ids)
        team_b_set = set(int(team_id) for team_id in team_b_ids)

        for row in rows:
            match_id = row.get("match_id")
            if match_id is None:
                continue
            mid = int(match_id)
            if mid <= 0:
                continue
            match_ids.append(mid)
            tournament_id = row.get("tournament_id")
            key = (mid, int(tournament_id) if tournament_id is not None else 0)
            match_lookup[key] = {
                "team_a": {"player_ids": [], "player_names": []},
                "team_b": {"player_ids": [], "player_names": []},
            }

        if not match_ids:
            return {}

        unique_match_ids = sorted(set(match_ids))
        all_team_ids = sorted(team_a_set | team_b_set)
        if not all_team_ids:
            return match_lookup

        query = f"""
            SELECT
                pat.match_id,
                pat.tournament_id,
                pat.team_id,
                ARRAY_AGG(DISTINCT pat.player_id ORDER BY pat.player_id) AS player_ids
            FROM {self.schema}.player_appearance_teams pat
            WHERE pat.match_id IN :match_ids
              AND pat.team_id IN :team_ids
            GROUP BY pat.match_id, pat.tournament_id, pat.team_id
        """

        rosters: Dict[tuple[int, int], Dict[str, Dict[str, object]]] = dict(match_lookup)
        player_ids_for_lookup: list[int] = []

        try:
            with self.engine.connect() as conn:
                roster_rows = conn.execute(
                    text(query).bindparams(
                        bindparam("match_ids", expanding=True),
                        bindparam("team_ids", expanding=True),
                    ),
                    {"match_ids": unique_match_ids, "team_ids": all_team_ids},
                ).mappings().all()
        except SQLAlchemyError:
            return match_lookup

        for row in roster_rows:
            match_id = int(row["match_id"])
            tournament_id = row.get("tournament_id")
            key = (match_id, int(tournament_id) if tournament_id is not None else 0)
            roster = rosters.get(key)
            if roster is None:
                continue

            team_id = int(row["team_id"])
            if team_id in team_a_set:
                target = "team_a"
            elif team_id in team_b_set:
                target = "team_b"
            else:
                continue

            ids = [int(player_id) for player_id in row.get("player_ids", []) if player_id is not None]
            roster[target]["player_ids"] = ids
            player_ids_for_lookup.extend(ids)

        if player_ids_for_lookup:
            id_to_name = self._fetch_player_names(player_ids_for_lookup)
        for match_roster in rosters.values():
            for side in ("team_a", "team_b"):
                ids = match_roster[side]["player_ids"]
                match_roster[side]["player_names"] = [
                    id_to_name.get(player_id, str(player_id)) for player_id in ids
                ]

        return rosters

    def _fetch_match_rounds(
        self,
        rows: Sequence[dict],
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
    ) -> Dict[tuple[int, int], List[Dict[str, object]]]:
        if not rows:
            return {}

        def to_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def to_float(value: Any) -> Optional[float]:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if np.isfinite(parsed) else None

        def to_text(value: Any) -> Optional[str]:
            if value is None:
                return None
            normalized = str(value).strip()
            return normalized or None

        def winner_side_from_row(
            winner_team_id: Optional[int],
            winner_side: Optional[str],
            team_a_id: Optional[int],
            team_b_id: Optional[int],
        ) -> Optional[str]:
            if winner_team_id is not None:
                if team_a_id is not None and winner_team_id == team_a_id:
                    return "team_a"
                if team_b_id is not None and winner_team_id == team_b_id:
                    return "team_b"
            if isinstance(winner_side, str):
                normalized = winner_side.strip().lower()
                if normalized in ("team_a", "a", "1", "left", "left_team", "team1"):
                    return "team_a"
                if normalized in ("team_b", "b", "2", "right", "right_team", "team2"):
                    return "team_b"
            return None

        match_context: dict[int, dict[str, object]] = {}
        team_a_set = set(int(team_id) for team_id in team_a_ids)
        team_b_set = set(int(team_id) for team_id in team_b_ids)

        for row in rows:
            match_id = to_int(row.get("match_id"))
            if match_id is None or match_id <= 0:
                continue

            team1_id = to_int(row.get("team1_id"))
            team2_id = to_int(row.get("team2_id"))
            if team1_id is not None and team2_id is not None:
                if team1_id in team_a_set:
                    team_a_id = team1_id
                    team_b_id = team2_id if team2_id in team_b_set else None
                    team1_is_team_a = True
                elif team2_id in team_a_set:
                    team_a_id = team2_id
                    team_b_id = team1_id if team1_id in team_b_set else None
                    team1_is_team_a = False
                else:
                    team_a_id = None
                    team_b_id = None
                    team1_is_team_a = False
            else:
                team_a_id = None
                team_b_id = None
                team1_is_team_a = False

            tournament_id = to_int(row.get("tournament_id")) or 0

            match_context[match_id] = {
                "team1_id": team1_id,
                "team2_id": team2_id,
                "team_a_id": team_a_id,
                "team_b_id": team_b_id,
                "team1_is_team_a": bool(team1_is_team_a),
                "team1_score": to_float(row.get("team1_score")),
                "team2_score": to_float(row.get("team2_score")),
                "winner_team_id": to_int(row.get("winner_team_id")),
                "tournament_id": tournament_id,
            }

        if not match_context:
            return {}

        unique_match_ids = sorted(match_context.keys())

        # In some ranking schemas, match-level rows do not have per-map child tables.
        # A `rounds` table exists that carries per-round metadata including
        # map count/type, which is enough to expand matches into per-map rows.
        rounds_columns = self._table_columns("rounds")
        if (
            self._table_exists("rounds")
            and "round_id" in rounds_columns
            and "number" in rounds_columns
            and ("maps_count" in rounds_columns or "maps_type" in rounds_columns)
        ):
            round_sql = f"""
                SELECT
                    m.{self._quote('match_id')} AS match_id,
                    m.{self._quote('tournament_id')} AS tournament_id,
                    m.{self._quote('team1_id')} AS team1_id,
                    m.{self._quote('team2_id')} AS team2_id,
                    m.{self._quote('team1_score')} AS team1_score,
                    m.{self._quote('team2_score')} AS team2_score,
                    m.{self._quote('winner_team_id')} AS winner_team_id_raw,
                    r.{self._quote('round_id')} AS round_id,
                    r.{self._quote('number')} AS round_no,
                    r.{self._quote('maps_count')} AS maps_count,
                    r.{self._quote('maps_type')} AS map_mode
                FROM {self.schema}.{self._quote('matches')} m
                LEFT JOIN {self.schema}.{self._quote('rounds')} r
                  ON m.{self._quote('round_id')} = r.{self._quote('round_id')}
                WHERE m.{self._quote('match_id')} IN :match_ids
                ORDER BY r.{self._quote('number')} NULLS LAST
            """

            query_rows: list[dict[str, object]] = []
            try:
                with self.engine.connect() as conn:
                    query_rows = [
                        dict(row)
                        for row in conn.execute(
                            text(round_sql).bindparams(bindparam("match_ids", expanding=True)),
                            {"match_ids": unique_match_ids},
                        ).mappings().all()
                    ]
            except SQLAlchemyError as exc:
                if not (
                    _is_missing_relation_error(exc)
                    or _is_missing_column_error(exc)
                    or _is_access_error(exc)
                ):
                    raise
                query_rows = []

            if query_rows:
                out = {}
                for row in query_rows:
                    match_id = to_int(row.get("match_id"))
                    if match_id is None or match_id <= 0:
                        continue

                    context = match_context.get(match_id)
                    if context is None:
                        continue

                    tournament_id = (
                        to_int(row.get("tournament_id"))
                        or to_int(context.get("tournament_id"))
                        or 0
                    )
                    team_a_id = to_int(context.get("team_a_id"))
                    team_b_id = to_int(context.get("team_b_id"))
                    team1_is_team_a = bool(context.get("team1_is_team_a"))
                    team1_score = to_float(row.get("team1_score"))
                    team2_score = to_float(row.get("team2_score"))
                    winner_team_id = to_int(row.get("winner_team_id_raw"))

                    if team1_is_team_a:
                        team_a_score = team1_score
                        team_b_score = team2_score
                    else:
                        team_a_score = team2_score
                        team_b_score = team1_score

                    winner_side = winner_side_from_row(
                        winner_team_id,
                        None,
                        team_a_id,
                        team_b_id,
                    )
                    if winner_side is None and team_a_score is not None and team_b_score is not None:
                        if team_a_score != team_b_score:
                            winner_side = "team_a" if team_a_score > team_b_score else "team_b"

                    maps_count = to_int(row.get("maps_count")) or 1
                    if maps_count <= 0:
                        maps_count = 1
                    if maps_count > 20:
                        maps_count = 20

                    raw_round_no = to_int(row.get("round_no"))
                    round_id = to_int(row.get("round_id"))

                    for map_index in range(1, maps_count + 1):
                        map_round_no = map_index
                        if raw_round_no is not None:
                            map_round_no = raw_round_no
                        map_name = None
                        normalized_round = {
                            "round_id": round_id,
                            "round_no": map_round_no,
                            "maps_count": maps_count,
                            "map_name": map_name,
                            "map_index": map_index,
                            "map_mode": to_text(row.get("map_mode")),
                            "team_a_score": team_a_score,
                            "team_b_score": team_b_score,
                            "winner_team_id": winner_team_id,
                            "winner_side": winner_side,
                        }
                        key = (match_id, tournament_id)
                        out.setdefault(key, []).append(normalized_round)

                for rounds in out.values():
                    rounds.sort(
                        key=lambda round_row: (
                            round_row.get("round_no") is None,
                            to_int(round_row.get("round_no")) if round_row.get("round_no") is not None else 1_000_000,
                            to_int(round_row.get("map_index"))
                            if round_row.get("map_index") is not None
                            else 1_000_000,
                        )
                    )

                return out

                # Fallback to the generic table resolution below.

        candidate_tables = (
            "match_rounds",
            "match_round",
            "match_maps",
            "match_round_map",
            "match_games",
            "match_map",
            "match_game",
        )

        best_candidate: Optional[tuple[int, str, dict[str, Optional[str]]]] = None

        score_pairs = (
            ("team1_score", "team2_score"),
            ("team_1_score", "team_2_score"),
            ("score1", "score2"),
            ("score_1", "score_2"),
            ("team1_points", "team2_points"),
            ("team1_goals", "team2_goals"),
            ("team_a_score", "team_b_score"),
            ("a_score", "b_score"),
            ("left_score", "right_score"),
        )
        round_no_candidates = (
            "round_no",
            "round_number",
            "round",
            "map_no",
            "map_number",
            "number",
            "index",
        )
        map_name_candidates = (
            "map_name",
            "map",
            "stage_name",
            "name",
            "source",
            "map_label",
        )
        maps_count_candidates = (
            "maps_count",
            "maps",
            "map_count",
            "match_maps_count",
            "match_map_count",
            "game_count",
        )
        map_mode_candidates = (
            "map_mode",
            "mode",
            "game_mode",
            "map_mode_hint",
            "type",
        )
        winner_team_candidates = (
            "winner_team_id",
            "winner_id",
            "winner_side_id",
            "winning_team_id",
            "winner",
        )
        winner_side_candidates = ("winner_side", "winner", "result", "victor")

        for table_name in candidate_tables:
            if not self._table_exists(table_name):
                continue

            columns = self._table_columns(table_name)
            if "match_id" not in columns:
                continue

            round_no_col = self._pick_first(columns, *round_no_candidates)
            map_name_col = self._pick_first(columns, *map_name_candidates)
            maps_count_col = self._pick_first(columns, *maps_count_candidates)
            map_mode_col = self._pick_first(columns, *map_mode_candidates)
            winner_team_col = self._pick_first(columns, *winner_team_candidates)
            winner_side_col = self._pick_first(columns, *winner_side_candidates)
            score_a_col, score_b_col = None, None
            for left_score_col, right_score_col in score_pairs:
                if left_score_col in columns and right_score_col in columns:
                    score_a_col = left_score_col
                    score_b_col = right_score_col
                    break

            useful_bits = 0
            if round_no_col:
                useful_bits += 2
            if map_name_col or map_mode_col:
                useful_bits += 1
            if score_a_col and score_b_col:
                useful_bits += 2
            if winner_team_col or winner_side_col:
                useful_bits += 1

            if useful_bits == 0:
                continue

            candidate = {
                "table": table_name,
                "columns": columns,
                "round_no_col": round_no_col,
                "map_name_col": map_name_col,
                "maps_count_col": maps_count_col,
                "map_mode_col": map_mode_col,
                "score_a_col": score_a_col,
                "score_b_col": score_b_col,
                "winner_team_col": winner_team_col,
                "winner_side_col": winner_side_col,
            }

            if best_candidate is None or useful_bits > best_candidate[0]:
                best_candidate = (useful_bits, table_name, candidate)

        if best_candidate is None:
            return {}

        _, selected_table, candidate_info = best_candidate
        row_columns = []
        row_columns.append(f"m.{self._quote('match_id')} AS match_id")
        if "tournament_id" in (candidate_info["columns"] or {}):
            row_columns.append(f"m.{self._quote('tournament_id')} AS tournament_id")
        else:
            row_columns.append("NULL::bigint AS tournament_id")

        if candidate_info["round_no_col"]:
            row_columns.append(
                f"m.{self._quote(candidate_info['round_no_col'])} AS round_no"
            )
        if candidate_info["map_name_col"]:
            row_columns.append(
                f"m.{self._quote(candidate_info['map_name_col'])} AS map_name"
            )
        if candidate_info["maps_count_col"]:
            row_columns.append(
                f"m.{self._quote(candidate_info['maps_count_col'])} AS maps_count"
            )
        if candidate_info["map_mode_col"]:
            row_columns.append(
                f"m.{self._quote(candidate_info['map_mode_col'])} AS map_mode"
            )

        if candidate_info["score_a_col"]:
            row_columns.append(
                f"m.{self._quote(candidate_info['score_a_col'])} AS team_a_round_score_raw"
            )
            row_columns.append(
                f"m.{self._quote(candidate_info['score_b_col'])} AS team_b_round_score_raw"
            )

        if candidate_info["winner_team_col"]:
            row_columns.append(
                f"m.{self._quote(candidate_info['winner_team_col'])} AS winner_team_id_raw"
            )
        elif candidate_info["winner_side_col"]:
            row_columns.append(
                f"m.{self._quote(candidate_info['winner_side_col'])} AS winner_side_raw"
            )

        if candidate_info["columns"] and self._pick_first(candidate_info["columns"], "round_id"):
            row_columns.append(f"m.{self._quote('round_id')} AS round_id")

        order_by_expr = "NULL"
        if candidate_info["round_no_col"]:
            order_by_expr = f"m.{self._quote(candidate_info['round_no_col'])} NULLS LAST"

        sql = f"""
            SELECT
                {', '.join(row_columns)}
            FROM {self.schema}.{self._quote(selected_table)} m
            WHERE m.match_id IN :match_ids
            ORDER BY {order_by_expr}
        """

        query_rows: list[dict[str, object]]
        try:
            with self.engine.connect() as conn:
                query_rows = [
                    dict(row)
                    for row in conn.execute(
                        text(sql).bindparams(bindparam("match_ids", expanding=True)),
                        {"match_ids": unique_match_ids},
                    ).mappings().all()
                ]
        except SQLAlchemyError as exc:
            if not (
                _is_missing_relation_error(exc)
                or _is_missing_column_error(exc)
                or _is_access_error(exc)
            ):
                raise
            return {}

        if not query_rows:
            return {}

        out: Dict[tuple[int, int], List[Dict[str, object]]] = {}
        row_index_by_match: dict[tuple[int, int], int] = {}

        for row in query_rows:
            match_id = to_int(row.get("match_id"))
            if match_id is None or match_id <= 0:
                continue

            context = match_context.get(match_id)
            if context is None:
                continue

            tournament_id = to_int(row.get("tournament_id")) or to_int(context.get("tournament_id")) or 0

            team_a_id = context.get("team_a_id")
            team_a_id = to_int(team_a_id)
            team_b_id = to_int(context.get("team_b_id"))

            team1_is_team_a = bool(context.get("team1_is_team_a"))
            team1_score = to_float(row.get("team_a_round_score_raw"))
            team2_score = to_float(row.get("team_b_round_score_raw"))

            if team1_is_team_a:
                team_a_score = team1_score
                team_b_score = team2_score
            else:
                team_a_score = team2_score
                team_b_score = team1_score

            winner_team_id = to_int(row.get("winner_team_id_raw"))
            winner_side = winner_side_from_row(
                winner_team_id,
                row.get("winner_side_raw") if winner_team_id is None else None,
                team_a_id,
                team_b_id,
            )

            if winner_side is None and team_a_score is not None and team_b_score is not None:
                if team_a_score != team_b_score:
                    winner_side = "team_a" if team_a_score > team_b_score else "team_b"

            round_no = to_int(row.get("round_no"))
            key = (match_id, tournament_id)
            map_index = round_no
            if map_index is None:
                map_index = row_index_by_match.get(key, 0) + 1
            row_index_by_match[key] = map_index

            normalized_round = {
                "round_id": to_int(row.get("round_id")),
                "round_no": round_no,
                "maps_count": to_int(row.get("maps_count")),
                "map_name": to_text(row.get("map_name")),
                "map_mode": to_text(row.get("map_mode")),
                "map_index": map_index,
                "team_a_score": team_a_score,
                "team_b_score": team_b_score,
                "winner_team_id": winner_team_id,
                "winner_side": winner_side,
            }

            key = (match_id, tournament_id)
            out.setdefault(key, []).append(normalized_round)

        for rounds in out.values():
            rounds.sort(
                key=lambda round_row: (
                    round_row.get("round_no") is None,
                    to_int(round_row.get("round_no")) if round_row.get("round_no") is not None else 1_000_000,
                    to_int(round_row.get("map_index")) if round_row.get("map_index") is not None else 1_000_000,
                )
            )

        return out

    def _fetch_tournament_scores(
        self, tournament_ids: Sequence[int]
    ) -> Dict[int, float]:
        rank_field = self._player_rankings_rank_field()

        def _build_score_sql(source_table: str, source_alias: str) -> str:
            # Prefer stored player rank when available; fall back to score-derived position.
            player_rank_expr = (
                f"NULLIF(pr.{rank_field}, 0)::double precision"
                if rank_field
                else "NULL::double precision"
            )
            if rank_field is None:
                rank_filter = "pr.score IS NOT NULL"
            else:
                rank_filter = f"(pr.score IS NOT NULL OR NULLIF(pr.{rank_field}, 0) IS NOT NULL)"

            return f"""
                WITH latest_rankings AS ({latest_timestamp_sql}),
                roster AS (
                    SELECT DISTINCT
                        {source_alias}.tournament_id::bigint AS tournament_id,
                        {source_alias}.player_id::bigint AS player_id
                    FROM {self.schema}.{source_table} {source_alias}
                    WHERE {source_alias}.tournament_id IN :tournament_ids
                ),
                scored_players AS (
                    SELECT
                        r.tournament_id,
                        {player_rank_expr} AS player_rank,
                        pr.score::double precision AS score,
                        ROW_NUMBER() OVER (
                          PARTITION BY r.tournament_id
                          ORDER BY
                            CASE
                              WHEN {player_rank_expr} IS NOT NULL THEN 0
                              ELSE 1
                            END,
                            CASE
                              WHEN {player_rank_expr} IS NOT NULL THEN {player_rank_expr}
                              ELSE -pr.score
                            END ASC,
                            r.player_id ASC
                        ) AS score_rank
                    FROM roster r
                    JOIN {self.schema}.player_rankings pr
                      ON pr.player_id = r.player_id
                     AND pr.calculated_at_ms = (SELECT calculated_at_ms FROM latest_rankings)
                    WHERE {rank_filter}
                ),
                ranked_players AS (
                    SELECT
                        tournament_id,
                        COALESCE(player_rank, score_rank::double precision) AS tournament_rank,
                        score_rank,
                        COUNT(*) OVER (PARTITION BY tournament_id) AS roster_player_count
                    FROM scored_players
                )
                SELECT
                    tournament_id,
                    MAX(tournament_rank) FILTER (
                        WHERE score_rank = LEAST(10, roster_player_count)
                    ) AS tournament_score
                FROM ranked_players
                GROUP BY tournament_id
            """

        normalized_tournament_ids = sorted(set(_normalize_id_sequence(tournament_ids)))
        if not normalized_tournament_ids:
            return {}

        latest_timestamp_sql = f"""
            SELECT MAX(calculated_at_ms) AS calculated_at_ms
            FROM {self.schema}.player_rankings
        """

        roster_queries = [
            {
                "table": "roster_entries",
                "base_sql": _build_score_sql("roster_entries", "re"),
            },
            {
                "table": "player_appearance_teams",
                "base_sql": _build_score_sql("player_appearance_teams", "pat"),
            },
        ]

        tournament_scores: Dict[int, float] = {}

        def _to_float(value: Any) -> Optional[float]:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if not np.isnan(parsed) and np.isfinite(parsed) else None

        for query_info in roster_queries:
            sql = query_info["base_sql"]
            source_scores: Dict[int, float] = {}

            try:
                with self.engine.connect() as conn:
                    rows = conn.execute(
                        text(sql).bindparams(bindparam("tournament_ids", expanding=True)),
                        {"tournament_ids": normalized_tournament_ids},
                    ).mappings().all()
                for row in rows:
                    tournament_id = row.get("tournament_id")
                    score = _to_float(row.get("tournament_score"))
                    if tournament_id is not None and score is not None:
                        source_scores[int(tournament_id)] = score

                if source_scores:
                    for source_tournament_id, source_score in source_scores.items():
                        tournament_scores.setdefault(source_tournament_id, source_score)
                    logger.debug(
                        "found %d tournament score entries from %s",
                        len(source_scores),
                        query_info["table"],
                    )
                    sample_scores = list(source_scores.items())[:3]
                    if sample_scores:
                        logger.debug(
                            "sample tournament scores from %s: %s",
                            query_info["table"],
                            sample_scores,
                        )
                else:
                    logger.info(
                        "no tournament score rows found from %s for %d tournaments",
                        query_info["table"],
                        len(normalized_tournament_ids),
                    )
            except SQLAlchemyError as exc:
                if not (_is_missing_relation_error(exc) or _is_missing_column_error(exc)):
                    if not _is_access_error(exc):
                        raise
                logger.info(
                    "failing tournament score source %s for %d tournaments: %s",
                    query_info["table"],
                    len(normalized_tournament_ids),
                    exc,
                )

        if tournament_scores:
            return tournament_scores

        return {}

    def _player_rankings_has_rank_column(self) -> bool:
        return self._player_rankings_rank_field() is not None

    def _player_rankings_rank_field(self) -> str | None:
        if self._player_rankings_rank_column_checked:
            return self._player_rankings_rank_column

        self._player_rankings_rank_column_checked = True
        if self._has_column("player_rankings", "player_rank"):
            self._player_rankings_rank_column = "player_rank"
        elif self._has_column("player_rankings", "rank"):
            self._player_rankings_rank_column = "rank"
        return self._player_rankings_rank_column

    def _has_column(self, table_name: str, column_name: str) -> bool:
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                          AND table_name = :table_name
                          AND column_name = :column_name
                        LIMIT 1
                        """
                    ),
                    {
                        "schema": self.schema,
                        "table_name": table_name,
                        "column_name": column_name,
                    },
                ).fetchone()
                return rows is not None
        except SQLAlchemyError:
            return False


    def _detect_vector_support(self) -> Dict[str, Any]:
        if self._vector_capability is None:
            self._vector_capability = get_vector_column_info(self.engine, self.schema)
        return self._vector_capability

    @staticmethod
    def _vector_literal(vector: np.ndarray) -> str:
        return "[" + ",".join(f"{float(x):.10g}" for x in vector) + "]"

    @staticmethod
    def _to_db_seconds(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        iv = int(value)
        return iv // 1000 if iv > 1_000_000_000_000 else iv

    def _fetch_tournament_window(self, snapshot_id: int) -> tuple[Optional[int], Optional[int]]:
        sql = f"""
            SELECT
                source_since_ms,
                source_until_ms
            FROM {self.schema}.team_search_refresh_runs
            WHERE run_id = :snapshot_id
        """
        row = self._fetch_first_row(sql, {"snapshot_id": int(snapshot_id)})
        if not row:
            return None, None
        return (
            self._to_db_seconds(row.get("source_since_ms")),
            self._to_db_seconds(row.get("source_until_ms")),
        )

    @staticmethod
    def _seconds_expr(column_expr: str) -> str:
        return (
            "CASE\n"
            f"    WHEN {column_expr} IS NULL THEN NULL\n"
            f"    WHEN {column_expr} >= 1000000000000 THEN {column_expr} / 1000\n"
            f"    ELSE {column_expr}\n"
            "END"
        )

    def _query_tournament_counts(
        self,
        sql: str,
        team_ids: Sequence[int],
        params: Dict[str, object],
    ) -> Optional[Dict[int, int]]:
        ids = sorted({int(team_id) for team_id in team_ids})
        if not ids:
            return {}

        counts: Dict[int, int] = {}
        try:
            with self.engine.connect() as conn:
                for offset in range(0, len(ids), 500):
                    chunk = ids[offset : offset + 500]
                    rows = conn.execute(
                        text(sql).bindparams(bindparam("team_ids", expanding=True)),
                        {"team_ids": chunk, **params},
                    ).mappings().all()
                    for row in rows:
                        counts[int(row["team_id"])] = int(row["tournament_count"])
        except SQLAlchemyError as exc:
            if (
                _is_missing_relation_error(exc)
                or _is_missing_column_error(exc)
                or _is_access_error(exc)
            ):
                return None
            raise

        return counts

    def _fetch_tournament_counts(self, snapshot_id: int, team_ids: Sequence[int]) -> Dict[int, int]:
        ids = sorted({int(team_id) for team_id in team_ids})
        if not ids:
            return {}

        since_ts, until_ts = self._fetch_tournament_window(snapshot_id)
        match_last_game = self._seconds_expr("m.last_game_finished_at_ms")
        match_created = self._seconds_expr("m.created_at_ms")
        match_start = self._seconds_expr("t.start_time_ms")
        match_time_expr = f"COALESCE({match_last_game}, {match_created}, {match_start})"

        filters: list[str] = []
        params: Dict[str, object] = {}
        if since_ts is not None:
            filters.append(f"{match_time_expr} >= :source_since_ts")
            params["source_since_ts"] = int(since_ts)
        if until_ts is not None:
            filters.append(f"{match_time_expr} <= :source_until_ts")
            params["source_until_ts"] = int(until_ts)

        match_where = ""
        if filters:
            match_where = " AND " + " AND ".join(filters)

        tournament_time_expr = (
            f"COALESCE({self._seconds_expr('tt.start_time_ms')}, "
            f"{self._seconds_expr('tt.created_at_ms')})"
        )
        tournament_filters: list[str] = []
        if since_ts is not None:
            tournament_filters.append(f"{tournament_time_expr} >= :source_since_ts")
        if until_ts is not None:
            tournament_filters.append(f"{tournament_time_expr} <= :source_until_ts")
        tournament_where = ""
        if tournament_filters:
            tournament_where = " AND " + " AND ".join(tournament_filters)

        primary_sql = f"""
            SELECT
                pat.team_id,
                COUNT(DISTINCT pat.tournament_id)::int AS tournament_count
            FROM {self.schema}.player_appearance_teams pat
            JOIN {self.schema}.matches m
              ON m.match_id = pat.match_id
             AND m.tournament_id = pat.tournament_id
            JOIN {self.schema}.tournaments t
              ON t.tournament_id = pat.tournament_id
            WHERE pat.team_id IN :team_ids
              {match_where}
            GROUP BY pat.team_id
        """
        primary_unbounded_sql = f"""
            SELECT
                pat.team_id,
                COUNT(DISTINCT pat.tournament_id)::int AS tournament_count
            FROM {self.schema}.player_appearance_teams pat
            JOIN {self.schema}.matches m
              ON m.match_id = pat.match_id
             AND m.tournament_id = pat.tournament_id
            JOIN {self.schema}.tournaments t
              ON t.tournament_id = pat.tournament_id
            WHERE pat.team_id IN :team_ids
            GROUP BY pat.team_id
        """
        fallback_sql = f"""
            SELECT
                tt.team_id,
                COUNT(DISTINCT tt.tournament_id)::int AS tournament_count
            FROM {self.schema}.tournament_teams tt
            JOIN {self.schema}.tournaments tt_t
              ON tt_t.tournament_id = tt.tournament_id
            WHERE tt.team_id IN :team_ids
              {tournament_where}
            GROUP BY tt.team_id
        """
        fallback_unbounded_sql = f"""
            SELECT
                tt.team_id,
                COUNT(DISTINCT tt.tournament_id)::int AS tournament_count
            FROM {self.schema}.tournament_teams tt
            WHERE tt.team_id IN :team_ids
            GROUP BY tt.team_id
        """

        query_plan = (
            (primary_sql, params),
            (primary_unbounded_sql, {}),
            (fallback_sql, params),
            (fallback_unbounded_sql, {}),
        )

        resolved: Dict[int, int] = {}
        for query_sql, query_params in query_plan:
            missing = [team_id for team_id in ids if team_id not in resolved]
            if not missing:
                return resolved

            result = self._query_tournament_counts(
                query_sql,
                missing,
                query_params,
            )
            if result is None:
                continue
            resolved.update(result)

        return resolved

    def _hydrate_tournament_counts(
        self, snapshot_id: int, rows: List[Dict[str, Any]]
    ) -> None:
        sid = int(snapshot_id)
        missing: List[int] = []
        cached_updates = 0
        for row in rows:
            if int(row.get("tournament_count") or 0) > 0:
                continue
            team_id = int(row["team_id"])
            cache_key = (sid, team_id)
            with self._cache_lock:
                cached_count = self._tournament_count_cache.get(cache_key)
            if cached_count is not None and int(cached_count) > 0:
                row["tournament_count"] = int(cached_count)
                cached_updates += 1
            else:
                missing.append(team_id)

        if cached_updates:
            logger.debug(
                "tournament_count_cache_hit schema=%s snapshot_id=%s teams=%s",
                self.schema,
                sid,
                cached_updates,
            )

        if not missing:
            return

        start = time.perf_counter()
        hydrated = self._fetch_tournament_counts(sid, missing)
        fetch_ms = (time.perf_counter() - start) * 1000.0
        if hydrated:
            with self._cache_lock:
                for team_id, tournament_count in hydrated.items():
                    self._tournament_count_cache[(sid, int(team_id))] = int(
                        tournament_count
                    )
        logger.debug(
            "tournament_count_hydrate schema=%s snapshot_id=%s queried=%s resolved=%s fetch_ms=%.1f",
            self.schema,
            sid,
            len(missing),
            len(hydrated),
            fetch_ms,
        )

        if not hydrated:
            # As a safe fallback, at least surface teams with match history as having
            # participated in one tournament when exact tournament counting can’t be resolved.
            for row in rows:
                if (
                    int(row.get("tournament_count") or 0) <= 0
                    and int(row.get("lineup_count") or 0) > 0
                ):
                    row["tournament_count"] = 1
            return

        for row in rows:
            team_id = int(row["team_id"])
            if int(row.get("tournament_count") or 0) <= 0 and team_id in hydrated:
                row["tournament_count"] = int(hydrated[team_id])

        for row in rows:
            if (
                int(row.get("tournament_count") or 0) <= 0
                and int(row.get("lineup_count") or 0) > 0
            ):
                row["tournament_count"] = 1

    def _fetch_embeddings_by_team_ids(
        self,
        snapshot_id: int,
        team_ids: Sequence[int],
    ) -> List[EmbeddingRow]:
        if not team_ids:
            return []

        unique_ids = sorted({int(tid) for tid in team_ids})
        if not unique_ids:
            return []

        sql_with_roster = f"""
            SELECT
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
                lineup_variant_counts
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND team_id IN :team_ids
        """
        sql_with_players = f"""
            SELECT
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
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND team_id IN :team_ids
        """
        sql_legacy = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND team_id IN :team_ids
        """
        sql_core = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND team_id IN :team_ids
        """

        with self.engine.connect() as conn:
            try:
                rows = conn.execute(
                    text(sql_with_roster).bindparams(bindparam("team_ids", expanding=True)),
                    {"snapshot_id": int(snapshot_id), "team_ids": unique_ids},
                ).mappings().all()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc):
                    logger.warning(
                        "Team-search embedding table missing in schema %s",
                        self.schema,
                    )
                    return []
                if _is_missing_column_error(exc):
                    try:
                        rows = conn.execute(
                            text(sql_with_players).bindparams(bindparam("team_ids", expanding=True)),
                            {"snapshot_id": int(snapshot_id), "team_ids": unique_ids},
                        ).mappings().all()
                    except SQLAlchemyError as nested_exc:
                        if not _is_missing_column_error(nested_exc):
                            raise
                        try:
                            rows = conn.execute(
                                text(sql_legacy).bindparams(
                                    bindparam("team_ids", expanding=True)
                                ),
                                {
                                    "snapshot_id": int(snapshot_id),
                                    "team_ids": unique_ids,
                                },
                            ).mappings().all()
                        except SQLAlchemyError as legacy_exc:
                            if _is_missing_column_error(legacy_exc):
                                rows = conn.execute(
                                    text(sql_core).bindparams(
                                        bindparam("team_ids", expanding=True)
                                    ),
                                    {
                                        "snapshot_id": int(snapshot_id),
                                        "team_ids": unique_ids,
                                    },
                                ).mappings().all()
                            else:
                                raise
                else:
                    raise

        out: List[EmbeddingRow] = []
        row_dicts = [dict(row) for row in rows]
        self._hydrate_tournament_counts(snapshot_id, row_dicts)

        for r in row_dicts:
            r.setdefault("top_lineup_player_ids", [])
            r.setdefault("top_lineup_player_names", [])
            r.setdefault("roster_player_ids", [])
            r.setdefault("roster_player_names", [])
            r.setdefault("roster_player_match_counts", [])
            r.setdefault("player_support", {})
            r.setdefault("pair_support", {})
            r.setdefault("lineup_variant_counts", {})
            out.append(self._row_to_embedding(r))
        return out

    def _query_vector_rank(
        self,
        snapshot_id: int,
        query_rows: Optional[List[EmbeddingRow]],
        target_team_ids: List[int],
        cluster_map: Dict[int, Dict[str, object]],
        top_n: int,
        min_relevance: float,
        candidate_top_n: Optional[int] = None,
        recency_weight: float = 0.0,
        query_team_weights: Optional[Dict[int, float]] = None,
        query_profile: Optional[Dict[str, object]] = None,
        rerank_candidate_limit: int = 200,
        rerank_weight_embed: float = 0.70,
        rerank_weight_player: float = 0.15,
        rerank_weight_pair: float = 0.15,
        use_pair_rerank: bool = True,
    ) -> Optional[Dict[str, object]]:
        if query_profile is None and not query_rows:
            return None

        from team_api.search_logic import build_query_similarity_profile

        vector_info = self._detect_vector_support()
        if not (
            vector_info.get("extension_enabled")
            and vector_info.get("has_final_vector_vec")
            and isinstance(vector_info.get("final_vector_vec_dim"), int)
        ):
            return None

        if query_profile is None:
            query_finals = np.stack([row.final_vector for row in query_rows], axis=0)
            query_semantics = np.stack([row.semantic_vector for row in query_rows], axis=0)
            query_identities = np.stack([row.identity_vector for row in query_rows], axis=0)
            query_profile = build_query_similarity_profile(
                query_rows,
                list(range(len(query_rows))),
                query_finals,
                query_semantics,
                query_identities,
                query_team_weights=query_team_weights,
            )
        else:
            query_profile = dict(query_profile)
        ann_query_vector = np.asarray(
            query_profile["ann_query_vector"],
            dtype=np.float64,
        )
        if ann_query_vector.size == 0:
            return None
        query_semantic_weight = float(query_profile.get("semantic_weight") or 0.5)

        vector_dim = int(vector_info["final_vector_vec_dim"])
        target_dim = int(ann_query_vector.size)
        if vector_dim != target_dim:
            logger.warning(
                "Skipping ANN query in %s due vector dimension mismatch: db=%s query=%s",
                self.schema,
                vector_dim,
                target_dim,
            )
            return None

        selection_top_n = (
            max(int(top_n), int(candidate_top_n))
            if candidate_top_n is not None
            else int(top_n)
        )
        candidate_limit = max(
            160,
            min(
                4000,
                selection_top_n * int(round(40.0 * (1.0 + query_semantic_weight))),
            ),
        )
        sql_with_roster = f"""
            SELECT
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
                lineup_variant_counts
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """
        sql_with_players = f"""
            SELECT
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
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """
        sql_without_tournament_count_with_roster = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
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
                lineup_variant_counts
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """
        sql_without_tournament_count_with_players = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
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
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """
        sql_core = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """

        with self.engine.connect() as conn:
            params = {
                "snapshot_id": int(snapshot_id),
                "query_vector": self._vector_literal(ann_query_vector),
                "candidate_limit": int(candidate_limit),
            }
            query_plan = (
                sql_with_roster,
                sql_with_players,
                sql_without_tournament_count_with_roster,
                sql_without_tournament_count_with_players,
                sql_core,
            )
            candidate_rows = None
            for query_sql in query_plan:
                try:
                    candidate_rows = conn.execute(
                        text(query_sql),
                        params,
                    ).mappings().all()
                    break
                except SQLAlchemyError as exc:
                    if _is_missing_column_error(exc):
                        continue
                    logger.warning(
                        "Vector ANN query failed in %s; falling back to in-memory search: %s",
                        self.schema,
                        exc,
                    )
                    return None

            if candidate_rows is None:
                return None

        candidates: List[EmbeddingRow] = []
        seen: set[int] = set()
        candidate_dicts = [dict(row) for row in candidate_rows]
        self._hydrate_tournament_counts(snapshot_id, candidate_dicts)
        for r in candidate_dicts:
            r.setdefault("top_lineup_player_ids", [])
            r.setdefault("top_lineup_player_names", [])
            r.setdefault("roster_player_ids", [])
            r.setdefault("roster_player_names", [])
            r.setdefault("roster_player_match_counts", [])
            r.setdefault("player_support", {})
            r.setdefault("pair_support", {})
            r.setdefault("lineup_variant_counts", {})
            item = self._row_to_embedding(r)
            seen.add(item.team_id)
            candidates.append(item)

        for row in query_rows or []:
            if int(row.team_id) not in seen:
                candidates.append(row)
                seen.add(int(row.team_id))

        if not candidates:
            return None

        ranked = self._rank_similar_teams(
            embeddings=candidates,
            target_team_ids=target_team_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            candidate_top_n=candidate_top_n,
            min_relevance=min_relevance,
            recency_weight=recency_weight,
            query_team_weights=query_team_weights,
            query_profile=query_profile,
            rerank_candidate_limit=rerank_candidate_limit,
            rerank_weight_embed=rerank_weight_embed,
            rerank_weight_player=rerank_weight_player,
            rerank_weight_pair=rerank_weight_pair,
            use_pair_rerank=use_pair_rerank,
        )
        if ranked["results"]:
            return ranked

        # Query centroid was built from only target teams. If top_n filtering removed all results,
        # retry with a very loose threshold on the ANN candidates to avoid empty outputs from
        # aggressive defaults.
        return self._rank_similar_teams(
            embeddings=candidates,
            target_team_ids=target_team_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            candidate_top_n=candidate_top_n,
            min_relevance=0.0,
            recency_weight=recency_weight,
            query_team_weights=query_team_weights,
            query_profile=query_profile,
            rerank_candidate_limit=rerank_candidate_limit,
            rerank_weight_embed=rerank_weight_embed,
            rerank_weight_player=rerank_weight_player,
            rerank_weight_pair=rerank_weight_pair,
            use_pair_rerank=use_pair_rerank,
        )

    @staticmethod
    def _build_seed_player_query_profile(
        cache_entry: EmbeddingSnapshotCacheEntry,
        player_ids: Sequence[int],
    ) -> Optional[Dict[str, object]]:
        seed_player_ids = _normalize_id_sequence(player_ids)
        if not seed_player_ids:
            return None

        from team_api.search_logic import build_player_query_profile

        return build_player_query_profile(
            seed_player_ids,
            semantic_dim=cache_entry.semantic_dim,
            identity_dim=cache_entry.identity_dim,
            identity_beta=cache_entry.identity_beta,
            idf_lookup=cache_entry.idf_lookup,
        )

    @staticmethod
    def _player_seed_query_profile(
        query_rows: Sequence[EmbeddingRow],
        player_ids: Sequence[int],
    ) -> tuple[List[EmbeddingRow], Optional[Dict[int, float]], Dict[str, int]]:
        seed_player_ids = _normalize_id_sequence(player_ids)
        if not query_rows or not seed_player_ids:
            return list(query_rows), None, {}

        seed_set = set(seed_player_ids)
        metrics: List[Dict[str, Any]] = []
        for row in query_rows:
            roster_ids = set(
                _normalize_id_sequence(row.roster_player_ids or row.top_lineup_player_ids)
            )
            if not roster_ids:
                continue

            overlap_count = len(seed_set & roster_ids)
            if overlap_count <= 0:
                continue

            roster_size = max(1, len(roster_ids))
            seed_size = max(1, len(seed_set))
            precision = overlap_count / float(roster_size)
            recall = overlap_count / float(seed_size)
            denom = precision + recall
            f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0

            # Strongly favor near-complete roster matches while still allowing
            # one-player drift for larger friend groups.
            score = (f1 * f1) * (1.0 + recall)
            metrics.append(
                {
                    "row": row,
                    "overlap_count": int(overlap_count),
                    "score": float(score),
                    "lineup_count": int(row.lineup_count or 0),
                }
            )

        if not metrics:
            return list(query_rows), None, {}

        metrics.sort(
            key=lambda item: (
                -int(item["overlap_count"]),
                -float(item["score"]),
                -int(item["lineup_count"]),
                int(item["row"].team_id),
            )
        )
        max_overlap = int(metrics[0]["overlap_count"])
        min_overlap = max(1, max_overlap - 1)
        max_score = float(metrics[0]["score"])

        strong = [
            item
            for item in metrics
            if int(item["overlap_count"]) >= min_overlap
            and float(item["score"]) >= (max_score * 0.40)
        ]
        selected = strong[:24] if strong else metrics[:12]

        selected_rows = [item["row"] for item in selected]
        query_team_weights = {
            int(item["row"].team_id): max(float(item["score"]), 1e-6)
            for item in selected
        }
        metadata = {
            "seed_player_max_overlap": max_overlap,
            "seed_player_query_team_count": len(selected_rows),
        }
        return selected_rows, query_team_weights, metadata

    @staticmethod
    def _match_targets_by_player_ids_from_cache(
        cache_entry: EmbeddingSnapshotCacheEntry,
        player_ids: Sequence[int],
        *,
        limit: int = 240,
    ) -> List[int]:
        normalized_player_ids = _normalize_id_sequence(player_ids)
        if not normalized_player_ids or not cache_entry.rows:
            return []

        overlap_count_by_team_id: Dict[int, int] = {}
        for player_id in normalized_player_ids:
            for team_id in cache_entry.team_ids_by_player_id.get(int(player_id), ()):
                overlap_count_by_team_id[int(team_id)] = (
                    overlap_count_by_team_id.get(int(team_id), 0) + 1
                )

        if not overlap_count_by_team_id:
            return []

        ordered_team_ids = sorted(
            overlap_count_by_team_id.keys(),
            key=lambda team_id: (
                -int(overlap_count_by_team_id[team_id]),
                -int(
                    cache_entry.rows[cache_entry.index_by_team_id[int(team_id)]].lineup_count
                ),
                int(team_id),
            ),
        )
        max_limit = max(1, min(int(limit), 2000))
        return ordered_team_ids[:max_limit]

    def _project_seed_players_to_proxy_teams(
        self,
        snapshot_id: int,
        player_ids: Sequence[int],
        *,
        subset_size: int = 4,
        per_subset_limit: int = 12,
        max_subsets: int = 96,
        final_limit: int = 64,
    ) -> tuple[List[EmbeddingRow], Optional[Dict[int, float]], Dict[str, int]]:
        normalized_player_ids = _normalize_id_sequence(player_ids)
        if len(normalized_player_ids) <= max(1, int(subset_size)):
            return [], None, {}

        cache_entry = self._get_cached_snapshot_entry(snapshot_id)
        if not cache_entry.rows:
            return [], None, {}

        subset_k = max(1, min(int(subset_size), len(normalized_player_ids)))
        total_possible_subsets = math.comb(len(normalized_player_ids), subset_k)
        team_score_by_id: Dict[int, float] = {}
        team_support_by_id: Dict[int, int] = {}
        subsets_processed = 0
        subsets_with_hits = 0

        for subset in combinations(normalized_player_ids, subset_k):
            if subsets_processed >= max_subsets:
                break
            subsets_processed += 1

            candidate_team_ids = self._match_targets_by_player_ids_from_cache(
                cache_entry,
                subset,
                limit=per_subset_limit,
            )
            if not candidate_team_ids:
                continue

            candidate_rows = [
                cache_entry.rows[cache_entry.index_by_team_id[int(team_id)]]
                for team_id in candidate_team_ids
                if int(team_id) in cache_entry.index_by_team_id
            ]
            subset_rows, subset_weights, _ = self._player_seed_query_profile(
                candidate_rows,
                subset,
            )
            if not subset_rows or not subset_weights:
                continue

            max_subset_weight = max(float(weight) for weight in subset_weights.values())
            if max_subset_weight <= 0.0:
                continue

            subsets_with_hits += 1
            for row in subset_rows:
                team_id = int(row.team_id)
                normalized_weight = (
                    float(subset_weights.get(team_id, 0.0)) / max_subset_weight
                )
                if normalized_weight <= 0.0:
                    continue
                team_score_by_id[team_id] = (
                    team_score_by_id.get(team_id, 0.0) + normalized_weight
                )
                team_support_by_id[team_id] = team_support_by_id.get(team_id, 0) + 1

        if not team_score_by_id:
            return [], None, {
                "seed_player_projection_subset_size": subset_k,
                "seed_player_projection_subset_count": subsets_processed,
                "seed_player_projection_hit_count": 0,
                "seed_player_projection_team_count": 0,
                "seed_player_projection_total_possible_subsets": total_possible_subsets,
            }

        ordered_team_ids = sorted(
            team_score_by_id.keys(),
            key=lambda team_id: (
                -float(team_score_by_id[team_id]),
                -int(team_support_by_id.get(team_id, 0)),
                -int(
                    cache_entry.rows[cache_entry.index_by_team_id[int(team_id)]].lineup_count
                ),
                int(team_id),
            ),
        )

        top_score = float(team_score_by_id[ordered_team_ids[0]])
        selected_team_ids = [
            int(team_id)
            for team_id in ordered_team_ids
            if float(team_score_by_id[team_id]) >= max(0.5, top_score * 0.25)
        ][: max(1, int(final_limit))]

        query_rows = [
            cache_entry.rows[cache_entry.index_by_team_id[int(team_id)]]
            for team_id in selected_team_ids
        ]
        query_team_weights = {
            int(team_id): float(team_score_by_id[team_id])
            for team_id in selected_team_ids
        }
        metadata = {
            "seed_player_projection_subset_size": subset_k,
            "seed_player_projection_subset_count": subsets_processed,
            "seed_player_projection_hit_count": subsets_with_hits,
            "seed_player_projection_team_count": len(selected_team_ids),
            "seed_player_projection_total_possible_subsets": total_possible_subsets,
        }
        if subsets_processed < total_possible_subsets:
            metadata["seed_player_projection_truncated"] = 1
        return query_rows, query_team_weights, metadata

    def search_similar_teams(
        self,
        snapshot_id: int,
        query: str,
        top_n: int,
        min_relevance: float,
        cluster_mode: str,
        include_clusters: bool,
        consolidate: bool = True,
        consolidate_min_overlap: float = 0.8,
        tournament_id: Optional[int] = None,
        seed_player_ids: Optional[Sequence[int]] = None,
        recency_weight: float = 0.0,
        query_mode: str = "whole_set",
        use_pair_rerank: bool = True,
        use_cluster_profile_scoring: bool = True,
        rerank_candidate_limit: int = 200,
        rerank_weight_embed: float = 0.70,
        rerank_weight_player: float = 0.15,
        rerank_weight_pair: float = 0.15,
    ) -> Dict[str, object]:
        from team_api.search_logic import (
            consolidate_ranked_results,
            rescore_consolidated_results,
            strip_internal_result_fields,
        )

        start_total = time.perf_counter()
        query_tournament_id = (
            int(tournament_id) if tournament_id is not None else None
        )
        explicit_seed_player_ids = _normalize_id_sequence(seed_player_ids or [])
        normalized_query_mode = (
            "subset_enum"
            if str(query_mode or "").strip().lower() == "subset_enum"
            else "whole_set"
        )
        query_source = "dataset"
        sendou_name_matches: List[str] = []
        sendou_player_ids_used: List[int] = []
        resolved_query_player_ids = list(explicit_seed_player_ids)
        query_team_weights: Optional[Dict[int, float]] = None
        seed_query_metadata: Dict[str, int] = {}
        direct_query_profile: Optional[Dict[str, object]] = None
        target_ids = self.match_targets(
            snapshot_id=snapshot_id,
            query=query,
            limit=max(1, min(int(top_n), 60)),
            tournament_id=query_tournament_id,
        )
        cluster_map = (
            self._get_cached_cluster_map(snapshot_id, cluster_mode)
            if include_clusters
            else {}
        )

        query_rows = self._fetch_embeddings_by_team_ids(snapshot_id, target_ids)
        if not query_rows and (
            query_tournament_id is not None or explicit_seed_player_ids
        ):
            sendou_player_ids = list(explicit_seed_player_ids)
            used_explicit_player_seed = bool(explicit_seed_player_ids)
            if query_tournament_id is not None:
                sendou_team_matches = self._match_sendou_tournament_teams(
                    query_tournament_id,
                    query,
                    limit=max(1, min(int(top_n) * 12, 700)),
                )
                sendou_name_matches = [
                    str(team.get("team_name") or team.get("display_name") or "").strip()
                    for team in sendou_team_matches
                    if str(team.get("team_name") or team.get("display_name") or "").strip()
                ]
                sendou_player_ids = _normalize_id_sequence(
                    [
                        *sendou_player_ids,
                        *[
                            player_id
                            for team in sendou_team_matches
                            for player_id in (team.get("member_user_ids") or [])
                        ],
                    ]
                )
                resolved_query_player_ids = list(sendou_player_ids)

            fallback_target_ids: List[int] = []
            if sendou_name_matches:
                for team_name in sendou_name_matches:
                    fallback_target_ids.extend(
                        self.match_targets(
                            snapshot_id=snapshot_id,
                            query=team_name,
                            limit=12,
                        )
                    )
            used_name_fallback = bool(fallback_target_ids)

            if sendou_player_ids:
                if normalized_query_mode == "whole_set" and not target_ids:
                    sendou_player_ids_used = sendou_player_ids
                else:
                    player_seed_ids = self._match_targets_by_player_ids(
                        snapshot_id=snapshot_id,
                        player_ids=sendou_player_ids,
                        limit=max(120, int(top_n) * 24),
                    )
                    if player_seed_ids:
                        sendou_player_ids_used = sendou_player_ids
                        fallback_target_ids.extend(player_seed_ids)
            used_player_fallback = bool(sendou_player_ids_used)

            deduped_target_ids = _normalize_id_sequence(fallback_target_ids)
            if deduped_target_ids:
                query_rows = self._fetch_embeddings_by_team_ids(
                    snapshot_id,
                    deduped_target_ids[:240],
                )
            if used_player_fallback:
                query_source = (
                    "seed_players"
                    if used_explicit_player_seed
                    else "sendou_players"
                )
            elif used_name_fallback:
                query_source = "sendou_names"
            elif query_tournament_id is not None:
                query_source = "sendou"

        player_seed_ids_for_weighting = (
            list(resolved_query_player_ids)
            if resolved_query_player_ids
            else list(sendou_player_ids_used)
        )
        cache_entry: Optional[EmbeddingSnapshotCacheEntry] = None
        if player_seed_ids_for_weighting and normalized_query_mode == "whole_set":
            cache_entry = self._get_cached_snapshot_entry(snapshot_id)
            direct_query_profile = self._build_seed_player_query_profile(
                cache_entry,
                player_seed_ids_for_weighting,
            )
            seed_query_metadata["seed_player_projection_subset_count"] = 0
            seed_query_metadata["subset_enumeration_count"] = 0
        elif not target_ids and len(player_seed_ids_for_weighting) > 4:
            (
                projected_rows,
                projected_weights,
                projected_metadata,
            ) = self._project_seed_players_to_proxy_teams(
                snapshot_id,
                player_seed_ids_for_weighting,
            )
            if projected_rows and projected_weights:
                query_rows = projected_rows
                query_team_weights = projected_weights
                seed_query_metadata.update(projected_metadata)
                seed_query_metadata["subset_enumeration_count"] = int(
                    projected_metadata.get("seed_player_projection_subset_count", 0)
                )
                if explicit_seed_player_ids:
                    query_source = "seed_player_subsets"
                elif sendou_player_ids_used:
                    query_source = "sendou_player_subsets"

        if (
            normalized_query_mode == "subset_enum"
            and query_rows
            and player_seed_ids_for_weighting
            and query_team_weights is None
        ):
            query_rows, query_team_weights, seed_query_metadata = self._player_seed_query_profile(
                query_rows,
                player_seed_ids_for_weighting,
            )

        if not query_rows and direct_query_profile is None:
            query_context: Dict[str, Any] = {
                "matched_team_ids": [],
                "matched_team_names": [],
                "query_mode": normalized_query_mode,
            }
            if query_tournament_id is not None:
                query_context["tournament_id"] = query_tournament_id
                query_context["tournament_source"] = query_source
                query_context["tournament_team_name_matches"] = sendou_name_matches
                query_context["tournament_player_id_count"] = len(sendou_player_ids_used)
            query_context["seed_player_id_count"] = len(explicit_seed_player_ids)
            query_context.update(seed_query_metadata)
            logger.debug(
                "team_search schema=%s snapshot_id=%s mode=empty query=%r elapsed_ms=%.1f",
                self.schema,
                int(snapshot_id),
                query,
                (time.perf_counter() - start_total) * 1000.0,
            )
            return {
                "query": query_context,
                "results": [],
            }

        target_ids = [int(row.team_id) for row in query_rows]
        consolidation_candidate_top_n = (
            self._consolidation_candidate_top_n(top_n, rerank_candidate_limit)
            if consolidate
            else top_n
        )

        ranked = self._query_vector_rank(
            snapshot_id=snapshot_id,
            query_rows=query_rows or None,
            target_team_ids=target_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            candidate_top_n=consolidation_candidate_top_n,
            min_relevance=min_relevance,
            recency_weight=recency_weight,
            query_team_weights=query_team_weights,
            query_profile=direct_query_profile,
            rerank_candidate_limit=rerank_candidate_limit,
            rerank_weight_embed=rerank_weight_embed,
            rerank_weight_player=rerank_weight_player,
            rerank_weight_pair=rerank_weight_pair,
            use_pair_rerank=use_pair_rerank,
        )

        if ranked is not None:
            # Adaptive relevance for ANN path.
            if (
                len(ranked.get("results", [])) < 3
                and min_relevance > 0.5
            ):
                relaxed = self._query_vector_rank(
                    snapshot_id=snapshot_id,
                    query_rows=query_rows or None,
                    target_team_ids=target_ids,
                    cluster_map=cluster_map,
                    top_n=top_n,
                    candidate_top_n=consolidation_candidate_top_n,
                    min_relevance=max(0.5, min_relevance - 0.2),
                    recency_weight=recency_weight,
                    query_team_weights=query_team_weights,
                    query_profile=direct_query_profile,
                    rerank_candidate_limit=rerank_candidate_limit,
                    rerank_weight_embed=rerank_weight_embed,
                    rerank_weight_player=rerank_weight_player,
                    rerank_weight_pair=rerank_weight_pair,
                    use_pair_rerank=use_pair_rerank,
                )
                if relaxed is not None and len(
                    relaxed.get("results", [])
                ) > len(ranked.get("results", [])):
                    ranked = relaxed
                    ranked.setdefault("query", {})["relevance_relaxed"] = True

            if consolidate:
                ranked["results"] = consolidate_ranked_results(
                    ranked["results"],
                    min_overlap=consolidate_min_overlap,
                )
                if direct_query_profile is not None and use_cluster_profile_scoring:
                    ranked["results"] = rescore_consolidated_results(
                        ranked["results"],
                        query_profile=direct_query_profile,
                        rerank_weight_embed=rerank_weight_embed,
                        rerank_weight_player=rerank_weight_player,
                        rerank_weight_pair=rerank_weight_pair,
                        use_pair_rerank=use_pair_rerank,
                        recency_weight=recency_weight,
                    )
            ranked = self._finalize_ranked_results(ranked, top_n=top_n)
            if query_tournament_id is not None:
                ranked_query = ranked.setdefault("query", {})
                ranked_query["tournament_id"] = query_tournament_id
                ranked_query["tournament_source"] = query_source
                ranked_query["tournament_team_name_matches"] = sendou_name_matches
                ranked_query["tournament_player_id_count"] = len(sendou_player_ids_used)
            ranked.setdefault("query", {})["seed_player_id_count"] = len(
                explicit_seed_player_ids
            )
            ranked.setdefault("query", {})["query_mode"] = normalized_query_mode
            ranked.setdefault("query", {})["use_pair_rerank"] = bool(use_pair_rerank)
            ranked.setdefault("query", {})["use_cluster_profile_scoring"] = bool(
                use_cluster_profile_scoring
            )
            ranked.setdefault("query", {}).update(seed_query_metadata)
            ranked["results"] = strip_internal_result_fields(ranked["results"])
            logger.debug(
                "team_search schema=%s snapshot_id=%s mode=ann query=%r targets=%s results=%s elapsed_ms=%.1f",
                self.schema,
                int(snapshot_id),
                query,
                len(target_ids),
                len(ranked.get("results", [])),
                (time.perf_counter() - start_total) * 1000.0,
            )
            return ranked

        cache_entry = cache_entry or self._get_cached_snapshot_entry(snapshot_id)
        all_rows = cache_entry.rows
        ranked = self._rank_similar_teams(
            embeddings=all_rows,
            target_team_ids=target_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            candidate_top_n=consolidation_candidate_top_n,
            min_relevance=min_relevance,
            cache_entry=cache_entry,
            recency_weight=recency_weight,
            query_team_weights=query_team_weights,
            query_profile=direct_query_profile,
            rerank_candidate_limit=rerank_candidate_limit,
            rerank_weight_embed=rerank_weight_embed,
            rerank_weight_player=rerank_weight_player,
            rerank_weight_pair=rerank_weight_pair,
            use_pair_rerank=use_pair_rerank,
        )
        # Adaptive relevance: retry with a relaxed threshold when results
        # are very sparse so the user doesn't see a confusing empty page.
        if (
            len(ranked.get("results", [])) < 3
            and min_relevance > 0.5
        ):
            relaxed = self._rank_similar_teams(
                embeddings=all_rows,
                target_team_ids=target_ids,
                cluster_map=cluster_map,
                top_n=top_n,
                candidate_top_n=consolidation_candidate_top_n,
                min_relevance=max(0.5, min_relevance - 0.2),
                cache_entry=cache_entry,
                recency_weight=recency_weight,
                query_team_weights=query_team_weights,
                query_profile=direct_query_profile,
                rerank_candidate_limit=rerank_candidate_limit,
                rerank_weight_embed=rerank_weight_embed,
                rerank_weight_player=rerank_weight_player,
                rerank_weight_pair=rerank_weight_pair,
                use_pair_rerank=use_pair_rerank,
            )
            if len(relaxed.get("results", [])) > len(
                ranked.get("results", [])
            ):
                ranked = relaxed
                ranked.setdefault("query", {})["relevance_relaxed"] = True

        if consolidate:
            ranked["results"] = consolidate_ranked_results(
                ranked["results"],
                min_overlap=consolidate_min_overlap,
            )
            if direct_query_profile is not None and use_cluster_profile_scoring:
                ranked["results"] = rescore_consolidated_results(
                    ranked["results"],
                    query_profile=direct_query_profile,
                    rerank_weight_embed=rerank_weight_embed,
                    rerank_weight_player=rerank_weight_player,
                    rerank_weight_pair=rerank_weight_pair,
                    use_pair_rerank=use_pair_rerank,
                    recency_weight=recency_weight,
                )
        ranked = self._finalize_ranked_results(ranked, top_n=top_n)
        if query_tournament_id is not None:
            ranked_query = ranked.setdefault("query", {})
            ranked_query["tournament_id"] = query_tournament_id
            ranked_query["tournament_source"] = query_source
            ranked_query["tournament_team_name_matches"] = sendou_name_matches
            ranked_query["tournament_player_id_count"] = len(sendou_player_ids_used)
        ranked.setdefault("query", {})["seed_player_id_count"] = len(
            explicit_seed_player_ids
        )
        ranked.setdefault("query", {})["query_mode"] = normalized_query_mode
        ranked.setdefault("query", {})["use_pair_rerank"] = bool(use_pair_rerank)
        ranked.setdefault("query", {})["use_cluster_profile_scoring"] = bool(
            use_cluster_profile_scoring
        )
        ranked.setdefault("query", {}).update(seed_query_metadata)
        ranked["results"] = strip_internal_result_fields(ranked["results"])
        logger.debug(
            "team_search schema=%s snapshot_id=%s mode=cache query=%r targets=%s results=%s elapsed_ms=%.1f",
            self.schema,
            int(snapshot_id),
            query,
            len(target_ids),
            len(ranked.get("results", [])),
            (time.perf_counter() - start_total) * 1000.0,
        )
        return ranked

    def _fetch_rows(
        self,
        sql: str,
        params: Dict[str, Any],
        *,
        missing_default: Any,
    ) -> Any:
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(sql), params).mappings().all()
            return rows
        except SQLAlchemyError as exc:
            if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                return missing_default
            raise

    def _fetch_first_row(self, sql: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(sql), params).mappings().first()
            return dict(row) if row else None
        except SQLAlchemyError as exc:
            if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                return None
            raise

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        sql = f"""
            SELECT run_id, finished_at, teams_indexed
            FROM {self.schema}.team_search_refresh_runs
            WHERE status = 'completed'
            ORDER BY finished_at DESC NULLS LAST, run_id DESC
            LIMIT 1
        """
        row = self._fetch_first_row(sql, {})
        return row

    def list_completed_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        sql = f"""
            SELECT
                run_id,
                started_at,
                finished_at,
                teams_indexed,
                clusters_strict,
                clusters_explore,
                message
            FROM {self.schema}.team_search_refresh_runs
            WHERE status = 'completed'
            ORDER BY finished_at DESC NULLS LAST, run_id DESC
            LIMIT :limit
        """
        rows = self._fetch_rows(
            sql,
            {"limit": max(1, min(int(limit), 50))},
            missing_default=[],
        )
        return [dict(row) for row in rows]

    def _fetch_snapshot_embedding_config(self, snapshot_id: int) -> Dict[str, Any]:
        sql_with_cap = f"""
            SELECT
                semantic_dim,
                identity_dim,
                identity_beta,
                identity_idf_cap
            FROM {self.schema}.team_search_refresh_runs
            WHERE run_id = :snapshot_id
            LIMIT 1
        """
        sql_without_cap = f"""
            SELECT
                semantic_dim,
                identity_dim,
                identity_beta
            FROM {self.schema}.team_search_refresh_runs
            WHERE run_id = :snapshot_id
            LIMIT 1
        """
        with self.engine.connect() as conn:
            try:
                row = conn.execute(
                    text(sql_with_cap),
                    {"snapshot_id": int(snapshot_id)},
                ).mappings().first()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc):
                    return {}
                if not _is_missing_column_error(exc):
                    raise
                try:
                    row = conn.execute(
                        text(sql_without_cap),
                        {"snapshot_id": int(snapshot_id)},
                    ).mappings().first()
                except SQLAlchemyError as fallback_exc:
                    if _is_missing_relation_error(fallback_exc):
                        return {}
                    raise

        if not row:
            return {}

        payload = dict(row)
        payload.setdefault("identity_idf_cap", 3.0)
        return payload

    def _row_to_embedding(self, row: Dict[str, Any]) -> EmbeddingRow:
        def _coerce_vector(value: Any) -> np.ndarray:
            try:
                arr = np.asarray(value or [], dtype=np.float64)
            except (TypeError, ValueError):
                arr = np.array([], dtype=np.float64)
            return arr.reshape(-1)

        def _coerce_player_id(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _coerce_support_map(value: Any) -> Dict[int, float]:
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    return {}
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return {}
            if not isinstance(value, dict):
                return {}

            out: Dict[int, float] = {}
            for raw_key, raw_value in value.items():
                try:
                    player_id = int(raw_key)
                    support = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if player_id <= 0 or support <= 0.0:
                    continue
                out[int(player_id)] = float(support)
            return out

        def _coerce_pair_support_map(value: Any) -> Dict[tuple[int, int], float]:
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    return {}
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return {}
            if not isinstance(value, dict):
                return {}

            out: Dict[tuple[int, int], float] = {}
            for raw_key, raw_value in value.items():
                parsed = parse_pair_key(raw_key)
                if parsed is None:
                    continue
                try:
                    support = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if support <= 0.0:
                    continue
                out[parsed] = float(support)
            return out

        def _coerce_lineup_variant_counts(value: Any) -> Dict[tuple[int, ...], int]:
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    return {}
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return {}
            if not isinstance(value, dict):
                return {}

            out: Dict[tuple[int, ...], int] = {}
            for raw_key, raw_value in value.items():
                parsed = parse_lineup_key(raw_key)
                if parsed is None:
                    continue
                try:
                    count = int(raw_value)
                except (TypeError, ValueError):
                    continue
                if count <= 0:
                    continue
                out[parsed] = int(count)
            return out

        return EmbeddingRow(
            team_id=int(row["team_id"]),
            tournament_id=(
                int(row["tournament_id"])
                if row.get("tournament_id") is not None
                else None
            ),
            team_name=str(row.get("team_name") or row["team_id"]),
            event_time_ms=(
                int(row["event_time_ms"])
                if row.get("event_time_ms") is not None
                else None
            ),
            lineup_count=int(row.get("lineup_count") or 0),
            tournament_count=int(row.get("tournament_count") or 0),
            unique_player_count=int(row.get("unique_player_count") or 0),
            distinct_lineup_count=int(row.get("distinct_lineup_count") or 0),
            top_lineup_share=float(row.get("top_lineup_share") or 0.0),
            lineup_entropy=float(row.get("lineup_entropy") or 0.0),
            effective_lineups=float(row.get("effective_lineups") or 0.0),
            semantic_vector=_coerce_vector(row.get("semantic_vector")),
            identity_vector=_coerce_vector(row.get("identity_vector")),
            final_vector=_coerce_vector(row.get("final_vector")),
            top_lineup_summary=row.get("top_lineup_summary"),
            top_lineup_player_ids=tuple(
                pid
                for raw_id in (row.get("top_lineup_player_ids") or [])
                if (pid := _coerce_player_id(raw_id)) is not None
            ),
            top_lineup_player_names=tuple(
                str(player_name)
                for player_name in (row.get("top_lineup_player_names") or [])
                if player_name is not None
            ),
            roster_player_ids=tuple(
                pid
                for raw_id in (row.get("roster_player_ids") or [])
                if (pid := _coerce_player_id(raw_id)) is not None
            ),
            roster_player_names=tuple(
                str(player_name)
                for player_name in (row.get("roster_player_names") or [])
                if player_name is not None
            ),
            roster_player_match_counts=tuple(
                int(value)
                for value in (row.get("roster_player_match_counts") or [])
                if value is not None
            ),
            player_support=_coerce_support_map(row.get("player_support")),
            pair_support=_coerce_pair_support_map(row.get("pair_support")),
            lineup_variant_counts=_coerce_lineup_variant_counts(
                row.get("lineup_variant_counts")
            ),
        )

    def load_embeddings(self, snapshot_id: int) -> List[EmbeddingRow]:
        sql_with_roster = f"""
            SELECT
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
                lineup_variant_counts
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
        """
        sql_with_players = f"""
            SELECT
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
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
        """
        sql_legacy = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
        """
        sql_core = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
        """
        with self.engine.connect() as conn:
            try:
                rows = conn.execute(
                    text(sql_with_roster), {"snapshot_id": int(snapshot_id)}
                ).mappings().all()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc):
                    logger.warning(
                        "No team-search embedding table available in schema %s",
                        self.schema,
                    )
                    rows = []
                elif not _is_missing_column_error(exc):
                    raise
                else:
                    try:
                        rows = conn.execute(
                            text(sql_with_players), {"snapshot_id": int(snapshot_id)}
                        ).mappings().all()
                    except SQLAlchemyError as nested_exc:
                        if not _is_missing_column_error(nested_exc):
                            raise
                        try:
                            rows = conn.execute(
                                text(sql_legacy),
                                {"snapshot_id": int(snapshot_id)},
                            ).mappings().all()
                        except SQLAlchemyError as legacy_exc:
                            if not _is_missing_column_error(legacy_exc):
                                raise
                            rows = conn.execute(
                                text(sql_core), {"snapshot_id": int(snapshot_id)}
                            ).mappings().all()
        normalized_rows = []
        expected_final = None
        expected_semantic = None
        expected_identity = None

        row_dicts = [dict(row) for row in rows]
        self._hydrate_tournament_counts(snapshot_id, row_dicts)
        for r in row_dicts:
            r.setdefault("top_lineup_player_ids", [])
            r.setdefault("top_lineup_player_names", [])
            r.setdefault("roster_player_ids", [])
            r.setdefault("roster_player_names", [])
            r.setdefault("roster_player_match_counts", [])
            r.setdefault("player_support", {})
            r.setdefault("pair_support", {})
            r.setdefault("lineup_variant_counts", {})
            r_obj = self._row_to_embedding(r)
            final_dim = int(r_obj.final_vector.size)
            semantic_dim = int(r_obj.semantic_vector.size)
            identity_dim = int(r_obj.identity_vector.size)
            if expected_final is None:
                expected_final = final_dim
                expected_semantic = semantic_dim
                expected_identity = identity_dim
            elif (
                final_dim != expected_final
                or semantic_dim != expected_semantic
                or identity_dim != expected_identity
            ):
                logger.warning(
                    "Skipping team_id=%s due mixed embedding dimensions in snapshot %s",
                    r_obj.team_id,
                    snapshot_id,
                )
                continue

            normalized_rows.append(r_obj)
        return normalized_rows

    def _build_snapshot_cache_entry(
        self, snapshot_id: int, rows: List[EmbeddingRow]
    ) -> EmbeddingSnapshotCacheEntry:
        config = self._fetch_snapshot_embedding_config(snapshot_id)
        if rows:
            finals = np.stack([row.final_vector for row in rows], axis=0)
            semantics = np.stack([row.semantic_vector for row in rows], axis=0)
            identities = np.stack([row.identity_vector for row in rows], axis=0)
            lineup_counts = np.asarray(
                [row.lineup_count for row in rows], dtype=np.float64
            )
            semantic_dim = int(config.get("semantic_dim") or rows[0].semantic_vector.size)
            identity_dim = int(config.get("identity_dim") or rows[0].identity_vector.size)
        else:
            finals = np.empty((0, 0), dtype=np.float64)
            semantics = np.empty((0, 0), dtype=np.float64)
            identities = np.empty((0, 0), dtype=np.float64)
            lineup_counts = np.empty((0,), dtype=np.float64)
            semantic_dim = int(config.get("semantic_dim") or 0)
            identity_dim = int(config.get("identity_dim") or 0)

        identity_beta = float(config.get("identity_beta") or 3.0)
        identity_idf_cap = float(config.get("identity_idf_cap") or 3.0)
        idf_lookup = build_identity_idf_lookup(
            (
                row.roster_player_ids or row.top_lineup_player_ids
                for row in rows
            ),
            idf_cap=identity_idf_cap,
        )

        return EmbeddingSnapshotCacheEntry(
            snapshot_id=int(snapshot_id),
            rows=rows,
            finals=finals,
            semantics=semantics,
            identities=identities,
            lineup_counts=lineup_counts,
            index_by_team_id={int(row.team_id): idx for idx, row in enumerate(rows)},
            team_ids_by_player_id={
                int(player_id): tuple(team_ids)
                for player_id, team_ids in self._build_player_team_index(rows).items()
            },
            idf_lookup=idf_lookup,
            semantic_dim=semantic_dim,
            identity_dim=identity_dim,
            identity_beta=identity_beta,
            identity_idf_cap=identity_idf_cap,
            built_at_ms=int(time.time() * 1000),
        )

    @staticmethod
    def _build_player_team_index(rows: Sequence[EmbeddingRow]) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for row in rows:
            player_ids = _normalize_id_sequence(
                row.roster_player_ids or row.top_lineup_player_ids
            )
            if not player_ids:
                continue
            for player_id in player_ids:
                out.setdefault(int(player_id), []).append(int(row.team_id))
        return out

    def _invalidate_snapshot_cache(self, snapshot_id: Optional[int] = None) -> None:
        with self._cache_lock:
            if snapshot_id is None:
                self._embedding_snapshot_cache = None
                self._cluster_map_cache.clear()
                self._tournament_count_cache.clear()
                return

            sid = int(snapshot_id)
            if (
                self._embedding_snapshot_cache is not None
                and int(self._embedding_snapshot_cache.snapshot_id) == sid
            ):
                self._embedding_snapshot_cache = None

            self._cluster_map_cache = {
                key: value
                for key, value in self._cluster_map_cache.items()
                if int(key[0]) != sid
            }
            self._tournament_count_cache = {
                key: value
                for key, value in self._tournament_count_cache.items()
                if int(key[0]) != sid
            }

    def _get_cached_snapshot_entry(self, snapshot_id: int) -> EmbeddingSnapshotCacheEntry:
        sid = int(snapshot_id)
        with self._cache_lock:
            cached = self._embedding_snapshot_cache
            if cached is not None and int(cached.snapshot_id) == sid:
                logger.debug(
                    "snapshot_cache_hit schema=%s snapshot_id=%s teams=%s",
                    self.schema,
                    sid,
                    len(cached.rows),
                )
                return cached

        start = time.perf_counter()
        rows = self.load_embeddings(sid)
        built = self._build_snapshot_cache_entry(sid, rows)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with self._cache_lock:
            self._embedding_snapshot_cache = built
            # Keep auxiliary caches bounded to the active snapshot.
            self._cluster_map_cache = {
                key: value
                for key, value in self._cluster_map_cache.items()
                if int(key[0]) == sid
            }
            self._tournament_count_cache = {
                key: value
                for key, value in self._tournament_count_cache.items()
                if int(key[0]) == sid
            }
        logger.debug(
            "snapshot_cache_miss schema=%s snapshot_id=%s teams=%s build_ms=%.1f",
            self.schema,
            sid,
            len(rows),
            elapsed_ms,
        )
        return built

    def _get_cached_cluster_map(
        self, snapshot_id: int, profile: str
    ) -> Dict[int, Dict[str, Any]]:
        sid = int(snapshot_id)
        key = (sid, str(profile))
        with self._cache_lock:
            cached = self._cluster_map_cache.get(key)
            if cached is not None:
                return cached

        cluster_map = self.load_cluster_map(sid, profile, use_cache=False)
        with self._cache_lock:
            # Keep only cluster maps for the active snapshot.
            self._cluster_map_cache = {
                cache_key: value
                for cache_key, value in self._cluster_map_cache.items()
                if int(cache_key[0]) == sid
            }
            self._cluster_map_cache[key] = cluster_map
        return cluster_map

    def _fetch_sendou_tournament_teams(
        self, tournament_id: int
    ) -> List[Dict[str, Any]]:
        tournament_id = int(tournament_id)
        url = f"https://sendou.ink/to/{tournament_id}/teams.data"
        request = urlrequest.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "SplatTopTeams/0.1",
            },
        )

        start = time.perf_counter()
        try:
            with urlrequest.urlopen(request, timeout=10) as response:
                raw_payload = response.read()
            payload = json.loads(raw_payload.decode("utf-8"))
        except (
            urlerror.HTTPError,
            urlerror.URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            logger.warning(
                "sendou_tournament_fetch_failed tournament_id=%s error=%s",
                tournament_id,
                exc,
            )
            return []

        teams = _extract_sendou_teams_from_turbo_payload(payload)
        logger.debug(
            "sendou_tournament_fetch tournament_id=%s teams=%s elapsed_ms=%.1f",
            tournament_id,
            len(teams),
            (time.perf_counter() - start) * 1000.0,
        )
        return teams

    def _get_cached_sendou_tournament_teams(
        self, tournament_id: int
    ) -> List[Dict[str, Any]]:
        tid = int(tournament_id)
        with self._cache_lock:
            cached = self._sendou_tournament_team_cache.get(tid)
            if cached is not None:
                return list(cached)

        teams = self._fetch_sendou_tournament_teams(tid)
        with self._cache_lock:
            self._sendou_tournament_team_cache[tid] = list(teams)
        return list(teams)

    def _match_sendou_tournament_teams(
        self, tournament_id: int, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        normalized_query = _normalize_query(query)
        token_query = set(_tokenized_query(query))
        rows = self._get_cached_sendou_tournament_teams(tournament_id)
        if not rows:
            return []

        seen: set[str] = set()
        matches: List[Dict[str, Any]] = []
        for row in rows:
            if not _sendou_team_matches_query(
                row,
                normalized_query=normalized_query,
                token_query=token_query,
            ):
                continue

            dedupe_key = (
                str(row.get("team_name") or "").strip().lower()
                or str(row.get("display_name") or "").strip().lower()
                or f"id:{row.get('team_id')}"
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            matches.append(dict(row))
            if len(matches) >= max(1, int(limit)):
                break

        return matches

    def _match_targets_by_player_ids(
        self,
        snapshot_id: int,
        player_ids: Sequence[int],
        *,
        limit: int = 240,
    ) -> List[int]:
        normalized_player_ids = _normalize_id_sequence(player_ids)
        if not normalized_player_ids:
            return []

        sql_with_roster = f"""
            SELECT
                e.team_id,
                COUNT(*)::int AS overlap_count,
                MAX(e.lineup_count)::int AS lineup_count
            FROM {self.schema}.team_search_embeddings e
            JOIN LATERAL unnest(e.roster_player_ids) AS p(player_id) ON TRUE
            WHERE e.snapshot_id = :snapshot_id
              AND p.player_id IN :player_ids
            GROUP BY e.team_id
            ORDER BY overlap_count DESC, lineup_count DESC, e.team_id ASC
            LIMIT :limit
        """
        sql_with_top_lineup = f"""
            SELECT
                e.team_id,
                COUNT(*)::int AS overlap_count,
                MAX(e.lineup_count)::int AS lineup_count
            FROM {self.schema}.team_search_embeddings e
            JOIN LATERAL unnest(e.top_lineup_player_ids) AS p(player_id) ON TRUE
            WHERE e.snapshot_id = :snapshot_id
              AND p.player_id IN :player_ids
            GROUP BY e.team_id
            ORDER BY overlap_count DESC, lineup_count DESC, e.team_id ASC
            LIMIT :limit
        """
        params = {
            "snapshot_id": int(snapshot_id),
            "player_ids": normalized_player_ids,
            "limit": max(1, min(int(limit), 2000)),
        }
        try:
            with self.engine.connect() as conn:
                try:
                    rows = conn.execute(
                        text(sql_with_roster).bindparams(
                            bindparam("player_ids", expanding=True)
                        ),
                        params,
                    ).mappings().all()
                except SQLAlchemyError as exc:
                    if _is_missing_relation_error(exc):
                        return []
                    if not _is_missing_column_error(exc):
                        raise
                    rows = conn.execute(
                        text(sql_with_top_lineup).bindparams(
                            bindparam("player_ids", expanding=True)
                        ),
                        params,
                    ).mappings().all()
        except SQLAlchemyError as exc:
            if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                return []
            raise

        return [int(row["team_id"]) for row in rows]

    def list_tournament_teams(
        self,
        snapshot_id: int,
        tournament_id: int,
        query: Optional[str],
        limit: int,
    ) -> Dict[str, Any]:
        sid = int(snapshot_id)
        tid = int(tournament_id)
        max_limit = max(1, min(int(limit), 700))

        normalized_query = _normalize_query(query or "")
        params: Dict[str, Any] = {
            "snapshot_id": sid,
            "tournament_id": tid,
            "limit": max_limit,
        }
        extra_where = ""
        if normalized_query:
            params["name_like"] = f"%{str(query or '').strip().lower()}%"
            params["name_norm"] = normalized_query
            extra_where = """
              AND (
                LOWER(COALESCE(team_name, '')) LIKE :name_like
                OR regexp_replace(LOWER(COALESCE(team_name, '')), '[^a-z0-9]+', '', 'g') = :name_norm
              )
            """

        sql = f"""
            SELECT
                team_id,
                team_name,
                lineup_count,
                event_time_ms
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND tournament_id = :tournament_id
              {extra_where}
            ORDER BY lineup_count DESC, team_name ASC, team_id ASC
            LIMIT :limit
        """
        local_rows = self._fetch_rows(sql, params, missing_default=[])
        if local_rows:
            teams = [
                {
                    "team_id": int(row["team_id"]),
                    "team_name": str(row.get("team_name") or row["team_id"]),
                    "display_name": str(row.get("team_name") or row["team_id"]),
                    "lineup_count": int(row.get("lineup_count") or 0),
                    "event_time_ms": (
                        int(row["event_time_ms"])
                        if row.get("event_time_ms") is not None
                        else None
                    ),
                    "member_user_ids": [],
                    "member_names": [],
                    "source": "dataset",
                }
                for row in local_rows
            ]
            return {
                "tournament_id": tid,
                "source": "dataset",
                "teams": teams,
            }

        remote_rows = self._get_cached_sendou_tournament_teams(tid)
        if normalized_query:
            token_query = set(_tokenized_query(query or ""))
            filtered_rows: List[Dict[str, Any]] = []
            for row in remote_rows:
                if _sendou_team_matches_query(
                    row,
                    normalized_query=normalized_query,
                    token_query=token_query,
                ):
                    filtered_rows.append(row)
            remote_rows = filtered_rows

        teams = []
        for row in remote_rows[:max_limit]:
            team_id = row.get("team_id")
            teams.append(
                {
                    "team_id": int(team_id) if isinstance(team_id, (int, float)) else None,
                    "team_name": str(row.get("team_name") or ""),
                    "display_name": str(row.get("display_name") or row.get("team_name") or ""),
                    "lineup_count": 0,
                    "event_time_ms": None,
                    "member_user_ids": _normalize_id_sequence(row.get("member_user_ids") or []),
                    "member_names": [
                        str(name).strip()
                        for name in (row.get("member_names") or [])
                        if str(name or "").strip()
                    ],
                    "source": "sendou",
                }
            )

        return {
            "tournament_id": tid,
            "source": "sendou" if teams else "none",
            "teams": teams,
        }

    def suggest_players(
        self,
        snapshot_id: int,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip().lower()
        if not q:
            return []
        limit = max(1, min(limit, 50))
        sid = int(snapshot_id)

        prefix_sql = f"""
            SELECT
                p.player_id,
                p.display_name,
                COUNT(DISTINCT e.team_id)::int AS team_count
            FROM {self.schema}.players p
            JOIN {self.schema}.team_search_embeddings e
                ON e.snapshot_id = :snapshot_id
            JOIN LATERAL unnest(e.roster_player_ids) AS rp(pid) ON TRUE
            WHERE rp.pid = p.player_id
              AND LOWER(p.display_name) LIKE :prefix
            GROUP BY p.player_id, p.display_name
            ORDER BY team_count DESC, p.display_name ASC
            LIMIT :limit
        """
        params: Dict[str, Any] = {
            "snapshot_id": sid,
            "prefix": f"{q}%",
            "limit": limit,
        }
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(prefix_sql), params).mappings().all()
        except SQLAlchemyError:
            rows = []

        suggestions = [
            {
                "player_id": int(r["player_id"]),
                "display_name": str(r["display_name"]),
                "team_count": int(r["team_count"]),
            }
            for r in rows
        ]

        if len(suggestions) >= limit:
            return suggestions[:limit]

        remaining = limit - len(suggestions)
        seen_ids = {s["player_id"] for s in suggestions}

        if self._detect_trgm_support() and len(q) >= 2:
            trgm_sql = f"""
                SELECT
                    p.player_id,
                    p.display_name,
                    COUNT(DISTINCT e.team_id)::int AS team_count,
                    similarity(LOWER(p.display_name), :q) AS sim
                FROM {self.schema}.players p
                JOIN {self.schema}.team_search_embeddings e
                    ON e.snapshot_id = :snapshot_id
                JOIN LATERAL unnest(e.roster_player_ids) AS rp(pid) ON TRUE
                WHERE rp.pid = p.player_id
                  AND LOWER(p.display_name) %% :q
                GROUP BY p.player_id, p.display_name
                ORDER BY sim DESC, team_count DESC
                LIMIT :fetch_limit
            """
            trgm_params = {
                "snapshot_id": sid,
                "q": q,
                "fetch_limit": remaining + 5,
            }
            try:
                with self.engine.connect() as conn:
                    trgm_rows = conn.execute(
                        text(trgm_sql), trgm_params
                    ).mappings().all()
                for r in trgm_rows:
                    pid = int(r["player_id"])
                    if pid not in seen_ids:
                        suggestions.append(
                            {
                                "player_id": pid,
                                "display_name": str(r["display_name"]),
                                "team_count": int(r["team_count"]),
                            }
                        )
                        seen_ids.add(pid)
                    if len(suggestions) >= limit:
                        break
            except SQLAlchemyError:
                pass

        return suggestions[:limit]

    def get_player_teams(
        self,
        snapshot_id: int,
        player_id: int,
        limit: int = 50,
    ) -> Dict[str, Any]:
        sid = int(snapshot_id)
        pid = int(player_id)
        limit = max(1, min(limit, 200))

        names = self._fetch_player_names([pid])
        display_name = names.get(pid, str(pid))

        sql = f"""
            SELECT
                e.team_id,
                e.team_name,
                e.lineup_count,
                e.event_time_ms,
                e.roster_player_ids,
                e.roster_player_names,
                e.roster_player_match_counts
            FROM {self.schema}.team_search_embeddings e
            JOIN LATERAL unnest(e.roster_player_ids) AS rp(pid) ON TRUE
            WHERE e.snapshot_id = :snapshot_id
              AND rp.pid = :player_id
            ORDER BY e.lineup_count DESC, e.team_id ASC
            LIMIT :limit
        """
        params = {"snapshot_id": sid, "player_id": pid, "limit": limit}
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(sql), params).mappings().all()
        except SQLAlchemyError:
            rows = []

        teams: List[Dict[str, Any]] = []
        for r in rows:
            roster_ids = tuple(
                rid
                for raw_id in (r.get("roster_player_ids") or [])
                if (rid := _coerce_player_id(raw_id)) is not None
            )
            roster_names = tuple(
                str(n) for n in (r.get("roster_player_names") or []) if n is not None
            )
            match_counts = tuple(
                int(v) for v in (r.get("roster_player_match_counts") or []) if v is not None
            )

            player_match_count = 0
            for idx, rid in enumerate(roster_ids):
                if rid == pid:
                    player_match_count = match_counts[idx] if idx < len(match_counts) else 0
                    break

            teams.append(
                {
                    "team_id": int(r["team_id"]),
                    "team_name": str(r["team_name"] or ""),
                    "lineup_count": int(r["lineup_count"]),
                    "event_time_ms": int(r["event_time_ms"] or 0),
                    "player_match_count": player_match_count,
                    "roster_player_names": list(roster_names),
                }
            )

        return {
            "player_id": pid,
            "display_name": display_name,
            "team_count": len(teams),
            "teams": teams,
        }

    def suggest_team_names(
        self,
        snapshot_id: int,
        query: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip().lower()
        if not q:
            return []
        limit = max(1, min(limit, 25))

        prefix_sql = f"""
            SELECT DISTINCT ON (LOWER(team_name))
                team_id, team_name, lineup_count
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND LOWER(team_name) LIKE :prefix
            ORDER BY LOWER(team_name), lineup_count DESC
            LIMIT :limit
        """
        params = {
            "snapshot_id": int(snapshot_id),
            "prefix": f"{q}%",
            "limit": limit,
        }
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(prefix_sql), params).mappings().all()
        except SQLAlchemyError:
            rows = []

        suggestions = [
            {
                "team_id": int(r["team_id"]),
                "team_name": r["team_name"],
                "lineup_count": int(r["lineup_count"]),
            }
            for r in rows
        ]

        if len(suggestions) >= limit:
            return suggestions[:limit]

        remaining = limit - len(suggestions)
        seen_names = {s["team_name"].lower() for s in suggestions}

        if self._detect_trgm_support() and len(q) >= 2:
            trgm_sql = f"""
                SELECT DISTINCT ON (LOWER(team_name))
                    team_id, team_name, lineup_count,
                    similarity(LOWER(COALESCE(team_name, '')), :q) AS sim
                FROM {self.schema}.team_search_embeddings
                WHERE snapshot_id = :snapshot_id
                  AND LOWER(COALESCE(team_name, '')) %% :q
                ORDER BY LOWER(team_name), lineup_count DESC
                LIMIT :fetch_limit
            """
            trgm_params = {
                "snapshot_id": int(snapshot_id),
                "q": q,
                "fetch_limit": remaining + 5,
            }
            try:
                with self.engine.connect() as conn:
                    trgm_rows = conn.execute(
                        text(trgm_sql), trgm_params
                    ).mappings().all()
                for r in trgm_rows:
                    if r["team_name"].lower() not in seen_names:
                        suggestions.append(
                            {
                                "team_id": int(r["team_id"]),
                                "team_name": r["team_name"],
                                "lineup_count": int(r["lineup_count"]),
                            }
                        )
                        seen_names.add(r["team_name"].lower())
                    if len(suggestions) >= limit:
                        break
            except SQLAlchemyError:
                pass

        return suggestions[:limit]

    def match_targets(
        self,
        snapshot_id: int,
        query: str,
        limit: int = 25,
        tournament_id: Optional[int] = None,
    ) -> List[int]:
        limit = max(1, min(limit, 200))
        q = (query or "").strip().lower()
        if not q:
            return []

        normalized_q = _normalize_query(q)
        tournament_filter = ""
        tournament_params: Dict[str, Any] = {}
        if tournament_id is not None:
            tournament_filter = "AND tournament_id = :tournament_id"
            tournament_params["tournament_id"] = int(tournament_id)

        params = {
            "snapshot_id": int(snapshot_id),
            "name_like": f"%{q}%",
            "name_eq": q,
            "limit": limit,
            **tournament_params,
        }
        where_clause = (
            "LOWER(team_name) = :name_eq "
            "OR LOWER(team_name) LIKE :name_like "
            "OR team_id = :team_id "
            "OR LOWER(team_id::text) = :team_id_text "
            "OR regexp_replace(LOWER(COALESCE(team_name, '')), '[^a-z0-9]+', '', 'g') = :name_norm"
        )

        team_id: Optional[int]
        try:
            team_id = int(query)
        except Exception:
            team_id = None
        params["team_id"] = team_id if team_id is not None else -1
        params["team_id_text"] = str(team_id) if team_id is not None else "__no-match__"
        params["name_norm"] = normalized_q

        sql = f"""
            SELECT team_id
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              {tournament_filter}
              AND ({where_clause})
            ORDER BY
                CASE
                    WHEN LOWER(team_name) = :name_eq THEN 0
                    WHEN LOWER(team_id::text) = :team_id_text THEN 1
                    WHEN regexp_replace(LOWER(COALESCE(team_name, '')), '[^a-z0-9]+', '', 'g') = :name_norm THEN 2
                    ELSE 3
                END,
                lineup_count DESC
            LIMIT :limit
        """
        with self.engine.connect() as conn:
            try:
                rows = conn.execute(text(sql), params).fetchall()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                    return []
                raise
        target_ids = [int(row[0]) for row in rows]

        if target_ids:
            return target_ids

        # Trigram fuzzy matching (catches typos like "Moonlgiht" → "Moonlight").
        if self._detect_trgm_support() and len(q) >= 2:
            trgm_sql = f"""
                SELECT team_id,
                       similarity(LOWER(COALESCE(team_name, '')), :query_lower) AS sim
                FROM {self.schema}.team_search_embeddings
                WHERE snapshot_id = :snapshot_id
                  {tournament_filter}
                  AND LOWER(COALESCE(team_name, '')) %% :query_lower
                ORDER BY sim DESC, lineup_count DESC
                LIMIT :limit
            """
            trgm_params = {
                "snapshot_id": int(snapshot_id),
                "query_lower": q,
                "limit": limit,
                **tournament_params,
            }
            try:
                with self.engine.connect() as conn:
                    trgm_rows = conn.execute(
                        text(trgm_sql), trgm_params
                    ).fetchall()
                trgm_ids = [int(row[0]) for row in trgm_rows]
                if trgm_ids:
                    return trgm_ids
            except SQLAlchemyError:
                pass

        # Last-resort fallback: scan top teams by lineup_count in Python.
        with self.engine.connect() as conn:
            try:
                fallback_rows = conn.execute(
                    text(
                        f"""
                        SELECT team_id, team_name
                        FROM {self.schema}.team_search_embeddings
                        WHERE snapshot_id = :snapshot_id
                          {tournament_filter}
                        ORDER BY lineup_count DESC
                        LIMIT :fallback_limit
                    """
                    ),
                    {
                        "snapshot_id": int(snapshot_id),
                        "fallback_limit": limit * 10,
                        **tournament_params,
                    },
                ).mappings().all()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                    return []
                raise

        fallback_ids: List[int] = []
        for row in fallback_rows:
            team_name = row["team_name"]
            if not team_name:
                continue
            normalized_name = _normalize_query(team_name)
            if not normalized_q:
                continue
            if len(normalized_q) <= 4:
                if normalized_q in _tokenized_query(team_name):
                    fallback_ids.append(int(row["team_id"]))
            elif normalized_q in normalized_name:
                fallback_ids.append(int(row["team_id"]))
            if len(fallback_ids) >= limit:
                break

        return fallback_ids

    def load_cluster_map(
        self, snapshot_id: int, profile: str, *, use_cache: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        sid = int(snapshot_id)
        mode = str(profile)
        cache_key = (sid, mode)
        if use_cache:
            with self._cache_lock:
                cached = self._cluster_map_cache.get(cache_key)
            if cached is not None:
                return cached

        sql = f"""
            SELECT team_id, cluster_id, cluster_size, representative_team_name
            FROM {self.schema}.team_search_clusters
            WHERE snapshot_id = :snapshot_id
              AND profile = :profile
        """
        rows = self._fetch_rows(
            sql,
            {"snapshot_id": sid, "profile": mode},
            missing_default=[],
        )
        resolved = {
            int(row["team_id"]): {
                "cluster_id": int(row["cluster_id"]),
                "cluster_size": int(row["cluster_size"]),
                "representative_team_name": row["representative_team_name"],
            }
            for row in rows
        }
        if use_cache:
            with self._cache_lock:
                self._cluster_map_cache = {
                    key: value
                    for key, value in self._cluster_map_cache.items()
                    if int(key[0]) == sid
                }
                self._cluster_map_cache[cache_key] = resolved
        return resolved

    def list_clusters(
        self, snapshot_id: int, profile: str, query: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "snapshot_id": int(snapshot_id),
            "profile": profile,
        }
        where_extra = ""
        if query:
            where_extra = (
                "AND (LOWER(COALESCE(c.representative_team_name, '')) LIKE :name_like "
                "OR LOWER(COALESCE(e.team_name, '')) LIKE :name_like)"
            )
            params["name_like"] = f"%{query.lower()}%"

        sql = f"""
            SELECT
                c.cluster_id,
                c.cluster_size,
                c.representative_team_name,
                c.team_id,
                e.team_name,
                e.lineup_count,
                e.event_time_ms,
                e.tournament_id
            FROM {self.schema}.team_search_clusters c
            JOIN {self.schema}.team_search_embeddings e
              ON e.snapshot_id = c.snapshot_id
             AND e.team_id = c.team_id
            WHERE c.snapshot_id = :snapshot_id
              AND c.profile = :profile
              AND c.cluster_id >= 0
              {where_extra}
            ORDER BY c.cluster_size DESC, c.cluster_id ASC, e.lineup_count DESC
        """

        rows = self._fetch_rows(sql, params, missing_default=[])

        by_cluster: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            cid = int(row["cluster_id"])
            entry = by_cluster.setdefault(
                cid,
                {
                    "cluster_id": cid,
                    "cluster_size": int(row["cluster_size"]),
                    "representative_team_name": row["representative_team_name"],
                    "top_teams": [],
                },
            )
            top_teams = entry["top_teams"]
            if len(top_teams) < 5:
                top_teams.append(
                    {
                        "team_id": int(row["team_id"]),
                        "team_name": row["team_name"],
                        "lineup_count": int(row["lineup_count"]),
                        "tournament_id": (
                            int(row["tournament_id"])
                            if row["tournament_id"] is not None
                            else None
                        ),
                        "event_time_ms": (
                            int(row["event_time_ms"])
                            if row["event_time_ms"] is not None
                            else None
                        ),
                    }
                )

        clusters = list(by_cluster.values())
        clusters.sort(key=lambda c: (-int(c["cluster_size"]), int(c["cluster_id"])))

        for cluster in clusters:
            size = int(cluster["cluster_size"])
            cluster["stability_hint"] = (
                "stable" if size >= 8 else "rotational" if size >= 4 else "small"
            )

        return clusters[: max(1, min(limit, 200))]

    def cluster_detail(
        self,
        snapshot_id: int,
        profile: str,
        cluster_id: int,
    ) -> Optional[Dict[str, Any]]:
        sql = f"""
            SELECT
                c.cluster_id,
                c.cluster_size,
                c.representative_team_name,
                e.team_id,
                e.team_name,
                e.tournament_id,
                e.event_time_ms,
                e.lineup_count,
                e.top_lineup_summary
            FROM {self.schema}.team_search_clusters c
            JOIN {self.schema}.team_search_embeddings e
              ON e.snapshot_id = c.snapshot_id
             AND e.team_id = c.team_id
            WHERE c.snapshot_id = :snapshot_id
              AND c.profile = :profile
              AND c.cluster_id = :cluster_id
            ORDER BY e.lineup_count DESC, e.team_id ASC
        """
        rows = self._fetch_rows(
            sql,
            {
                "snapshot_id": int(snapshot_id),
                "profile": profile,
                "cluster_id": int(cluster_id),
            },
            missing_default=[],
        )

        if not rows:
            return None

        header = rows[0]
        members = []
        for row in rows:
            members.append(
                {
                    "team_id": int(row["team_id"]),
                    "team_name": row["team_name"],
                    "tournament_id": (
                        int(row["tournament_id"])
                        if row["tournament_id"] is not None
                        else None
                    ),
                    "event_time_ms": (
                        int(row["event_time_ms"])
                        if row["event_time_ms"] is not None
                        else None
                    ),
                    "lineup_count": int(row["lineup_count"]),
                    "top_lineup_summary": row["top_lineup_summary"],
                }
            )

        return {
            "cluster_id": int(header["cluster_id"]),
            "cluster_size": int(header["cluster_size"]),
            "representative_team_name": header["representative_team_name"],
            "members": members,
        }

    def get_refresh_run(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        sql = f"""
            SELECT
                run_id,
                status,
                started_at,
                finished_at,
                days_window,
                semantic_dim,
                identity_dim,
                identity_beta,
                source_since_ms,
                source_until_ms,
                teams_indexed,
                clusters_strict,
                clusters_explore,
                message
            FROM {self.schema}.team_search_refresh_runs
            WHERE run_id = :snapshot_id
            LIMIT 1
        """
        return self._fetch_first_row(sql, {"snapshot_id": int(snapshot_id)})

    def _cluster_names(self, snapshot_id: int, profile: str) -> Dict[int, str]:
        sql = f"""
            SELECT cluster_id, representative_team_name
            FROM {self.schema}.team_search_clusters
            WHERE snapshot_id = :snapshot_id
              AND profile = :profile
              AND cluster_id >= 0
        """
        rows = self._fetch_rows(
            sql,
            {"snapshot_id": int(snapshot_id), "profile": profile},
            missing_default=[],
        )
        out: Dict[int, str] = {}
        for row in rows:
            cid = int(row["cluster_id"])
            if cid not in out and row.get("representative_team_name"):
                out[cid] = str(row["representative_team_name"])
        return out

    def analytics_overview(
        self,
        *,
        snapshot_id: int,
        profile: str,
        limit_clusters: int,
        volatile_limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_overview

        start_total = time.perf_counter()
        cache_entry = self._get_cached_snapshot_entry(snapshot_id)
        embeddings = cache_entry.rows
        cluster_map = self._get_cached_cluster_map(snapshot_id, profile)
        overview = compute_overview(
            embeddings,
            cluster_map,
            limit_clusters=limit_clusters,
            volatile_limit=volatile_limit,
        )
        run = self.get_refresh_run(snapshot_id)
        payload = {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            "run": run,
            **overview,
        }
        logger.debug(
            "analytics_overview schema=%s snapshot_id=%s profile=%s elapsed_ms=%.1f",
            self.schema,
            int(snapshot_id),
            profile,
            (time.perf_counter() - start_total) * 1000.0,
        )
        return payload

    def analytics_matchups(
        self,
        *,
        snapshot_id: int,
        profile: str,
        min_matches: int,
        limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import summarize_matchups

        sql = f"""
            SELECT
                m.match_id,
                m.team1_id,
                m.team2_id,
                m.winner_team_id,
                c1.cluster_id AS cluster_1,
                c2.cluster_id AS cluster_2
            FROM {self.schema}.matches m
            JOIN {self.schema}.team_search_clusters c1
              ON c1.snapshot_id = :snapshot_id
             AND c1.profile = :profile
             AND c1.team_id = m.team1_id
            JOIN {self.schema}.team_search_clusters c2
              ON c2.snapshot_id = :snapshot_id
             AND c2.profile = :profile
             AND c2.team_id = m.team2_id
            WHERE c1.cluster_id >= 0
              AND c2.cluster_id >= 0
              AND c1.cluster_id <> c2.cluster_id
              AND m.winner_team_id IS NOT NULL
        """
        rows = self._fetch_rows(
            sql,
            {"snapshot_id": int(snapshot_id), "profile": profile},
            missing_default=[],
        )
        cluster_names = self._cluster_names(snapshot_id, profile)
        summarized = summarize_matchups(
            rows,
            cluster_names,
            min_matches=max(1, int(min_matches)),
            limit=max(1, int(limit)),
        )
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            "total_cross_cluster_matches": len(rows),
            "matchups": summarized,
        }

    def analytics_head_to_head(
        self,
        *,
        snapshot_id: Optional[int],
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        limit: int,
    ) -> Dict[str, Any]:
        team_a = _normalize_id_sequence(team_a_ids)
        team_b = _normalize_id_sequence(team_b_ids)

        if not team_a or not team_b:
            return {
                "snapshot_id": int(snapshot_id),
                "summary": {
                    "team_a_id": None,
                    "team_b_id": None,
                    "team_a_ids": team_a,
                    "team_b_ids": team_b,
                    "team_a_name": "N/A",
                    "team_b_name": "N/A",
                    "team_a_names": [],
                    "team_b_names": [],
                    "total_matches": 0,
                    "team_a_wins": 0,
                    "team_b_wins": 0,
                    "unresolved_matches": 0,
                    "decided_matches": 0,
                    "team_a_win_rate": 0.0,
                    "team_b_win_rate": 0.0,
                    "tournaments": 0,
                    "tournament_tier_distribution": {
                        "X": 0,
                        "S+": 0,
                        "S": 0,
                        "A+": 0,
                        "A": 0,
                        "A-": 0,
                        "Unscored": 0,
                    },
                    "tournament_tier_match_distribution": {
                        "X": 0,
                        "S+": 0,
                        "S": 0,
                        "A+": 0,
                        "A": 0,
                        "A-": 0,
                        "Unscored": 0,
                    },
                },
                "matches": [],
            }

        team_a_set = set(team_a)
        team_b_set = set(team_b)
        team_a_ids_sorted = sorted(team_a_set)
        team_b_ids_sorted = sorted(team_b_set)

        if snapshot_id is not None:
            since_ts, until_ts = self._fetch_tournament_window(snapshot_id)
        else:
            since_ts, until_ts = None, None
        match_last_game = self._seconds_expr("m.last_game_finished_at_ms")
        match_created = self._seconds_expr("m.created_at_ms")
        tournament_start = self._seconds_expr("t.start_time_ms")
        match_time_expr = f"COALESCE({match_last_game}, {match_created}, {tournament_start})"
        match_time_expr_without_tournament = f"COALESCE({match_last_game}, {match_created})"

        filters_with_tournament: list[str] = []
        filters_without_tournament: list[str] = []
        params: Dict[str, object] = {
            "snapshot_id": int(snapshot_id) if snapshot_id is not None else None,
            "team_a_ids": team_a_ids_sorted,
            "team_b_ids": team_b_ids_sorted,
            "limit": max(1, int(limit)),
        }

        score_columns = self._resolve_match_score_columns()
        if score_columns is not None:
            score_col_a, score_col_b = score_columns
            score_select = f"""
                    CASE
                        WHEN m.team1_id IN :team_a_ids THEN m.{score_col_a}
                        ELSE m.{score_col_b}
                      END AS team_a_score,
                    CASE
                        WHEN m.team1_id IN :team_a_ids THEN m.{score_col_b}
                        ELSE m.{score_col_a}
                      END AS team_b_score
            """.strip()
        else:
            score_select = None

        winner_score_select = "m.winner_team_id"
        if score_select:
            winner_score_select = f"""
                {winner_score_select},
                {score_select}
            """.strip()

        def _safe_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        if since_ts is not None:
            filters_with_tournament.append(f"{match_time_expr} >= :source_since_ts")
            filters_without_tournament.append(
                f"{match_time_expr_without_tournament} >= :source_since_ts"
            )
            params["source_since_ts"] = int(since_ts)

        if until_ts is not None:
            filters_with_tournament.append(f"{match_time_expr} <= :source_until_ts")
            filters_without_tournament.append(
                f"{match_time_expr_without_tournament} <= :source_until_ts"
            )
            params["source_until_ts"] = int(until_ts)

        match_where = " AND " + " AND ".join(filters_with_tournament) if filters_with_tournament else ""
        match_where_no_tournament = (
            " AND " + " AND ".join(filters_without_tournament)
            if filters_without_tournament
            else ""
        )

        def _run_head_to_head_query(where_with_tournament: str, where_without_tournament: str) -> list[Dict[str, object]]:
            sql_with_tournament_name = f"""
                SELECT
                    m.match_id,
                    m.tournament_id,
                    t.tournament_name,
                    m.team1_id,
                    m.team2_id,
                    {winner_score_select}
                    ,
                    NULLIF(btrim(t.format_hint), '') AS tournament_mode,
                    NULLIF(btrim(t.map_picking_style), '') AS map_picking_style,
                    t.tags AS tournament_tags,
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END AS event_time_ms
                FROM {self.schema}.matches m
                LEFT JOIN {self.schema}.tournaments t
                  ON t.tournament_id = m.tournament_id
                WHERE ((m.team1_id IN :team_a_ids AND m.team2_id IN :team_b_ids)
                       OR (m.team1_id IN :team_b_ids AND m.team2_id IN :team_a_ids))
                  {where_with_tournament}
                ORDER BY
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END DESC NULLS LAST
                LIMIT :limit
            """

            sql_with_tournament_name_column = f"""
                SELECT
                    m.match_id,
                    m.tournament_id,
                    t.name AS tournament_name,
                    m.team1_id,
                    m.team2_id,
                    {winner_score_select}
                    ,
                    NULLIF(btrim(t.format_hint), '') AS tournament_mode,
                    NULLIF(btrim(t.map_picking_style), '') AS map_picking_style,
                    t.tags AS tournament_tags,
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END AS event_time_ms
                FROM {self.schema}.matches m
                LEFT JOIN {self.schema}.tournaments t
                  ON t.tournament_id = m.tournament_id
                WHERE ((m.team1_id IN :team_a_ids AND m.team2_id IN :team_b_ids)
                       OR (m.team1_id IN :team_b_ids AND m.team2_id IN :team_a_ids))
                  {where_with_tournament}
                ORDER BY
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END DESC NULLS LAST
                LIMIT :limit
            """

            sql_without_tournament = f"""
                SELECT
                    m.match_id,
                    m.tournament_id,
                    NULL::text AS tournament_name,
                    m.team1_id,
                    m.team2_id,
                    {winner_score_select}
                    ,
                    NULL::text AS tournament_mode,
                    NULL::text AS map_picking_style,
                    NULL::jsonb AS tournament_tags,
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        ELSE NULL
                    END AS event_time_ms
                FROM {self.schema}.matches m
                WHERE ((m.team1_id IN :team_a_ids AND m.team2_id IN :team_b_ids)
                       OR (m.team1_id IN :team_b_ids AND m.team2_id IN :team_a_ids))
                  {where_without_tournament}
                ORDER BY
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        ELSE NULL
                    END DESC NULLS LAST
                LIMIT :limit
            """

            query_results: list[Dict[str, object]] = []
            for query in (
                sql_with_tournament_name,
                sql_with_tournament_name_column,
                sql_without_tournament,
            ):
                try:
                    with self.engine.connect() as conn:
                        query_results = [
                            dict(row)
                            for row in conn.execute(
                                text(query).bindparams(
                                    bindparam("team_a_ids", expanding=True),
                                    bindparam("team_b_ids", expanding=True),
                                ),
                                params,
                            ).mappings().all()
                        ]
                    break
                except SQLAlchemyError as exc:
                    if not (_is_missing_relation_error(exc) or _is_missing_column_error(exc)):
                        raise

            return query_results

        rows = _run_head_to_head_query(match_where, match_where_no_tournament)
        snapshot_window_enabled = bool(since_ts is not None or until_ts is not None)
        window_match_count = len(rows)
        fallback_to_all_time = False
        if not rows and snapshot_window_enabled:
            rows = _run_head_to_head_query("", "")

            if rows:
                fallback_to_all_time = True
                logger.info(
                    "head-to-head fallback to all-time matchups after snapshot window returned no matches"
                    " (snapshot_id=%s, team_a_count=%s, team_b_count=%s)",
                    snapshot_id,
                    len(team_a_ids_sorted),
                    len(team_b_ids_sorted),
                )

        match_rosters = self._fetch_match_rosters(
            rows, team_a_ids_sorted, team_b_ids_sorted
        )
        match_rounds = self._fetch_match_rounds(
            rows, team_a_ids_sorted, team_b_ids_sorted
        )
        tournament_scores = self._fetch_tournament_scores(
            [
                int(row["tournament_id"])
                for row in rows
                if row.get("tournament_id") is not None
            ]
        )
        team_name_rows = []
        if snapshot_id is not None:
            try:
                with self.engine.connect() as conn:
                    team_name_rows = conn.execute(
                        text(
                            f"""
                            SELECT team_id, team_name
                              FROM {self.schema}.team_search_embeddings
                             WHERE snapshot_id = :snapshot_id
                               AND team_id IN :team_ids
                            """
                        ).bindparams(bindparam("team_ids", expanding=True)),
                        {
                            "snapshot_id": int(snapshot_id),
                            "team_ids": team_a_ids_sorted + team_b_ids_sorted,
                        },
                    ).mappings().all()
            except SQLAlchemyError as exc:
                if not (_is_missing_relation_error(exc) or _is_missing_column_error(exc)):
                    raise

        team_names = {
            int(row["team_id"]): str(row["team_name"])
            for row in team_name_rows
            if row is not None and row.get("team_id") is not None
        }

        summary = {
            "team_a_id": team_a_ids_sorted[0],
            "team_b_id": team_b_ids_sorted[0],
            "team_a_ids": team_a_ids_sorted,
            "team_b_ids": team_b_ids_sorted,
            "team_a_names": [team_names.get(team_id, f"Team {team_id}") for team_id in team_a_ids_sorted],
            "team_b_names": [team_names.get(team_id, f"Team {team_id}") for team_id in team_b_ids_sorted],
            "team_a_name": None,
            "team_b_name": None,
            "total_matches": 0,
            "team_a_wins": 0,
            "team_b_wins": 0,
            "unresolved_matches": 0,
            "decided_matches": 0,
            "team_a_win_rate": 0.0,
            "team_b_win_rate": 0.0,
            "tournaments": 0,
            "tournament_tier_distribution": {},
            "tournament_tier_match_distribution": {},
            "snapshot_window_applied": bool(snapshot_id is not None and snapshot_window_enabled),
            "matched_all_time": bool(
                snapshot_id is not None
                and snapshot_window_enabled
                and window_match_count == 0
                and fallback_to_all_time
            ),
        }
        summary["team_a_name"] = summary["team_a_names"][0] if summary["team_a_names"] else "Team n/a"
        summary["team_b_name"] = summary["team_b_names"][0] if summary["team_b_names"] else "Team n/a"

        tournaments_seen: set[int] = set()
        tournaments_by_tier: Dict[str, int] = {}
        matches_by_tier: Dict[str, int] = {}
        tournament_tier_by_id: Dict[int, dict[str, str]] = {}
        matchups = []
        for row in rows:
            tournament_id = (
                int(row["tournament_id"]) if row["tournament_id"] is not None else None
            )
            winner_team_id = (
                int(row["winner_team_id"]) if row["winner_team_id"] is not None else None
            )
            team_a_score = _safe_float(row.get("team_a_score"))
            team_b_score = _safe_float(row.get("team_b_score"))
            roster_key = (int(row["match_id"]), tournament_id or 0)
            roster = match_rosters.get(
                roster_key,
                {"team_a": {"player_ids": [], "player_names": []}, "team_b": {"player_ids": [], "player_names": []}},
            )
            team_a_roster_ids = list(roster.get("team_a", {}).get("player_ids", []))
            team_a_roster_names = list(roster.get("team_a", {}).get("player_names", []))
            team_b_roster_ids = list(roster.get("team_b", {}).get("player_ids", []))
            team_b_roster_names = list(roster.get("team_b", {}).get("player_names", []))
            match_round_rows = match_rounds.get(roster_key, [])
            side = None
            if winner_team_id in team_a_set:
                side = "team_a"
            elif winner_team_id in team_b_set:
                side = "team_b"
            elif (
                team_a_score is not None
                and team_b_score is not None
                and team_a_score != team_b_score
            ):
                if team_a_score > team_b_score:
                    side = "team_a"
                else:
                    side = "team_b"

            if not match_round_rows and team_a_score is not None:
                match_round_rows = [
                    {
                        "round_no": None,
                        "maps_count": None,
                        "map_index": 1,
                        "map_name": None,
                        "map_mode": None,
                        "team_a_score": team_a_score,
                        "team_b_score": team_b_score,
                        "winner_team_id": winner_team_id,
                        "winner_side": side,
                    }
                ]

            if side is None:
                summary["unresolved_matches"] += 1
            elif side == "team_a":
                summary["team_a_wins"] += 1
            elif side == "team_b":
                summary["team_b_wins"] += 1
            else:
                summary["unresolved_matches"] += 1

            tournament_strength = tournament_scores.get(tournament_id)
            tier = _tournament_tier(tournament_strength)
            if tournament_id is not None:
                tournaments_seen.add(tournament_id)
                previous_tier = tournament_tier_by_id.get(int(tournament_id))
                if previous_tier is None:
                    tournaments_by_tier[tier["tier_id"]] = tournaments_by_tier.get(
                        tier["tier_id"], 0
                    ) + 1
                    tournament_tier_by_id[int(tournament_id)] = tier
                elif previous_tier["tier_id"] != tier["tier_id"]:
                    tournaments_by_tier[previous_tier["tier_id"]] = max(
                        0, tournaments_by_tier.get(previous_tier["tier_id"], 0) - 1
                    )
                    tournaments_by_tier[tier["tier_id"]] = tournaments_by_tier.get(
                        tier["tier_id"], 0
                    ) + 1
                    tournament_tier_by_id[int(tournament_id)] = tier

            matches_by_tier[tier["tier_id"]] = matches_by_tier.get(tier["tier_id"], 0) + 1

            matchups.append(
                {
                    "match_id": int(row["match_id"]),
                    "tournament_id": tournament_id,
                    "tournament_name": row["tournament_name"],
                    "tournament_mode": row.get("tournament_mode"),
                    "map_picking_style": row.get("map_picking_style"),
                    "tournament_tags": row.get("tournament_tags"),
                    "tournament_score": (
                        None if tournament_strength is None else round(float(tournament_strength), 4)
                    ),
                    "tournament_score_tier_id": tier["tier_id"],
                    "tournament_score_tier": tier["tier_label"],
                    "winner_team_id": winner_team_id,
                    "winner_side": side,
                    "team_a_score": team_a_score,
                    "team_b_score": team_b_score,
                    "team_a_roster": [
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                        }
                        for player_id, player_name in zip(
                            team_a_roster_ids,
                            team_a_roster_names,
                        )
                    ],
                    "team_b_roster": [
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                        }
                        for player_id, player_name in zip(
                            team_b_roster_ids,
                            team_b_roster_names,
                        )
                    ],
                    "event_time_ms": (
                        int(row["event_time_ms"]) if row["event_time_ms"] is not None else None
                    ),
                    "match_rounds": match_round_rows,
                    "team_a_is_winner": bool(side == "team_a"),
                    "team_b_is_winner": bool(side == "team_b"),
                }
            )

        summary["total_matches"] = len(rows)
        summary["tournaments"] = len(tournaments_seen)
        summary["tournament_tier_distribution"] = {
            "X": tournaments_by_tier.get("x", 0),
            "S+": tournaments_by_tier.get("s_plus", 0),
            "S": tournaments_by_tier.get("s", 0),
            "A+": tournaments_by_tier.get("a_plus", 0),
            "A": tournaments_by_tier.get("a", 0),
            "A-": tournaments_by_tier.get("a_minus", 0),
            "Unscored": tournaments_by_tier.get("unscored", 0),
        }
        summary["tournament_tier_match_distribution"] = {
            "X": matches_by_tier.get("x", 0),
            "S+": matches_by_tier.get("s_plus", 0),
            "S": matches_by_tier.get("s", 0),
            "A+": matches_by_tier.get("a_plus", 0),
            "A": matches_by_tier.get("a", 0),
            "A-": matches_by_tier.get("a_minus", 0),
            "Unscored": matches_by_tier.get("unscored", 0),
        }

        if summary["total_matches"] > 0:
            decided = summary["team_a_wins"] + summary["team_b_wins"]
            summary["decided_matches"] = decided
            if decided > 0:
                summary["team_a_win_rate"] = round(summary["team_a_wins"] / decided, 4)
                summary["team_b_win_rate"] = round(summary["team_b_wins"] / decided, 4)
            else:
                summary["team_a_win_rate"] = 0.0
                summary["team_b_win_rate"] = 0.0

        return {
            "snapshot_id": snapshot_id,
            "summary": summary,
            "matches": matchups,
        }

    def analytics_roster_diversity(
        self,
        *,
        snapshot_id: int,
        profile: str,
        min_similarity: float,
        max_player_overlap: float,
        min_cluster_size: int,
        limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_roster_diversity_candidates

        start_total = time.perf_counter()
        cache_entry = self._get_cached_snapshot_entry(snapshot_id)
        embeddings = cache_entry.rows
        cluster_map = self._get_cached_cluster_map(snapshot_id, profile)
        result = compute_roster_diversity_candidates(
            embeddings=embeddings,
            cluster_map=cluster_map,
            min_similarity=float(min_similarity),
            max_player_overlap=float(max_player_overlap),
            min_cluster_size=max(2, int(min_cluster_size)),
            limit=max(1, int(limit)),
        )
        logger.debug(
            "analytics_roster schema=%s snapshot_id=%s profile=%s elapsed_ms=%.1f",
            self.schema,
            int(snapshot_id),
            profile,
            (time.perf_counter() - start_total) * 1000.0,
        )
        return result

    def analytics_team_lab(
        self,
        *,
        snapshot_id: int,
        profile: str,
        team_id: int,
        neighbors: int,
    ) -> Optional[Dict[str, Any]]:
        from team_api.analytics_logic import build_team_lab

        cache_entry = self._get_cached_snapshot_entry(snapshot_id)
        embeddings = cache_entry.rows
        cluster_map = self._get_cached_cluster_map(snapshot_id, profile)

        sql = f"""
            SELECT
                CASE
                    WHEN m.team1_id = :team_id THEN m.team2_id
                    ELSE m.team1_id
                END AS opponent_team_id,
                CASE
                    WHEN m.winner_team_id = :team_id THEN 1
                    ELSE 0
                END AS is_win
            FROM {self.schema}.matches m
            WHERE (m.team1_id = :team_id OR m.team2_id = :team_id)
              AND m.winner_team_id IS NOT NULL
        """
        match_rows = self._fetch_rows(
            sql,
            {"team_id": int(team_id)},
            missing_default=[],
        )

        result = build_team_lab(
            team_id=int(team_id),
            embeddings=embeddings,
            cluster_map=cluster_map,
            matches=match_rows,
            neighbors=max(1, int(neighbors)),
        )
        if result is None:
            return None
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            **result,
        }

    def analytics_team_blend(
        self,
        *,
        snapshot_id: int,
        profile: str,
        team_id: int,
        semantic_weight: float,
        neighbors: int,
    ) -> Optional[Dict[str, Any]]:
        from team_api.analytics_logic import build_blended_neighbors

        cache_entry = self._get_cached_snapshot_entry(snapshot_id)
        embeddings = cache_entry.rows
        cluster_map = self._get_cached_cluster_map(snapshot_id, profile)
        result = build_blended_neighbors(
            team_id=int(team_id),
            embeddings=embeddings,
            cluster_map=cluster_map,
            semantic_weight=float(semantic_weight),
            neighbors=max(1, int(neighbors)),
        )
        if result is None:
            return None
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            **result,
        }

    def analytics_outliers(
        self,
        *,
        snapshot_id: int,
        profile: str,
        limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_outliers

        start_total = time.perf_counter()
        cache_entry = self._get_cached_snapshot_entry(snapshot_id)
        embeddings = cache_entry.rows
        cluster_map = self._get_cached_cluster_map(snapshot_id, profile)
        outliers = compute_outliers(
            embeddings,
            cluster_map,
            limit=max(1, int(limit)),
        )
        payload = {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            "count": len(outliers),
            "outliers": outliers,
        }
        logger.debug(
            "analytics_outliers schema=%s snapshot_id=%s profile=%s elapsed_ms=%.1f",
            self.schema,
            int(snapshot_id),
            profile,
            (time.perf_counter() - start_total) * 1000.0,
        )
        return payload

    def analytics_space(
        self,
        *,
        snapshot_id: int,
        profile: str,
        max_points: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_space_projection

        start_total = time.perf_counter()
        cache_entry = self._get_cached_snapshot_entry(snapshot_id)
        embeddings = cache_entry.rows
        cluster_map = self._get_cached_cluster_map(snapshot_id, profile)
        projection = compute_space_projection(
            embeddings,
            cluster_map,
            max_points=max(50, min(int(max_points), 3000)),
        )
        payload = {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            **projection,
        }
        logger.debug(
            "analytics_space schema=%s snapshot_id=%s profile=%s elapsed_ms=%.1f",
            self.schema,
            int(snapshot_id),
            profile,
            (time.perf_counter() - start_total) * 1000.0,
        )
        return payload

    def analytics_drift(
        self,
        *,
        profile: str,
        current_snapshot_id: Optional[int],
        previous_snapshot_id: Optional[int],
        top_movers: int,
    ) -> Optional[Dict[str, Any]]:
        from team_api.analytics_logic import compute_snapshot_drift

        start_total = time.perf_counter()
        snapshots = self.list_completed_snapshots(limit=10)
        if not snapshots:
            return None

        if current_snapshot_id is None:
            current_snapshot_id = int(snapshots[0]["run_id"])

        if previous_snapshot_id is None:
            previous_snapshot_id = None
            for snap in snapshots:
                rid = int(snap["run_id"])
                if rid != int(current_snapshot_id):
                    previous_snapshot_id = rid
                    break

        if previous_snapshot_id is None:
            payload = {
                "current_snapshot_id": int(current_snapshot_id),
                "previous_snapshot_id": None,
                "summary": {
                    "shared_teams": 0,
                    "new_teams": 0,
                    "dropped_teams": 0,
                    "cluster_switches": 0,
                    "newly_clustered": 0,
                    "newly_noise": 0,
                    "avg_embedding_drift": 0.0,
                    "p90_embedding_drift": 0.0,
                },
                "top_embedding_movers": [],
                "top_volatility_shifts": [],
            }
            logger.debug(
                "analytics_drift schema=%s profile=%s current=%s previous=%s elapsed_ms=%.1f",
                self.schema,
                profile,
                int(current_snapshot_id),
                None,
                (time.perf_counter() - start_total) * 1000.0,
            )
            return payload

        current_embeddings = self._get_cached_snapshot_entry(
            int(current_snapshot_id)
        ).rows
        previous_embeddings = self._get_cached_snapshot_entry(
            int(previous_snapshot_id)
        ).rows
        current_cluster_map = self._get_cached_cluster_map(
            int(current_snapshot_id), profile
        )
        previous_cluster_map = self._get_cached_cluster_map(
            int(previous_snapshot_id), profile
        )

        drift = compute_snapshot_drift(
            current_snapshot_id=int(current_snapshot_id),
            previous_snapshot_id=int(previous_snapshot_id),
            current_embeddings=current_embeddings,
            previous_embeddings=previous_embeddings,
            current_cluster_map=current_cluster_map,
            previous_cluster_map=previous_cluster_map,
            top_movers=max(1, int(top_movers)),
        )
        payload = {
            "cluster_mode": profile,
            **drift,
        }
        logger.debug(
            "analytics_drift schema=%s profile=%s current=%s previous=%s elapsed_ms=%.1f",
            self.schema,
            profile,
            int(current_snapshot_id),
            int(previous_snapshot_id),
            (time.perf_counter() - start_total) * 1000.0,
        )
        return payload
