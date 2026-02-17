from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from team_api.store import EmbeddingRow


def _normalize_embeddings(
    embeddings: Sequence[EmbeddingRow],
) -> List[EmbeddingRow]:
    normalized: List[EmbeddingRow] = []
    final_dim: int | None = None
    semantic_dim: int | None = None
    identity_dim: int | None = None

    for row in embeddings:
        final_vector = np.asarray(row.final_vector)
        semantic_vector = np.asarray(row.semantic_vector)
        identity_vector = np.asarray(row.identity_vector)

        if final_vector.ndim != 1 or semantic_vector.ndim != 1 or identity_vector.ndim != 1:
            continue

        if final_dim is None:
            final_dim = int(final_vector.size)
            semantic_dim = int(semantic_vector.size)
            identity_dim = int(identity_vector.size)
        elif (
            int(final_vector.size) != final_dim
            or int(semantic_vector.size) != semantic_dim
            or int(identity_vector.size) != identity_dim
        ):
            continue

        normalized.append(
            EmbeddingRow(
                team_id=int(row.team_id),
                tournament_id=row.tournament_id,
                team_name=row.team_name,
                event_time_ms=row.event_time_ms,
                lineup_count=row.lineup_count,
                tournament_count=row.tournament_count,
                semantic_vector=semantic_vector.astype(np.float64),
                identity_vector=identity_vector.astype(np.float64),
                final_vector=final_vector.astype(np.float64),
                top_lineup_summary=row.top_lineup_summary,
                unique_player_count=row.unique_player_count,
                distinct_lineup_count=row.distinct_lineup_count,
                top_lineup_share=row.top_lineup_share,
                lineup_entropy=row.lineup_entropy,
                effective_lineups=row.effective_lineups,
                top_lineup_player_ids=row.top_lineup_player_ids,
                top_lineup_player_names=row.top_lineup_player_names,
            )
        )

    return normalized


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def _weighted_centroid(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=np.float64)
    if weights.size == 0 or float(weights.sum()) <= 0:
        return _normalize(vectors.mean(axis=0))
    centroid = (vectors * weights[:, None]).sum(axis=0) / float(weights.sum())
    return _normalize(centroid)


def _parse_player_ids(result: Dict[str, object]) -> tuple[int, ...]:
    raw = result.get("top_lineup_player_ids")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        values = raw.split(",")
        parsed = []
        for value in values:
            text = value.strip().strip('"').strip("'")
            if not text:
                continue
            try:
                parsed.append(int(text))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(set(parsed)))
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(out)))


def _parse_player_set(result: Dict[str, object]) -> set[int]:
    return set(_parse_player_ids(result))


def _normalize_player_name(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return "Unknown Player" if not text else text


def _parse_player_id_sequence(result: Dict[str, object]) -> tuple[int, ...]:
    raw = result.get("top_lineup_player_ids")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        values = raw.split(",")
        parsed: list[int] = []
        for value in values:
            text = value.strip().strip('"').strip("'")
            if not text:
                continue
            try:
                parsed.append(int(text))
            except (TypeError, ValueError):
                continue
        return tuple(dict.fromkeys(parsed))

    parsed: list[int] = []
    for value in raw:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(dict.fromkeys(parsed))


def _parse_player_name_sequence(result: Dict[str, object]) -> tuple[str, ...]:
    raw = result.get("top_lineup_player_names")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        values = raw.split(",")
        return tuple(_normalize_player_name(value) for value in values)
    return tuple(_normalize_player_name(value) for value in raw)


def _parse_top_lineup_summary_count(value: object) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    x_idx = raw.find("x")
    if x_idx <= 0:
        return None
    prefix = raw[:x_idx].strip()
    try:
        parsed = int(prefix)
    except (TypeError, ValueError):
        return None
    return parsed


def _estimate_top_lineup_matches(result: Dict[str, object]) -> int:
    parsed_summary = _parse_top_lineup_summary_count(result.get("top_lineup_summary"))
    if parsed_summary is not None:
        return max(0, parsed_summary)

    lineup_count = int(result.get("lineup_count") or 0)
    if lineup_count <= 0:
        return 0

    share = float(result.get("top_lineup_share") or 0.0)
    if share <= 0.0:
        return 0

    count = int(round(lineup_count * share))
    return 1 if count <= 0 else count


def _build_core_lineup_players(
    group: list[Dict[str, object]],
) -> list[Dict[str, object]]:
    player_metrics: Dict[int, Dict[str, object]] = {}
    for result in group:
        top_matches = _estimate_top_lineup_matches(result)
        player_ids = _parse_player_id_sequence(result)
        if not player_ids:
            continue

        player_names = _parse_player_name_sequence(result)
        for i, player_id in enumerate(player_ids):
            name = (
                player_names[i]
                if i < len(player_names)
                else f"Player {player_id}"
            )
            entry = player_metrics.get(int(player_id))
            if entry is None:
                player_metrics[int(player_id)] = {
                    "player_id": int(player_id),
                    "player_name": name,
                    "matches_played": 0,
                }
                entry = player_metrics[int(player_id)]
            else:
                current_name = str(entry.get("player_name") or "")
                if current_name == "Unknown Player" and name != "Unknown Player":
                    entry["player_name"] = name

            entry["matches_played"] = int(entry["matches_played"]) + top_matches

    output: list[Dict[str, object]] = []
    for player_id, payload in player_metrics.items():
        output.append(
            {
                "player_id": player_id,
                "player_name": str(payload.get("player_name") or "Unknown Player"),
                "matches_played": int(payload.get("matches_played") or 0),
                "sendou_url": f"https://sendou.ink/u/{player_id}",
            }
        )

    output.sort(key=lambda row: (-int(row["matches_played"]), int(row["player_id"])))
    return output


def _lineup_overlap_fraction(set_a: set[int], set_b: set[int]) -> float:
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b)) / float(len(union))


def _allowed_player_substitutions(player_count: int) -> int:
    """Return how many lineup player substitutions are allowed during consolidation.

    We start with one tolerated substitute for base 4-player lineups and only
    loosen this slowly as team size grows, so larger rosters require proportionally
    more evidence to match.
    """

    if player_count <= 4:
        return 1
    return 1 + ((player_count - 4) // 3)


def _lineup_nearly_matches(set_a: set[int], set_b: set[int]) -> bool:
    if not set_a or not set_b:
        return False
    if len(set_a) < 3 or len(set_b) < 3:
        return False

    len_a = len(set_a)
    len_b = len(set_b)
    size_delta = abs(len_a - len_b)
    if size_delta > 2:
        return False

    min_len = min(len_a, len_b)
    allowed_substitutions = _allowed_player_substitutions(min_len)
    max_symmetric_diff = (2 * allowed_substitutions) + min(size_delta, 1)

    if len(set_a ^ set_b) > max_symmetric_diff:
        return False

    overlap = len(set_a & set_b)
    required_overlap = max(3, min_len - allowed_substitutions)
    return overlap >= required_overlap


def _lineup_signature(result: Dict[str, object]) -> tuple[int, ...]:
    return _parse_player_ids(result)


def _eligible_for_consolidation(
    a: Dict[str, object], b: Dict[str, object], *, min_overlap: float
) -> bool:
    players_a = _parse_player_set(a)
    players_b = _parse_player_set(b)
    signature_a = _lineup_signature(a)
    signature_b = _lineup_signature(b)

    if signature_a and signature_b and signature_a == signature_b:
        return True

    if _lineup_nearly_matches(players_a, players_b):
        return True

    if players_a and players_b and len(players_a) >= 3 and len(players_b) >= 3:
        overlap = _lineup_overlap_fraction(players_a, players_b)
        if overlap < min_overlap:
            return False

        count_a = int(a.get("lineup_count") or 0)
        count_b = int(b.get("lineup_count") or 0)
        count_delta = abs(count_a - count_b)
        count_threshold = max(5, int(0.35 * max(max(count_a, count_b), 1)))
        if count_delta > count_threshold:
            return False

        distinct_a = float(a.get("distinct_lineup_count") or 0.0)
        distinct_b = float(b.get("distinct_lineup_count") or 0.0)
        if abs(distinct_a - distinct_b) > 2.5:
            return False

        top_a = float(a.get("top_lineup_share") or 0.0)
        top_b = float(b.get("top_lineup_share") or 0.0)
        if abs(top_a - top_b) > 0.3:
            return False
        return True

    return False


def consolidate_ranked_results(
    results: List[Dict[str, object]],
    *,
    min_overlap: float = 0.72,
) -> List[Dict[str, object]]:
    if not results:
        return []
    if min_overlap <= 0.0:
        return results

    n_results = len(results)
    if n_results <= 1:
        return [dict(row) for row in results]

    parent = list(range(n_results))

    def find(idx: int) -> int:
        node = idx
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        parent[right_root] = left_root

    for left in range(n_results):
        base = results[left]
        for right in range(left + 1, n_results):
            candidate = results[right]
            if _eligible_for_consolidation(base, candidate, min_overlap=min_overlap):
                union(left, right)

    groups_by_root: Dict[int, List[Dict[str, object]]] = {}
    for idx, result in enumerate(results):
        root = find(idx)
        groups_by_root.setdefault(root, []).append(result)

    grouped = []
    for root in sorted(groups_by_root):
        group = groups_by_root[root]
        if len(group) == 1:
            grouped.append(dict(group[0]))
            continue

        rep = dict(group[0])
        base_match_count = int(rep.get("match_count") or rep.get("lineup_count", 0) or 0)
        base_tournament_count = int(rep.get("tournament_count", 0) or 0)
        rep_top_lineup_signature_matches: dict[tuple[int, ...], int] = {}
        rep_top_lineup_signature_names: dict[tuple[int, ...], tuple[str, ...]] = {}

        rep_signature = tuple(_parse_player_id_sequence(rep))
        if rep_signature:
            rep_top_lineup_signature_matches[rep_signature] = _estimate_top_lineup_matches(
                rep
            )
            rep_top_lineup_signature_names[rep_signature] = tuple(_parse_player_name_sequence(rep))

        rep["is_consolidated"] = True
        rep["consolidated_team_count"] = len(group)

        total_match_count = base_match_count
        total_tournament_count = base_tournament_count
        max_distinct_lineup_count = int(rep.get("distinct_lineup_count", 0) or 0)
        aliases: list[Dict[str, object]] = []
        for alias in group[1:]:
            alias_match_count = int(alias.get("match_count") or alias.get("lineup_count", 0) or 0)
            alias_tournament_count = int(alias.get("tournament_count", 0) or 0)
            alias_signature = tuple(_parse_player_id_sequence(alias))
            if alias_signature:
                rep_top_lineup_signature_matches[alias_signature] = rep_top_lineup_signature_matches.get(
                    alias_signature,
                    0,
                ) + _estimate_top_lineup_matches(alias)
                rep_top_lineup_signature_names.setdefault(
                    alias_signature,
                    tuple(_parse_player_name_sequence(alias)),
                )
            alias_distinct_lineups = int(alias.get("distinct_lineup_count", 0) or 0)
            if alias_distinct_lineups > max_distinct_lineup_count:
                max_distinct_lineup_count = alias_distinct_lineups
            total_match_count += alias_match_count
            total_tournament_count += alias_tournament_count
            aliases.append(
                {
                    "team_id": alias["team_id"],
                    "team_name": alias["team_name"],
                    "tournament_id": alias.get("tournament_id"),
                    "event_time_ms": alias.get("event_time_ms"),
                    "match_count": alias_match_count,
                    "tournament_count": alias_tournament_count,
                }
            )

        chosen_signature = ()
        chosen_lineup_match_count = 0
        if rep_top_lineup_signature_matches:
            chosen_signature, chosen_lineup_match_count = max(
                rep_top_lineup_signature_matches.items(),
                key=lambda item: (item[1], len(item[0])),
            )
        rep["top_lineup_match_count"] = chosen_lineup_match_count
        rep["top_lineup_match_share"] = (
            chosen_lineup_match_count / total_match_count if total_match_count > 0 else 0.0
        )
        rep["top_lineup_player_ids"] = list(chosen_signature)
        rep["top_lineup_player_names"] = list(
            rep_top_lineup_signature_names.get(chosen_signature, ())
        )
        rep["distinct_lineup_count"] = max_distinct_lineup_count
        rep["match_count"] = total_match_count
        rep["tournament_count"] = total_tournament_count
        rep["consolidated_teams"] = aliases
        rep["core_lineup_players"] = _build_core_lineup_players(group)
        rep["consolidated_team_ids"] = [rep["team_id"]] + [
            alias["team_id"] for alias in aliases
        ]
        rep["consolidated_team_names"] = [rep["team_name"]] + [
            alias["team_name"] for alias in aliases
        ]
        grouped.append(rep)

    return grouped


def rank_similar_teams(
    embeddings: Sequence[EmbeddingRow],
    target_team_ids: Sequence[int],
    cluster_map: Dict[int, Dict[str, object]],
    top_n: int,
    min_relevance: float = 0.0,
) -> Dict[str, object]:
    embeddings = _normalize_embeddings(embeddings)
    if not embeddings:
        return {"query": {}, "results": []}

    index = {row.team_id: i for i, row in enumerate(embeddings)}
    target_idx = [index[tid] for tid in target_team_ids if tid in index]
    if not target_idx:
        return {"query": {"matched_team_ids": []}, "results": []}

    finals = np.stack([row.final_vector for row in embeddings], axis=0)
    semantics = np.stack([row.semantic_vector for row in embeddings], axis=0)
    identities = np.stack([row.identity_vector for row in embeddings], axis=0)
    weights = np.array([embeddings[i].lineup_count for i in target_idx], dtype=np.float64)

    query_final = _weighted_centroid(finals[target_idx], weights)
    query_sem = _weighted_centroid(semantics[target_idx], weights)
    query_id = _weighted_centroid(identities[target_idx], weights)

    sims_final = finals @ query_final
    sims_sem = semantics @ query_sem
    sims_id = identities @ query_id

    order = np.argsort(-sims_final)
    min_relevance = max(0.0, min(float(min_relevance), 1.0))
    top_n = max(1, min(int(top_n), len(order)))
    target_set = set(int(tid) for tid in target_team_ids)

    results: List[Dict[str, object]] = []
    filtered = [
        int(idx) for idx in order if float(sims_final[idx]) >= min_relevance
    ]
    filtered = filtered[:top_n]

    for rank, idx in enumerate(filtered, start=1):
        row = embeddings[int(idx)]
        cluster = cluster_map.get(row.team_id)

        top_lineup_players: List[Dict[str, object]] = []
        core_lineup_players: List[Dict[str, object]] = []
        player_ids = list(row.top_lineup_player_ids)
        player_names = list(row.top_lineup_player_names)
        row_dict = {
            "lineup_count": int(row.lineup_count),
            "top_lineup_share": float(row.top_lineup_share),
            "top_lineup_summary": row.top_lineup_summary,
        }
        top_lineup_match_count = _estimate_top_lineup_matches(row_dict)
        result_top_lineup_player_ids = list(player_ids)
        if top_lineup_match_count > int(row.lineup_count):
            top_lineup_match_count = int(row.lineup_count)
        if player_ids:
            for i, player_id in enumerate(player_ids):
                raw_name = player_names[i] if i < len(player_names) else None
                name = str(raw_name).strip() if raw_name else "Unknown Player"
                pid = int(player_id)
                core_lineup_players.append(
                    {
                        "player_id": pid,
                        "player_name": name or "Unknown Player",
                        "matches_played": top_lineup_match_count,
                        "sendou_url": f"https://sendou.ink/u/{pid}",
                    }
                )
                top_lineup_players.append(
                    {
                        "player_id": pid,
                        "player_name": name or "Unknown Player",
                        "sendou_url": f"https://sendou.ink/u/{pid}",
                    }
                )
        else:
            for raw_name in player_names:
                name = str(raw_name).strip() if raw_name else "Unknown Player"
                top_lineup_players.append(
                    {
                        "player_id": None,
                        "player_name": name or "Unknown Player",
                        "sendou_url": None,
                    }
                )

        results.append(
            {
                "rank": rank,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "tournament_id": row.tournament_id,
                "event_time_ms": row.event_time_ms,
                "lineup_count": row.lineup_count,
                "match_count": row.lineup_count,
                "tournament_count": row.tournament_count,
                "unique_player_count": row.unique_player_count,
                "distinct_lineup_count": row.distinct_lineup_count,
                "top_lineup_match_count": top_lineup_match_count,
                "top_lineup_match_share": (
                    top_lineup_match_count / int(row.lineup_count)
                    if int(row.lineup_count) > 0
                    else 0.0
                ),
                "top_lineup_share": row.top_lineup_share,
                "top_lineup_player_count": len(result_top_lineup_player_ids),
                "lineup_entropy": row.lineup_entropy,
                "effective_lineups": row.effective_lineups,
                "sim_to_query": float(sims_final[idx]),
                "sim_semantic_to_query": float(sims_sem[idx]),
                "sim_identity_to_query": float(sims_id[idx]),
                "top_lineup_summary": row.top_lineup_summary,
                "top_lineup_player_ids": result_top_lineup_player_ids,
                "top_lineup_player_names": list(row.top_lineup_player_names),
                "top_lineup_players": top_lineup_players,
                "core_lineup_players": core_lineup_players,
                "cluster_id": int(cluster["cluster_id"]) if cluster else None,
                "cluster_size": int(cluster["cluster_size"]) if cluster else None,
                "representative_team_name": (
                    cluster.get("representative_team_name") if cluster else None
                ),
                "is_clustered": bool(cluster),
                "is_query_match": row.team_id in target_set,
            }
        )

    return {
        "query": {
            "matched_team_ids": [embeddings[i].team_id for i in target_idx],
            "matched_team_names": [embeddings[i].team_name for i in target_idx],
        },
        "results": results,
    }
