from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from shared_lib.team_vector_utils import (
    canonicalize_player_ids,
    hash_index,
    hash_sign,
    parse_lineup_key,
    unordered_player_pairs,
)
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
                roster_player_ids=row.roster_player_ids,
                roster_player_names=row.roster_player_names,
                roster_player_match_counts=row.roster_player_match_counts,
                player_support=dict(row.player_support),
                pair_support=dict(row.pair_support),
                lineup_variant_counts=dict(row.lineup_variant_counts),
            )
        )

    return normalized


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def _clamp01(value: float, *, default: float = 0.0) -> float:
    if not np.isfinite(value):
        return default
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _blend_with_best_member(
    group_value: float,
    best_member_value: float,
    *,
    strength: float = 0.45,
) -> float:
    group = _clamp01(group_value)
    best = _clamp01(best_member_value)
    if best <= group:
        return group
    weight = _clamp01(float(strength), default=0.45)
    return _clamp01(group + (weight * (best - group)))


def _apply_lineup_boost(
    base_score: float,
    lineup_overlap: float,
    *,
    strength: float = 0.35,
) -> float:
    base = _clamp01(base_score)
    lineup = _clamp01(lineup_overlap)
    if lineup <= 0.0:
        return base
    weight = _clamp01(float(strength), default=0.35)
    return _clamp01(base + (weight * lineup * (1.0 - base)))


def _team_stability(row: EmbeddingRow) -> float:
    top_share = _clamp01(float(row.top_lineup_share or 0.0))
    entropy = _clamp01(float(row.lineup_entropy or 0.0))
    distinct = int(row.distinct_lineup_count or 0)
    effective = float(row.effective_lineups or 0.0)

    # Unit tests and partially-populated rows often omit lineup-distribution
    # metadata. Treat those as neutral instead of artificially "stable".
    if top_share <= 0.0 and distinct <= 0 and effective <= 0.0:
        return 0.5

    return _clamp01(
        (0.60 * top_share) + (0.40 * (1.0 - entropy)),
        default=0.5,
    )


def _weighted_centroid(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=np.float64)
    if weights.size == 0 or float(weights.sum()) <= 0:
        return _normalize(vectors.mean(axis=0))
    centroid = (vectors * weights[:, None]).sum(axis=0) / float(weights.sum())
    return _normalize(centroid)


def _build_query_lineup_signatures(
    player_ids: Sequence[int],
    *,
    lineup_size: int = 4,
    max_query_players: int = 8,
) -> tuple[tuple[int, ...], ...]:
    canonical = canonicalize_player_ids(player_ids)
    if len(canonical) < lineup_size:
        return ()
    if len(canonical) == lineup_size:
        return (canonical,)
    if len(canonical) > max_query_players:
        return ()
    return tuple(
        tuple(int(player_id) for player_id in subset)
        for subset in combinations(canonical, lineup_size)
    )


def build_query_similarity_profile(
    embeddings: Sequence[EmbeddingRow],
    target_idx: Sequence[int],
    finals: np.ndarray,
    semantics: np.ndarray,
    identities: np.ndarray,
    query_team_weights: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    if not target_idx:
        empty = np.array([], dtype=np.float64)
        return {
            "centroid_weights": empty,
            "query_final": empty,
            "query_semantic": empty,
            "query_identity": empty,
            "ann_query_vector": empty,
            "query_stability": 0.5,
            "semantic_weight": 0.5,
            "identity_weight": 0.5,
        }

    target_rows = [embeddings[int(idx)] for idx in target_idx]
    stability = np.asarray(
        [_team_stability(row) for row in target_rows],
        dtype=np.float64,
    )
    lineup_weights = np.asarray(
        [max(float(row.lineup_count or 0), 1.0) for row in target_rows],
        dtype=np.float64,
    )
    centroid_weights = np.sqrt(lineup_weights) * (0.75 + (0.25 * stability))
    if query_team_weights:
        external_weights = np.asarray(
            [
                max(float(query_team_weights.get(int(row.team_id), 1.0)), 0.0)
                for row in target_rows
            ],
            dtype=np.float64,
        )
        if external_weights.size and float(np.max(external_weights)) > 0.0:
            centroid_weights = centroid_weights * external_weights

    if centroid_weights.size == 0 or float(np.sum(centroid_weights)) <= 0.0:
        centroid_weights = np.ones(len(target_idx), dtype=np.float64)

    query_stability = float(np.average(stability, weights=centroid_weights))
    identity_weight = min(0.85, max(0.20, 0.15 + (0.70 * query_stability)))
    semantic_weight = 1.0 - identity_weight

    query_final = _weighted_centroid(finals[list(target_idx)], centroid_weights)
    query_semantic = _weighted_centroid(semantics[list(target_idx)], centroid_weights)
    query_identity = _weighted_centroid(identities[list(target_idx)], centroid_weights)
    ann_query_vector = _normalize(
        np.concatenate(
            [
                semantic_weight * query_semantic,
                identity_weight * query_identity,
            ]
        )
    )

    return {
        "centroid_weights": centroid_weights,
        "query_final": query_final,
        "query_semantic": query_semantic,
        "query_identity": query_identity,
        "ann_query_vector": ann_query_vector,
        "query_stability": query_stability,
        "semantic_weight": semantic_weight,
        "identity_weight": identity_weight,
    }


def build_query_semantic_vector(
    player_ids: Sequence[int],
    semantic_dim: int,
) -> np.ndarray:
    dim = max(0, int(semantic_dim))
    vec = np.zeros(dim, dtype=np.float64)
    if dim <= 0:
        return vec

    for player_id in canonicalize_player_ids(player_ids):
        idx = hash_index(player_id, dim, "sem_idx")
        sign = hash_sign(player_id, "sem_sign")
        vec[idx] += sign
    return _normalize(vec)


def build_query_identity_vector(
    player_ids: Sequence[int],
    identity_dim: int,
    idf_lookup: Dict[int, float],
) -> np.ndarray:
    dim = max(0, int(identity_dim))
    vec = np.zeros(dim, dtype=np.float64)
    if dim <= 0:
        return vec

    for player_id in canonicalize_player_ids(player_ids):
        idx = hash_index(player_id, dim, "id_idx")
        sign = hash_sign(player_id, "id_sign")
        vec[idx] += sign * float(idf_lookup.get(int(player_id), 1.0))
    return _normalize(vec)


def build_query_final_vector(
    player_ids: Sequence[int],
    *,
    semantic_dim: int,
    identity_dim: int,
    identity_beta: float,
    idf_lookup: Dict[int, float],
) -> np.ndarray:
    semantic = build_query_semantic_vector(player_ids, semantic_dim)
    identity = build_query_identity_vector(player_ids, identity_dim, idf_lookup)
    return _normalize(
        np.concatenate(
            [
                semantic,
                float(identity_beta) * identity,
            ]
        )
    )


def build_player_query_profile(
    player_ids: Sequence[int],
    *,
    semantic_dim: int,
    identity_dim: int,
    identity_beta: float,
    idf_lookup: Dict[int, float],
) -> Dict[str, object]:
    canonical_player_ids = canonicalize_player_ids(player_ids)
    semantic = build_query_semantic_vector(canonical_player_ids, semantic_dim)
    identity = build_query_identity_vector(
        canonical_player_ids,
        identity_dim,
        idf_lookup,
    )
    final = _normalize(
        np.concatenate(
            [
                semantic,
                float(identity_beta) * identity,
            ]
        )
    )
    query_pairs = unordered_player_pairs(canonical_player_ids)
    query_lineup_signatures = _build_query_lineup_signatures(canonical_player_ids)
    player_weights = {
        int(player_id): max(float(idf_lookup.get(int(player_id), 1.0)), 1e-9)
        for player_id in canonical_player_ids
    }
    pair_weights = {
        (int(left), int(right)): player_weights.get(int(left), 1e-9)
        * player_weights.get(int(right), 1e-9)
        for left, right in query_pairs
    }

    norm_denom = 1.0 + max(float(identity_beta), 0.0)
    return {
        "query_mode": "whole_set",
        "query_final": final,
        "query_semantic": semantic,
        "query_identity": identity,
        "ann_query_vector": final,
        "query_player_ids": canonical_player_ids,
        "query_pairs": query_pairs,
        "query_lineup_signatures": query_lineup_signatures,
        "query_player_weights": player_weights,
        "query_pair_weights": pair_weights,
        "query_stability": None,
        "semantic_weight": 1.0 / norm_denom,
        "identity_weight": max(float(identity_beta), 0.0) / norm_denom,
        "ranking_algorithm": "whole_set_hashed_v1",
    }


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
    lineup_variant_counts = _extract_lineup_variant_counts(result)
    if lineup_variant_counts:
        _, count = _select_top_lineup_variant(lineup_variant_counts)
        if count > 0:
            return int(count)

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


def _coerce_lineup_signature(value: object) -> tuple[int, ...]:
    if isinstance(value, tuple):
        try:
            parsed = tuple(int(part) for part in value)
        except (TypeError, ValueError):
            return ()
        canonical = canonicalize_player_ids(parsed)
        return canonical if len(canonical) == len(parsed) else ()
    if isinstance(value, list):
        try:
            parsed = tuple(int(part) for part in value)
        except (TypeError, ValueError):
            return ()
        canonical = canonicalize_player_ids(parsed)
        return canonical if len(canonical) == len(parsed) else ()
    parsed_key = parse_lineup_key(value)
    return parsed_key or ()


def _extract_lineup_variant_counts(result: Dict[str, object]) -> Dict[tuple[int, ...], int]:
    raw_row = result.get("_embedding_row")
    if isinstance(raw_row, EmbeddingRow) and raw_row.lineup_variant_counts:
        return {
            tuple(signature): int(count)
            for signature, count in raw_row.lineup_variant_counts.items()
            if signature and int(count) > 0
        }

    raw_counts = result.get("_lineup_variant_counts")
    if isinstance(raw_counts, dict):
        out: Dict[tuple[int, ...], int] = {}
        for raw_key, raw_value in raw_counts.items():
            signature = _coerce_lineup_signature(raw_key)
            if not signature:
                continue
            try:
                count = int(raw_value)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            out[signature] = out.get(signature, 0) + int(count)
        if out:
            return out

    signature = _parse_player_id_sequence(result)
    top_matches = _estimate_top_lineup_matches_fallback(result)
    if signature and top_matches > 0:
        return {signature: int(top_matches)}
    return {}


def _estimate_top_lineup_matches_fallback(result: Dict[str, object]) -> int:
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


def _select_top_lineup_variant(
    lineup_variant_counts: Dict[tuple[int, ...], int],
) -> tuple[tuple[int, ...], int]:
    if not lineup_variant_counts:
        return (), 0
    signature, count = max(
        lineup_variant_counts.items(),
        key=lambda item: (int(item[1]), len(item[0]), item[0]),
    )
    return tuple(signature), int(count)


def _group_player_name_lookup(group: List[Dict[str, object]]) -> Dict[int, str]:
    player_names: Dict[int, str] = {}
    for result in group:
        row = result.get("_embedding_row")
        if isinstance(row, EmbeddingRow):
            for idx, player_id in enumerate(row.roster_player_ids):
                if int(player_id) <= 0:
                    continue
                raw_name = (
                    row.roster_player_names[idx]
                    if idx < len(row.roster_player_names)
                    else None
                )
                name = _normalize_player_name(raw_name)
                if name != "Unknown Player":
                    player_names[int(player_id)] = name

        ids = _parse_player_id_sequence(result)
        names = _parse_player_name_sequence(result)
        for idx, player_id in enumerate(ids):
            if idx >= len(names):
                continue
            name = _normalize_player_name(names[idx])
            if name != "Unknown Player":
                player_names[int(player_id)] = name
    return player_names


def _build_core_lineup_players(
    group: list[Dict[str, object]],
) -> list[Dict[str, object]]:
    player_metrics: Dict[int, Dict[str, object]] = {}
    for result in group:
        if isinstance(result.get("core_lineup_players"), list):
            for player in result.get("core_lineup_players") or []:
                if not isinstance(player, dict):
                    continue
                player_id = player.get("player_id")
                if player_id is None:
                    continue
                try:
                    pid = int(player_id)
                except (TypeError, ValueError):
                    continue
                if pid <= 0:
                    continue
                name = _normalize_player_name(player.get("player_name"))
                matches_played = int(player.get("matches_played") or 0)

                entry = player_metrics.get(pid)
                if entry is None:
                    entry = {
                        "player_id": pid,
                        "player_name": name,
                        "matches_played": 0,
                    }
                    player_metrics[pid] = entry
                else:
                    current_name = str(entry.get("player_name") or "")
                    if current_name == "Unknown Player" and name != "Unknown Player":
                        entry["player_name"] = name

                entry["matches_played"] = int(entry["matches_played"]) + matches_played
            continue

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
            rep = dict(group[0])
            rep["_consolidated_member_results"] = [dict(group[0])]
            grouped.append(rep)
            continue

        rep = dict(group[0])
        rep["_consolidated_member_results"] = [dict(item) for item in group]
        base_match_count = int(rep.get("match_count") or rep.get("lineup_count", 0) or 0)
        base_tournament_count = int(rep.get("tournament_count", 0) or 0)
        merged_lineup_variant_counts = _extract_lineup_variant_counts(rep)

        rep_signature = tuple(_parse_player_id_sequence(rep))
        if rep_signature:
            merged_lineup_variant_counts.setdefault(
                rep_signature,
                _estimate_top_lineup_matches_fallback(rep),
            )

        rep["is_consolidated"] = True
        rep["consolidated_team_count"] = len(group)

        total_match_count = base_match_count
        total_tournament_count = base_tournament_count
        aliases: list[Dict[str, object]] = []
        for alias in group[1:]:
            alias_match_count = int(alias.get("match_count") or alias.get("lineup_count", 0) or 0)
            alias_tournament_count = int(alias.get("tournament_count", 0) or 0)
            for signature, count in _extract_lineup_variant_counts(alias).items():
                merged_lineup_variant_counts[signature] = (
                    merged_lineup_variant_counts.get(signature, 0) + int(count)
                )
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

        chosen_signature, chosen_lineup_match_count = _select_top_lineup_variant(
            merged_lineup_variant_counts
        )
        player_name_lookup = _group_player_name_lookup(group)
        rep["top_lineup_match_count"] = chosen_lineup_match_count
        rep["top_lineup_match_share"] = (
            chosen_lineup_match_count / total_match_count if total_match_count > 0 else 0.0
        )
        rep["top_lineup_share"] = rep["top_lineup_match_share"]
        rep["top_lineup_player_ids"] = list(chosen_signature)
        chosen_player_names = list(
            player_name_lookup.get(int(player_id), "Unknown Player")
            for player_id in chosen_signature
        )
        rep["top_lineup_player_names"] = chosen_player_names
        rep["top_lineup_player_count"] = len(chosen_signature)
        rep["top_lineup_summary"] = (
            f"{chosen_lineup_match_count}x:{','.join(chosen_player_names)}"
            if chosen_signature and chosen_lineup_match_count > 0
            else rep.get("top_lineup_summary", "")
        )
        rep["distinct_lineup_count"] = (
            len(merged_lineup_variant_counts)
            if merged_lineup_variant_counts
            else int(rep.get("distinct_lineup_count", 0) or 0)
        )
        rep["lineup_count"] = total_match_count
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
        rep["_lineup_variant_counts"] = dict(merged_lineup_variant_counts)
        grouped.append(rep)

    return grouped


def _build_recency_scores(
    embeddings: Sequence[EmbeddingRow],
    *,
    recency_weight: float,
    max_time_ms: float | None = None,
) -> tuple[np.ndarray, float]:
    n_rows = len(embeddings)
    scores = np.full(n_rows, 0.5, dtype=np.float64)
    weight = max(0.0, min(float(recency_weight), 1.0))
    if n_rows <= 0 or weight <= 0.0:
        return scores, 0.0

    event_times = np.asarray(
        [float(row.event_time_ms or 0) for row in embeddings],
        dtype=np.float64,
    )
    max_time = float(max_time_ms) if max_time_ms is not None else (
        float(np.max(event_times)) if event_times.size > 0 else 0.0
    )
    if max_time <= 0.0:
        return scores, weight

    half_life_ms = 90.0 * 24 * 3600 * 1000
    age_ms = np.maximum(max_time - event_times, 0.0)
    scores = np.exp(-0.693 * age_ms / half_life_ms)
    scores = np.where(event_times > 0, scores, 0.5)
    return scores, weight


def _blend_with_recency(
    base_scores: np.ndarray,
    recency_scores: np.ndarray,
    *,
    recency_weight: float,
) -> np.ndarray:
    weight = max(0.0, min(float(recency_weight), 1.0))
    if weight <= 0.0:
        return base_scores
    return ((1.0 - weight) * base_scores) + (weight * recency_scores)


def _compute_player_overlap(
    row: EmbeddingRow,
    query_player_weights: Dict[int, float],
) -> float:
    if not query_player_weights:
        return 0.0
    denom = float(sum(query_player_weights.values()))
    if denom <= 0.0:
        return 0.0
    numer = 0.0
    for player_id, weight in query_player_weights.items():
        numer += float(weight) * float(row.player_support.get(int(player_id), 0.0))
    return numer / denom


def _compute_pair_overlap(
    row: EmbeddingRow,
    query_pair_weights: Dict[tuple[int, int], float],
) -> float:
    if not query_pair_weights:
        return 0.0
    denom = float(sum(query_pair_weights.values()))
    if denom <= 0.0:
        return 0.0
    numer = 0.0
    for pair, weight in query_pair_weights.items():
        numer += float(weight) * float(row.pair_support.get(pair, 0.0))
    return numer / denom


def _lineup_match_fraction(set_a: set[int], set_b: set[int]) -> float:
    if not set_a or not set_b:
        return 0.0
    min_size = min(len(set_a), len(set_b))
    if min_size <= 0:
        return 0.0
    return float(len(set_a & set_b)) / float(min_size)


def _lineup_variant_total_count(
    lineup_variant_counts: Dict[tuple[int, ...], int],
    *,
    fallback_lineup_count: int = 0,
) -> int:
    total = sum(max(int(count), 0) for count in lineup_variant_counts.values())
    if total > 0:
        return int(total)
    return max(int(fallback_lineup_count), 0)


def _compute_lineup_overlap_from_counts(
    lineup_variant_counts: Dict[tuple[int, ...], int],
    query_lineup_signatures: Sequence[tuple[int, ...]],
    *,
    fallback_lineup_count: int = 0,
) -> float:
    if not lineup_variant_counts or not query_lineup_signatures:
        return 0.0

    total = _lineup_variant_total_count(
        lineup_variant_counts,
        fallback_lineup_count=fallback_lineup_count,
    )
    if total <= 0:
        return 0.0

    query_sets = [
        set(signature)
        for signature in query_lineup_signatures
        if signature
    ]
    if not query_sets:
        return 0.0

    best = 0.0
    for signature, raw_count in lineup_variant_counts.items():
        count = max(int(raw_count), 0)
        if count <= 0 or not signature:
            continue
        support = float(count) / float(total)
        lineup_set = set(signature)
        for query_signature, query_set in zip(query_lineup_signatures, query_sets):
            if tuple(signature) == tuple(query_signature):
                best = max(best, support)
                continue
            if not _lineup_nearly_matches(lineup_set, query_set):
                continue
            best = max(best, support * _lineup_match_fraction(lineup_set, query_set))
    return _clamp01(best)


def _compute_lineup_overlap(
    row: EmbeddingRow,
    query_lineup_signatures: Sequence[tuple[int, ...]],
) -> float:
    if not query_lineup_signatures:
        return 0.0
    return _compute_lineup_overlap_from_counts(
        dict(row.lineup_variant_counts),
        query_lineup_signatures,
        fallback_lineup_count=int(row.lineup_count or 0),
    )


def _update_rank_fields(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def _aggregate_support_map(
    rows: Sequence[EmbeddingRow],
    *,
    use_pairs: bool,
) -> Dict[Any, float]:
    if not rows:
        return {}

    total_weight = 0.0
    aggregate: Dict[Any, float] = {}
    for row in rows:
        weight = max(float(row.lineup_count or 0), 1.0)
        total_weight += weight
        source = row.pair_support if use_pairs else row.player_support
        for key, value in source.items():
            aggregate[key] = aggregate.get(key, 0.0) + (weight * float(value))

    if total_weight <= 0.0:
        return {}
    return {
        key: float(value) / total_weight
        for key, value in aggregate.items()
        if float(value) > 0.0
    }


def _aggregate_lineup_variant_counts(
    rows: Sequence[EmbeddingRow],
) -> Dict[tuple[int, ...], int]:
    aggregate: Dict[tuple[int, ...], int] = {}
    for row in rows:
        for signature, count in row.lineup_variant_counts.items():
            if not signature:
                continue
            aggregate[tuple(signature)] = aggregate.get(tuple(signature), 0) + max(int(count), 0)
    return aggregate


def rescore_consolidated_results(
    results: List[Dict[str, object]],
    *,
    query_profile: Dict[str, object],
    rerank_weight_embed: float,
    rerank_weight_player: float,
    rerank_weight_pair: float,
    use_pair_rerank: bool,
    recency_weight: float = 0.0,
) -> List[Dict[str, object]]:
    query_final_raw = query_profile.get("query_final")
    query_final = np.asarray(
        query_final_raw if query_final_raw is not None else [],
        dtype=np.float64,
    )
    query_player_weights = dict(
        query_profile.get("query_player_weights") or {}
    )
    query_pair_weights = dict(
        query_profile.get("query_pair_weights") or {}
    )
    query_lineup_signatures = tuple(
        query_profile.get("query_lineup_signatures") or ()
    )
    max_event_time_ms = query_profile.get("max_event_time_ms")
    if query_final.size == 0 or not results:
        return results

    rescored: List[Dict[str, object]] = []
    for result in results:
        member_payloads = result.get("_consolidated_member_results") or [result]
        member_rows = [
            member.get("_embedding_row")
            for member in member_payloads
            if isinstance(member, dict) and isinstance(member.get("_embedding_row"), EmbeddingRow)
        ]
        if not member_rows:
            rescored.append(result)
            continue

        weights = np.asarray(
            [max(float(row.lineup_count or 0), 1.0) for row in member_rows],
            dtype=np.float64,
        )
        final_vectors = np.stack([row.final_vector for row in member_rows], axis=0)
        centroid = _weighted_centroid(final_vectors, weights)
        base_embed_score = float(centroid @ query_final)
        recency_scores, _ = _build_recency_scores(
            member_rows,
            recency_weight=recency_weight,
            max_time_ms=(
                float(max_event_time_ms)
                if isinstance(max_event_time_ms, (int, float))
                else None
            ),
        )
        group_recency = (
            float(np.average(recency_scores, weights=weights))
            if recency_scores.size
            else 0.5
        )
        group_embed_score = float(
            _blend_with_recency(
                np.asarray([base_embed_score], dtype=np.float64),
                np.asarray([group_recency], dtype=np.float64),
                recency_weight=recency_weight,
            )[0]
        )
        member_embed_scores = _blend_with_recency(
            final_vectors @ query_final,
            recency_scores,
            recency_weight=recency_weight,
        )
        best_member_embed_score = (
            float(np.max(member_embed_scores))
            if member_embed_scores.size
            else group_embed_score
        )
        embed_score = _blend_with_best_member(
            group_embed_score,
            best_member_embed_score,
        )

        aggregate_player_support = _aggregate_support_map(member_rows, use_pairs=False)
        aggregate_pair_support = _aggregate_support_map(member_rows, use_pairs=True)
        aggregate_lineup_variant_counts = _aggregate_lineup_variant_counts(member_rows)
        player_overlap = 0.0
        if query_player_weights:
            denom = float(sum(query_player_weights.values()))
            if denom > 0.0:
                player_overlap = sum(
                    float(weight)
                    * float(aggregate_player_support.get(int(player_id), 0.0))
                    for player_id, weight in query_player_weights.items()
                ) / denom
            best_member_player_overlap = max(
                _compute_player_overlap(row, query_player_weights)
                for row in member_rows
            )
            player_overlap = _blend_with_best_member(
                player_overlap,
                best_member_player_overlap,
            )

        pair_overlap = 0.0
        if use_pair_rerank and query_pair_weights:
            denom = float(sum(query_pair_weights.values()))
            if denom > 0.0:
                pair_overlap = sum(
                    float(weight)
                    * float(aggregate_pair_support.get(pair, 0.0))
                    for pair, weight in query_pair_weights.items()
                ) / denom
            best_member_pair_overlap = max(
                _compute_pair_overlap(row, query_pair_weights)
                for row in member_rows
            )
            pair_overlap = _blend_with_best_member(
                pair_overlap,
                best_member_pair_overlap,
            )

        lineup_overlap = 0.0
        if query_lineup_signatures:
            total_lineups = sum(max(int(row.lineup_count or 0), 0) for row in member_rows)
            lineup_overlap = _compute_lineup_overlap_from_counts(
                aggregate_lineup_variant_counts,
                query_lineup_signatures,
                fallback_lineup_count=total_lineups,
            )
            best_member_lineup_overlap = max(
                _compute_lineup_overlap(row, query_lineup_signatures)
                for row in member_rows
            )
            lineup_overlap = _blend_with_best_member(
                lineup_overlap,
                best_member_lineup_overlap,
            )

        base_rerank_score = (
            float(rerank_weight_embed) * embed_score
            + float(rerank_weight_player) * player_overlap
            + float(rerank_weight_pair) * pair_overlap
        )
        rerank_score = _apply_lineup_boost(base_rerank_score, lineup_overlap)
        updated = dict(result)
        updated["embed_score"] = float(embed_score)
        updated["player_overlap"] = float(player_overlap)
        updated["pair_overlap"] = float(pair_overlap)
        updated["lineup_overlap"] = float(lineup_overlap)
        updated["rerank_score"] = float(rerank_score)
        updated["sim_to_query"] = float(rerank_score)
        rescored.append(updated)

    rescored.sort(
        key=lambda item: (
            -float(item.get("rerank_score", item.get("sim_to_query", 0.0)) or 0.0),
            -float(item.get("embed_score", item.get("sim_final_to_query", 0.0)) or 0.0),
            -int(item.get("lineup_count", 0) or 0),
            int(item.get("team_id", 0) or 0),
        )
    )
    return _update_rank_fields(rescored)


def strip_internal_result_fields(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def _strip_value(value: object) -> object:
        if isinstance(value, list):
            return [_strip_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: _strip_value(inner)
                for key, inner in value.items()
                if not str(key).startswith("_")
            }
        return value

    return [
        _strip_value(dict(result))
        for result in results
    ]


def rank_similar_teams(
    embeddings: Sequence[EmbeddingRow],
    target_team_ids: Sequence[int],
    cluster_map: Dict[int, Dict[str, object]],
    top_n: int,
    min_relevance: float = 0.0,
    candidate_top_n: Optional[int] = None,
    recency_weight: float = 0.0,
    query_team_weights: Optional[Dict[int, float]] = None,
    query_profile: Optional[Dict[str, object]] = None,
    rerank_candidate_limit: int = 200,
    rerank_weight_embed: float = 0.70,
    rerank_weight_player: float = 0.15,
    rerank_weight_pair: float = 0.15,
    use_pair_rerank: bool = True,
    precomputed_finals: Optional[np.ndarray] = None,
    precomputed_semantics: Optional[np.ndarray] = None,
    precomputed_identities: Optional[np.ndarray] = None,
    precomputed_index: Optional[Dict[int, int]] = None,
) -> Dict[str, object]:
    embedding_rows = list(embeddings)
    use_precomputed = (
        bool(embedding_rows)
        and precomputed_finals is not None
        and precomputed_semantics is not None
        and precomputed_identities is not None
    )

    finals: np.ndarray
    semantics: np.ndarray
    identities: np.ndarray
    index: Dict[int, int]
    if use_precomputed:
        finals = np.asarray(precomputed_finals, dtype=np.float64)
        semantics = np.asarray(precomputed_semantics, dtype=np.float64)
        identities = np.asarray(precomputed_identities, dtype=np.float64)
        if (
            finals.ndim != 2
            or semantics.ndim != 2
            or identities.ndim != 2
            or finals.shape[0] != len(embedding_rows)
            or semantics.shape[0] != len(embedding_rows)
            or identities.shape[0] != len(embedding_rows)
        ):
            use_precomputed = False
        else:
            index = (
                {
                    int(team_id): int(idx)
                    for team_id, idx in precomputed_index.items()
                    if 0 <= int(idx) < len(embedding_rows)
                }
                if precomputed_index
                else {row.team_id: i for i, row in enumerate(embedding_rows)}
            )

    if not use_precomputed:
        embedding_rows = _normalize_embeddings(embedding_rows)
        if not embedding_rows:
            return {"query": {}, "results": []}
        index = {row.team_id: i for i, row in enumerate(embedding_rows)}
        finals = np.stack([row.final_vector for row in embedding_rows], axis=0)
        semantics = np.stack([row.semantic_vector for row in embedding_rows], axis=0)
        identities = np.stack([row.identity_vector for row in embedding_rows], axis=0)

    target_idx = [index[tid] for tid in target_team_ids if tid in index]
    if query_profile is None:
        if not target_idx:
            return {"query": {"matched_team_ids": []}, "results": []}
        query_profile = build_query_similarity_profile(
            embedding_rows,
            target_idx,
            finals,
            semantics,
            identities,
            query_team_weights=query_team_weights,
        )
        dense_algorithm = "adaptive_blend"
    else:
        query_profile = dict(query_profile)
        dense_algorithm = "whole_set_final"

    query_final_raw = query_profile.get("query_final")
    query_sem_raw = query_profile.get("query_semantic")
    query_id_raw = query_profile.get("query_identity")
    query_final = np.asarray(
        query_final_raw if query_final_raw is not None else [],
        dtype=np.float64,
    )
    query_sem = np.asarray(
        query_sem_raw if query_sem_raw is not None else [],
        dtype=np.float64,
    )
    query_id = np.asarray(
        query_id_raw if query_id_raw is not None else [],
        dtype=np.float64,
    )
    if query_final.size == 0:
        return {"query": {"matched_team_ids": []}, "results": []}

    query_stability = query_profile.get("query_stability")
    semantic_weight = float(query_profile.get("semantic_weight") or 0.5)
    identity_weight = float(query_profile.get("identity_weight") or 0.5)

    sims_final = finals @ query_final
    sims_sem = (
        semantics @ query_sem
        if query_sem.size == semantics.shape[1]
        else np.zeros(len(embedding_rows), dtype=np.float64)
    )
    sims_id = (
        identities @ query_id
        if query_id.size == identities.shape[1]
        else np.zeros(len(embedding_rows), dtype=np.float64)
    )
    if dense_algorithm == "adaptive_blend":
        dense_scores = (semantic_weight * sims_sem) + (identity_weight * sims_id)
    else:
        dense_scores = sims_final

    recency_scores, recency_weight = _build_recency_scores(
        embedding_rows,
        recency_weight=recency_weight,
    )
    if recency_scores.size:
        query_profile["max_event_time_ms"] = float(
            np.max([float(row.event_time_ms or 0) for row in embedding_rows]) or 0.0
        )
    embed_scores = _blend_with_recency(
        dense_scores,
        recency_scores,
        recency_weight=recency_weight,
    )

    order = np.argsort(-embed_scores)
    min_relevance = max(0.0, min(float(min_relevance), 1.0))
    top_n = max(1, min(int(top_n), len(order)))
    target_set = set(int(tid) for tid in target_team_ids)
    query_player_weights = dict(query_profile.get("query_player_weights") or {})
    query_pair_weights = dict(query_profile.get("query_pair_weights") or {})
    query_lineup_signatures = tuple(query_profile.get("query_lineup_signatures") or ())
    rerank_enabled = bool(query_player_weights)

    filtered = [int(idx) for idx in order if float(embed_scores[idx]) >= min_relevance]
    if not filtered:
        return {
            "query": {
                "matched_team_ids": [embedding_rows[i].team_id for i in target_idx],
                "matched_team_names": [embedding_rows[i].team_name for i in target_idx],
                "ranking_algorithm": str(query_profile.get("ranking_algorithm") or dense_algorithm),
            },
            "results": [],
        }

    score_rows: list[tuple[int, float, float, float, float, float]] = []
    rerank_limit = max(1, min(int(rerank_candidate_limit), len(filtered)))
    candidate_idx = filtered[:rerank_limit] if rerank_enabled else filtered[:top_n]
    for idx in candidate_idx:
        row = embedding_rows[int(idx)]
        embed_score = float(embed_scores[idx])
        player_overlap = (
            _compute_player_overlap(row, query_player_weights)
            if rerank_enabled
            else 0.0
        )
        pair_overlap = (
            _compute_pair_overlap(row, query_pair_weights)
            if rerank_enabled and use_pair_rerank
            else 0.0
        )
        lineup_overlap = (
            _compute_lineup_overlap(row, query_lineup_signatures)
            if rerank_enabled and query_lineup_signatures
            else 0.0
        )
        base_rerank_score = (
            float(rerank_weight_embed) * embed_score
            + float(rerank_weight_player) * player_overlap
            + float(rerank_weight_pair) * pair_overlap
            if rerank_enabled
            else embed_score
        )
        rerank_score = (
            _apply_lineup_boost(base_rerank_score, lineup_overlap)
            if rerank_enabled
            else embed_score
        )
        score_rows.append(
            (
                int(idx),
                float(embed_score),
                float(player_overlap),
                float(pair_overlap),
                float(lineup_overlap),
                float(rerank_score),
            )
        )

    score_rows.sort(
        key=lambda item: (
            -float(item[5]),
            -float(item[4]),
            -float(item[1]),
            -int(embedding_rows[int(item[0])].lineup_count),
            int(embedding_rows[int(item[0])].team_id),
        )
    )
    score_by_idx = {
        int(idx): {
            "embed_score": float(embed_score),
            "player_overlap": float(player_overlap),
            "pair_overlap": float(pair_overlap),
            "lineup_overlap": float(lineup_overlap),
            "rerank_score": float(rerank_score),
        }
        for idx, embed_score, player_overlap, pair_overlap, lineup_overlap, rerank_score in score_rows
    }
    selected_limit = top_n
    if candidate_top_n is not None:
        selected_limit = max(top_n, min(int(candidate_top_n), len(score_rows)))

    selected = [int(item[0]) for item in score_rows[:selected_limit]]

    results: List[Dict[str, object]] = []

    for rank, idx in enumerate(selected, start=1):
        row = embedding_rows[int(idx)]
        cluster = cluster_map.get(row.team_id)
        score_payload = score_by_idx.get(
            int(idx),
            {
                "embed_score": float(embed_scores[idx]),
                "player_overlap": 0.0,
                "pair_overlap": 0.0,
                "lineup_overlap": 0.0,
                "rerank_score": float(embed_scores[idx]),
            },
        )

        top_lineup_players: List[Dict[str, object]] = []
        core_lineup_players: List[Dict[str, object]] = []
        player_ids = list(row.top_lineup_player_ids)
        player_names = list(row.top_lineup_player_names)
        row_dict = {
            "lineup_count": int(row.lineup_count),
            "top_lineup_share": float(row.top_lineup_share),
            "top_lineup_summary": row.top_lineup_summary,
        }
        if row.lineup_variant_counts:
            _, top_lineup_match_count = _select_top_lineup_variant(
                dict(row.lineup_variant_counts)
            )
        else:
            top_lineup_match_count = _estimate_top_lineup_matches_fallback(row_dict)
        result_top_lineup_player_ids = list(player_ids)
        if top_lineup_match_count > int(row.lineup_count):
            top_lineup_match_count = int(row.lineup_count)
        if player_ids:
            for i, player_id in enumerate(player_ids):
                raw_name = player_names[i] if i < len(player_names) else None
                name = str(raw_name).strip() if raw_name else "Unknown Player"
                pid = int(player_id)
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

        if row.roster_player_ids and row.roster_player_match_counts:
            roster_ids = list(row.roster_player_ids)
            roster_names = list(row.roster_player_names)
            roster_counts = list(row.roster_player_match_counts)
            max_len = min(len(roster_ids), len(roster_counts))
            for i in range(max_len):
                pid = int(roster_ids[i])
                if pid <= 0:
                    continue
                raw_name = roster_names[i] if i < len(roster_names) else None
                name = str(raw_name).strip() if raw_name else "Unknown Player"
                matches = int(roster_counts[i] or 0)
                core_lineup_players.append(
                    {
                        "player_id": pid,
                        "player_name": name or "Unknown Player",
                        "matches_played": matches,
                        "sendou_url": f"https://sendou.ink/u/{pid}",
                    }
                )
        else:
            for player in top_lineup_players:
                if player.get("player_id") is None:
                    continue
                pid = int(player["player_id"])
                core_lineup_players.append(
                    {
                        "player_id": pid,
                        "player_name": player.get("player_name") or "Unknown Player",
                        "matches_played": top_lineup_match_count,
                        "sendou_url": f"https://sendou.ink/u/{pid}",
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
                "sim_to_query": float(score_payload["rerank_score"]),
                "sim_final_to_query": float(sims_final[idx]),
                "sim_semantic_to_query": float(sims_sem[idx]),
                "sim_identity_to_query": float(sims_id[idx]),
                "embed_score": float(score_payload["embed_score"]),
                "player_overlap": float(score_payload["player_overlap"]),
                "pair_overlap": float(score_payload["pair_overlap"]),
                "lineup_overlap": float(score_payload["lineup_overlap"]),
                "rerank_score": float(score_payload["rerank_score"]),
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
                "_lineup_variant_counts": dict(row.lineup_variant_counts),
                "_embedding_row": row,
            }
        )

    return {
        "query": {
            "matched_team_ids": [embedding_rows[i].team_id for i in target_idx],
            "matched_team_names": [embedding_rows[i].team_name for i in target_idx],
            "ranking_algorithm": str(
                query_profile.get("ranking_algorithm")
                or ("adaptive_blend_v1" if dense_algorithm == "adaptive_blend" else "whole_set_hashed_v1")
            ),
            "query_mode": str(query_profile.get("query_mode") or "team_centroid"),
            "query_player_ids": list(query_profile.get("query_player_ids") or ()),
            "query_player_count": len(query_profile.get("query_player_ids") or ()),
            "query_pair_count": len(query_profile.get("query_pairs") or ()),
            "query_lineup_signature_count": len(query_lineup_signatures),
            "rerank_candidate_limit": rerank_limit if rerank_enabled else 0,
            "candidate_top_n": selected_limit,
            "use_pair_rerank": bool(use_pair_rerank),
            "query_stability": (
                round(float(query_stability), 4)
                if isinstance(query_stability, (int, float))
                else None
            ),
            "query_semantic_weight": round(semantic_weight, 4),
            "query_identity_weight": round(identity_weight, 4),
        },
        "results": results,
    }
