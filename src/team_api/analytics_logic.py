from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from team_api.store import EmbeddingRow


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _to_percent(value: float) -> float:
    return float(round(value * 100.0, 2))


def _volatility_score(row: EmbeddingRow) -> float:
    lineup_ratio = 0.0
    if row.lineup_count > 0:
        lineup_ratio = row.distinct_lineup_count / float(row.lineup_count)
    return float(
        0.45 * row.lineup_entropy
        + 0.35 * (1.0 - row.top_lineup_share)
        + 0.20 * min(1.0, lineup_ratio)
    )


def _cluster_centroids(
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
) -> Dict[int, np.ndarray]:
    if not embeddings:
        return {}
    vectors = np.stack([row.final_vector for row in embeddings], axis=0)
    index_by_team_id = {row.team_id: i for i, row in enumerate(embeddings)}

    by_cluster: Dict[int, List[int]] = defaultdict(list)
    for team_id, info in cluster_map.items():
        cid = int(info["cluster_id"])
        if cid >= 0 and team_id in index_by_team_id:
            by_cluster[cid].append(team_id)

    centroids: Dict[int, np.ndarray] = {}
    for cid, team_ids in by_cluster.items():
        idx = [index_by_team_id[team_id] for team_id in team_ids]
        centroids[cid] = _normalize(vectors[idx].mean(axis=0))
    return centroids


def _resolve_cluster_id(
    team_id: int,
    cluster_map: Dict[int, Dict[str, Any]],
) -> Optional[int]:
    if team_id not in cluster_map:
        return None
    return int(cluster_map[team_id]["cluster_id"])


def _resolve_shared_cluster_id(
    team_ids: Iterable[int],
    cluster_map: Dict[int, Dict[str, Any]],
) -> Optional[int]:
    cluster_ids = {
        cluster_id
        for team_id in team_ids
        if (cluster_id := _resolve_cluster_id(int(team_id), cluster_map)) is not None
    }
    if len(cluster_ids) != 1:
        return None
    return next(iter(cluster_ids))


def _player_ids(row: EmbeddingRow) -> tuple[int, ...]:
    ids = []
    for value in row.top_lineup_player_ids:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        ids.append(pid)
    return tuple(ids)


def _player_names(row: EmbeddingRow) -> Dict[int, str]:
    names = {}
    for idx, value in enumerate(row.top_lineup_player_ids):
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        name = ""
        try:
            if idx < len(row.top_lineup_player_names):
                name = str(row.top_lineup_player_names[idx]).strip()
        except Exception:
            name = ""
        if not name:
            name = "Unknown Player"
        names.setdefault(pid, name)
    return names


def _player_details(
    player_ids: Iterable[int],
    primary: Dict[int, str],
    secondary: Dict[int, str],
) -> List[Dict[str, object]]:
    details: List[Dict[str, object]] = []
    seen: set[int] = set()
    for pid in sorted(set(player_ids)):
        if pid in seen:
            continue
        seen.add(pid)
        details.append(
            {
                "player_id": int(pid),
                "player_name": str(
                    primary.get(pid, secondary.get(pid, f"Player {int(pid)}"))
                ),
            }
        )
    return details


def _estimate_top_lineup_matches(row: EmbeddingRow) -> int:
    if row.lineup_variant_counts:
        return max(int(count) for count in row.lineup_variant_counts.values())
    if row.lineup_count <= 0:
        return 0
    estimated = int(round(float(row.top_lineup_share or 0.0) * float(row.lineup_count)))
    if estimated <= 0 and row.top_lineup_player_ids:
        estimated = 1
    return max(0, min(int(row.lineup_count), estimated))


def _merge_lineup_variant_counts(
    rows: Sequence[EmbeddingRow],
) -> Dict[tuple[int, ...], int]:
    merged: Dict[tuple[int, ...], int] = defaultdict(int)
    for row in rows:
        if row.lineup_variant_counts:
            for signature, count in row.lineup_variant_counts.items():
                if not signature:
                    continue
                merged[tuple(int(pid) for pid in signature)] += int(count)
            continue
        signature = tuple(int(pid) for pid in row.top_lineup_player_ids if int(pid) > 0)
        if not signature:
            continue
        merged[signature] += _estimate_top_lineup_matches(row)
    return dict(merged)


def _lineup_metrics(
    lineup_variant_counts: Dict[tuple[int, ...], int],
    total_lineup_count: int,
) -> tuple[int, float, float, float, tuple[int, ...], int]:
    if total_lineup_count <= 0 or not lineup_variant_counts:
        return 0, 0.0, 0.0, 0.0, (), 0

    counts = np.asarray(
        [max(0, int(count)) for count in lineup_variant_counts.values()],
        dtype=np.float64,
    )
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0, 0.0, 0.0, 0.0, (), 0

    total = float(max(1, total_lineup_count))
    probabilities = counts / total
    entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    effective = float(np.exp(entropy)) if entropy > 0.0 else (1.0 if counts.size > 0 else 0.0)
    chosen_signature, chosen_count = max(
        lineup_variant_counts.items(),
        key=lambda item: (int(item[1]), tuple(int(pid) for pid in item[0])),
    )
    top_count = int(max(0, chosen_count))
    top_share = top_count / total if total > 0 else 0.0
    return len(lineup_variant_counts), float(top_share), entropy, effective, tuple(int(pid) for pid in chosen_signature), top_count


def _aggregate_roster_rows(
    rows: Sequence[EmbeddingRow],
) -> tuple[Dict[int, int], Dict[int, str]]:
    roster_counts: Dict[int, int] = defaultdict(int)
    name_lookup: Dict[int, str] = {}

    for row in rows:
        if row.roster_player_ids and row.roster_player_match_counts:
            for idx, raw_pid in enumerate(row.roster_player_ids):
                try:
                    pid = int(raw_pid)
                except (TypeError, ValueError):
                    continue
                if pid <= 0:
                    continue
                matches = (
                    int(row.roster_player_match_counts[idx])
                    if idx < len(row.roster_player_match_counts)
                    else 0
                )
                roster_counts[pid] += matches
                if pid not in name_lookup:
                    name = ""
                    if idx < len(row.roster_player_names):
                        name = str(row.roster_player_names[idx] or "").strip()
                    if not name:
                        name = _player_names(row).get(pid, f"Player {pid}")
                    name_lookup[pid] = name
            continue

        estimated = _estimate_top_lineup_matches(row)
        for pid, name in _player_names(row).items():
            roster_counts[pid] += estimated
            name_lookup.setdefault(pid, name)

    return dict(roster_counts), name_lookup


def _aggregate_embedding_rows(
    *,
    primary_team_id: int,
    team_name: str,
    rows: Sequence[EmbeddingRow],
) -> EmbeddingRow:
    if not rows:
        raise ValueError("rows must not be empty")

    finals = np.stack([row.final_vector for row in rows], axis=0)
    semantics = np.stack([row.semantic_vector for row in rows], axis=0)
    identities = np.stack([row.identity_vector for row in rows], axis=0)

    final_vector = _normalize(np.mean(finals, axis=0))
    semantic_vector = _normalize(np.mean(semantics, axis=0))
    identity_vector = _normalize(np.mean(identities, axis=0))

    total_lineup_count = int(sum(max(0, int(row.lineup_count)) for row in rows))
    lineup_variant_counts = _merge_lineup_variant_counts(rows)
    (
        distinct_lineup_count,
        top_lineup_share,
        lineup_entropy,
        effective_lineups,
        chosen_signature,
        chosen_count,
    ) = _lineup_metrics(lineup_variant_counts, total_lineup_count)

    roster_counts, roster_name_lookup = _aggregate_roster_rows(rows)
    if not roster_counts:
        for row in rows:
            for pid, name in _player_names(row).items():
                roster_counts[pid] = roster_counts.get(pid, 0) + _estimate_top_lineup_matches(row)
                roster_name_lookup.setdefault(pid, name)

    roster_sorted = sorted(
        roster_counts.items(),
        key=lambda item: (-int(item[1]), int(item[0])),
    )
    roster_player_ids = tuple(int(pid) for pid, _ in roster_sorted)
    roster_player_match_counts = tuple(int(count) for _, count in roster_sorted)
    roster_player_names = tuple(
        str(roster_name_lookup.get(pid, f"Player {pid}"))
        for pid in roster_player_ids
    )

    top_lineup_player_names = tuple(
        str(roster_name_lookup.get(pid, f"Player {pid}"))
        for pid in chosen_signature
    )
    tournament_ids = {
        int(row.tournament_id)
        for row in rows
        if row.tournament_id is not None
    }
    tournament_count = len(tournament_ids) if tournament_ids else int(
        sum(max(0, int(row.tournament_count)) for row in rows)
    )
    event_time_ms = max(int(row.event_time_ms or 0) for row in rows) or None

    return EmbeddingRow(
        team_id=int(primary_team_id),
        tournament_id=None,
        team_name=str(team_name or rows[0].team_name),
        event_time_ms=event_time_ms,
        lineup_count=int(total_lineup_count),
        semantic_vector=semantic_vector,
        identity_vector=identity_vector,
        final_vector=final_vector,
        top_lineup_summary=(
            f"{chosen_count}x:{','.join(top_lineup_player_names)}"
            if chosen_signature and chosen_count > 0
            else ""
        ),
        unique_player_count=len(roster_player_ids),
        distinct_lineup_count=int(distinct_lineup_count),
        top_lineup_share=float(top_lineup_share),
        lineup_entropy=float(lineup_entropy),
        effective_lineups=float(effective_lineups),
        top_lineup_player_ids=tuple(int(pid) for pid in chosen_signature),
        top_lineup_player_names=top_lineup_player_names,
        roster_player_ids=roster_player_ids,
        roster_player_names=roster_player_names,
        roster_player_match_counts=roster_player_match_counts,
        lineup_variant_counts=lineup_variant_counts,
        tournament_count=int(tournament_count),
    )


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def _find(parent: Dict[int, int], node: int) -> int:
    value = parent[node]
    if value != node:
        parent[node] = _find(parent, value)
    return parent[node]


def _union(parent: Dict[int, int], rank: Dict[int, int], a: int, b: int) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a == root_b:
        return
    rank_a = rank[root_a]
    rank_b = rank[root_b]
    if rank_a < rank_b:
        parent[root_a] = root_b
    elif rank_b < rank_a:
        parent[root_b] = root_a
    else:
        parent[root_b] = root_a
        rank[root_a] = rank_a + 1


def compute_roster_diversity_candidates(
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
    *,
    min_similarity: float,
    max_player_overlap: float,
    min_cluster_size: int,
    limit: int,
) -> Dict[str, Any]:
    if min_cluster_size < 2:
        min_cluster_size = 2
    if limit < 1:
        limit = 1

    if not embeddings:
        return {
            "total_pairs_found": 0,
            "clusters_considered": 0,
            "pairs": [],
            "cohorts": [],
        }

    by_cluster: Dict[int, List[EmbeddingRow]] = defaultdict(list)
    for row in embeddings:
        cid = _resolve_cluster_id(row.team_id, cluster_map)
        if cid is None or cid < 0:
            continue
        if not _player_ids(row):
            continue
        by_cluster[cid].append(row)

    candidate_pairs: List[Dict[str, Any]] = []
    candidate_cohorts: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for cid, rows in by_cluster.items():
        if len(rows) < min_cluster_size:
            continue

        vectors = np.stack([row.final_vector for row in rows], axis=0)
        similarity = vectors @ vectors.T

        row_lookup: Dict[int, EmbeddingRow] = {int(row.team_id): row for row in rows}
        team_players = {int(row.team_id): set(_player_ids(row)) for row in rows}
        player_names = {int(row.team_id): _player_names(row) for row in rows}
        team_tournament_counts = {int(row.team_id): int(row.tournament_count) for row in rows}
        team_lineup_counts = {int(row.team_id): int(row.lineup_count) for row in rows}
        cluster_pairs: List[Dict[str, Any]] = []

        candidate_teams: set[int] = set()
        for idx_a in range(len(rows)):
            team_a = int(rows[idx_a].team_id)
            players_a = team_players[team_a]
            names_a = player_names[team_a]
            for idx_b in range(idx_a + 1, len(rows)):
                sim = float(similarity[idx_a, idx_b])
                if sim < min_similarity:
                    continue

                team_b = int(rows[idx_b].team_id)
                players_b = team_players[team_b]
                names_b = player_names[team_b]
                overlap = _jaccard(players_a, players_b)
                if overlap > max_player_overlap:
                    continue

                shared_players = sorted(players_a & players_b)
                union_players = sorted(players_a | players_b)
                if not union_players:
                    continue

                candidate_teams.update({team_a, team_b})
                candidate_pairs.append(
                    {
                        "cluster_id": int(cid),
                        "team_a_id": team_a,
                        "team_a_name": rows[idx_a].team_name,
                        "team_a_tournament_count": team_tournament_counts.get(team_a, 0),
                        "team_a_lineup_count": int(rows[idx_a].lineup_count),
                        "team_b_id": team_b,
                        "team_b_name": rows[idx_b].team_name,
                        "team_b_tournament_count": team_tournament_counts.get(team_b, 0),
                        "team_b_lineup_count": int(rows[idx_b].lineup_count),
                        "cluster_size": len(rows),
                        "similarity": round(sim, 4),
                        "overlap_fraction": round(overlap, 4),
                        "overlap_count": len(shared_players),
                        "roster_pool_size": len(union_players),
                        "potential_squad_size": len(union_players),
                        "min_lineup_count": min(
                            team_lineup_counts.get(team_a, 0),
                            team_lineup_counts.get(team_b, 0),
                        ),
                        "shared_player_ids": shared_players,
                        "shared_player_names": _player_details(shared_players, names_a, names_b),
                        "roster_player_ids": union_players,
                        "roster_player_names": _player_details(
                            union_players, names_a, names_b
                        ),
                    }
                )
                cluster_pairs.append(
                    candidate_pairs[-1]
                )

        if not cluster_pairs:
            continue

        # Build connected components of teams in this cluster for "can-be-in-same-team" cohorts.
        ds_parent: Dict[int, int] = {team_id: team_id for team_id in candidate_teams}
        ds_rank: Dict[int, int] = {team_id: 0 for team_id in candidate_teams}
        for pair in cluster_pairs:
            _union(ds_parent, ds_rank, pair["team_a_id"], pair["team_b_id"])

        group_map: Dict[int, List[int]] = defaultdict(list)
        for team_id in candidate_teams:
            group_map[_find(ds_parent, team_id)].append(team_id)

        for group_team_ids in group_map.values():
            if len(group_team_ids) < 2:
                continue
            group_pairs = [
                pair
                for pair in cluster_pairs
                if pair["team_a_id"] in group_team_ids
                and pair["team_b_id"] in group_team_ids
            ]
            if not group_pairs:
                continue

            pool_ids: set[int] = set()
            sim_values: list[float] = []
            overlap_values: list[float] = []
            names_by_team = row_lookup
            for team_id in group_team_ids:
                pool_ids |= team_players.get(team_id, set())
            for pair in group_pairs:
                sim_values.append(float(pair["similarity"]))
                overlap_values.append(float(pair["overlap_fraction"]))

            common_roster: set[int] | None = None
            for team_id in group_team_ids:
                if common_roster is None:
                    common_roster = set(team_players[team_id])
                else:
                    common_roster &= team_players[team_id]
            common_roster = common_roster or set()

            merged_names: Dict[int, str] = {}
            for team_id in group_team_ids:
                merged_names.update(_player_names(names_by_team[team_id]))

            candidate_cohorts[int(cid)].append(
                {
                    "cluster_id": int(cid),
                    "team_count": len(group_team_ids),
                    "cluster_size": len(rows),
                    "team_ids": sorted(group_team_ids),
                    "team_names": [names_by_team[team_id].team_name for team_id in group_team_ids],
                    "pair_count": len(group_pairs),
                    "roster_pool_size": len(pool_ids),
                    "potential_squad_size": len(pool_ids),
                    "roster_player_ids": sorted(pool_ids),
                    "roster_player_names": _player_details(
                        sorted(pool_ids), merged_names, {}
                    ),
                    "shared_all_player_ids": sorted(common_roster),
                    "shared_all_player_names": _player_details(
                        sorted(common_roster), merged_names, {}
                    ),
                    "team_tournament_counts": {
                        team_id: team_tournament_counts.get(team_id, 0)
                        for team_id in group_team_ids
                    },
                    "team_lineup_counts": {
                        team_id: team_lineup_counts.get(team_id, 0)
                        for team_id in group_team_ids
                    },
                    "pair_similarity_avg": round(float(np.mean(sim_values)), 4),
                    "pair_similarity_min": round(float(np.min(sim_values)), 4),
                    "pair_similarity_max": round(float(np.max(sim_values)), 4),
                    "pair_overlap_min": round(float(np.min(overlap_values)), 4),
                    "pair_overlap_max": round(float(np.max(overlap_values)), 4),
                }
            )

    if not candidate_pairs:
        return {
            "total_pairs_found": 0,
            "clusters_considered": len(by_cluster),
            "pairs": [],
            "cohorts": [],
        }

    # Best candidates are those with the biggest combined roster pool while retaining strong
    # similarity, so they are likely to expose truly distinct but related team compositions.
    candidate_pairs.sort(
        key=lambda row: (
            -row["roster_pool_size"],
            -row["similarity"],
            row["overlap_fraction"],
        )
    )
    all_cohorts = [cohort for groups in candidate_cohorts.values() for cohort in groups]
    all_cohorts.sort(
        key=lambda row: (
            -row["roster_pool_size"],
            -row["pair_count"],
            -row["pair_similarity_avg"],
        )
    )

    return {
        "total_pairs_found": len(candidate_pairs),
        "clusters_considered": len(by_cluster),
        "pairs": candidate_pairs[: max(1, limit)],
        "cohorts": all_cohorts[: max(1, limit)],
    }


def compute_overview(
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
    *,
    limit_clusters: int,
    volatile_limit: int,
) -> Dict[str, Any]:
    if not embeddings:
        return {
            "summary": {
                "teams_indexed": 0,
                "clustered_teams": 0,
                "noise_teams": 0,
                "cluster_count": 0,
                "coverage_pct": 0.0,
                "avg_lineups_per_team": 0.0,
                "median_lineups_per_team": 0.0,
            },
            "cluster_health": [],
            "volatility_leaders": [],
            "volume_leaders": [],
            "archetypes": {},
        }

    team_by_id = {row.team_id: row for row in embeddings}
    clustered_ids = {
        tid for tid, info in cluster_map.items() if int(info["cluster_id"]) >= 0
    }
    noise_ids = set(team_by_id.keys()) - clustered_ids

    vectors = np.stack([row.final_vector for row in embeddings], axis=0)
    index_by_team_id = {row.team_id: i for i, row in enumerate(embeddings)}

    by_cluster: Dict[int, List[int]] = defaultdict(list)
    for team_id, info in cluster_map.items():
        cid = int(info["cluster_id"])
        if cid < 0:
            continue
        if team_id in team_by_id:
            by_cluster[cid].append(team_id)

    cluster_centroids = _cluster_centroids(embeddings, cluster_map)

    cluster_health: List[Dict[str, Any]] = []
    for cid, team_ids in by_cluster.items():
        idx = [index_by_team_id[team_id] for team_id in team_ids]
        member_vecs = vectors[idx]
        centroid = cluster_centroids[cid]
        centroid_sims = member_vecs @ centroid
        cohesion_mean = float(np.mean(centroid_sims))
        cohesion_p10 = float(np.percentile(centroid_sims, 10))

        nearest_other = 0.0
        if len(cluster_centroids) > 1:
            other_sims = [
                float(centroid @ other_centroid)
                for other_cid, other_centroid in cluster_centroids.items()
                if other_cid != cid
            ]
            nearest_other = max(other_sims) if other_sims else 0.0

        members = [team_by_id[team_id] for team_id in team_ids]
        members.sort(key=lambda r: r.lineup_count, reverse=True)
        top_members = [
            {
                "team_id": row.team_id,
                "team_name": row.team_name,
                "lineup_count": row.lineup_count,
            }
            for row in members[:3]
        ]
        avg_entropy = float(np.mean([row.lineup_entropy for row in members]))

        cluster_health.append(
            {
                "cluster_id": int(cid),
                "cluster_size": len(team_ids),
                "cohesion_mean": round(cohesion_mean, 4),
                "cohesion_p10": round(cohesion_p10, 4),
                "nearest_cluster_similarity": round(float(nearest_other), 4),
                "separation_gap": round(float(cohesion_mean - nearest_other), 4),
                "avg_lineup_entropy": round(avg_entropy, 4),
                "top_members": top_members,
            }
        )

    cluster_health.sort(
        key=lambda row: (
            -row["cluster_size"],
            -row["separation_gap"],
            -row["cohesion_mean"],
        )
    )

    volatility_rows = []
    for row in embeddings:
        score = _volatility_score(row)
        volatility_rows.append(
            {
                "team_id": row.team_id,
                "team_name": row.team_name,
                "lineup_count": row.lineup_count,
                "distinct_lineup_count": row.distinct_lineup_count,
                "lineup_entropy": round(row.lineup_entropy, 4),
                "top_lineup_share_pct": _to_percent(row.top_lineup_share),
                "volatility_score": round(score, 4),
            }
        )
    volatility_rows.sort(key=lambda row: row["volatility_score"], reverse=True)

    volume_leaders = sorted(
        embeddings,
        key=lambda row: row.lineup_count,
        reverse=True,
    )

    archetypes = {
        "most_unique": None,
        "most_stable": None,
        "largest_volume": None,
    }
    if embeddings:
        most_unique = max(embeddings, key=_volatility_score)
        most_stable = min(embeddings, key=_volatility_score)
        largest_volume = max(embeddings, key=lambda row: row.lineup_count)
        archetypes = {
            "most_unique": {
                "team_id": most_unique.team_id,
                "team_name": most_unique.team_name,
                "volatility_score": round(_volatility_score(most_unique), 4),
            },
            "most_stable": {
                "team_id": most_stable.team_id,
                "team_name": most_stable.team_name,
                "volatility_score": round(_volatility_score(most_stable), 4),
            },
            "largest_volume": {
                "team_id": largest_volume.team_id,
                "team_name": largest_volume.team_name,
                "lineup_count": largest_volume.lineup_count,
            },
        }

    lineup_counts = [row.lineup_count for row in embeddings]
    return {
        "summary": {
            "teams_indexed": len(embeddings),
            "clustered_teams": len(clustered_ids),
            "noise_teams": len(noise_ids),
            "cluster_count": len(by_cluster),
            "coverage_pct": _to_percent(
                len(clustered_ids) / float(max(1, len(embeddings)))
            ),
            "avg_lineups_per_team": round(float(np.mean(lineup_counts)), 2),
            "median_lineups_per_team": round(float(np.median(lineup_counts)), 2),
        },
        "cluster_health": cluster_health[: max(1, limit_clusters)],
        "volatility_leaders": volatility_rows[: max(1, volatile_limit)],
        "volume_leaders": [
            {
                "team_id": row.team_id,
                "team_name": row.team_name,
                "lineup_count": row.lineup_count,
                "distinct_lineup_count": row.distinct_lineup_count,
            }
            for row in volume_leaders[:10]
        ],
        "archetypes": archetypes,
    }


def summarize_matchups(
    rows: Iterable[Dict[str, Any]],
    cluster_names: Dict[int, str],
    *,
    min_matches: int,
    limit: int,
) -> List[Dict[str, Any]]:
    accum: Dict[tuple[int, int], Dict[str, Any]] = {}

    for row in rows:
        c1 = int(row["cluster_1"])
        c2 = int(row["cluster_2"])
        if c1 == c2:
            continue

        a, b = (c1, c2) if c1 < c2 else (c2, c1)
        key = (a, b)
        bucket = accum.setdefault(
            key,
            {
                "cluster_a": a,
                "cluster_b": b,
                "matches": 0,
                "wins_a": 0,
                "wins_b": 0,
            },
        )
        bucket["matches"] += 1

        winner_team_id = row.get("winner_team_id")
        if winner_team_id is None:
            continue

        winner_cluster = None
        if int(winner_team_id) == int(row["team1_id"]):
            winner_cluster = c1
        elif int(winner_team_id) == int(row["team2_id"]):
            winner_cluster = c2

        if winner_cluster == a:
            bucket["wins_a"] += 1
        elif winner_cluster == b:
            bucket["wins_b"] += 1

    out: List[Dict[str, Any]] = []
    for bucket in accum.values():
        matches = int(bucket["matches"])
        if matches < min_matches:
            continue
        wins_a = int(bucket["wins_a"])
        wins_b = int(bucket["wins_b"])
        win_rate_a = wins_a / float(matches)
        close_factor = 1.0 - abs(win_rate_a - 0.5) * 2.0
        rivalry_score = matches * (0.4 + 0.6 * close_factor)
        out.append(
            {
                "cluster_a": bucket["cluster_a"],
                "cluster_b": bucket["cluster_b"],
                "cluster_a_name": cluster_names.get(
                    bucket["cluster_a"], f"Cluster {bucket['cluster_a']}"
                ),
                "cluster_b_name": cluster_names.get(
                    bucket["cluster_b"], f"Cluster {bucket['cluster_b']}"
                ),
                "matches": matches,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "win_rate_a": round(win_rate_a, 4),
                "win_rate_b": round(1.0 - win_rate_a, 4),
                "rivalry_score": round(rivalry_score, 4),
                "close_factor": round(close_factor, 4),
            }
        )

    out.sort(key=lambda row: (row["rivalry_score"], row["matches"]), reverse=True)
    return out[: max(1, limit)]


def _neighbors_by_query(
    *,
    excluded_team_ids: Sequence[int],
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
    query_vec: np.ndarray,
    query_semantic_vec: np.ndarray,
    query_identity_vec: np.ndarray,
    candidate_vectors: np.ndarray,
    neighbors: int,
) -> List[Dict[str, Any]]:
    sims = candidate_vectors @ query_vec
    sem_vecs = np.stack([row.semantic_vector for row in embeddings], axis=0)
    id_vecs = np.stack([row.identity_vector for row in embeddings], axis=0)
    excluded_ids = {int(team_id) for team_id in excluded_team_ids}
    sem_sims = sem_vecs @ query_semantic_vec
    id_sims = id_vecs @ query_identity_vec

    order = np.argsort(-sims)
    out: List[Dict[str, Any]] = []
    for position in order:
        idx = int(position)
        row = embeddings[idx]
        if row.team_id in excluded_ids:
            continue
        out.append(
            {
                "team_id": row.team_id,
                "team_name": row.team_name,
                "lineup_count": row.lineup_count,
                "sim_to_query": round(float(sims[idx]), 4),
                "sim_semantic": round(float(sem_sims[idx]), 4),
                "sim_identity": round(float(id_sims[idx]), 4),
                "identity_delta": round(float(id_sims[idx] - sem_sims[idx]), 4),
                "cluster_id": _resolve_cluster_id(row.team_id, cluster_map),
            }
        )
        if len(out) >= max(1, neighbors):
            break
    return out


def build_team_lab(
    *,
    team_id: int,
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
    matches: Iterable[Dict[str, Any]],
    neighbors: int,
    team_ids: Sequence[int] | None = None,
    team_name: str | None = None,
) -> Dict[str, Any] | None:
    team_lookup = {row.team_id: row for row in embeddings}
    selected_team_ids = [int(team_id)]
    if team_ids:
        selected_team_ids = sorted({int(value) for value in team_ids if int(value) > 0})

    target_rows = [
        team_lookup[int(selected_team_id)]
        for selected_team_id in selected_team_ids
        if int(selected_team_id) in team_lookup
    ]
    if not target_rows:
        return None

    if len(target_rows) == 1 and int(target_rows[0].team_id) == int(team_id):
        target = target_rows[0]
    else:
        target = _aggregate_embedding_rows(
            primary_team_id=int(team_id),
            team_name=str(team_name or target_rows[0].team_name),
            rows=target_rows,
        )
    excluded_team_ids = sorted({int(row.team_id) for row in target_rows})

    vectors = np.stack([row.final_vector for row in embeddings], axis=0)
    neighbors_out = _neighbors_by_query(
        excluded_team_ids=excluded_team_ids,
        embeddings=embeddings,
        cluster_map=cluster_map,
        query_vec=target.final_vector,
        query_semantic_vec=target.semantic_vector,
        query_identity_vec=target.identity_vector,
        candidate_vectors=vectors,
        neighbors=neighbors,
    )

    top_neighbor_sims = [row["sim_to_query"] for row in neighbors_out[:5]]
    uniqueness_score = (
        1.0 - float(np.mean(top_neighbor_sims)) if top_neighbor_sims else 0.0
    )

    own_cluster_id = _resolve_shared_cluster_id(excluded_team_ids, cluster_map)

    total_matches = 0
    total_wins = 0
    same_cluster_matches = 0
    same_cluster_wins = 0
    cross_cluster_matches = 0
    cross_cluster_wins = 0
    by_opponent_cluster: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"matches": 0, "wins": 0}
    )

    for row in matches:
        opponent_team_id = int(row["opponent_team_id"])
        if opponent_team_id in excluded_team_ids:
            continue
        is_win = int(row["is_win"])
        total_matches += 1
        total_wins += is_win

        opponent_cluster_id = _resolve_cluster_id(opponent_team_id, cluster_map)
        opponent_cluster_id = -1 if opponent_cluster_id is None else opponent_cluster_id

        same_cluster = (
            own_cluster_id is not None
            and opponent_cluster_id == own_cluster_id
        )
        if same_cluster:
            same_cluster_matches += 1
            same_cluster_wins += is_win
        else:
            cross_cluster_matches += 1
            cross_cluster_wins += is_win

        bucket = by_opponent_cluster[opponent_cluster_id]
        bucket["matches"] += 1
        bucket["wins"] += is_win

    opponent_cluster_breakdown = []
    for cluster_id, bucket in by_opponent_cluster.items():
        m = bucket["matches"]
        w = bucket["wins"]
        opponent_cluster_breakdown.append(
            {
                "cluster_id": int(cluster_id),
                "matches": int(m),
                "wins": int(w),
                "win_rate": round(w / float(max(1, m)), 4),
            }
        )
    opponent_cluster_breakdown.sort(key=lambda row: row["matches"], reverse=True)

    target_info = {
        "team_id": target.team_id,
        "team_name": target.team_name,
        "lineup_count": target.lineup_count,
        "match_count": target.lineup_count,
        "distinct_lineup_count": target.distinct_lineup_count,
        "lineup_entropy": round(target.lineup_entropy, 4),
        "top_lineup_share_pct": _to_percent(target.top_lineup_share),
        "effective_lineups": round(target.effective_lineups, 4),
        "volatility_score": round(_volatility_score(target), 4),
        "uniqueness_score": round(max(0.0, uniqueness_score), 4),
        "team_ids": excluded_team_ids,
        "selected_team_count": len(excluded_team_ids),
        "cluster_id": own_cluster_id,
        "cluster_size": (
            int(
                next(
                    cluster_map[int(member_team_id)]["cluster_size"]
                    for member_team_id in excluded_team_ids
                    if int(member_team_id) in cluster_map
                )
            )
            if own_cluster_id is not None
            else None
        ),
    }

    match_summary = {
        "matches": int(total_matches),
        "wins": int(total_wins),
        "win_rate": round(total_wins / float(max(1, total_matches)), 4),
        "same_cluster_matches": int(same_cluster_matches),
        "same_cluster_win_rate": round(
            same_cluster_wins / float(max(1, same_cluster_matches)),
            4,
        ),
        "cross_cluster_matches": int(cross_cluster_matches),
        "cross_cluster_win_rate": round(
            cross_cluster_wins / float(max(1, cross_cluster_matches)),
            4,
        ),
    }

    return {
        "team": target_info,
        "neighbors": neighbors_out,
        "match_summary": match_summary,
        "opponent_cluster_breakdown": opponent_cluster_breakdown[:10],
    }


def build_blended_neighbors(
    *,
    team_id: int,
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
    semantic_weight: float,
    neighbors: int,
) -> Dict[str, Any] | None:
    team_lookup = {row.team_id: row for row in embeddings}
    target = team_lookup.get(team_id)
    if target is None:
        return None

    semantic_weight = float(max(0.0, min(1.0, semantic_weight)))
    identity_weight = 1.0 - semantic_weight

    sem_matrix = np.stack([row.semantic_vector for row in embeddings], axis=0)
    id_matrix = np.stack([row.identity_vector for row in embeddings], axis=0)
    blend_matrix = _normalize_rows(
        semantic_weight * sem_matrix + identity_weight * id_matrix
    )

    blend_query = _normalize(
        semantic_weight * target.semantic_vector
        + identity_weight * target.identity_vector
    )

    neighbors_out = _neighbors_by_query(
        excluded_team_ids=[team_id],
        embeddings=embeddings,
        cluster_map=cluster_map,
        query_vec=blend_query,
        query_semantic_vec=target.semantic_vector,
        query_identity_vec=target.identity_vector,
        candidate_vectors=blend_matrix,
        neighbors=neighbors,
    )

    return {
        "team_id": target.team_id,
        "team_name": target.team_name,
        "semantic_weight": round(semantic_weight, 4),
        "identity_weight": round(identity_weight, 4),
        "neighbors": neighbors_out,
    }


def compute_space_projection(
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
    *,
    max_points: int,
) -> Dict[str, Any]:
    if not embeddings:
        return {"points": [], "centroids": [], "bounds": {"x": 1.0, "y": 1.0}}

    ordered = sorted(embeddings, key=lambda row: row.lineup_count, reverse=True)
    sample = ordered[: max(10, min(max_points, len(ordered)))]

    vectors = np.stack([row.final_vector for row in sample], axis=0)
    centered = vectors - vectors.mean(axis=0, keepdims=True)

    if centered.shape[1] >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2].T
        projected = centered @ basis
    else:
        projected = np.pad(centered, ((0, 0), (0, 2 - centered.shape[1])))

    x_abs = float(np.max(np.abs(projected[:, 0]))) if projected.size else 1.0
    y_abs = float(np.max(np.abs(projected[:, 1]))) if projected.size else 1.0
    x_abs = max(1e-6, x_abs)
    y_abs = max(1e-6, y_abs)

    points = []
    for i, row in enumerate(sample):
        cluster_id = _resolve_cluster_id(row.team_id, cluster_map)
        points.append(
            {
                "team_id": row.team_id,
                "team_name": row.team_name,
                "lineup_count": row.lineup_count,
                "cluster_id": cluster_id,
                "x": round(float(projected[i, 0] / x_abs), 5),
                "y": round(float(projected[i, 1] / y_abs), 5),
            }
        )

    centroid_accum: Dict[int, List[tuple[float, float]]] = defaultdict(list)
    for point in points:
        cid = point["cluster_id"]
        if cid is None or cid < 0:
            continue
        centroid_accum[cid].append((float(point["x"]), float(point["y"])))

    centroids = []
    for cid, values in centroid_accum.items():
        xs = [v[0] for v in values]
        ys = [v[1] for v in values]
        centroids.append(
            {
                "cluster_id": int(cid),
                "x": round(float(np.mean(xs)), 5),
                "y": round(float(np.mean(ys)), 5),
                "count": len(values),
            }
        )
    centroids.sort(key=lambda row: row["count"], reverse=True)

    return {
        "points": points,
        "centroids": centroids,
        "bounds": {"x": x_abs, "y": y_abs},
    }


def compute_outliers(
    embeddings: Sequence[EmbeddingRow],
    cluster_map: Dict[int, Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    if not embeddings:
        return []

    centroids = _cluster_centroids(embeddings, cluster_map)
    if not centroids:
        return []

    by_cluster_sims: Dict[int, List[float]] = defaultdict(list)
    rows = []
    for row in embeddings:
        cid = _resolve_cluster_id(row.team_id, cluster_map)
        if cid is None or cid < 0 or cid not in centroids:
            continue
        sim = float(row.final_vector @ centroids[cid])
        by_cluster_sims[cid].append(sim)
        rows.append((row, cid, sim))

    cluster_stats: Dict[int, tuple[float, float]] = {}
    for cid, sims in by_cluster_sims.items():
        mean = float(np.mean(sims))
        std = float(np.std(sims))
        cluster_stats[cid] = (mean, max(std, 1e-6))

    out = []
    for row, cid, sim in rows:
        mean, std = cluster_stats[cid]
        z = (mean - sim) / std
        rarity = max(0.0, z) + 0.35 * (1.0 - sim)
        out.append(
            {
                "team_id": row.team_id,
                "team_name": row.team_name,
                "cluster_id": cid,
                "lineup_count": row.lineup_count,
                "cluster_similarity": round(sim, 4),
                "cluster_mean_similarity": round(mean, 4),
                "outlier_z": round(z, 4),
                "outlier_score": round(rarity, 4),
                "lineup_entropy": round(row.lineup_entropy, 4),
                "top_lineup_share_pct": _to_percent(row.top_lineup_share),
            }
        )

    out.sort(key=lambda row: row["outlier_score"], reverse=True)
    return out[: max(1, limit)]


def compute_snapshot_drift(
    *,
    current_snapshot_id: int,
    previous_snapshot_id: int,
    current_embeddings: Sequence[EmbeddingRow],
    previous_embeddings: Sequence[EmbeddingRow],
    current_cluster_map: Dict[int, Dict[str, Any]],
    previous_cluster_map: Dict[int, Dict[str, Any]],
    top_movers: int,
) -> Dict[str, Any]:
    current_by_id = {row.team_id: row for row in current_embeddings}
    previous_by_id = {row.team_id: row for row in previous_embeddings}

    common_ids = sorted(set(current_by_id).intersection(previous_by_id))
    new_ids = sorted(set(current_by_id) - set(previous_by_id))
    dropped_ids = sorted(set(previous_by_id) - set(current_by_id))

    if not common_ids:
        return {
            "current_snapshot_id": current_snapshot_id,
            "previous_snapshot_id": previous_snapshot_id,
            "summary": {
                "shared_teams": 0,
                "new_teams": len(new_ids),
                "dropped_teams": len(dropped_ids),
                "cluster_switches": 0,
                "newly_clustered": 0,
                "newly_noise": 0,
                "avg_embedding_drift": 0.0,
                "p90_embedding_drift": 0.0,
            },
            "top_embedding_movers": [],
            "top_volatility_shifts": [],
        }

    drift_rows = []
    volatility_shift_rows = []

    cluster_switches = 0
    newly_clustered = 0
    newly_noise = 0
    drift_values = []

    for team_id in common_ids:
        cur = current_by_id[team_id]
        prev = previous_by_id[team_id]

        sim = float(cur.final_vector @ prev.final_vector)
        drift = max(0.0, 1.0 - sim)
        drift_values.append(drift)

        cur_cluster = _resolve_cluster_id(team_id, current_cluster_map)
        prev_cluster = _resolve_cluster_id(team_id, previous_cluster_map)

        cur_clustered = cur_cluster is not None and cur_cluster >= 0
        prev_clustered = prev_cluster is not None and prev_cluster >= 0

        if cur_clustered and prev_clustered and cur_cluster != prev_cluster:
            cluster_switches += 1
        elif cur_clustered and not prev_clustered:
            newly_clustered += 1
        elif prev_clustered and not cur_clustered:
            newly_noise += 1

        volatility_delta = _volatility_score(cur) - _volatility_score(prev)

        drift_rows.append(
            {
                "team_id": team_id,
                "team_name": cur.team_name,
                "drift": round(drift, 4),
                "prev_cluster_id": prev_cluster,
                "current_cluster_id": cur_cluster,
                "lineups_prev": prev.lineup_count,
                "lineups_current": cur.lineup_count,
                "lineup_delta": cur.lineup_count - prev.lineup_count,
            }
        )
        volatility_shift_rows.append(
            {
                "team_id": team_id,
                "team_name": cur.team_name,
                "volatility_prev": round(_volatility_score(prev), 4),
                "volatility_current": round(_volatility_score(cur), 4),
                "volatility_delta": round(volatility_delta, 4),
            }
        )

    drift_rows.sort(key=lambda row: row["drift"], reverse=True)
    volatility_shift_rows.sort(
        key=lambda row: abs(row["volatility_delta"]),
        reverse=True,
    )

    return {
        "current_snapshot_id": current_snapshot_id,
        "previous_snapshot_id": previous_snapshot_id,
        "summary": {
            "shared_teams": len(common_ids),
            "new_teams": len(new_ids),
            "dropped_teams": len(dropped_ids),
            "cluster_switches": cluster_switches,
            "newly_clustered": newly_clustered,
            "newly_noise": newly_noise,
            "avg_embedding_drift": round(float(np.mean(drift_values)), 4),
            "p90_embedding_drift": round(float(np.percentile(drift_values, 90)), 4),
        },
        "top_embedding_movers": drift_rows[: max(1, top_movers)],
        "top_volatility_shifts": volatility_shift_rows[: max(1, top_movers)],
    }
