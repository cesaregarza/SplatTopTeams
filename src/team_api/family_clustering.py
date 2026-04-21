from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Sequence

import numpy as np

from team_api.search_logic import _eligible_for_consolidation
from team_api.store import EmbeddingRow


ClusterMap = Dict[int, Dict[str, int]]


def _normalized_matrix(
    rows: Sequence[EmbeddingRow],
    *,
    vector_attr: str,
) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=float)

    matrix = np.vstack(
        [
            np.asarray(getattr(row, vector_attr), dtype=float)
            for row in rows
        ]
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return matrix / norms


def _build_cluster_map_from_components(
    rows: Sequence[EmbeddingRow],
    parent: list[int],
    *,
    min_cluster_size: int,
) -> ClusterMap:
    groups: dict[int, list[int]] = defaultdict(list)

    def find(idx: int) -> int:
        node = idx
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for idx in range(len(rows)):
        groups[find(idx)].append(idx)

    cluster_map: ClusterMap = {}
    next_cluster_id = 0
    for members in sorted(
        groups.values(),
        key=lambda member_idxs: (-len(member_idxs), min(member_idxs)),
    ):
        if len(members) < max(1, int(min_cluster_size)):
            continue
        member_team_ids = sorted(int(rows[idx].team_id) for idx in members)
        cluster_size = len(member_team_ids)
        for team_id in member_team_ids:
            cluster_map[team_id] = {
                "cluster_id": next_cluster_id,
                "cluster_size": cluster_size,
            }
        next_cluster_id += 1

    return cluster_map


def cluster_members_by_team(
    rows: Sequence[EmbeddingRow],
    cluster_map: ClusterMap,
) -> dict[int, frozenset[int]]:
    team_ids = [int(row.team_id) for row in rows]
    members_by_cluster: dict[int, frozenset[int]] = {}
    grouped: dict[int, set[int]] = defaultdict(set)

    for team_id, meta in cluster_map.items():
        grouped[int(meta["cluster_id"])].add(int(team_id))

    for cluster_id, members in grouped.items():
        members_by_cluster[int(cluster_id)] = frozenset(sorted(members))

    components: dict[int, frozenset[int]] = {}
    for team_id in team_ids:
        meta = cluster_map.get(team_id)
        if not meta:
            components[team_id] = frozenset({team_id})
            continue
        components[team_id] = members_by_cluster[int(meta["cluster_id"])]
    return components


def cluster_rows_vector_greedy(
    rows: Sequence[EmbeddingRow],
    *,
    threshold: float = 0.90,
    min_cluster_size: int = 2,
    max_teams: int = 2500,
    vector_attr: str = "final_vector",
) -> ClusterMap:
    if not rows:
        return {}

    vectors = _normalized_matrix(rows, vector_attr=vector_attr)
    team_ids = [int(row.team_id) for row in rows]
    lineup_counts = np.asarray(
        [int(row.lineup_count) for row in rows],
        dtype=float,
    )
    order_all = np.argsort(-lineup_counts)
    selected = order_all[: max(1, min(int(max_teams), len(order_all)))]
    assigned = np.zeros(len(selected), dtype=bool)

    cluster_map: ClusterMap = {}
    next_cluster_id = 0

    for pos, idx in enumerate(selected):
        if assigned[pos]:
            continue

        sims = vectors[selected] @ vectors[idx]
        member_positions = np.where((sims >= float(threshold)) & (~assigned))[0]

        if len(member_positions) < max(1, int(min_cluster_size)):
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


def cluster_rows_vector_union(
    rows: Sequence[EmbeddingRow],
    *,
    threshold: float = 0.90,
    min_cluster_size: int = 2,
    vector_attr: str = "final_vector",
) -> ClusterMap:
    if not rows:
        return {}

    vectors = _normalized_matrix(rows, vector_attr=vector_attr)
    sims = vectors @ vectors.T
    n_rows = len(rows)
    parent = list(range(n_rows))

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

    for left in range(n_rows):
        for right in range(left + 1, n_rows):
            if float(sims[left, right]) >= float(threshold):
                union(left, right)

    return _build_cluster_map_from_components(
        rows,
        parent,
        min_cluster_size=min_cluster_size,
    )


def cluster_rows_consolidation_graph(
    rows: Sequence[EmbeddingRow],
    *,
    min_overlap: float = 0.72,
    min_cluster_size: int = 2,
) -> ClusterMap:
    if not rows:
        return {}

    n_rows = len(rows)
    parent = list(range(n_rows))
    payloads = [dict(vars(row)) for row in rows]

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

    for left in range(n_rows):
        for right in range(left + 1, n_rows):
            if _eligible_for_consolidation(
                payloads[left],
                payloads[right],
                min_overlap=min_overlap,
            ):
                union(left, right)

    return _build_cluster_map_from_components(
        rows,
        parent,
        min_cluster_size=min_cluster_size,
    )


def _shared_player_fraction(a: EmbeddingRow, b: EmbeddingRow) -> float:
    players_a = set(int(player_id) for player_id in (a.roster_player_ids or ()))
    players_b = set(int(player_id) for player_id in (b.roster_player_ids or ()))
    if not players_a or not players_b:
        return 0.0
    return float(len(players_a & players_b)) / float(min(len(players_a), len(players_b)))


def cluster_rows_hybrid_graph(
    rows: Sequence[EmbeddingRow],
    *,
    min_overlap: float = 0.72,
    vector_threshold: float = 0.88,
    min_shared_player_fraction: float = 0.60,
    min_cluster_size: int = 2,
    vector_attr: str = "final_vector",
) -> ClusterMap:
    if not rows:
        return {}

    n_rows = len(rows)
    parent = list(range(n_rows))
    payloads = [dict(vars(row)) for row in rows]
    vectors = _normalized_matrix(rows, vector_attr=vector_attr)
    sims = vectors @ vectors.T

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

    for left in range(n_rows):
        for right in range(left + 1, n_rows):
            if _eligible_for_consolidation(
                payloads[left],
                payloads[right],
                min_overlap=min_overlap,
            ):
                union(left, right)
                continue

            if float(sims[left, right]) < float(vector_threshold):
                continue

            if _shared_player_fraction(rows[left], rows[right]) < float(
                min_shared_player_fraction
            ):
                continue

            union(left, right)

    return _build_cluster_map_from_components(
        rows,
        parent,
        min_cluster_size=min_cluster_size,
    )


CLUSTERING_APPROACHES: dict[str, Callable[..., ClusterMap]] = {
    "vector_greedy": cluster_rows_vector_greedy,
    "vector_union": cluster_rows_vector_union,
    "consolidation_graph": cluster_rows_consolidation_graph,
    "hybrid_graph": cluster_rows_hybrid_graph,
}
