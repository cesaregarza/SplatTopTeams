from __future__ import annotations

import numpy as np

from shared_lib.team_vector_utils import hash_index, hash_sign
from team_api.search_logic import (
    build_player_query_profile,
    build_query_final_vector,
    build_query_identity_vector,
    build_query_semantic_vector,
    consolidate_ranked_results,
    rank_similar_teams,
    rescore_consolidated_results,
)
from team_api.store import EmbeddingRow


def _row(
    team_id: int,
    name: str,
    vec: list[float],
    *,
    semantic: list[float] | None = None,
    identity: list[float] | None = None,
    lineup_count: int = 10,
    top_lineup_player_ids=(),
    player_support=None,
    pair_support=None,
    lineup_variant_counts=None,
) -> EmbeddingRow:
    arr = np.asarray(vec, dtype=np.float64)
    semantic_arr = np.asarray(semantic if semantic is not None else vec, dtype=np.float64)
    identity_arr = np.asarray(identity if identity is not None else vec, dtype=np.float64)
    return EmbeddingRow(
        team_id=team_id,
        tournament_id=1,
        team_name=name,
        event_time_ms=1,
        lineup_count=lineup_count,
        semantic_vector=semantic_arr,
        identity_vector=identity_arr,
        final_vector=arr,
        top_lineup_summary="",
        top_lineup_player_ids=tuple(int(pid) for pid in top_lineup_player_ids),
        player_support=dict(player_support or {}),
        pair_support=dict(pair_support or {}),
        lineup_variant_counts=dict(lineup_variant_counts or {}),
    )


def _final_vector(semantic: list[float], identity: list[float], beta: float = 3.0) -> np.ndarray:
    sem = np.asarray(semantic, dtype=np.float64)
    ident = np.asarray(identity, dtype=np.float64)
    combined = np.concatenate([sem, beta * ident])
    return combined / np.linalg.norm(combined)


def test_rank_similar_teams_orders_by_similarity():
    rows = [
        _row(1, "Alpha", [1.0, 0.0]),
        _row(2, "Bravo", [0.9, 0.1]),
        _row(3, "Charlie", [0.0, 1.0]),
    ]
    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[1],
        cluster_map={},
        top_n=3,
    )
    ordered_ids = [r["team_id"] for r in out["results"]]
    assert ordered_ids[:2] == [1, 2]
    assert ordered_ids[2] == 3


def test_build_whole_set_query_vectors_dedupes_and_matches_hash_scheme():
    player_ids = [7, 3, 7, 11]
    idf_lookup = {3: 1.2, 7: 2.0, 11: 0.7}

    semantic = build_query_semantic_vector(player_ids, 8)
    identity = build_query_identity_vector(player_ids, 8, idf_lookup)
    final = build_query_final_vector(
        player_ids,
        semantic_dim=8,
        identity_dim=8,
        identity_beta=3.0,
        idf_lookup=idf_lookup,
    )

    expected_semantic = np.zeros(8, dtype=np.float64)
    expected_identity = np.zeros(8, dtype=np.float64)
    for player_id in (3, 7, 11):
        expected_semantic[hash_index(player_id, 8, "sem_idx")] += hash_sign(
            player_id, "sem_sign"
        )
        expected_identity[hash_index(player_id, 8, "id_idx")] += hash_sign(
            player_id, "id_sign"
        ) * idf_lookup[player_id]

    expected_semantic = expected_semantic / np.linalg.norm(expected_semantic)
    expected_identity = expected_identity / np.linalg.norm(expected_identity)
    expected_final = np.concatenate([expected_semantic, 3.0 * expected_identity])
    expected_final = expected_final / np.linalg.norm(expected_final)

    assert np.allclose(semantic, expected_semantic)
    assert np.allclose(identity, expected_identity)
    assert np.allclose(final, expected_final)


def test_rank_similar_teams_pair_rerank_breaks_dense_ties():
    profile = build_player_query_profile(
        [1, 2, 3, 4],
        semantic_dim=8,
        identity_dim=8,
        identity_beta=3.0,
        idf_lookup={1: 2.5, 2: 2.0, 3: 1.5, 4: 1.0},
    )
    query_final = np.asarray(profile["query_final"], dtype=np.float64)
    rows = [
        _row(
            1,
            "Pair Exact",
            query_final.tolist(),
            semantic=np.asarray(profile["query_semantic"], dtype=np.float64).tolist(),
            identity=np.asarray(profile["query_identity"], dtype=np.float64).tolist(),
            player_support={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            pair_support={
                (1, 2): 1.0,
                (1, 3): 1.0,
                (1, 4): 1.0,
                (2, 3): 1.0,
                (2, 4): 1.0,
                (3, 4): 1.0,
            },
        ),
        _row(
            2,
            "Pair Weak",
            query_final.tolist(),
            semantic=np.asarray(profile["query_semantic"], dtype=np.float64).tolist(),
            identity=np.asarray(profile["query_identity"], dtype=np.float64).tolist(),
            player_support={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            pair_support={},
        ),
    ]

    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[],
        cluster_map={},
        top_n=2,
        query_profile=profile,
    )

    assert [row["team_id"] for row in out["results"]] == [1, 2]
    assert out["results"][0]["pair_overlap"] > out["results"][1]["pair_overlap"]
    assert out["results"][0]["rerank_score"] > out["results"][1]["rerank_score"]


def test_rank_similar_teams_uses_precomputed_vectors_when_provided():
    rows = [
        _row(1, "Alpha", [1.0, 0.0]),
        _row(2, "Bravo", [0.9, 0.1]),
        _row(3, "Charlie", [0.0, 1.0]),
    ]
    finals = np.stack([row.final_vector for row in rows], axis=0)
    semantics = np.stack([row.semantic_vector for row in rows], axis=0)
    identities = np.stack([row.identity_vector for row in rows], axis=0)
    index = {row.team_id: i for i, row in enumerate(rows)}

    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[1],
        cluster_map={},
        top_n=3,
        precomputed_finals=finals,
        precomputed_semantics=semantics,
        precomputed_identities=identities,
        precomputed_index=index,
    )
    ordered_ids = [r["team_id"] for r in out["results"]]
    assert ordered_ids[:2] == [1, 2]
    assert ordered_ids[2] == 3


def test_rank_similar_teams_respects_relevance_threshold():
    rows = [
        _row(1, "Alpha", [1.0, 0.0]),
        _row(2, "Bravo", [0.99, 0.1]),
        _row(3, "Charlie", [0.2, 0.98]),
    ]
    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[1],
        cluster_map={},
        top_n=3,
        min_relevance=0.85,
    )
    ordered_ids = [r["team_id"] for r in out["results"]]
    assert ordered_ids == [1, 2]


def test_rank_similar_teams_skips_inconsistent_embedding_dims():
    rows = [
        EmbeddingRow(
            team_id=1,
            tournament_id=1,
            team_name="Alpha",
            event_time_ms=1,
            lineup_count=10,
            semantic_vector=np.array(1.0, dtype=np.float64),
            identity_vector=np.array(1.0, dtype=np.float64),
            final_vector=np.array(1.0, dtype=np.float64),
            top_lineup_summary="",
        ),
        EmbeddingRow(
            team_id=2,
            tournament_id=1,
            team_name="Bravo",
            event_time_ms=1,
            lineup_count=10,
            semantic_vector=np.array(1.0, dtype=np.float64),
            identity_vector=np.array(1.0, dtype=np.float64),
            final_vector=np.array(1.0, dtype=np.float64),
            top_lineup_summary="",
        ),
    ]
    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[1],
        cluster_map={},
        top_n=10,
    )
    assert out["results"] == []

    rows_with_mixed = [
        EmbeddingRow(
            team_id=1,
            tournament_id=1,
            team_name="Alpha",
            event_time_ms=1,
            lineup_count=10,
            semantic_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            identity_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            final_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            top_lineup_summary="",
        ),
        EmbeddingRow(
            team_id=2,
            tournament_id=1,
            team_name="Bravo",
            event_time_ms=1,
            lineup_count=10,
            semantic_vector=np.asarray([0.9, 0.1, 0.0], dtype=np.float64),
            identity_vector=np.asarray([0.9, 0.0, 0.1], dtype=np.float64),
            final_vector=np.asarray([0.9, 0.1, 0.0], dtype=np.float64),
            top_lineup_summary="",
        ),
    ]
    out2 = rank_similar_teams(
        embeddings=rows_with_mixed,
        target_team_ids=[1],
        cluster_map={},
        top_n=10,
    )
    assert [r["team_id"] for r in out2["results"]] == [1]


def test_consolidate_ranked_results_groups_similar_teams():
    rows = [
        {
            "team_id": 1,
            "team_name": "FTW",
            "lineup_count": 42,
            "distinct_lineup_count": 7,
            "top_lineup_share": 0.61,
            "top_lineup_player_ids": [101, 102, 103, 104, 105],
            "top_lineup_player_names": ["A", "B", "C", "D", "E"],
            "sim_to_query": 0.99,
        },
        {
            "team_id": 2,
            "team_name": "FTW#2",
            "lineup_count": 45,
            "distinct_lineup_count": 8,
            "top_lineup_share": 0.58,
                "top_lineup_player_ids": [101, 102, 103, 104, 105, 106],
            "top_lineup_player_names": ["A", "B", "C", "D", "F"],
            "sim_to_query": 0.93,
        },
        {
            "team_id": 3,
            "team_name": "Different",
            "lineup_count": 12,
            "distinct_lineup_count": 3,
            "top_lineup_share": 0.33,
            "top_lineup_player_ids": [201, 202, 203, 204, 205],
            "top_lineup_player_names": ["X", "Y", "Z", "W", "Q"],
            "sim_to_query": 0.81,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.7)
    assert len(out) == 2
    rep = out[0]
    assert rep["is_consolidated"] is True
    assert rep["consolidated_team_count"] == 2
    assert rep["match_count"] == 87
    assert rep["tournament_count"] == 0
    assert rep["consolidated_team_names"] == ["FTW", "FTW#2"]
    assert rep["consolidated_team_ids"] == [1, 2]


def test_consolidate_ranked_results_unions_exact_lineup_variants():
    rows = [
        {
            "team_id": 1,
            "team_name": "Healbook",
            "lineup_count": 6,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.5,
            "top_lineup_player_ids": [1, 2, 3, 4],
            "top_lineup_player_names": ["A", "B", "C", "D"],
            "_lineup_variant_counts": {
                (1, 2, 3, 4): 3,
                (1, 2, 3, 5): 3,
            },
        },
        {
            "team_id": 2,
            "team_name": "Healbook#2",
            "lineup_count": 5,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.6,
            "top_lineup_player_ids": [1, 2, 3, 4],
            "top_lineup_player_names": ["A", "B", "C", "D"],
            "_lineup_variant_counts": {
                (1, 2, 3, 4): 2,
                (1, 2, 4, 5): 3,
            },
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)

    assert len(out) == 1
    rep = out[0]
    assert rep["distinct_lineup_count"] == 3
    assert rep["top_lineup_match_count"] == 5
    assert rep["top_lineup_player_ids"] == [1, 2, 3, 4]
    assert rep["top_lineup_share"] == 5 / 11


def test_consolidate_ranked_results_requires_strong_overlap():
    rows = [
        {
            "team_id": 1,
            "team_name": "FTW",
            "lineup_count": 42,
            "distinct_lineup_count": 7,
            "top_lineup_share": 0.61,
            "top_lineup_player_ids": [101, 102, 103, 104, 105],
            "top_lineup_player_names": ["A", "B", "C", "D", "E"],
            "sim_to_query": 0.99,
        },
        {
            "team_id": 2,
            "team_name": "Moonlight",
            "lineup_count": 41,
            "distinct_lineup_count": 7,
            "top_lineup_share": 0.6,
            "top_lineup_player_ids": [101, 106, 107, 108, 109],
            "top_lineup_player_names": ["A", "F", "G", "H", "I"],
            "sim_to_query": 0.97,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.72)
    assert len(out) == 2
    assert "is_consolidated" not in out[0]
    assert "is_consolidated" not in out[1]


def test_consolidate_ranked_results_merges_transitive_matches():
    rows = [
        {
            "team_id": 10_001,
            "team_name": "BlankZ",
            "lineup_count": 30,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.66,
            "top_lineup_player_ids": [1, 2, 3, 4, 5],
            "top_lineup_player_names": ["A", "B", "C", "D", "E"],
            "sim_to_query": 0.99,
        },
        {
            "team_id": 10_002,
            "team_name": "BlankZ#2",
            "lineup_count": 29,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.64,
            "top_lineup_player_ids": [1, 2, 3, 4, 6],
            "top_lineup_player_names": ["A", "B", "C", "D", "F"],
            "sim_to_query": 0.98,
        },
        {
            "team_id": 10_003,
            "team_name": "BlankZ#3",
            "lineup_count": 31,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.65,
            "top_lineup_player_ids": [1, 2, 3, 6, 7],
            "top_lineup_player_names": ["A", "B", "C", "F", "G"],
            "sim_to_query": 0.97,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)
    assert len(out) == 1
    rep = out[0]
    assert rep["is_consolidated"] is True
    assert rep["consolidated_team_count"] == 3
    assert rep["match_count"] == 90
    assert set(rep["consolidated_team_ids"]) == {10_001, 10_002, 10_003}


def test_consolidate_ranked_results_keeps_exact_lineup_together():
    rows = [
        {
            "team_id": 10,
            "team_name": "BlankZ",
            "lineup_count": 30,
            "distinct_lineup_count": 4,
            "top_lineup_share": 0.12,
            "top_lineup_player_ids": [901, 902, 903, 904, 905, 906],
            "top_lineup_player_names": ["A", "B", "C", "D", "E", "F"],
            "sim_to_query": 0.99,
        },
        {
            "team_id": 11,
            "team_name": "BlankZ#2",
            "lineup_count": 500,
            "distinct_lineup_count": 35,
            "top_lineup_share": 0.89,
            "top_lineup_player_ids": [901, 902, 903, 904, 905, 906],
            "top_lineup_player_names": ["A", "B", "C", "D", "E", "F"],
            "sim_to_query": 0.97,
        },
        {
            "team_id": 12,
            "team_name": "Different",
            "lineup_count": 120,
            "distinct_lineup_count": 16,
            "top_lineup_share": 0.31,
            "top_lineup_player_ids": [700, 701, 702, 703, 704, 705],
            "top_lineup_player_names": ["X", "Y", "Z", "W", "Q", "R"],
            "sim_to_query": 0.95,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.72)
    assert len(out) == 2
    rep = next(result for result in out if result["team_id"] == 10)
    assert rep["is_consolidated"] is True
    assert rep["consolidated_team_count"] == 2
    assert set(rep["consolidated_team_ids"]) == {10, 11}


def test_consolidate_ranked_results_consolidates_single_player_drift():
    rows = [
        {
            "team_id": 20,
            "team_name": "BlankZ",
            "lineup_count": 12,
            "distinct_lineup_count": 1,
            "top_lineup_share": 1.0,
            "top_lineup_player_ids": [10, 20, 30, 40],
            "top_lineup_player_names": ["A", "B", "C", "D"],
            "sim_to_query": 0.91,
        },
        {
            "team_id": 21,
            "team_name": "BlankZ#2",
            "lineup_count": 11,
            "distinct_lineup_count": 1,
            "top_lineup_share": 1.0,
            "top_lineup_player_ids": [10, 20, 30, 50],
            "top_lineup_player_names": ["A", "B", "C", "E"],
            "sim_to_query": 0.89,
        },
        {
            "team_id": 22,
            "team_name": "Not Same",
            "lineup_count": 12,
            "distinct_lineup_count": 1,
            "top_lineup_share": 1.0,
            "top_lineup_player_ids": [11, 21, 31, 41],
            "top_lineup_player_names": ["F", "G", "H", "I"],
            "sim_to_query": 0.88,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)
    assert len(out) == 2
    rep = next(result for result in out if result["team_id"] == 20)
    assert rep["is_consolidated"] is True
    assert rep["consolidated_team_count"] == 2
    assert set(rep["consolidated_team_ids"]) == {20, 21}


def test_consolidate_ranked_results_allows_one_player_drift_for_six_player_lineups():
    rows = [
        {
            "team_id": 30,
            "team_name": "CoreA",
            "lineup_count": 40,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.76,
            "top_lineup_player_ids": [1, 2, 3, 4, 5, 6],
            "top_lineup_player_names": ["A", "B", "C", "D", "E", "F"],
            "sim_to_query": 0.98,
        },
        {
            "team_id": 31,
            "team_name": "CoreA#2",
            "lineup_count": 41,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.75,
            "top_lineup_player_ids": [1, 2, 3, 4, 5, 7],
            "top_lineup_player_names": ["A", "B", "C", "D", "E", "G"],
            "sim_to_query": 0.97,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)
    assert len(out) == 1
    rep = out[0]
    assert rep["is_consolidated"] is True
    assert rep["consolidated_team_count"] == 2
    assert rep["match_count"] == 81


def test_consolidate_ranked_results_blocks_two_player_drift_for_six_player_lineups():
    rows = [
        {
            "team_id": 40,
            "team_name": "CoreB",
            "lineup_count": 40,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.76,
            "top_lineup_player_ids": [1, 2, 3, 4, 5, 6],
            "top_lineup_player_names": ["A", "B", "C", "D", "E", "F"],
            "sim_to_query": 0.98,
        },
        {
            "team_id": 41,
            "team_name": "CoreB#2",
            "lineup_count": 41,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.75,
            "top_lineup_player_ids": [1, 2, 3, 4, 7, 8, 9],
            "top_lineup_player_names": ["A", "B", "C", "D", "G", "H"],
            "sim_to_query": 0.97,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)
    assert len(out) == 2
    for result in out:
        assert not result.get("is_consolidated", False)


def test_consolidate_ranked_results_scales_to_larger_core_groups():
    rows = [
        {
            "team_id": 50,
            "team_name": "CoreC",
            "lineup_count": 40,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.76,
            "top_lineup_player_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            "top_lineup_player_names": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "sim_to_query": 0.98,
        },
        {
            "team_id": 51,
            "team_name": "CoreC#2",
            "lineup_count": 41,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.75,
            "top_lineup_player_ids": [1, 2, 3, 4, 5, 6, 9, 10, 11],
            "top_lineup_player_names": ["A", "B", "C", "D", "E", "F", "I", "J"],
            "sim_to_query": 0.97,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)
    assert len(out) == 1
    rep = out[0]
    assert rep["is_consolidated"] is True
    assert rep["consolidated_team_count"] == 2
    assert set(rep["consolidated_team_ids"]) == {50, 51}


def test_consolidate_ranked_results_exposes_sorted_core_lineup_players():
    rows = [
        {
            "team_id": 60,
            "team_name": "BlankZ",
            "lineup_count": 10,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.6,
            "top_lineup_player_ids": [1, 2, 3, 4, 5],
            "top_lineup_player_names": ["A", "B", "C", "D", "E"],
            "sim_to_query": 0.99,
        },
        {
            "team_id": 61,
            "team_name": "BlankZ#2",
            "lineup_count": 10,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.3,
            "top_lineup_player_ids": [1, 2, 3, 4, 6],
            "top_lineup_player_names": ["A", "B", "C", "D", "F"],
            "sim_to_query": 0.98,
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)
    assert len(out) == 1
    rep = out[0]
    assert rep["is_consolidated"] is True
    assert rep["lineup_count"] == 20
    assert rep["match_count"] == 20
    assert rep["top_lineup_match_count"] <= rep["lineup_count"]
    core_players = rep["core_lineup_players"]
    assert [row["player_id"] for row in core_players] == [1, 2, 3, 4, 5, 6]
    assert [row["matches_played"] for row in core_players] == [9, 9, 9, 9, 6, 3]


def test_rescore_consolidated_results_uses_aggregate_support_profiles():
    profile = build_player_query_profile(
        [1, 2, 3, 4, 5, 6],
        semantic_dim=8,
        identity_dim=8,
        identity_beta=3.0,
        idf_lookup={1: 2.5, 2: 2.4, 3: 2.3, 4: 2.2, 5: 2.1, 6: 2.0},
    )
    query_final = np.asarray(profile["query_final"], dtype=np.float64)
    query_semantic = np.asarray(profile["query_semantic"], dtype=np.float64)
    query_identity = np.asarray(profile["query_identity"], dtype=np.float64)

    row_a = _row(
        101,
        "Alias A",
        query_final.tolist(),
        semantic=query_semantic.tolist(),
        identity=query_identity.tolist(),
        lineup_count=20,
        top_lineup_player_ids=(1, 2, 3, 4, 5),
        player_support={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
        pair_support={(1, 2): 1.0, (2, 3): 1.0, (3, 4): 1.0, (4, 5): 1.0},
    )
    row_b = _row(
        102,
        "Alias B",
        query_final.tolist(),
        semantic=query_semantic.tolist(),
        identity=query_identity.tolist(),
        lineup_count=18,
        top_lineup_player_ids=(1, 2, 3, 4, 6),
        player_support={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 6: 1.0},
        pair_support={(1, 2): 1.0, (2, 3): 1.0, (3, 4): 1.0, (4, 6): 1.0},
    )
    row_c = _row(
        201,
        "Single",
        query_final.tolist(),
        semantic=query_semantic.tolist(),
        identity=query_identity.tolist(),
        lineup_count=40,
        top_lineup_player_ids=(1, 2, 7, 8, 9),
        player_support={1: 1.0, 2: 1.0, 7: 1.0, 8: 1.0, 9: 1.0},
        pair_support={(1, 2): 1.0},
    )

    ranked = rank_similar_teams(
        embeddings=[row_a, row_b, row_c],
        target_team_ids=[],
        cluster_map={},
        top_n=3,
        query_profile=profile,
    )
    consolidated = consolidate_ranked_results(ranked["results"], min_overlap=0.72)
    rescored = rescore_consolidated_results(
        consolidated,
        query_profile=profile,
        rerank_weight_embed=0.70,
        rerank_weight_player=0.15,
        rerank_weight_pair=0.15,
        use_pair_rerank=True,
    )

    assert rescored[0]["consolidated_team_count"] == 2
    assert rescored[0]["team_id"] == 101
    assert rescored[0]["rerank_score"] >= rescored[1]["rerank_score"]
    assert rescored[0]["player_overlap"] > rescored[1]["player_overlap"]


def test_rank_similar_teams_candidate_top_n_widens_alias_pool_for_consolidation():
    query_pairs = {
        (1, 2): 1.0,
        (1, 3): 1.0,
        (1, 4): 1.0,
        (2, 3): 1.0,
        (2, 4): 1.0,
        (3, 4): 1.0,
    }
    query_profile = {
        "query_final": np.asarray([1.0, 0.0], dtype=np.float64),
        "query_semantic": np.asarray([1.0, 0.0], dtype=np.float64),
        "query_identity": np.asarray([1.0, 0.0], dtype=np.float64),
        "query_player_weights": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        "query_pair_weights": query_pairs,
        "query_pairs": tuple(query_pairs),
    }

    rows = [
        _row(
            1,
            "Alias A",
            [1.0, 0.0],
            lineup_count=20,
            top_lineup_player_ids=(1, 2, 3, 4),
            player_support={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            pair_support={pair: 1.0 for pair in query_pairs},
        ),
        _row(
            2,
            "Distractor",
            [1.0, 0.0],
            lineup_count=19,
            top_lineup_player_ids=(1, 5, 6, 7),
            player_support={1: 1.0, 5: 1.0, 6: 1.0, 7: 1.0},
            pair_support={},
        ),
        _row(
            3,
            "Alias B",
            [0.78, 0.6257795138864806],
            lineup_count=18,
            top_lineup_player_ids=(1, 2, 3, 5),
            player_support={1: 1.0, 2: 1.0, 3: 1.0, 5: 1.0},
            pair_support={
                (1, 2): 1.0,
                (1, 3): 1.0,
                (1, 5): 1.0,
                (2, 3): 1.0,
                (2, 5): 1.0,
                (3, 5): 1.0,
            },
        ),
    ]

    narrow = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[],
        cluster_map={},
        top_n=2,
        query_profile=query_profile,
    )
    narrow_consolidated = consolidate_ranked_results(
        narrow["results"],
        min_overlap=0.72,
    )
    assert all(
        set(result.get("consolidated_team_ids") or [result["team_id"]]) != {1, 3}
        for result in narrow_consolidated
    )

    wide = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[],
        cluster_map={},
        top_n=2,
        candidate_top_n=3,
        query_profile=query_profile,
    )
    wide_consolidated = consolidate_ranked_results(
        wide["results"],
        min_overlap=0.72,
    )
    wide_rescored = rescore_consolidated_results(
        wide_consolidated,
        query_profile=query_profile,
        rerank_weight_embed=0.70,
        rerank_weight_player=0.15,
        rerank_weight_pair=0.15,
        use_pair_rerank=True,
    )

    assert set(wide_rescored[0]["consolidated_team_ids"]) == {1, 3}
    assert wide_rescored[0]["consolidated_team_count"] == 2
    assert wide_rescored[0]["rerank_score"] > wide_rescored[1]["rerank_score"]


def test_rank_similar_teams_lineup_overlap_breaks_structural_ties():
    profile = build_player_query_profile(
        [1, 2, 3, 4],
        semantic_dim=8,
        identity_dim=8,
        identity_beta=3.0,
        idf_lookup={1: 2.0, 2: 1.8, 3: 1.6, 4: 1.4},
    )
    query_final = np.asarray(profile["query_final"], dtype=np.float64)
    query_semantic = np.asarray(profile["query_semantic"], dtype=np.float64)
    query_identity = np.asarray(profile["query_identity"], dtype=np.float64)
    shared_player_support = {1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8}
    shared_pair_support = {
        (1, 2): 0.8,
        (1, 3): 0.8,
        (1, 4): 0.8,
        (2, 3): 0.8,
        (2, 4): 0.8,
        (3, 4): 0.8,
    }

    rows = [
        _row(
            1,
            "Exact Lineup",
            query_final.tolist(),
            semantic=query_semantic.tolist(),
            identity=query_identity.tolist(),
            lineup_count=8,
            top_lineup_player_ids=(1, 2, 3, 4),
            player_support=shared_player_support,
            pair_support=shared_pair_support,
            lineup_variant_counts={(1, 2, 3, 4): 8},
        ),
        _row(
            2,
            "Near Lineup",
            query_final.tolist(),
            semantic=query_semantic.tolist(),
            identity=query_identity.tolist(),
            lineup_count=8,
            top_lineup_player_ids=(1, 2, 3, 5),
            player_support=shared_player_support,
            pair_support=shared_pair_support,
            lineup_variant_counts={(1, 2, 3, 5): 8},
        ),
    ]

    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[],
        cluster_map={},
        top_n=2,
        query_profile=profile,
    )

    assert [row["team_id"] for row in out["results"]] == [1, 2]
    assert out["results"][0]["player_overlap"] == out["results"][1]["player_overlap"]
    assert out["results"][0]["pair_overlap"] == out["results"][1]["pair_overlap"]
    assert out["results"][0]["lineup_overlap"] > out["results"][1]["lineup_overlap"]
    assert out["results"][0]["rerank_score"] > out["results"][1]["rerank_score"]


def test_rescore_consolidated_results_preserves_best_member_signal():
    query_pairs = {
        (1, 2): 1.0,
        (1, 3): 1.0,
        (1, 4): 1.0,
        (2, 3): 1.0,
        (2, 4): 1.0,
        (3, 4): 1.0,
    }
    query_profile = {
        "query_final": np.asarray([1.0, 0.0], dtype=np.float64),
        "query_semantic": np.asarray([1.0, 0.0], dtype=np.float64),
        "query_identity": np.asarray([1.0, 0.0], dtype=np.float64),
        "query_player_weights": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        "query_pair_weights": query_pairs,
        "query_pairs": tuple(query_pairs),
    }

    rows = [
        _row(
            101,
            "Strong Alias",
            [1.0, 0.0],
            lineup_count=20,
            top_lineup_player_ids=(1, 2, 3, 4),
            player_support={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            pair_support={pair: 1.0 for pair in query_pairs},
        ),
        *[
            _row(
                200 + idx,
                f"Weak Alias {idx}",
                [0.3, 0.9539392014169457],
                lineup_count=18,
                top_lineup_player_ids=(1, 2, 3, 5),
                player_support={1: 1.0, 2: 1.0, 3: 1.0, 5: 1.0},
                pair_support={
                    (1, 2): 1.0,
                    (1, 3): 1.0,
                    (1, 5): 1.0,
                    (2, 3): 1.0,
                    (2, 5): 1.0,
                    (3, 5): 1.0,
                },
            )
            for idx in range(1, 6)
        ],
        _row(
            999,
            "Distractor",
            [0.92, 0.3919183588453085],
            lineup_count=19,
            top_lineup_player_ids=(1, 6, 7, 8),
            player_support={1: 1.0, 6: 1.0, 7: 1.0, 8: 1.0},
            pair_support={},
        ),
    ]

    ranked = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[],
        cluster_map={},
        top_n=len(rows),
        query_profile=query_profile,
    )
    best_member_score = next(
        result["rerank_score"] for result in ranked["results"] if result["team_id"] == 101
    )

    consolidated = consolidate_ranked_results(ranked["results"], min_overlap=0.72)
    rescored = rescore_consolidated_results(
        consolidated,
        query_profile=query_profile,
        rerank_weight_embed=0.70,
        rerank_weight_player=0.15,
        rerank_weight_pair=0.15,
        use_pair_rerank=True,
    )

    assert set(rescored[0]["consolidated_team_ids"]) == {101, 201, 202, 203, 204, 205}
    assert rescored[0]["rerank_score"] < best_member_score
    assert rescored[0]["rerank_score"] > 0.70
    assert rescored[0]["rerank_score"] > rescored[1]["rerank_score"]


def test_rank_similar_teams_recency_weight_zero_is_identity():
    """recency_weight=0 should produce the same ordering as default."""
    rows = [
        _row(1, "Alpha", [1.0, 0.0]),
        _row(2, "Bravo", [0.9, 0.1]),
        _row(3, "Charlie", [0.0, 1.0]),
    ]
    base = rank_similar_teams(
        embeddings=rows, target_team_ids=[1], cluster_map={}, top_n=3,
    )
    with_zero = rank_similar_teams(
        embeddings=rows, target_team_ids=[1], cluster_map={}, top_n=3,
        recency_weight=0.0,
    )
    assert (
        [r["team_id"] for r in base["results"]]
        == [r["team_id"] for r in with_zero["results"]]
    )


def test_rank_similar_teams_recency_boosts_recent_team():
    """With recency_weight > 0, a more recent team should rank higher."""
    now_ms = 1_700_000_000_000
    old_ms = 1_600_000_000_000  # ~115 days earlier
    vec = [0.5, 0.5]
    arr = np.asarray(vec, dtype=np.float64)
    rows = [
        EmbeddingRow(
            team_id=1, tournament_id=1, team_name="Query",
            event_time_ms=now_ms, lineup_count=10,
            semantic_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            identity_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            final_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            top_lineup_summary="",
        ),
        EmbeddingRow(
            team_id=2, tournament_id=1, team_name="Recent",
            event_time_ms=now_ms, lineup_count=10,
            semantic_vector=arr, identity_vector=arr, final_vector=arr,
            top_lineup_summary="",
        ),
        EmbeddingRow(
            team_id=3, tournament_id=1, team_name="Old",
            event_time_ms=old_ms, lineup_count=10,
            semantic_vector=arr, identity_vector=arr, final_vector=arr,
            top_lineup_summary="",
        ),
    ]

    out_no_recency = rank_similar_teams(
        embeddings=rows, target_team_ids=[1], cluster_map={}, top_n=3,
    )
    ids_no = [r["team_id"] for r in out_no_recency["results"]]
    # Without recency, same vectors → same similarity → same rank
    # (both team 2 and 3 have identical final_vector)

    out_with_recency = rank_similar_teams(
        embeddings=rows, target_team_ids=[1], cluster_map={}, top_n=3,
        recency_weight=0.5,
    )
    ids_rec = [r["team_id"] for r in out_with_recency["results"]]
    # The recent team (id=2) should rank ahead of the old team (id=3)
    idx_recent = ids_rec.index(2)
    idx_old = ids_rec.index(3)
    assert idx_recent < idx_old


def test_rank_similar_teams_prefers_semantic_match_for_volatile_query():
    query_sem = [1.0, 0.0]
    query_id = [1.0, 0.0]
    rows = [
        EmbeddingRow(
            team_id=1,
            tournament_id=1,
            team_name="Volatile Query",
            event_time_ms=1_700_000_000_000,
            lineup_count=40,
            semantic_vector=np.asarray(query_sem, dtype=np.float64),
            identity_vector=np.asarray(query_id, dtype=np.float64),
            final_vector=_final_vector(query_sem, query_id),
            top_lineup_summary="",
            distinct_lineup_count=8,
            top_lineup_share=0.22,
            lineup_entropy=0.88,
            effective_lineups=6.4,
        ),
        EmbeddingRow(
            team_id=2,
            tournament_id=1,
            team_name="Semantic Match",
            event_time_ms=1_700_000_000_000,
            lineup_count=18,
            semantic_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            identity_vector=np.asarray([0.0, 1.0], dtype=np.float64),
            final_vector=_final_vector([1.0, 0.0], [0.0, 1.0]),
            top_lineup_summary="",
            distinct_lineup_count=6,
            top_lineup_share=0.25,
            lineup_entropy=0.82,
            effective_lineups=5.2,
        ),
        EmbeddingRow(
            team_id=3,
            tournament_id=1,
            team_name="Identity Match",
            event_time_ms=1_700_000_000_000,
            lineup_count=18,
            semantic_vector=np.asarray([0.4, 0.916515], dtype=np.float64),
            identity_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            final_vector=_final_vector([0.4, 0.916515], [1.0, 0.0]),
            top_lineup_summary="",
            distinct_lineup_count=2,
            top_lineup_share=0.78,
            lineup_entropy=0.18,
            effective_lineups=1.5,
        ),
    ]

    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[1],
        cluster_map={},
        top_n=3,
    )
    ids = [r["team_id"] for r in out["results"]]
    assert ids == [1, 2, 3]
    assert out["query"]["query_semantic_weight"] > out["query"]["query_identity_weight"]


def test_rank_similar_teams_prefers_identity_match_for_stable_query():
    query_sem = [1.0, 0.0]
    query_id = [1.0, 0.0]
    rows = [
        EmbeddingRow(
            team_id=1,
            tournament_id=1,
            team_name="Stable Query",
            event_time_ms=1_700_000_000_000,
            lineup_count=40,
            semantic_vector=np.asarray(query_sem, dtype=np.float64),
            identity_vector=np.asarray(query_id, dtype=np.float64),
            final_vector=_final_vector(query_sem, query_id),
            top_lineup_summary="",
            distinct_lineup_count=1,
            top_lineup_share=0.97,
            lineup_entropy=0.02,
            effective_lineups=1.0,
        ),
        EmbeddingRow(
            team_id=2,
            tournament_id=1,
            team_name="Semantic Match",
            event_time_ms=1_700_000_000_000,
            lineup_count=18,
            semantic_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            identity_vector=np.asarray([0.0, 1.0], dtype=np.float64),
            final_vector=_final_vector([1.0, 0.0], [0.0, 1.0]),
            top_lineup_summary="",
            distinct_lineup_count=6,
            top_lineup_share=0.25,
            lineup_entropy=0.82,
            effective_lineups=5.2,
        ),
        EmbeddingRow(
            team_id=3,
            tournament_id=1,
            team_name="Identity Match",
            event_time_ms=1_700_000_000_000,
            lineup_count=18,
            semantic_vector=np.asarray([0.4, 0.916515], dtype=np.float64),
            identity_vector=np.asarray([1.0, 0.0], dtype=np.float64),
            final_vector=_final_vector([0.4, 0.916515], [1.0, 0.0]),
            top_lineup_summary="",
            distinct_lineup_count=2,
            top_lineup_share=0.78,
            lineup_entropy=0.18,
            effective_lineups=1.5,
        ),
    ]

    out = rank_similar_teams(
        embeddings=rows,
        target_team_ids=[1],
        cluster_map={},
        top_n=3,
    )
    ids = [r["team_id"] for r in out["results"]]
    assert ids == [1, 3, 2]
    assert out["query"]["query_identity_weight"] > out["query"]["query_semantic_weight"]
