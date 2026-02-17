from __future__ import annotations

import numpy as np

from team_api.search_logic import consolidate_ranked_results, rank_similar_teams
from team_api.store import EmbeddingRow


def _row(team_id: int, name: str, vec: list[float]) -> EmbeddingRow:
    arr = np.asarray(vec, dtype=np.float64)
    return EmbeddingRow(
        team_id=team_id,
        tournament_id=1,
        team_name=name,
        event_time_ms=1,
        lineup_count=10,
        semantic_vector=arr,
        identity_vector=arr,
        final_vector=arr,
        top_lineup_summary="",
    )


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
    assert rep["match_count"] == 20
    core_players = rep["core_lineup_players"]
    assert [row["player_id"] for row in core_players] == [1, 2, 3, 4, 5, 6]
    assert [row["matches_played"] for row in core_players] == [9, 9, 9, 9, 6, 3]
