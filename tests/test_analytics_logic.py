from __future__ import annotations

import numpy as np

from team_api.analytics_logic import (
    build_blended_neighbors,
    build_team_lab,
    compute_outliers,
    compute_overview,
    compute_roster_diversity_candidates,
    compute_snapshot_drift,
    compute_space_projection,
    summarize_matchups,
)
from team_api.store import EmbeddingRow


def _row(
    team_id: int,
    name: str,
    vec: list[float],
    *,
    top_player_ids: tuple[int, ...] = (),
    top_player_names: tuple[str, ...] = (),
    lineup_count: int,
    distinct: int,
    top_share: float,
    entropy: float,
) -> EmbeddingRow:
    arr = np.asarray(vec, dtype=np.float64)
    return EmbeddingRow(
        team_id=team_id,
        tournament_id=1,
        team_name=name,
        event_time_ms=1,
        lineup_count=lineup_count,
        semantic_vector=arr,
        identity_vector=arr,
        final_vector=arr,
        top_lineup_summary="",
        top_lineup_player_ids=top_player_ids,
        top_lineup_player_names=top_player_names,
        unique_player_count=4,
        distinct_lineup_count=distinct,
        top_lineup_share=top_share,
        lineup_entropy=entropy,
        effective_lineups=1.0,
    )


def test_compute_overview_has_expected_summary_and_cluster_health():
    rows = [
        _row(1, "A", [1.0, 0.0], lineup_count=20, distinct=5, top_share=0.4, entropy=0.8),
        _row(2, "B", [0.95, 0.05], lineup_count=18, distinct=4, top_share=0.45, entropy=0.7),
        _row(3, "C", [0.0, 1.0], lineup_count=8, distinct=2, top_share=0.7, entropy=0.3),
    ]
    cluster_map = {
        1: {"cluster_id": 10, "cluster_size": 2},
        2: {"cluster_id": 10, "cluster_size": 2},
    }

    out = compute_overview(
        rows,
        cluster_map,
        limit_clusters=10,
        volatile_limit=10,
    )
    assert out["summary"]["teams_indexed"] == 3
    assert out["summary"]["cluster_count"] == 1
    assert out["summary"]["noise_teams"] == 1
    assert out["cluster_health"][0]["cluster_id"] == 10


def test_summarize_matchups_canonicalizes_pairs_and_counts_wins():
    rows = [
        {
            "cluster_1": 1,
            "cluster_2": 2,
            "team1_id": 11,
            "team2_id": 22,
            "winner_team_id": 11,
        },
        {
            "cluster_1": 2,
            "cluster_2": 1,
            "team1_id": 22,
            "team2_id": 11,
            "winner_team_id": 22,
        },
        {
            "cluster_1": 1,
            "cluster_2": 2,
            "team1_id": 11,
            "team2_id": 22,
            "winner_team_id": 22,
        },
    ]

    out = summarize_matchups(
        rows,
        cluster_names={1: "Alpha", 2: "Beta"},
        min_matches=1,
        limit=10,
    )
    assert len(out) == 1
    assert out[0]["cluster_a"] == 1
    assert out[0]["cluster_b"] == 2
    assert out[0]["matches"] == 3
    assert out[0]["wins_a"] == 1
    assert out[0]["wins_b"] == 2


def test_build_team_lab_returns_neighbors_and_match_summary():
    rows = [
        _row(1, "A", [1.0, 0.0], lineup_count=12, distinct=3, top_share=0.5, entropy=0.5),
        _row(2, "B", [0.95, 0.05], lineup_count=11, distinct=3, top_share=0.5, entropy=0.4),
        _row(3, "C", [0.2, 0.8], lineup_count=10, distinct=2, top_share=0.7, entropy=0.2),
    ]
    cluster_map = {
        1: {"cluster_id": 8, "cluster_size": 2},
        2: {"cluster_id": 8, "cluster_size": 2},
        3: {"cluster_id": 4, "cluster_size": 1},
    }
    matches = [
        {"opponent_team_id": 2, "is_win": 1},
        {"opponent_team_id": 3, "is_win": 0},
    ]

    out = build_team_lab(
        team_id=1,
        embeddings=rows,
        cluster_map=cluster_map,
        matches=matches,
        neighbors=5,
    )
    assert out is not None
    assert out["team"]["team_id"] == 1
    assert out["match_summary"]["matches"] == 2
    assert out["neighbors"][0]["team_id"] == 2


def test_compute_space_projection_returns_points_and_centroids():
    rows = [
        _row(1, "A", [1.0, 0.0], lineup_count=20, distinct=5, top_share=0.4, entropy=0.8),
        _row(2, "B", [0.95, 0.1], lineup_count=18, distinct=4, top_share=0.45, entropy=0.7),
        _row(3, "C", [0.1, 0.95], lineup_count=10, distinct=2, top_share=0.7, entropy=0.2),
    ]
    cluster_map = {
        1: {"cluster_id": 1, "cluster_size": 2},
        2: {"cluster_id": 1, "cluster_size": 2},
        3: {"cluster_id": 2, "cluster_size": 1},
    }
    out = compute_space_projection(rows, cluster_map, max_points=100)
    assert len(out["points"]) == 3
    assert len(out["centroids"]) >= 1


def test_compute_roster_diversity_candidates_finds_low_overlap_high_sim_teams():
    rows = [
        _row(
            1,
            "Alpha",
            [1.0, 0.0],
            top_player_ids=(1, 2, 3, 4),
            top_player_names=("A1", "A2", "A3", "A4"),
            lineup_count=10,
            distinct=4,
            top_share=0.5,
            entropy=0.5,
        ),
        _row(
            2,
            "Bravo",
            [0.99, 0.1],
            top_player_ids=(1, 2, 5, 6),
            top_player_names=("A1", "A2", "B1", "B2"),
            lineup_count=11,
            distinct=4,
            top_share=0.5,
            entropy=0.4,
        ),
        _row(
            3,
            "Charlie",
            [0.0, 1.0],
            top_player_ids=(7, 8, 9, 10),
            top_player_names=("C1", "C2", "C3", "C4"),
            lineup_count=12,
            distinct=4,
            top_share=0.6,
            entropy=0.3,
        ),
    ]
    cluster_map = {
        1: {"cluster_id": 3, "cluster_size": 3},
        2: {"cluster_id": 3, "cluster_size": 3},
        3: {"cluster_id": 3, "cluster_size": 3},
    }
    result = compute_roster_diversity_candidates(
        rows,
        cluster_map,
        min_similarity=0.8,
        max_player_overlap=0.4,
        min_cluster_size=2,
        limit=10,
    )
    assert result["total_pairs_found"] == 1
    assert len(result["cohorts"]) == 1
    assert result["cohorts"][0]["team_count"] == 2
    assert result["cohorts"][0]["roster_pool_size"] == 6


def test_compute_outliers_returns_ranked_rows():
    rows = [
        _row(1, "A", [1.0, 0.0], lineup_count=20, distinct=5, top_share=0.4, entropy=0.8),
        _row(2, "B", [0.98, 0.05], lineup_count=18, distinct=4, top_share=0.45, entropy=0.7),
        _row(3, "C", [0.5, 0.4], lineup_count=12, distinct=5, top_share=0.3, entropy=0.9),
    ]
    cluster_map = {
        1: {"cluster_id": 7, "cluster_size": 3},
        2: {"cluster_id": 7, "cluster_size": 3},
        3: {"cluster_id": 7, "cluster_size": 3},
    }
    out = compute_outliers(rows, cluster_map, limit=5)
    assert len(out) == 3
    assert out[0]["outlier_score"] >= out[1]["outlier_score"]


def test_compute_snapshot_drift_tracks_changes():
    prev_rows = [
        _row(1, "A", [1.0, 0.0], lineup_count=12, distinct=3, top_share=0.5, entropy=0.5),
        _row(2, "B", [0.95, 0.1], lineup_count=11, distinct=2, top_share=0.7, entropy=0.2),
    ]
    curr_rows = [
        _row(1, "A", [0.8, 0.2], lineup_count=15, distinct=4, top_share=0.4, entropy=0.7),
        _row(2, "B", [0.95, 0.1], lineup_count=8, distinct=2, top_share=0.8, entropy=0.1),
        _row(3, "C", [0.1, 1.0], lineup_count=9, distinct=3, top_share=0.5, entropy=0.4),
    ]
    prev_clusters = {
        1: {"cluster_id": 1, "cluster_size": 2},
        2: {"cluster_id": 1, "cluster_size": 2},
    }
    curr_clusters = {
        1: {"cluster_id": 2, "cluster_size": 1},
        2: {"cluster_id": 1, "cluster_size": 2},
        3: {"cluster_id": 1, "cluster_size": 2},
    }
    out = compute_snapshot_drift(
        current_snapshot_id=20,
        previous_snapshot_id=19,
        current_embeddings=curr_rows,
        previous_embeddings=prev_rows,
        current_cluster_map=curr_clusters,
        previous_cluster_map=prev_clusters,
        top_movers=10,
    )
    assert out["summary"]["shared_teams"] == 2
    assert out["summary"]["new_teams"] == 1
    assert out["summary"]["cluster_switches"] >= 1
    assert len(out["top_embedding_movers"]) == 2


def test_build_blended_neighbors_respects_target():
    rows = [
        _row(1, "A", [1.0, 0.0], lineup_count=10, distinct=3, top_share=0.5, entropy=0.4),
        _row(2, "B", [0.9, 0.1], lineup_count=9, distinct=3, top_share=0.5, entropy=0.4),
        _row(3, "C", [0.0, 1.0], lineup_count=8, distinct=2, top_share=0.7, entropy=0.2),
    ]
    cluster_map = {
        1: {"cluster_id": 1, "cluster_size": 2},
        2: {"cluster_id": 1, "cluster_size": 2},
        3: {"cluster_id": 2, "cluster_size": 1},
    }
    out = build_blended_neighbors(
        team_id=1,
        embeddings=rows,
        cluster_map=cluster_map,
        semantic_weight=0.7,
        neighbors=5,
    )
    assert out is not None
    assert out["team_id"] == 1
    assert out["neighbors"][0]["team_id"] == 2
