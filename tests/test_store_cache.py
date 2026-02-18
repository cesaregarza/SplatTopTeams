from __future__ import annotations

from typing import List

import numpy as np
from sqlalchemy import create_engine

from team_api.store import EmbeddingRow, TeamSearchStore


def _row(team_id: int, lineup_count: int = 10) -> EmbeddingRow:
    semantic = np.array([0.3, 0.4], dtype=np.float64)
    identity = np.array([0.5, 0.6], dtype=np.float64)
    final = np.array([0.7, 0.8], dtype=np.float64)
    return EmbeddingRow(
        team_id=team_id,
        tournament_id=1,
        team_name=f"Team {team_id}",
        event_time_ms=1_700_000_000_000,
        lineup_count=lineup_count,
        semantic_vector=semantic,
        identity_vector=identity,
        final_vector=final,
        tournament_count=1,
    )


def _store() -> TeamSearchStore:
    return TeamSearchStore(create_engine("sqlite:///:memory:"), "comp_rankings")


def test_snapshot_cache_reuses_same_snapshot_and_evicts_previous():
    store = _store()
    calls: List[int] = []

    def fake_load_embeddings(snapshot_id: int):
        calls.append(int(snapshot_id))
        return [_row(snapshot_id * 10 + 1), _row(snapshot_id * 10 + 2)]

    store.load_embeddings = fake_load_embeddings  # type: ignore[assignment]

    entry_one_first = store._get_cached_snapshot_entry(1)
    entry_one_second = store._get_cached_snapshot_entry(1)
    assert calls == [1]
    assert entry_one_first is entry_one_second
    assert entry_one_first.snapshot_id == 1
    assert entry_one_first.finals.shape == (2, 2)

    store._cluster_map_cache[(1, "explore")] = {11: {"cluster_id": 0, "cluster_size": 2}}
    store._tournament_count_cache[(1, 11)] = 3
    store._tournament_count_cache[(2, 21)] = 5

    entry_two = store._get_cached_snapshot_entry(2)
    assert calls == [1, 2]
    assert entry_two.snapshot_id == 2
    assert list(store._cluster_map_cache.keys()) == []
    assert list(store._tournament_count_cache.keys()) == [(2, 21)]


def test_cluster_map_cache_hits_without_requery():
    store = _store()
    calls: List[tuple[int, str, bool]] = []

    def fake_load_cluster_map(snapshot_id: int, profile: str, *, use_cache: bool = True):
        calls.append((int(snapshot_id), str(profile), bool(use_cache)))
        return {42: {"cluster_id": 1, "cluster_size": 3, "representative_team_name": "FTW"}}

    store.load_cluster_map = fake_load_cluster_map  # type: ignore[assignment]

    first = store._get_cached_cluster_map(7, "explore")
    second = store._get_cached_cluster_map(7, "explore")
    assert first == second
    assert calls == [(7, "explore", False)]


def test_hydrate_tournament_counts_uses_memo_cache_before_query():
    store = _store()
    store._tournament_count_cache[(7, 100)] = 9
    requested: List[int] = []

    def fake_fetch_tournament_counts(snapshot_id: int, team_ids):
        requested.extend(sorted(int(team_id) for team_id in team_ids))
        return {200: 4}

    store._fetch_tournament_counts = fake_fetch_tournament_counts  # type: ignore[assignment]

    rows = [
        {"team_id": 100, "tournament_count": 0, "lineup_count": 11},
        {"team_id": 200, "tournament_count": 0, "lineup_count": 7},
    ]
    store._hydrate_tournament_counts(7, rows)

    assert requested == [200]
    assert rows[0]["tournament_count"] == 9
    assert rows[1]["tournament_count"] == 4
    assert store._tournament_count_cache[(7, 200)] == 4


def test_search_fallback_uses_cached_snapshot_rows():
    store = _store()
    target_row = _row(101, lineup_count=15)
    candidate_row = _row(202, lineup_count=20)
    cached_rows = [target_row, candidate_row]
    snapshot_cache_entry = store._build_snapshot_cache_entry(7, cached_rows)

    store.match_targets = lambda snapshot_id, query, limit=25: [101]  # type: ignore[assignment]
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]
    store._fetch_embeddings_by_team_ids = lambda snapshot_id, team_ids: [target_row]  # type: ignore[assignment]
    store._query_vector_rank = lambda **kwargs: None  # type: ignore[assignment]
    store._get_cached_snapshot_entry = lambda snapshot_id: snapshot_cache_entry  # type: ignore[assignment]

    observed_embeddings: List[List[EmbeddingRow]] = []

    def fake_rank_similar_teams(
        embeddings,
        target_team_ids,
        cluster_map,
        top_n,
        min_relevance,
        cache_entry=None,
    ):
        assert cache_entry is snapshot_cache_entry
        observed_embeddings.append(list(embeddings))
        return {
            "query": {"matched_team_ids": [101], "matched_team_names": ["Team 101"]},
            "results": [{"team_id": 202, "sim_to_query": 0.9}],
        }

    store._rank_similar_teams = fake_rank_similar_teams  # type: ignore[assignment]

    result = store.search_similar_teams(
        snapshot_id=7,
        query="FTW",
        top_n=20,
        min_relevance=0.8,
        cluster_mode="explore",
        include_clusters=True,
        consolidate=False,
    )
    assert result["results"][0]["team_id"] == 202
    assert len(observed_embeddings) == 1
    assert observed_embeddings[0][0].team_id == 101
    assert observed_embeddings[0][1].team_id == 202
