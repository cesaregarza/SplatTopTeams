from __future__ import annotations

from typing import List

import numpy as np
from sqlalchemy import create_engine

from team_api.store import EmbeddingRow, TeamSearchStore


def _row(
    team_id: int,
    lineup_count: int = 10,
    *,
    roster_player_ids=(),
    player_support=None,
    pair_support=None,
) -> EmbeddingRow:
    semantic = np.array([0.3, 0.4], dtype=np.float64)
    identity = np.array([0.5, 0.6], dtype=np.float64)
    final = np.array([0.3, 0.4, 1.5, 1.8], dtype=np.float64)
    roster_ids = tuple(int(player_id) for player_id in roster_player_ids)
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
        unique_player_count=len(roster_ids),
        roster_player_ids=roster_ids,
        player_support=dict(player_support or {}),
        pair_support=dict(pair_support or {}),
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
    assert entry_one_first.finals.shape == (2, 4)

    store._cluster_map_cache[(1, "explore")] = {11: {"cluster_id": 0, "cluster_size": 2}}
    store._tournament_count_cache[(1, 11)] = 3
    store._tournament_count_cache[(2, 21)] = 5

    entry_two = store._get_cached_snapshot_entry(2)
    assert calls == [1, 2]
    assert entry_two.snapshot_id == 2
    assert list(store._cluster_map_cache.keys()) == []
    assert list(store._tournament_count_cache.keys()) == [(2, 21)]


def test_latest_snapshot_uses_ttl_cache():
    store = _store()
    calls: List[int] = []
    rows = [
        {"run_id": 7, "teams_indexed": 100},
        {"run_id": 8, "teams_indexed": 125},
    ]
    times = iter([100.0, 105.0, 121.0])

    store._monotonic_now = lambda: next(times)  # type: ignore[assignment]

    def fake_fetch_first_row(sql, params):
        calls.append(1)
        return dict(rows[len(calls) - 1])

    store._fetch_first_row = fake_fetch_first_row  # type: ignore[assignment]

    first = store.latest_snapshot()
    second = store.latest_snapshot()
    third = store.latest_snapshot()

    assert first == {"run_id": 7, "teams_indexed": 100}
    assert second == {"run_id": 7, "teams_indexed": 100}
    assert third == {"run_id": 8, "teams_indexed": 125}
    assert len(calls) == 2


def test_row_to_embedding_parses_lineup_variant_counts():
    store = _store()

    row = store._row_to_embedding(
        {
            "team_id": 1,
            "tournament_id": 2,
            "team_name": "Healbook",
            "event_time_ms": 123,
            "lineup_count": 6,
            "semantic_vector": [0.3, 0.4],
            "identity_vector": [0.5, 0.6],
            "final_vector": [0.3, 0.4, 1.5, 1.8],
            "lineup_variant_counts": {
                "1|2|3|4": 3,
                "1|2|3|5": 2,
            },
        }
    )

    assert row.lineup_variant_counts == {
        (1, 2, 3, 4): 3,
        (1, 2, 3, 5): 2,
    }


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

    store.match_targets = (  # type: ignore[assignment]
        lambda snapshot_id, query, limit=25, tournament_id=None: [101]
    )
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
        recency_weight=0.0,
        query_team_weights=None,
        **kwargs,
    ):
        assert cache_entry is snapshot_cache_entry
        observed_embeddings.append(list(embeddings))
        return {
            "query": {"matched_team_ids": [101], "matched_team_names": ["Team 101"]},
            "results": [
                {"team_id": 202, "sim_to_query": 0.9},
                {"team_id": 203, "sim_to_query": 0.8},
                {"team_id": 204, "sim_to_query": 0.7},
            ],
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


def test_analytics_team_lab_uses_family_scope_resolution():
    store = _store()
    family_a = _row(101, lineup_count=6, roster_player_ids=(1, 2, 3, 4))
    family_b = _row(102, lineup_count=4, roster_player_ids=(1, 2, 3, 5))
    outsider = _row(202, lineup_count=8, roster_player_ids=(7, 8, 9, 10))
    cache_entry = store._build_snapshot_cache_entry(7, [family_a, family_b, outsider])

    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]
    store._resolve_team_lab_scope = lambda snapshot_id, profile, team_id: ([101, 102], "Healbook")  # type: ignore[assignment]
    store._fetch_team_lab_match_rows = lambda team_ids: [  # type: ignore[assignment]
        {"opponent_team_id": 202, "is_win": 1},
        {"opponent_team_id": 202, "is_win": 0},
    ]

    result = store.analytics_team_lab(
        snapshot_id=7,
        profile="family",
        team_id=101,
        neighbors=5,
    )

    assert result is not None
    assert result["cluster_mode"] == "family"
    assert result["team"]["team_id"] == 101
    assert result["team"]["team_name"] == "Healbook"
    assert result["team"]["team_ids"] == [101, 102]
    assert result["team"]["selected_team_count"] == 2
    assert result["team"]["lineup_count"] == 10
    assert result["match_summary"]["matches"] == 2
    assert result["neighbors"][0]["team_id"] == 202


def test_team_lab_scope_cache_reuses_family_resolution():
    store = _store()
    cache_entry = store._build_snapshot_cache_entry(7, [_row(101)])
    calls: List[int] = []

    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]

    def fake_search_similar_teams(**kwargs):
        calls.append(int(kwargs["team_id"]) if "team_id" in kwargs else 101)
        return {
            "results": [
                {
                    "team_id": 101,
                    "team_name": "Healbook",
                    "consolidated_team_ids": [102, 103],
                }
            ]
        }

    store.search_similar_teams = fake_search_similar_teams  # type: ignore[assignment]

    first = store._resolve_team_lab_scope(snapshot_id=7, profile="family", team_id=101)
    second = store._resolve_team_lab_scope(snapshot_id=7, profile="family", team_id=101)

    assert first == ([101, 102, 103], "Healbook")
    assert second == ([101, 102, 103], "Healbook")
    assert len(calls) == 1


def test_analytics_team_lab_response_cache_avoids_repeat_work():
    store = _store()
    family_a = _row(101, lineup_count=6, roster_player_ids=(1, 2, 3, 4))
    family_b = _row(102, lineup_count=4, roster_player_ids=(1, 2, 3, 5))
    outsider = _row(202, lineup_count=8, roster_player_ids=(7, 8, 9, 10))
    cache_entry = store._build_snapshot_cache_entry(7, [family_a, family_b, outsider])

    scope_calls: List[tuple[int, str, int]] = []
    match_calls: List[tuple[int, ...]] = []

    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]

    def fake_scope(snapshot_id, profile, team_id):
        scope_calls.append((int(snapshot_id), str(profile), int(team_id)))
        return [101, 102], "Healbook"

    def fake_match_rows(team_ids):
        match_calls.append(tuple(int(team_id) for team_id in team_ids))
        return [
            {"opponent_team_id": 202, "is_win": 1},
            {"opponent_team_id": 202, "is_win": 0},
        ]

    store._resolve_team_lab_scope = fake_scope  # type: ignore[assignment]
    store._fetch_team_lab_match_rows = fake_match_rows  # type: ignore[assignment]

    first = store.analytics_team_lab(snapshot_id=7, profile="family", team_id=101, neighbors=5)
    second = store.analytics_team_lab(snapshot_id=7, profile="family", team_id=101, neighbors=5)

    assert first == second
    assert len(scope_calls) == 1
    assert len(match_calls) == 1


def test_analytics_team_matches_response_cache_avoids_repeat_queries():
    store = _store()
    base_calls: List[tuple[int, ...]] = []
    roster_calls: List[int] = []
    round_calls: List[int] = []
    score_calls: List[int] = []
    name_calls: List[tuple[int, ...]] = []

    store._fetch_team_match_base_rows = lambda team_ids, limit: base_calls.append(tuple(int(team_id) for team_id in team_ids)) or [  # type: ignore[assignment]
        {
            "match_id": 77,
            "team1_id": 101,
            "team2_id": 202,
            "tournament_id": 19,
            "winner_team_id": 101,
            "team_score": 3.0,
            "opponent_score": 1.0,
            "tournament_name": "LUTI",
            "tournament_mode": None,
            "map_picking_style": None,
            "tournament_tags": None,
            "event_time_ms": 1_700_000_000_000,
        }
    ]
    store._fetch_match_rosters = lambda rows, team_ids, opponent_ids: roster_calls.append(len(rows)) or {  # type: ignore[assignment]
        (77, 19): {
            "team_a": {"player_ids": [1, 2, 3, 4], "player_names": ["A", "B", "C", "D"]},
            "team_b": {"player_ids": [5, 6, 7, 8], "player_names": ["E", "F", "G", "H"]},
        }
    }
    store._fetch_match_rounds = lambda rows, team_ids, opponent_ids: round_calls.append(len(rows)) or {}  # type: ignore[assignment]
    store._fetch_tournament_scores = lambda tournament_ids: score_calls.append(len(tournament_ids)) or {19: 8.0}  # type: ignore[assignment]
    store._get_cached_snapshot_team_names = lambda snapshot_id, team_ids: name_calls.append(tuple(int(team_id) for team_id in team_ids)) or {101: "Alpha", 202: "Bravo"}  # type: ignore[assignment]

    first = store.analytics_team_matches(snapshot_id=7, team_ids=[101], limit=25)
    second = store.analytics_team_matches(snapshot_id=7, team_ids=[101], limit=25)

    assert first == second
    assert base_calls == [(101,)]
    assert roster_calls == [1]
    assert round_calls == [1]
    assert score_calls == [1]
    assert name_calls == [(101, 202)]


def test_list_tournament_teams_falls_back_to_sendou_when_snapshot_missing():
    store = _store()
    store._fetch_rows = lambda sql, params, missing_default=None: []  # type: ignore[assignment]
    store._get_cached_sendou_tournament_teams = lambda tournament_id: [  # type: ignore[assignment]
        {
            "team_id": 31921,
            "team_name": "Moonlight",
            "display_name": "Moonlight",
            "member_user_ids": [1, 2],
            "member_names": ["A", "B"],
        },
        {
            "team_id": 31922,
            "team_name": "FTW",
            "display_name": "FTW",
            "member_user_ids": [],
            "member_names": [],
        },
    ]

    payload = store.list_tournament_teams(
        snapshot_id=7,
        tournament_id=3192,
        query="moon",
        limit=50,
    )

    assert payload["source"] == "sendou"
    assert payload["tournament_id"] == 3192
    assert len(payload["teams"]) == 1
    assert payload["teams"][0]["team_name"] == "Moonlight"
    assert payload["teams"][0]["display_name"] == "Moonlight"
    assert payload["teams"][0]["member_user_ids"] == [1, 2]
    assert payload["teams"][0]["source"] == "sendou"


def test_tournament_scoped_search_uses_sendou_name_fallback():
    store = _store()
    target_row = _row(101, lineup_count=22)
    candidate_row = _row(202, lineup_count=30)
    cache_entry = store._build_snapshot_cache_entry(7, [target_row, candidate_row])

    def fake_match_targets(snapshot_id, query, limit=25, tournament_id=None):
        if tournament_id is not None:
            return []
        if str(query).lower() == "moonlight":
            return [101]
        return []

    store.match_targets = fake_match_targets  # type: ignore[assignment]
    store._match_sendou_tournament_teams = (  # type: ignore[assignment]
        lambda tournament_id, query, limit: [
            {
                "team_id": 31921,
                "team_name": "Moonlight",
                "display_name": "Moonlight",
                "member_user_ids": [],
                "member_names": [],
            }
        ]
    )
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]
    store._fetch_embeddings_by_team_ids = (  # type: ignore[assignment]
        lambda snapshot_id, team_ids: [target_row] if list(team_ids) else []
    )
    store._query_vector_rank = lambda **kwargs: None  # type: ignore[assignment]
    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]

    result = store.search_similar_teams(
        snapshot_id=7,
        query="moon",
        top_n=20,
        min_relevance=0.8,
        cluster_mode="explore",
        include_clusters=True,
        consolidate=False,
        tournament_id=3192,
    )

    assert result["query"]["tournament_id"] == 3192
    assert result["query"]["tournament_source"] == "sendou_names"
    assert result["query"]["tournament_team_name_matches"] == ["Moonlight"]
    assert result["query"]["tournament_player_id_count"] == 0


def test_tournament_scoped_search_uses_sendou_player_fallback():
    store = _store()
    target_row = _row(101, lineup_count=22)
    candidate_row = _row(202, lineup_count=30)
    cache_entry = store._build_snapshot_cache_entry(7, [target_row, candidate_row])

    store.match_targets = (  # type: ignore[assignment]
        lambda snapshot_id, query, limit=25, tournament_id=None: []
    )
    store._match_sendou_tournament_teams = (  # type: ignore[assignment]
        lambda tournament_id, query, limit: [
            {
                "team_id": 31921,
                "team_name": "",
                "display_name": "[No team name] player-a / player-b",
                "member_user_ids": [998, 997],
                "member_names": ["player-a", "player-b"],
            }
        ]
    )
    store._match_targets_by_player_ids = (  # type: ignore[assignment]
        lambda snapshot_id, player_ids, limit=240: [101]
    )
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]
    store._fetch_embeddings_by_team_ids = (  # type: ignore[assignment]
        lambda snapshot_id, team_ids: [target_row] if list(team_ids) else []
    )
    store._query_vector_rank = lambda **kwargs: None  # type: ignore[assignment]
    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]

    result = store.search_similar_teams(
        snapshot_id=7,
        query="player-a",
        top_n=20,
        min_relevance=0.8,
        cluster_mode="explore",
        include_clusters=True,
        consolidate=False,
        tournament_id=3192,
    )

    assert result["query"]["tournament_id"] == 3192
    assert result["query"]["tournament_source"] == "sendou_players"
    assert result["query"]["tournament_player_id_count"] == 2


def test_seed_player_ids_search_without_tournament_uses_player_fallback():
    store = _store()
    target_row = _row(101, lineup_count=22)

    store.match_targets = (  # type: ignore[assignment]
        lambda snapshot_id, query, limit=25, tournament_id=None: []
    )
    store._match_targets_by_player_ids = (  # type: ignore[assignment]
        lambda snapshot_id, player_ids, limit=240: [101]
    )
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]
    store._fetch_embeddings_by_team_ids = (  # type: ignore[assignment]
        lambda snapshot_id, team_ids: [target_row] if list(team_ids) else []
    )
    store._query_vector_rank = lambda **kwargs: {  # type: ignore[assignment]
        "query": {"matched_team_ids": [101], "matched_team_names": ["Team 101"]},
        "results": [{"team_id": 101, "sim_to_query": 1.0}],
    }

    result = store.search_similar_teams(
        snapshot_id=7,
        query="anything",
        top_n=20,
        min_relevance=0.8,
        cluster_mode="explore",
        include_clusters=True,
        consolidate=False,
        seed_player_ids=[998, 997],
    )

    assert result["query"]["seed_player_id_count"] == 2
    assert result["query"]["matched_team_ids"] == [101]


def test_seed_player_ids_default_to_whole_set_query_profile():
    store = _store()
    exact_row = _row(
        101,
        lineup_count=12,
        roster_player_ids=(1, 2, 3, 4),
        player_support={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
    )
    near_row = _row(
        202,
        lineup_count=20,
        roster_player_ids=(1, 2, 3, 5),
        player_support={1: 1.0, 2: 1.0, 3: 1.0, 5: 1.0},
    )
    weak_row = _row(
        303,
        lineup_count=50,
        roster_player_ids=(1, 8, 9, 10),
        player_support={1: 1.0, 8: 1.0, 9: 1.0, 10: 1.0},
    )

    store.match_targets = (  # type: ignore[assignment]
        lambda snapshot_id, query, limit=25, tournament_id=None: []
    )
    cache_entry = store._build_snapshot_cache_entry(7, [exact_row, near_row, weak_row])
    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]
    store._project_seed_players_to_proxy_teams = (  # type: ignore[assignment]
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subset projection should not run"))
    )

    captured = {}

    def fake_query_vector_rank(**kwargs):
        captured.update(kwargs)
        return {
            "query": {"matched_team_ids": [], "matched_team_names": []},
            "results": [{"team_id": 101, "sim_to_query": 1.0}],
        }

    store._query_vector_rank = fake_query_vector_rank  # type: ignore[assignment]

    result = store.search_similar_teams(
        snapshot_id=7,
        query="anything",
        top_n=20,
        min_relevance=0.8,
        cluster_mode="explore",
        include_clusters=True,
        consolidate=False,
        seed_player_ids=[1, 2, 3, 4, 5],
    )

    assert captured["query_profile"] is not None
    assert captured["query_rows"] is None
    assert result["query"]["query_mode"] == "whole_set"
    assert result["query"]["seed_player_projection_subset_count"] == 0
    assert result["query"]["subset_enumeration_count"] == 0


def test_seed_player_subset_projection_builds_weighted_proxy_query():
    store = _store()
    exact_row = _row(101, lineup_count=18, roster_player_ids=(1, 2, 3, 4))
    alias_row = _row(202, lineup_count=15, roster_player_ids=(1, 2, 3, 5))
    weak_row = _row(303, lineup_count=40, roster_player_ids=(1, 2, 8, 9))
    cache_entry = store._build_snapshot_cache_entry(7, [exact_row, alias_row, weak_row])

    store.match_targets = (  # type: ignore[assignment]
        lambda snapshot_id, query, limit=25, tournament_id=None: []
    )
    store._match_targets_by_player_ids = (  # type: ignore[assignment]
        lambda snapshot_id, player_ids, limit=240: []
    )
    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]
    store._get_cached_cluster_map = lambda snapshot_id, profile: {}  # type: ignore[assignment]

    captured = {}

    def fake_query_vector_rank(**kwargs):
        captured.update(kwargs)
        return {
            "query": {"matched_team_ids": kwargs["target_team_ids"], "matched_team_names": ["Team"]},
            "results": [{"team_id": 101, "sim_to_query": 1.0}],
        }

    store._query_vector_rank = fake_query_vector_rank  # type: ignore[assignment]

    result = store.search_similar_teams(
        snapshot_id=7,
        query="anything",
        top_n=20,
        min_relevance=0.8,
        cluster_mode="explore",
        include_clusters=True,
        consolidate=False,
        seed_player_ids=[1, 2, 3, 4, 5, 6, 7],
        query_mode="subset_enum",
    )

    assert captured["target_team_ids"] == [101, 202]
    assert captured["query_team_weights"][101] >= captured["query_team_weights"][202] > 0.0
    assert result["query"]["seed_player_projection_subset_size"] == 4
    assert result["query"]["seed_player_projection_subset_count"] == 35
    assert result["query"]["seed_player_projection_hit_count"] > 0
    assert result["query"]["seed_player_projection_team_count"] == 2
    assert result["query"]["query_mode"] == "subset_enum"
