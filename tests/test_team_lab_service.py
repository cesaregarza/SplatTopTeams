from types import SimpleNamespace

from team_api.team_lab_service import resolve_team_lab_scope


def test_resolve_team_lab_scope_returns_single_team_for_non_family_profile():
    rows = [SimpleNamespace(team_id=101, team_name="Healbook")]

    team_ids, team_name = resolve_team_lab_scope(
        snapshot_rows=rows,
        family_members_by_team={},
        search_similar_teams=lambda **kwargs: {"results": []},
        normalize_ids=lambda values: [int(value) for value in values if int(value) > 0],
        snapshot_id=7,
        profile="explore",
        team_id=101,
    )

    assert team_ids == [101]
    assert team_name == "Healbook"


def test_resolve_team_lab_scope_prefers_family_cluster_membership_when_available():
    rows = [
        SimpleNamespace(team_id=101, team_name="WE as in HealBook", lineup_count=4),
        SimpleNamespace(team_id=102, team_name="Healbook", lineup_count=6),
        SimpleNamespace(team_id=103, team_name="Healbook", lineup_count=5),
    ]
    calls = []

    def fake_search_similar_teams(**kwargs):
        calls.append(kwargs)
        return {"results": []}

    team_ids, team_name = resolve_team_lab_scope(
        snapshot_rows=rows,
        family_members_by_team={
            101: frozenset({101, 102, 103}),
            102: frozenset({101, 102, 103}),
            103: frozenset({101, 102, 103}),
            999: frozenset({999}),
        },
        search_similar_teams=fake_search_similar_teams,
        normalize_ids=lambda values: [int(value) for value in values if int(value) > 0],
        snapshot_id=7,
        profile="family",
        team_id=101,
    )

    assert team_ids == [101, 102, 103]
    assert team_name == "Healbook"
    assert calls == []


def test_resolve_team_lab_scope_falls_back_to_family_search_when_cluster_map_missing():
    rows = [SimpleNamespace(team_id=101, team_name="Healbook")]

    def fake_search_similar_teams(**kwargs):
        assert kwargs["snapshot_id"] == 7
        assert kwargs["query"] == "101"
        return {
            "results": [
                {
                    "team_id": 101,
                    "team_name": "Healbook",
                    "consolidated_team_ids": [102, 103],
                }
            ]
        }

    team_ids, team_name = resolve_team_lab_scope(
        snapshot_rows=rows,
        family_members_by_team={},
        search_similar_teams=fake_search_similar_teams,
        normalize_ids=lambda values: [int(value) for value in values if int(value) > 0],
        snapshot_id=7,
        profile="family",
        team_id=101,
    )

    assert team_ids == [101, 102, 103]
    assert team_name == "Healbook"


def test_resolve_team_lab_scope_builds_stable_family_from_search_consensus():
    rows = [
        SimpleNamespace(team_id=101, team_name="Healbook", lineup_count=6),
        SimpleNamespace(team_id=102, team_name="Healbook", lineup_count=5),
        SimpleNamespace(team_id=103, team_name="WE as in HealBook", lineup_count=4),
        SimpleNamespace(team_id=104, team_name="GO!", lineup_count=3),
        SimpleNamespace(team_id=105, team_name="My Pants", lineup_count=2),
    ]

    groups = {
        101: [101, 102, 103, 104],
        102: [101, 102, 103, 105],
        103: [101, 102, 103],
        104: [101, 104],
        105: [102, 105],
    }

    def fake_search_similar_teams(**kwargs):
        seed = int(kwargs["query"])
        group = groups.get(seed, [seed])
        return {
            "results": [
                {
                    "team_id": seed,
                    "team_name": "Healbook",
                    "consolidated_team_ids": [candidate for candidate in group if candidate != seed],
                }
            ]
        }

    first_ids, first_name = resolve_team_lab_scope(
        snapshot_rows=rows,
        family_members_by_team={},
        search_similar_teams=fake_search_similar_teams,
        normalize_ids=lambda values: [int(value) for value in values if int(value) > 0],
        snapshot_id=7,
        profile="family",
        team_id=101,
    )
    second_ids, second_name = resolve_team_lab_scope(
        snapshot_rows=rows,
        family_members_by_team={},
        search_similar_teams=fake_search_similar_teams,
        normalize_ids=lambda values: [int(value) for value in values if int(value) > 0],
        snapshot_id=7,
        profile="family",
        team_id=102,
    )

    assert first_ids == [101, 102, 103]
    assert second_ids == [101, 102, 103]
    assert first_name == "Healbook"
    assert second_name == "Healbook"
