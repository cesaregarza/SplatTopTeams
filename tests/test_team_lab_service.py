from types import SimpleNamespace

from team_api.team_lab_service import resolve_team_lab_scope


def test_resolve_team_lab_scope_returns_single_team_for_non_family_profile():
    rows = [SimpleNamespace(team_id=101, team_name="Healbook")]

    team_ids, team_name = resolve_team_lab_scope(
        snapshot_rows=rows,
        search_similar_teams=lambda **kwargs: {"results": []},
        normalize_ids=lambda values: [int(value) for value in values if int(value) > 0],
        snapshot_id=7,
        profile="explore",
        team_id=101,
    )

    assert team_ids == [101]
    assert team_name == "Healbook"


def test_resolve_team_lab_scope_uses_family_search_result_when_available():
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
        search_similar_teams=fake_search_similar_teams,
        normalize_ids=lambda values: [int(value) for value in values if int(value) > 0],
        snapshot_id=7,
        profile="family",
        team_id=101,
    )

    assert team_ids == [101, 102, 103]
    assert team_name == "Healbook"
