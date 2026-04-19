from __future__ import annotations

from team_api.store import (
    _extract_sendou_teams_from_turbo_payload,
    _normalize_query,
    _tournament_tier,
)


def test_normalize_query_strips_non_alphanumerics():
    assert _normalize_query("FTW") == "ftw"
    assert _normalize_query(" Moonlight ") == "moonlight"
    assert _normalize_query("#rock the-hill") == "rockthehill"
    assert _normalize_query("") == ""
    assert _normalize_query("  ") == ""


def test_tournament_tier_uses_rank_cutoffs():
    assert _tournament_tier(5.0)["tier_id"] == "x"
    assert _tournament_tier(5.1)["tier_id"] == "s_plus"
    assert _tournament_tier(10.0)["tier_id"] == "s_plus"
    assert _tournament_tier(10.1)["tier_id"] == "s"
    assert _tournament_tier(20.0)["tier_id"] == "s"
    assert _tournament_tier(20.1)["tier_id"] == "a_plus"
    assert _tournament_tier(40.0)["tier_id"] == "a_plus"
    assert _tournament_tier(40.1)["tier_id"] == "a"
    assert _tournament_tier(80.0)["tier_id"] == "a"
    assert _tournament_tier(80.1)["tier_id"] == "a_minus"
    assert _tournament_tier(160.0)["tier_id"] == "a_minus"
    assert _tournament_tier(None)["tier_id"] == "unscored"


def test_extract_sendou_teams_from_turbo_payload_reads_route_data():
    payload = [
        {"_1": 2},
        "features/tournament/routes/to.$id",
        {"_3": 4},
        "data",
        "{\"tournament\":{\"ctx\":{\"teams\":[{\"id\":31921,\"name\":\"Moonlight\"},{\"id\":31922,\"name\":\"FTW\"}]}}}",
    ]

    teams = _extract_sendou_teams_from_turbo_payload(payload)
    assert teams == [
        {
            "team_id": 31921,
            "team_name": "Moonlight",
            "display_name": "Moonlight",
            "member_user_ids": [],
            "member_names": [],
        },
        {
            "team_id": 31922,
            "team_name": "FTW",
            "display_name": "FTW",
            "member_user_ids": [],
            "member_names": [],
        },
    ]
