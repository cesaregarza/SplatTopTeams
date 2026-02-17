from __future__ import annotations

from team_api.store import _normalize_query, _tournament_tier


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
