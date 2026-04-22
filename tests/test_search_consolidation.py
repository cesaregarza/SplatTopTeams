from team_api.search_consolidation import consolidate_ranked_results


def test_consolidate_ranked_results_updates_lineup_count_for_family_groups():
    rows = [
        {
            "team_id": 101,
            "team_name": "Healbook",
            "lineup_count": 6,
            "match_count": 6,
            "tournament_count": 2,
            "distinct_lineup_count": 2,
            "top_lineup_share": 4 / 6,
            "top_lineup_player_ids": [1, 2, 3, 4],
            "top_lineup_player_names": ["A", "B", "C", "D"],
            "top_lineup_summary": "4x:A,B,C,D",
        },
        {
            "team_id": 102,
            "team_name": "WE as in HealBook",
            "lineup_count": 4,
            "match_count": 4,
            "tournament_count": 1,
            "distinct_lineup_count": 2,
            "top_lineup_share": 0.75,
            "top_lineup_player_ids": [1, 2, 3, 4],
            "top_lineup_player_names": ["A", "B", "C", "D"],
            "top_lineup_summary": "3x:A,B,C,D",
        },
    ]

    out = consolidate_ranked_results(rows, min_overlap=0.8)

    assert len(out) == 1
    result = out[0]
    assert result["lineup_count"] == 10
    assert result["match_count"] == 10
    assert result["top_lineup_match_count"] == 7
    assert result["top_lineup_match_count"] <= result["lineup_count"]
    assert result["consolidated_team_ids"] == [101, 102]

