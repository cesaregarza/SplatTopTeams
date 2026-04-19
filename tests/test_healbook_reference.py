from __future__ import annotations

import pytest

from team_api.healbook_reference import (
    HEALBOOK_QUERY_CASES,
    build_healbook_reference_store,
    run_healbook_reference_query,
)


@pytest.fixture(scope="module")
def healbook_context():
    return build_healbook_reference_store()


@pytest.mark.parametrize(
    "case",
    HEALBOOK_QUERY_CASES,
    ids=lambda case: case.name,
)
def test_healbook_reference_queries_pull_expected_top_entity(healbook_context, case):
    store, cache_entry = healbook_context
    results = run_healbook_reference_query(store, cache_entry, case.player_ids)

    assert results
    assert int(results[0]["team_id"]) == case.top_team_id


@pytest.mark.parametrize(
    "case",
    HEALBOOK_QUERY_CASES,
    ids=lambda case: case.name,
)
def test_healbook_reference_queries_respect_group_boundaries(healbook_context, case):
    store, cache_entry = healbook_context
    results = run_healbook_reference_query(store, cache_entry, case.player_ids)

    assert results
    top_group_ids = set(results[0].get("consolidated_team_ids") or [results[0]["team_id"]])
    assert case.required_top_group_ids <= top_group_ids
    assert top_group_ids.isdisjoint(case.forbidden_top_group_ids)
    assert bool(results[0].get("is_consolidated", False)) is case.expect_consolidated


def test_healbook_devade_branch_stays_separate_from_core_family(healbook_context):
    store, cache_entry = healbook_context
    results = run_healbook_reference_query(
        store,
        cache_entry,
        (1954, 7898, 10403, 27966),
    )

    assert results
    top_group_ids = set(results[0].get("consolidated_team_ids") or [results[0]["team_id"]])
    assert top_group_ids == {47527}
    assert {31888, 46624, 49106, 54634, 54749}.isdisjoint(top_group_ids)
