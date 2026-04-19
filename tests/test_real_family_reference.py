from __future__ import annotations

import pytest

from team_api.real_family_reference import (
    REAL_FAMILY_QUERY_CASES,
    build_real_family_reference_store,
    run_real_family_reference_query,
)


@pytest.fixture(scope="module")
def real_family_context():
    return build_real_family_reference_store()


@pytest.mark.parametrize(
    "case",
    REAL_FAMILY_QUERY_CASES,
    ids=lambda case: case.name,
)
def test_real_family_queries_pull_expected_top_entity(real_family_context, case):
    store, cache_entry = real_family_context
    results = run_real_family_reference_query(store, cache_entry, case.player_ids)

    assert results
    assert int(results[0]["team_id"]) == case.top_team_id


@pytest.mark.parametrize(
    "case",
    REAL_FAMILY_QUERY_CASES,
    ids=lambda case: case.name,
)
def test_real_family_queries_respect_family_boundaries(real_family_context, case):
    store, cache_entry = real_family_context
    results = run_real_family_reference_query(store, cache_entry, case.player_ids)

    assert results
    top_group_ids = set(results[0].get("consolidated_team_ids") or [results[0]["team_id"]])
    assert case.required_top_group_ids <= top_group_ids
    assert top_group_ids.isdisjoint(case.forbidden_top_group_ids)
    assert bool(results[0].get("is_consolidated", False)) is case.expect_consolidated


def test_seapunks_eider_branch_stays_separate(real_family_context):
    store, cache_entry = real_family_context
    results = run_real_family_reference_query(
        store,
        cache_entry,
        (9163, 10841, 12280, 14605),
    )

    assert results
    top_group_ids = set(results[0].get("consolidated_team_ids") or [results[0]["team_id"]])
    assert top_group_ids == {24043}
