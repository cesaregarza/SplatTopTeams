from __future__ import annotations

import pytest

from team_api.entity_resolution_reference import (
    REFERENCE_NO_BRIDGE_CASE,
    REFERENCE_QUERY_CASES,
    build_reference_store,
    run_reference_query,
    top_hit_rank,
)


@pytest.fixture(scope="module")
def reference_context():
    return build_reference_store()


@pytest.mark.parametrize(
    "case",
    REFERENCE_QUERY_CASES,
    ids=lambda case: case.name,
)
def test_reference_queries_pull_expected_entities(reference_context, case):
    store, cache_entry = reference_context
    results = run_reference_query(store, cache_entry, case.player_ids)

    assert results
    assert top_hit_rank(results, case.expected_ids) == 1


@pytest.mark.parametrize(
    "case",
    [case for case in REFERENCE_QUERY_CASES if case.expected_consolidated_ids is not None],
    ids=lambda case: case.name,
)
def test_reference_queries_pull_expected_consolidated_groups(reference_context, case):
    store, cache_entry = reference_context
    results = run_reference_query(store, cache_entry, case.player_ids)

    assert results
    top = results[0]
    assert set(top.get("consolidated_team_ids") or []) == set(case.expected_consolidated_ids or [])
    assert int(top.get("consolidated_team_count") or 0) == len(case.expected_consolidated_ids or ())


def test_large_fifteen_reference_stays_unconsolidated(reference_context):
    store, cache_entry = reference_context
    case = next(case for case in REFERENCE_QUERY_CASES if case.name == "Large 15-player roster")

    results = run_reference_query(store, cache_entry, case.player_ids)

    assert results
    assert top_hit_rank(results, case.expected_ids) == 1
    assert results[0].get("consolidated_team_ids") in (None, [])
    assert not results[0].get("is_consolidated", False)


def test_no_bridge_reference_does_not_merge_entities(reference_context):
    store, cache_entry = reference_context
    results = run_reference_query(store, cache_entry, REFERENCE_NO_BRIDGE_CASE.player_ids)

    assert results
    assert top_hit_rank(results, REFERENCE_NO_BRIDGE_CASE.expected_ids) == 1
    assert all(
        set(result.get("consolidated_team_ids") or []) != {601, 602}
        for result in results
    )
    assert int(results[0]["team_id"]) == 601
