from __future__ import annotations

import pytest

from team_api.entity_resolution_reference import build_reference_store
from team_api.family_clustering import (
    cluster_members_by_team,
    cluster_rows_consolidation_graph,
)
from team_api.healbook_reference import (
    HEALBOOK_QUERY_CASES,
    build_healbook_reference_store,
)
from team_api.real_family_reference import (
    REAL_FAMILY_QUERY_CASES,
    build_real_family_reference_store,
)


@pytest.fixture(scope="module")
def family_components():
    _, generic_cache = build_reference_store()
    _, healbook_cache = build_healbook_reference_store()
    _, real_family_cache = build_real_family_reference_store()

    rows = (
        list(generic_cache.rows)
        + list(healbook_cache.rows)
        + list(real_family_cache.rows)
    )
    cluster_map = cluster_rows_consolidation_graph(rows)
    return cluster_members_by_team(rows, cluster_map)


def test_consolidation_graph_clusters_synthetic_exact_alias(family_components):
    assert family_components[101] == frozenset({101, 102})


def test_consolidation_graph_keeps_synthetic_no_bridge_separate(family_components):
    assert family_components[601] == frozenset({601})
    assert family_components[602] == frozenset({602})


@pytest.mark.parametrize(
    "case",
    HEALBOOK_QUERY_CASES,
    ids=lambda case: case.name,
)
def test_consolidation_graph_respects_healbook_reference_boundaries(
    family_components,
    case,
):
    component = family_components[int(case.top_team_id)]
    assert set(case.required_top_group_ids) <= component
    assert component.isdisjoint(case.forbidden_top_group_ids)
    assert (len(component) > 1) is bool(case.expect_consolidated)


@pytest.mark.parametrize(
    "case",
    REAL_FAMILY_QUERY_CASES,
    ids=lambda case: f"{case.family}: {case.name}",
)
def test_consolidation_graph_respects_real_family_reference_boundaries(
    family_components,
    case,
):
    component = family_components[int(case.top_team_id)]
    assert set(case.required_top_group_ids) <= component
    assert component.isdisjoint(case.forbidden_top_group_ids)
    assert (len(component) > 1) is bool(case.expect_consolidated)
