from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

from team_api.entity_resolution_reference import build_reference_store
from team_api.family_clustering import (
    CLUSTERING_APPROACHES,
    cluster_members_by_team,
)
from team_api.healbook_reference import (
    HEALBOOK_QUERY_CASES,
    build_healbook_reference_store,
)
from team_api.real_family_reference import (
    REAL_FAMILY_QUERY_CASES,
    build_real_family_reference_store,
)
from team_api.store import EmbeddingRow


@dataclass(frozen=True)
class ClusterExpectation:
    name: str
    top_team_id: int
    required_ids: frozenset[int]
    forbidden_ids: frozenset[int]
    expect_clustered: bool


@dataclass(frozen=True)
class CaseEvaluation:
    case: ClusterExpectation
    component: frozenset[int]
    required_ok: bool
    purity_ok: bool
    clustered_ok: bool

    @property
    def passed(self) -> bool:
        return self.required_ok and self.purity_ok and self.clustered_ok


def _reference_rows() -> list[EmbeddingRow]:
    _, generic_cache = build_reference_store()
    _, healbook_cache = build_healbook_reference_store()
    _, real_family_cache = build_real_family_reference_store()

    rows = list(generic_cache.rows) + list(healbook_cache.rows) + list(real_family_cache.rows)
    team_ids = [int(row.team_id) for row in rows]
    if len(team_ids) != len(set(team_ids)):
        raise ValueError("reference rows contain duplicate team ids")
    return rows


def _reference_cases() -> tuple[ClusterExpectation, ...]:
    cases: list[ClusterExpectation] = [
        ClusterExpectation(
            name="synthetic exact alias",
            top_team_id=101,
            required_ids=frozenset({101, 102}),
            forbidden_ids=frozenset({601, 602}),
            expect_clustered=True,
        ),
        ClusterExpectation(
            name="synthetic historical reunion",
            top_team_id=501,
            required_ids=frozenset({501, 502}),
            forbidden_ids=frozenset({601, 602}),
            expect_clustered=True,
        ),
        ClusterExpectation(
            name="synthetic no bridge",
            top_team_id=601,
            required_ids=frozenset({601}),
            forbidden_ids=frozenset({602}),
            expect_clustered=False,
        ),
    ]

    cases.extend(
        ClusterExpectation(
            name=f"healbook: {case.name}",
            top_team_id=int(case.top_team_id),
            required_ids=frozenset(int(team_id) for team_id in case.required_top_group_ids),
            forbidden_ids=frozenset(int(team_id) for team_id in case.forbidden_top_group_ids),
            expect_clustered=bool(case.expect_consolidated),
        )
        for case in HEALBOOK_QUERY_CASES
    )
    cases.extend(
        ClusterExpectation(
            name=f"{case.family}: {case.name}",
            top_team_id=int(case.top_team_id),
            required_ids=frozenset(int(team_id) for team_id in case.required_top_group_ids),
            forbidden_ids=frozenset(int(team_id) for team_id in case.forbidden_top_group_ids),
            expect_clustered=bool(case.expect_consolidated),
        )
        for case in REAL_FAMILY_QUERY_CASES
    )
    return tuple(cases)


def _evaluate_cases(
    rows: Sequence[EmbeddingRow],
    cases: Sequence[ClusterExpectation],
    *,
    approach_name: str,
) -> tuple[dict[str, int], list[CaseEvaluation], dict[int, frozenset[int]]]:
    cluster_fn = CLUSTERING_APPROACHES[approach_name]
    cluster_map = cluster_fn(rows)
    components = cluster_members_by_team(rows, cluster_map)

    evaluations: list[CaseEvaluation] = []
    for case in cases:
        component = components.get(int(case.top_team_id), frozenset({int(case.top_team_id)}))
        required_ok = case.required_ids <= component
        purity_ok = component.isdisjoint(case.forbidden_ids)
        clustered_ok = (len(component) > 1) is bool(case.expect_clustered)
        evaluations.append(
            CaseEvaluation(
                case=case,
                component=component,
                required_ok=required_ok,
                purity_ok=purity_ok,
                clustered_ok=clustered_ok,
            )
        )

    cluster_count = len(
        {
            int(meta["cluster_id"])
            for meta in cluster_map.values()
        }
    )
    summary = {
        "cases": len(evaluations),
        "passed": sum(1 for evaluation in evaluations if evaluation.passed),
        "required_ok": sum(1 for evaluation in evaluations if evaluation.required_ok),
        "purity_ok": sum(1 for evaluation in evaluations if evaluation.purity_ok),
        "clustered_ok": sum(1 for evaluation in evaluations if evaluation.clustered_ok),
        "clustered_teams": len(cluster_map),
        "cluster_count": cluster_count,
    }
    return summary, evaluations, components


def _format_members(team_ids: Iterable[int]) -> str:
    return ", ".join(str(team_id) for team_id in sorted(team_ids))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare candidate team-family clustering approaches against reference cases."
    )
    parser.add_argument(
        "--approach",
        action="append",
        choices=sorted(CLUSTERING_APPROACHES),
        help="Only run the named approach. Repeat to run more than one.",
    )
    args = parser.parse_args()

    rows = _reference_rows()
    cases = _reference_cases()
    approach_names = args.approach or sorted(CLUSTERING_APPROACHES)

    print("| approach | pass | required | purity | clustered | clusters | clustered teams |")
    print("| --- | --- | --- | --- | --- | --- | --- |")

    full_results: list[tuple[str, dict[str, int], list[CaseEvaluation], dict[int, frozenset[int]]]] = []
    for approach_name in approach_names:
        summary, evaluations, components = _evaluate_cases(
            rows,
            cases,
            approach_name=approach_name,
        )
        full_results.append((approach_name, summary, evaluations, components))
        print(
            "| {approach} | {passed}/{cases} | {required_ok}/{cases} | "
            "{purity_ok}/{cases} | {clustered_ok}/{cases} | {cluster_count} | "
            "{clustered_teams} |".format(
                approach=approach_name,
                **summary,
            )
        )

    print()
    for approach_name, summary, evaluations, components in full_results:
        print(f"## {approach_name}")
        print(
            f"- Passes: {summary['passed']}/{summary['cases']}"
        )
        print(
            "- Representative components: "
            f"Healbook core 46624 -> [{_format_members(components.get(46624, frozenset({46624})))}]; "
            f"Healbook devade 47527 -> [{_format_members(components.get(47527, frozenset({47527})))}]; "
            f"Seapunks eider 24043 -> [{_format_members(components.get(24043, frozenset({24043})))}]"
        )
        failures = [evaluation for evaluation in evaluations if not evaluation.passed]
        if not failures:
            print("- Failures: none")
            print()
            continue
        print("- Failures:")
        for evaluation in failures:
            reasons: list[str] = []
            if not evaluation.required_ok:
                missing = sorted(evaluation.case.required_ids - evaluation.component)
                reasons.append(f"missing required {missing}")
            if not evaluation.purity_ok:
                forbidden = sorted(evaluation.component & evaluation.case.forbidden_ids)
                reasons.append(f"included forbidden {forbidden}")
            if not evaluation.clustered_ok:
                reasons.append(
                    f"expected clustered={evaluation.case.expect_clustered} got size={len(evaluation.component)}"
                )
            print(
                f"  - {evaluation.case.name}: "
                f"[{_format_members(evaluation.component)}] ({'; '.join(reasons)})"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
