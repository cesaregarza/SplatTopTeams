from __future__ import annotations

from team_api.healbook_reference import (
    HEALBOOK_QUERY_CASES,
    build_healbook_reference_store,
    run_healbook_reference_query,
)
from team_api.real_family_reference import (
    REAL_FAMILY_QUERY_CASES,
    build_real_family_reference_store,
    run_real_family_reference_query,
)


def _format_group(result: dict[str, object]) -> str:
    group = result.get("consolidated_team_ids") or [result.get("team_id")]
    return ",".join(str(team_id) for team_id in group if team_id is not None)


def main() -> int:
    lines: list[str] = []
    lines.append("# Real Family Reference Evaluation")
    lines.append("")
    lines.append("## Healbook")
    lines.append("")
    lines.append("| Query | Top Team | Group | Consolidated | Score |")
    lines.append("| --- | ---: | --- | --- | ---: |")
    healbook_store, healbook_cache = build_healbook_reference_store()
    for case in HEALBOOK_QUERY_CASES:
        results = run_healbook_reference_query(
            healbook_store,
            healbook_cache,
            case.player_ids,
        )
        top = results[0]
        lines.append(
            f"| {case.name} | {top['team_id']} | {_format_group(top)} | "
            f"{'yes' if top.get('is_consolidated', False) else 'no'} | "
            f"{float(top.get('sim_to_query') or 0.0):.3f} |"
        )

    lines.append("")
    lines.append("## Combined Families")
    lines.append("")
    lines.append("| Query | Family | Top Team | Group | Consolidated | Score |")
    lines.append("| --- | --- | ---: | --- | --- | ---: |")
    store, cache = build_real_family_reference_store()
    for case in REAL_FAMILY_QUERY_CASES:
        results = run_real_family_reference_query(store, cache, case.player_ids)
        top = results[0]
        lines.append(
            f"| {case.name} | {case.family} | {top['team_id']} | {_format_group(top)} | "
            f"{'yes' if top.get('is_consolidated', False) else 'no'} | "
            f"{float(top.get('sim_to_query') or 0.0):.3f} |"
        )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
