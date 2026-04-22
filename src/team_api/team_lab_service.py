from __future__ import annotations

from typing import Any, Callable, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


def resolve_team_lab_scope(
    *,
    snapshot_rows: Sequence[Any],
    search_similar_teams: Callable[..., dict[str, Any]],
    normalize_ids: Callable[[Any], list[int]],
    snapshot_id: int,
    profile: str,
    team_id: int,
) -> tuple[list[int], str | None]:
    primary_team_id = int(team_id)
    base_row = next(
        (row for row in snapshot_rows if int(row.team_id) == primary_team_id),
        None,
    )

    if str(profile) != "family":
        return [primary_team_id], (base_row.team_name if base_row is not None else None)

    try:
        ranked = search_similar_teams(
            snapshot_id=snapshot_id,
            query=str(primary_team_id),
            top_n=60,
            min_relevance=0.0,
            cluster_mode=profile,
            include_clusters=True,
            consolidate=True,
            consolidate_min_overlap=0.8,
        )
    except Exception:
        ranked = {"results": []}

    for result in ranked.get("results", []):
        resolved_team_ids = normalize_ids(
            [
                result.get("team_id"),
                *(result.get("consolidated_team_ids") or []),
            ]
        )
        if primary_team_id not in resolved_team_ids:
            continue
        return resolved_team_ids, str(
            result.get("team_name")
            or (base_row.team_name if base_row is not None else f"Team {primary_team_id}")
        )

    return [primary_team_id], (base_row.team_name if base_row is not None else None)


def fetch_team_lab_match_rows(
    *,
    engine: Engine,
    schema: str,
    team_ids: Sequence[int],
    normalize_ids: Callable[[Any], list[int]],
    is_missing_relation_error: Callable[[Exception], bool],
) -> list[dict[str, Any]]:
    scoped_team_ids = normalize_ids(team_ids)
    if not scoped_team_ids:
        return []

    exclude_internal_alias_matches = ""
    if len(scoped_team_ids) > 1:
        exclude_internal_alias_matches = """
          AND NOT (m.team1_id IN :team_ids AND m.team2_id IN :team_ids)
        """.rstrip()

    sql = f"""
        SELECT
            CASE
                WHEN m.team1_id IN :team_ids THEN m.team2_id
                ELSE m.team1_id
            END AS opponent_team_id,
            CASE
                WHEN m.winner_team_id IN :team_ids THEN 1
                ELSE 0
            END AS is_win
        FROM {schema}.matches m
        WHERE (m.team1_id IN :team_ids OR m.team2_id IN :team_ids)
          AND m.winner_team_id IS NOT NULL
          {exclude_internal_alias_matches}
    """
    try:
        with engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    text(sql).bindparams(bindparam("team_ids", expanding=True)),
                    {"team_ids": scoped_team_ids},
                ).mappings().all()
            ]
    except SQLAlchemyError as exc:
        if is_missing_relation_error(exc):
            return []
        raise
