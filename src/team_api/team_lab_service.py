from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


def resolve_team_lab_scope(
    *,
    snapshot_rows: Sequence[Any],
    family_members_by_team: dict[int, frozenset[int]] | None,
    search_similar_teams: Callable[..., dict[str, Any]],
    normalize_ids: Callable[[Any], list[int]],
    snapshot_id: int,
    profile: str,
    team_id: int,
) -> tuple[list[int], str | None]:
    def canonical_team_name(team_ids: Sequence[int]) -> str | None:
        weights: dict[str, int] = {}
        selected_ids = {int(candidate_id) for candidate_id in team_ids}
        for row in snapshot_rows:
            if int(getattr(row, "team_id", 0)) not in selected_ids:
                continue
            team_name = str(getattr(row, "team_name", "") or "").strip()
            if not team_name:
                continue
            try:
                weight = int(getattr(row, "lineup_count", 0) or 0)
            except (TypeError, ValueError):
                weight = 0
            weights[team_name] = weights.get(team_name, 0) + max(1, weight)

        if not weights:
            return None
        return max(
            weights.items(),
            key=lambda item: (int(item[1]), -len(item[0]), item[0].lower()),
        )[0]

    def normalized_team_name(value: object) -> str:
        lowered = str(value or "").strip().lower()
        return re.sub(r"[^a-z0-9]+", "", lowered)

    primary_team_id = int(team_id)
    base_row = next(
        (row for row in snapshot_rows if int(row.team_id) == primary_team_id),
        None,
    )

    if str(profile) != "family":
        return [primary_team_id], (base_row.team_name if base_row is not None else None)

    resolved_members = (family_members_by_team or {}).get(primary_team_id)
    if resolved_members:
        resolved_team_ids = sorted(int(candidate_team_id) for candidate_team_id in resolved_members)
        representative_team_name = (
            canonical_team_name(resolved_team_ids)
            or (base_row.team_name if base_row is not None else None)
            or f"Team {primary_team_id}"
        )
        return resolved_team_ids, str(representative_team_name)

    def containing_search_group(seed_team_id: int) -> list[int]:
        try:
            ranked = search_similar_teams(
                snapshot_id=snapshot_id,
                query=str(seed_team_id),
                top_n=60,
                min_relevance=0.0,
                cluster_mode=profile,
                include_clusters=True,
                consolidate=True,
                consolidate_min_overlap=0.8,
            )
        except Exception:
            return [int(seed_team_id)]

        for result in ranked.get("results", []):
            resolved_team_ids = normalize_ids(
                [
                    result.get("team_id"),
                    *(result.get("consolidated_team_ids") or []),
                ]
            )
            if int(seed_team_id) in resolved_team_ids:
                return resolved_team_ids
        return [int(seed_team_id)]

    base_group = containing_search_group(primary_team_id)
    if len(base_group) > 1:
        representative_team_name = (
            canonical_team_name(base_group)
            or (base_row.team_name if base_row is not None else None)
            or f"Team {primary_team_id}"
        )
        normalized_representative = normalized_team_name(representative_team_name)
        snapshot_rows_by_id = {
            int(getattr(row, "team_id", 0)): row
            for row in snapshot_rows
            if getattr(row, "team_id", None) is not None
        }
        anchor_candidates = [
            int(candidate_team_id)
            for candidate_team_id in sorted(set(int(candidate_team_id) for candidate_team_id in base_group))
            if normalized_team_name(getattr(snapshot_rows_by_id.get(int(candidate_team_id)), "team_name", "")) == normalized_representative
        ]
        if not anchor_candidates:
            anchor_candidates = sorted(set(int(candidate_team_id) for candidate_team_id in base_group))

        anchor_group_by_team_id: dict[int, list[int]] = {}
        for candidate_team_id in anchor_candidates[:12]:
            anchor_group_by_team_id[int(candidate_team_id)] = containing_search_group(int(candidate_team_id))

        consensus_team_ids: list[int] = []
        if anchor_group_by_team_id:
            candidate_counts: dict[int, int] = {}
            for resolved_ids in anchor_group_by_team_id.values():
                for candidate_team_id in set(int(candidate_team_id) for candidate_team_id in resolved_ids):
                    candidate_counts[int(candidate_team_id)] = (
                        candidate_counts.get(int(candidate_team_id), 0) + 1
                    )

            consensus_threshold = max(2, (len(anchor_group_by_team_id) // 2) + 1)
            consensus_team_ids = sorted(
                candidate_team_id
                for candidate_team_id, count in candidate_counts.items()
                if int(count) >= consensus_threshold
            )

        if primary_team_id in consensus_team_ids and len(consensus_team_ids) > 1:
            resolved_team_ids = consensus_team_ids
        else:
            anchor_team_id = min(
                anchor_group_by_team_id,
                key=lambda candidate_team_id: (
                    len(anchor_group_by_team_id[candidate_team_id]),
                    int(candidate_team_id),
                ),
            )
            resolved_team_ids = sorted(anchor_group_by_team_id[anchor_team_id])
        representative_team_name = (
            canonical_team_name(resolved_team_ids)
            or representative_team_name
        )
        return resolved_team_ids, str(representative_team_name)

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
        representative_team_name = (
            canonical_team_name(resolved_team_ids)
            or result.get("team_name")
            or (base_row.team_name if base_row is not None else f"Team {primary_team_id}")
        )
        return resolved_team_ids, str(representative_team_name)

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


def fetch_snapshot_team_names(
    *,
    engine: Engine,
    schema: str,
    snapshot_id: int,
    team_ids: Sequence[int],
    normalize_ids: Callable[[Any], list[int]],
    is_missing_relation_error: Callable[[Exception], bool],
    is_missing_column_error: Callable[[Exception], bool],
) -> dict[int, str]:
    scoped_team_ids = normalize_ids(team_ids)
    if not scoped_team_ids:
        return {}

    sql = f"""
        SELECT team_id, team_name
        FROM {schema}.team_search_embeddings
        WHERE snapshot_id = :snapshot_id
          AND team_id IN :team_ids
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(sql).bindparams(bindparam("team_ids", expanding=True)),
                {
                    "snapshot_id": int(snapshot_id),
                    "team_ids": scoped_team_ids,
                },
            ).mappings().all()
    except SQLAlchemyError as exc:
        if is_missing_relation_error(exc) or is_missing_column_error(exc):
            return {}
        raise

    return {
        int(row["team_id"]): str(row["team_name"])
        for row in rows
        if row.get("team_id") is not None and row.get("team_name") is not None
    }
