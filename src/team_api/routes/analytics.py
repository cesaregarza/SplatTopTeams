from __future__ import annotations

from typing import Literal
import re

from fastapi import APIRouter, Depends, HTTPException, Query

from team_api.dependencies import get_store
from team_api.store import TeamSearchStore

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _resolve_snapshot_id(store: TeamSearchStore, snapshot_id: int | None) -> int:
    if snapshot_id is not None:
        return int(snapshot_id)
    snapshot = store.latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )
    return int(snapshot["run_id"])


def _parse_team_ids(raw: str | None, fallback_id: int | None) -> list[int]:
    tokens = re.findall(r"\d+", str(raw)) if raw else []
    if tokens:
        parsed: list[int] = []
        seen = set()
        for value in tokens:
            team_id = int(value)
            if team_id <= 0 or team_id in seen:
                continue
            seen.add(team_id)
            parsed.append(team_id)
        if parsed:
            return parsed

    if fallback_id is not None:
        return [int(fallback_id)]
    return []


@router.get("/overview")
def analytics_overview(
    snapshot_id: int | None = Query(default=None, ge=1),
    cluster_mode: Literal["strict", "explore"] = "explore",
    limit_clusters: int = Query(default=20, ge=1, le=100),
    volatile_limit: int = Query(default=15, ge=1, le=50),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    payload = store.analytics_overview(
        snapshot_id=sid,
        profile=cluster_mode,
        limit_clusters=limit_clusters,
        volatile_limit=volatile_limit,
    )
    return payload


@router.get("/matchups")
def analytics_matchups(
    snapshot_id: int | None = Query(default=None, ge=1),
    cluster_mode: Literal["strict", "explore"] = "explore",
    min_matches: int = Query(default=3, ge=1, le=100),
    limit: int = Query(default=30, ge=1, le=100),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    payload = store.analytics_matchups(
        snapshot_id=sid,
        profile=cluster_mode,
        min_matches=min_matches,
        limit=limit,
    )
    return payload


@router.get("/head-to-head")
def analytics_head_to_head(
    team_a_id: int | None = Query(default=None, ge=1),
    team_b_id: int | None = Query(default=None, ge=1),
    team_a_ids: str | None = Query(default=None),
    team_b_ids: str | None = Query(default=None),
    snapshot_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=200, ge=1, le=1000),
    store: TeamSearchStore = Depends(get_store),
):
    parsed_team_a_ids = _parse_team_ids(team_a_ids, team_a_id)
    parsed_team_b_ids = _parse_team_ids(team_b_ids, team_b_id)

    if not parsed_team_a_ids or not parsed_team_b_ids:
        raise HTTPException(
            status_code=400,
            detail="team_a_id/team_a_ids and team_b_id/team_b_ids are required.",
        )

    if set(parsed_team_a_ids) & set(parsed_team_b_ids):
        raise HTTPException(
            status_code=400,
            detail="team_a_id(s) and team_b_id(s) must be disjoint.",
        )

    sid = int(snapshot_id) if snapshot_id is not None else None
    return store.analytics_head_to_head(
        snapshot_id=sid,
        team_a_ids=parsed_team_a_ids,
        team_b_ids=parsed_team_b_ids,
        limit=limit,
    )


@router.get("/roster-overlap")
def analytics_roster_overlap(
    snapshot_id: int | None = Query(default=None, ge=1),
    cluster_mode: Literal["strict", "explore"] = "explore",
    min_similarity: float = Query(default=0.80, ge=0.0, le=1.0),
    max_player_overlap: float = Query(default=0.30, ge=0.0, le=1.0),
    min_cluster_size: int = Query(default=2, ge=2, le=200),
    limit: int = Query(default=30, ge=1, le=200),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    return store.analytics_roster_diversity(
        snapshot_id=sid,
        profile=cluster_mode,
        min_similarity=min_similarity,
        max_player_overlap=max_player_overlap,
        min_cluster_size=min_cluster_size,
        limit=limit,
    )


@router.get("/team/{team_id}")
def analytics_team_lab(
    team_id: int,
    snapshot_id: int | None = Query(default=None, ge=1),
    cluster_mode: Literal["strict", "explore"] = "explore",
    neighbors: int = Query(default=12, ge=3, le=40),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    payload = store.analytics_team_lab(
        snapshot_id=sid,
        profile=cluster_mode,
        team_id=team_id,
        neighbors=neighbors,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Team not found in snapshot")
    return payload


@router.get("/team/{team_id}/matches")
def analytics_team_matches(
    team_id: int,
    team_ids: str | None = Query(default=None),
    snapshot_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=25, ge=1, le=200),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    parsed_team_ids = _parse_team_ids(team_ids, team_id)
    if int(team_id) not in parsed_team_ids:
        parsed_team_ids = [int(team_id), *parsed_team_ids]

    payload = store.analytics_team_matches(
        snapshot_id=sid,
        team_ids=parsed_team_ids,
        limit=limit,
    )
    return payload


@router.get("/team/{team_id}/blend")
def analytics_team_blend(
    team_id: int,
    snapshot_id: int | None = Query(default=None, ge=1),
    cluster_mode: Literal["strict", "explore"] = "explore",
    semantic_weight: float = Query(default=0.5, ge=0.0, le=1.0),
    neighbors: int = Query(default=12, ge=3, le=40),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    payload = store.analytics_team_blend(
        snapshot_id=sid,
        profile=cluster_mode,
        team_id=team_id,
        semantic_weight=semantic_weight,
        neighbors=neighbors,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Team not found in snapshot")
    return payload


@router.get("/outliers")
def analytics_outliers(
    snapshot_id: int | None = Query(default=None, ge=1),
    cluster_mode: Literal["strict", "explore"] = "explore",
    limit: int = Query(default=30, ge=1, le=200),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    return store.analytics_outliers(
        snapshot_id=sid,
        profile=cluster_mode,
        limit=limit,
    )


@router.get("/space")
def analytics_space(
    snapshot_id: int | None = Query(default=None, ge=1),
    cluster_mode: Literal["strict", "explore"] = "explore",
    max_points: int = Query(default=800, ge=50, le=3000),
    store: TeamSearchStore = Depends(get_store),
):
    sid = _resolve_snapshot_id(store, snapshot_id)
    return store.analytics_space(
        snapshot_id=sid,
        profile=cluster_mode,
        max_points=max_points,
    )


@router.get("/drift")
def analytics_drift(
    cluster_mode: Literal["strict", "explore"] = "explore",
    current_snapshot_id: int | None = Query(default=None, ge=1),
    previous_snapshot_id: int | None = Query(default=None, ge=1),
    top_movers: int = Query(default=20, ge=1, le=100),
    store: TeamSearchStore = Depends(get_store),
):
    payload = store.analytics_drift(
        profile=cluster_mode,
        current_snapshot_id=current_snapshot_id,
        previous_snapshot_id=previous_snapshot_id,
        top_movers=top_movers,
    )
    if payload is None:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )
    return payload
