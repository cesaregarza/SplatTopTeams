from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from team_api.search_logic import strip_internal_result_fields
from team_api.dependencies import get_store
from team_api.store import TeamSearchStore

router = APIRouter(prefix="/api", tags=["team-search"])
ClusterMode = Literal["strict", "explore", "family"]


@router.get("/team-search")
def team_search(
    q: str = Query(..., min_length=1, max_length=120),
    top_n: int = Query(default=20, ge=1, le=100),
    min_relevance: float = Query(default=0.8, ge=0.0, le=1.0),
    tournament_id: int | None = Query(default=None, ge=1),
    seed_player_ids: list[int] = Query(default=[]),
    consolidate: bool = True,
    consolidate_min_overlap: float = Query(default=0.8, ge=0.0, le=1.0),
    cluster_mode: ClusterMode = "strict",
    include_clusters: bool = True,
    recency_weight: float = Query(default=0.0, ge=0.0, le=1.0),
    store: TeamSearchStore = Depends(get_store),
):
    snapshot = store.latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )

    snapshot_id = int(snapshot["run_id"])
    ranked = store.search_similar_teams(
        snapshot_id=snapshot_id,
        query=q,
        top_n=top_n,
        min_relevance=min_relevance,
        consolidate=consolidate,
        consolidate_min_overlap=consolidate_min_overlap,
        cluster_mode=cluster_mode,
        include_clusters=include_clusters,
        tournament_id=tournament_id,
        seed_player_ids=seed_player_ids,
        recency_weight=recency_weight,
    )

    return {
        "snapshot_id": snapshot_id,
        "cluster_mode": cluster_mode,
        "tournament_id": tournament_id,
        "seed_player_ids": seed_player_ids,
        "consolidate": consolidate,
        "consolidate_min_overlap": consolidate_min_overlap,
        "query_context": ranked["query"],
        "result_count": len(ranked["results"]),
        "results": strip_internal_result_fields(list(ranked["results"])),
    }


@router.get("/team-search/suggest")
def team_search_suggest(
    q: str = Query(..., min_length=1, max_length=120),
    limit: int = Query(default=8, ge=1, le=25),
    store: TeamSearchStore = Depends(get_store),
):
    snapshot = store.latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )

    snapshot_id = int(snapshot["run_id"])
    suggestions = store.suggest_team_names(
        snapshot_id=snapshot_id,
        query=q,
        limit=limit,
    )
    return {
        "snapshot_id": snapshot_id,
        "query": q,
        "suggestions": suggestions,
    }


@router.get("/tournaments/{tournament_id}/teams")
def tournament_teams(
    tournament_id: int,
    q: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=200, ge=1, le=700),
    store: TeamSearchStore = Depends(get_store),
):
    snapshot = store.latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )

    snapshot_id = int(snapshot["run_id"])
    payload = store.list_tournament_teams(
        snapshot_id=snapshot_id,
        tournament_id=tournament_id,
        query=q,
        limit=limit,
    )

    return {
        "snapshot_id": snapshot_id,
        "tournament_id": tournament_id,
        "query": q,
        "source": payload["source"],
        "count": len(payload["teams"]),
        "teams": payload["teams"],
    }
