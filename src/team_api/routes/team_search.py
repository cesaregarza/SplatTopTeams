from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from team_api.dependencies import get_store
from team_api.store import TeamSearchStore

router = APIRouter(prefix="/api", tags=["team-search"])


@router.get("/team-search")
def team_search(
    q: str = Query(..., min_length=1, max_length=120),
    top_n: int = Query(default=20, ge=1, le=100),
    min_relevance: float = Query(default=0.8, ge=0.0, le=1.0),
    consolidate: bool = True,
    consolidate_min_overlap: float = Query(default=0.8, ge=0.0, le=1.0),
    cluster_mode: Literal["strict", "explore"] = "strict",
    include_clusters: bool = True,
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
    )

    return {
        "snapshot_id": snapshot_id,
        "cluster_mode": cluster_mode,
        "consolidate": consolidate,
        "consolidate_min_overlap": consolidate_min_overlap,
        "query_context": ranked["query"],
        "result_count": len(ranked["results"]),
        "results": ranked["results"],
    }
