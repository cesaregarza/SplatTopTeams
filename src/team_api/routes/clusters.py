from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from team_api.dependencies import get_store
from team_api.store import TeamSearchStore

router = APIRouter(prefix="/api", tags=["clusters"])
ClusterMode = Literal["strict", "explore", "family"]


@router.get("/clusters")
def list_clusters(
    q: Optional[str] = Query(default=None, min_length=1, max_length=120),
    cluster_mode: ClusterMode = "explore",
    limit: int = Query(default=50, ge=1, le=200),
    store: TeamSearchStore = Depends(get_store),
):
    snapshot = store.latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )
    snapshot_id = int(snapshot["run_id"])
    clusters = store.list_clusters(
        snapshot_id=snapshot_id,
        profile=cluster_mode,
        query=q,
        limit=limit,
    )
    return {
        "snapshot_id": snapshot_id,
        "cluster_mode": cluster_mode,
        "count": len(clusters),
        "clusters": clusters,
    }


@router.get("/clusters/{cluster_id}")
def cluster_detail(
    cluster_id: int,
    cluster_mode: ClusterMode = "explore",
    store: TeamSearchStore = Depends(get_store),
):
    snapshot = store.latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )
    snapshot_id = int(snapshot["run_id"])
    detail = store.cluster_detail(
        snapshot_id=snapshot_id,
        profile=cluster_mode,
        cluster_id=cluster_id,
    )
    if not detail:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return {
        "snapshot_id": snapshot_id,
        "cluster_mode": cluster_mode,
        **detail,
    }
