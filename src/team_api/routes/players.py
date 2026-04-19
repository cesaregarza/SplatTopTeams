from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from team_api.dependencies import get_store
from team_api.store import TeamSearchStore

router = APIRouter(prefix="/api", tags=["players"])


@router.get("/players/suggest")
def player_suggest(
    q: str = Query(..., min_length=1, max_length=120),
    limit: int = Query(default=10, ge=1, le=50),
    store: TeamSearchStore = Depends(get_store),
):
    snapshot = store.latest_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No completed team-search snapshot available yet.",
        )

    snapshot_id = int(snapshot["run_id"])
    suggestions = store.suggest_players(
        snapshot_id=snapshot_id,
        query=q,
        limit=limit,
    )
    return {
        "snapshot_id": snapshot_id,
        "query": q,
        "suggestions": suggestions,
    }


@router.get("/players/{player_id}/teams")
def player_teams(
    player_id: int,
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
    payload = store.get_player_teams(
        snapshot_id=snapshot_id,
        player_id=player_id,
        limit=limit,
    )
    return {
        "snapshot_id": snapshot_id,
        **payload,
    }
