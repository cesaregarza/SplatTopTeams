from __future__ import annotations

from fastapi import APIRouter, Depends

from team_api.dependencies import get_store
from team_api.store import TeamSearchStore

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(store: TeamSearchStore = Depends(get_store)):
    ok = store.ping()
    latest = store.latest_snapshot()
    return {
        "status": "ok" if ok else "degraded",
        "latest_snapshot": latest,
    }


@router.get("/ready")
def ready(store: TeamSearchStore = Depends(get_store)):
    _ = store.ping()
    return {"ready": True}
