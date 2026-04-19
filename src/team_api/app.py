from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from team_api.middleware import RateLimitMiddleware
from team_api.routes.analytics import router as analytics_router
from team_api.routes.clusters import router as clusters_router
from team_api.routes.health import router as health_router
from team_api.routes.players import router as players_router
from team_api.routes.team_search import router as team_search_router
from team_api.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Public read API for SplatTop team search and latent clusters.",
)


def _is_db_schema_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "does not exist" in msg
        or "undefinedtable" in msg
        or "undefinedcolumn" in msg
    )


@app.exception_handler(SQLAlchemyError)
async def _sql_exception_handler(request: Request, exc: SQLAlchemyError):
    detail = "Database query failed."
    if "permission denied" in str(exc).lower():
        detail = "Database permission denied."
    elif _is_db_schema_error(exc):
        detail = "Required analytics tables are not available yet."
    return JSONResponse(status_code=503, content={"detail": detail})

app.add_middleware(RateLimitMiddleware, per_minute=settings.api_rl_per_min)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(team_search_router)
app.include_router(clusters_router)
app.include_router(analytics_router)
app.include_router(players_router)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/api/health",
    }
