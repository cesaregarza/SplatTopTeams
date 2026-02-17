from __future__ import annotations

from functools import lru_cache
import logging

from sqlalchemy.exc import SQLAlchemyError
from shared_lib.db import create_engine
from team_api.settings import get_settings
from team_api.sql import ensure_search_tables
from team_api.store import TeamSearchStore

logger = logging.getLogger(__name__)


def _is_permission_denied(exc: Exception) -> bool:
    message = str(exc).lower()
    return "insufficientprivilege" in message or "permission denied" in message


@lru_cache(maxsize=1)
def get_store() -> TeamSearchStore:
    settings = get_settings()
    engine = create_engine()
    try:
        ensure_search_tables(engine, settings.rankings_db_schema)
    except SQLAlchemyError as exc:
        if not _is_permission_denied(exc):
            raise
        logger.warning(
            "Skipping schema bootstrap for %s due to insufficient privileges.",
            settings.rankings_db_schema,
        )
    return TeamSearchStore(engine=engine, schema=settings.rankings_db_schema)
