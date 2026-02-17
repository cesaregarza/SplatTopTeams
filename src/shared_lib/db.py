from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy.engine import Engine


def _build_url_from_parts() -> Optional[str]:
    host = (
        os.getenv("RANKINGS_DB_HOST")
        or os.getenv("DB_HOST")
        or os.getenv("POSTGRES_HOST")
    )
    user = (
        os.getenv("RANKINGS_DB_USER")
        or os.getenv("DB_USER")
        or os.getenv("POSTGRES_USER")
    )
    if not host or not user:
        return None

    port = (
        os.getenv("RANKINGS_DB_PORT")
        or os.getenv("DB_PORT")
        or os.getenv("POSTGRES_PORT")
        or "5432"
    )
    name = (
        os.getenv("RANKINGS_DB_NAME")
        or os.getenv("DB_NAME")
        or os.getenv("POSTGRES_DB")
        or "rankings_db"
    )
    password = (
        os.getenv("RANKINGS_DB_PASSWORD")
        or os.getenv("DB_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
        or ""
    )
    sslmode = (
        os.getenv("RANKINGS_DB_SSLMODE")
        or os.getenv("DB_SSLMODE")
        or os.getenv("POSTGRES_SSLMODE")
    )

    auth = f"{user}:{password}" if password else user
    url = f"postgresql://{auth}@{host}:{port}/{name}"
    if sslmode:
        url = f"{url}?sslmode={sslmode}"
    return url


def resolve_database_url(explicit: Optional[str] = None) -> str:
    url = (
        explicit
        or os.getenv("RANKINGS_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or _build_url_from_parts()
    )
    if not url:
        raise RuntimeError(
            "No database URL configured. Set RANKINGS_DATABASE_URL or DATABASE_URL "
            "or DB/RANKINGS_DB component env variables."
        )
    return url


def create_engine(url: Optional[str] = None, *, echo: bool = False) -> Engine:
    return sa_create_engine(
        resolve_database_url(url),
        echo=echo,
        future=True,
        pool_pre_ping=True,
    )


def get_schema() -> str:
    return os.getenv("RANKINGS_DB_SCHEMA", "comp_rankings")
