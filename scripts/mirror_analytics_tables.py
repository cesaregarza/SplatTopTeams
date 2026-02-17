from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import ForeignKeyConstraint, MetaData, Table, create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable
from sqlalchemy.engine import Engine

from shared_lib.db import resolve_database_url


DEFAULT_MIRROR_TABLES: tuple[str, ...] = (
    "matches",
    "tournaments",
    "player_appearance_teams",
    "tournament_teams",
    "team_search_refresh_runs",
    "team_search_clusters",
    "team_search_embeddings",
    "players",
    "rounds",
    "stages",
    "player_appearances",
    "player_rankings",
    "player_ranking_stats",
    "external_ids",
    "roster_entries",
    "groups",
)


@dataclass(frozen=True)
class MirrorResult:
    table: str
    rows_source: int
    rows_target: int
    status: str


def _parse_tables(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_MIRROR_TABLES)
    out: list[str] = []
    for raw in value.split(","):
        table = raw.strip()
        if table:
            out.append(table)
    return out


def _resolve_urls(args: argparse.Namespace) -> tuple[str, str]:
    source = (
        args.source_db_url
        or os.getenv("RANKINGS_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    if not source:
        raise RuntimeError("source db URL missing: set RANKINGS_DATABASE_URL or pass --source-db-url")

    target = (
        args.target_db_url
        or os.getenv("DOCKER_LOCAL_DB_URL")
        or os.getenv("LOCAL_DB_URL")
        or "postgresql://postgres:postgres@localhost:5432/rankings_db?sslmode=disable"
    )
    return source, target


def _engine(url: str) -> Engine:
    return create_engine(resolve_database_url(url))


def _fetch_table_names(engine: Engine, schema: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_type = 'BASE TABLE'
                """
            ),
            {"schema": schema},
        ).scalars().all()
    return set(rows)


def _quote_identifier(value: str) -> str:
    return '"{}"'.format(value.replace('"', '""'))


def _qualified_table(schema: str, table: str) -> str:
    return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"


def _load_source_table(schema: str, table_name: str, source_engine: Engine) -> Table:
    metadata = MetaData(schema=schema)
    return Table(table_name, metadata, autoload_with=source_engine)


def _create_table(
    schema: str,
    source_table: Table,
    target_engine: Engine,
) -> tuple[Table, list[str]]:
    metadata = MetaData(schema=schema)
    target_table = source_table.to_metadata(metadata)

    removed_constraints: list[str] = []
    for constraint in list(target_table.constraints):
        if isinstance(constraint, ForeignKeyConstraint):
            removed_constraints.append(str(constraint))
            target_table.constraints.remove(constraint)
    for column in target_table.columns:
        if column.foreign_keys:
            removed_constraints.append(f"{column.name}: dropped foreign key(s)")
            column.foreign_keys.clear()

    create_sql = str(
        CreateTable(target_table).compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"include_foreign_key_constraints": False},
        )
    )

    with target_engine.begin() as conn:
        conn.execute(
            text(
                f"DROP TABLE IF EXISTS {_qualified_table(schema, source_table.name)} CASCADE"
            )
        )
        conn.execute(text(create_sql))

    return target_table, removed_constraints


def _copy_table_rows(
    schema: str,
    source_table: Table,
    target_table: Table,
    source_engine: Engine,
    target_engine: Engine,
    chunk_size: int = 5000,
) -> int:
    copied_rows = 0
    with source_engine.connect() as src_conn, target_engine.begin() as dst_conn:
        result = src_conn.execute(source_table.select())

        while True:
            rows = result.fetchmany(chunk_size)
            if not rows:
                break

            payload = [dict(row._mapping) for row in rows]
            dst_conn.execute(target_table.insert(), payload)
            copied_rows += len(payload)

    return copied_rows


def mirror_tables(
    source_url: str,
    target_url: str,
    schema: str,
    table_names: Sequence[str],
    chunk_size: int,
    allow_missing: bool = False,
) -> list[MirrorResult]:
    source_engine = _engine(source_url)
    target_engine = _engine(target_url)

    source_table_names = _fetch_table_names(source_engine, schema)

    missing: list[str] = []
    for table in table_names:
        if table not in source_table_names:
            if allow_missing:
                missing.append(table)
                continue
            raise RuntimeError(f"source missing requested table: {schema}.{table}")

    with target_engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {_quote_identifier(schema)}"))

    results: list[MirrorResult] = []
    for table_name in table_names:
        if table_name not in source_table_names:
            continue

        source_table = _load_source_table(schema, table_name, source_engine)
        target_table, removed_constraints = _create_table(schema, source_table, target_engine)

        copied = _copy_table_rows(
            schema=schema,
            source_table=source_table,
            target_table=target_table,
            source_engine=source_engine,
            target_engine=target_engine,
            chunk_size=chunk_size,
        )

        with source_engine.connect() as conn:
            source_rows = int(
                conn.execute(
                    text(
                        f"SELECT COUNT(*) FROM {_qualified_table(schema, table_name)}"
                    )
                ).scalar_one()
            )

        with target_engine.connect() as conn:
            target_rows = int(
                conn.execute(
                    text(
                        f"SELECT COUNT(*) FROM {_qualified_table(schema, table_name)}"
                    )
                ).scalar_one()
            )

        status = (
            f"ok; removed {len(removed_constraints)} foreign keys"
            if removed_constraints
            else "ok"
        )
        if copied != source_rows or copied != target_rows:
            status = f"{status}; count mismatch (copied={copied})"
        logging.getLogger(__name__).info(
            "[%s] source=%s target=%s (%s)",
            table_name,
            source_rows,
            target_rows,
            status,
        )
        results.append(
            MirrorResult(
                table=table_name,
                rows_source=source_rows,
                rows_target=target_rows,
                status=status,
            )
        )

    if missing:
        logging.getLogger(__name__).warning("missing source tables ignored: %s", ", ".join(missing))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror key comp_rankings tables from source Postgres to local analytics DB."
    )
    parser.add_argument("--source-db-url", default=None, help="Source Postgres URL.")
    parser.add_argument("--target-db-url", default=None, help="Target Postgres URL.")
    parser.add_argument(
        "--schema",
        default=os.getenv("RANKINGS_DB_SCHEMA", "comp_rankings"),
        help="Postgres schema containing source/target tables.",
    )
    parser.add_argument(
        "--tables",
        default=",".join(DEFAULT_MIRROR_TABLES),
        help=(
            "Comma-separated table names. Defaults to a curated set for team search + "
            "head-to-head analytics."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Ignore missing source tables instead of failing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-row progress logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    table_names = _parse_tables(args.tables)
    source_url, target_url = _resolve_urls(args)

    logging.getLogger(__name__).info("source: %s", source_url)
    logging.getLogger(__name__).info("target: %s", target_url)
    logging.getLogger(__name__).info("schema: %s", args.schema)
    logging.getLogger(__name__).info("tables: %s", ", ".join(table_names))

    results = mirror_tables(
        source_url=source_url,
        target_url=target_url,
        schema=args.schema,
        table_names=table_names,
        chunk_size=args.chunk_size,
        allow_missing=args.allow_missing,
    )

    for result in results:
        logging.getLogger(__name__).info(
            "%s: source=%s target=%s %s",
            result.table,
            result.rows_source,
            result.rows_target,
            result.status,
        )

    failed = [result for result in results if result.rows_source != result.rows_target]
    if failed:
        logging.getLogger(__name__).error("count mismatch in %d tables", len(failed))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
