from __future__ import annotations

import re
import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VECTOR_DIM_RE = re.compile(r"^vector\((\d+)\)$", re.IGNORECASE)

logger = logging.getLogger(__name__)


def validate_identifier(value: str) -> str:
    if not value:
        raise ValueError("identifier cannot be empty")
    value = value.strip()
    if not _VALID_IDENTIFIER_RE.match(value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def _extract_vector_dim(type_name: str | None) -> int | None:
    if not type_name:
        return None
    match = _VECTOR_DIM_RE.match(type_name.strip())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _has_vector_columns(conn, schema: str) -> dict:
    type_query = """
        SELECT format_type(a.atttypid, a.atttypmod) AS type_name
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema
          AND c.relname = 'team_search_embeddings'
          AND a.attname = 'final_vector_vec'
          AND a.attnum > 0
          AND NOT a.attisdropped
    """
    has_extension = bool(
        conn.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')")
        ).scalar_one()
    )
    has_column = bool(
        conn.execute(
            text(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = 'team_search_embeddings'
                      AND column_name = 'final_vector_vec'
                )
                """
            ),
            {"schema": schema},
        ).scalar_one()
    )
    vector_type = (
        conn.execute(text(type_query), {"schema": schema}).scalar_one_or_none()
    )
    return {
        "extension_enabled": has_extension,
        "has_final_vector_vec": has_column,
        "final_vector_vec_dim": _extract_vector_dim(vector_type),
    }


def _execute_ddl_in_savepoint(conn, statement: str) -> bool:
    """Run optional DDL without aborting the outer transaction on failure."""
    try:
        with conn.begin_nested():
            conn.execute(text(statement))
    except SQLAlchemyError:
        return False
    return True


def ensure_search_tables(
    engine: Engine,
    schema: str,
    final_vector_dim: int | None = None,
) -> None:
    schema = validate_identifier(schema)

    statements = [
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.team_search_refresh_runs (
            run_id BIGSERIAL PRIMARY KEY,
            status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ NULL,
            days_window INTEGER NOT NULL,
            semantic_dim INTEGER NOT NULL,
            identity_dim INTEGER NOT NULL,
            identity_beta DOUBLE PRECISION NOT NULL,
            identity_idf_cap DOUBLE PRECISION NOT NULL DEFAULT 3.0,
            source_since_ms BIGINT NULL,
            source_until_ms BIGINT NULL,
            teams_indexed INTEGER NULL,
            clusters_strict INTEGER NULL,
            clusters_explore INTEGER NULL,
            message TEXT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.team_search_embeddings (
            snapshot_id BIGINT NOT NULL REFERENCES {schema}.team_search_refresh_runs(run_id) ON DELETE CASCADE,
            team_id BIGINT NOT NULL,
            tournament_id BIGINT NULL,
            team_name TEXT NULL,
            event_time_ms BIGINT NULL,
            lineup_count INTEGER NOT NULL,
            tournament_count INTEGER NOT NULL DEFAULT 0,
            unique_player_count INTEGER NULL,
            distinct_lineup_count INTEGER NULL,
            top_lineup_share DOUBLE PRECISION NULL,
            lineup_entropy DOUBLE PRECISION NULL,
            effective_lineups DOUBLE PRECISION NULL,
            semantic_vector DOUBLE PRECISION[] NOT NULL,
            identity_vector DOUBLE PRECISION[] NOT NULL,
            final_vector DOUBLE PRECISION[] NOT NULL,
            top_lineup_summary TEXT NULL,
            top_lineup_player_ids BIGINT[] NULL,
            top_lineup_player_names TEXT[] NULL,
            player_support JSONB NULL,
            pair_support JSONB NULL,
            lineup_variant_counts JSONB NULL,
            semantic_dim INTEGER NOT NULL,
            identity_dim INTEGER NOT NULL,
            identity_beta DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (snapshot_id, team_id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.team_search_clusters (
            snapshot_id BIGINT NOT NULL REFERENCES {schema}.team_search_refresh_runs(run_id) ON DELETE CASCADE,
            profile TEXT NOT NULL,
            cluster_id BIGINT NOT NULL,
            team_id BIGINT NOT NULL,
            cluster_size INTEGER NOT NULL,
            representative_team_name TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (snapshot_id, profile, team_id)
        )
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_team_search_embeddings_snapshot
            ON {schema}.team_search_embeddings(snapshot_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_team_search_embeddings_snapshot_lineup_count
            ON {schema}.team_search_embeddings(snapshot_id, lineup_count DESC)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_team_search_embeddings_name
            ON {schema}.team_search_embeddings(LOWER(team_name))
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_team_search_clusters_snapshot_profile
            ON {schema}.team_search_clusters(snapshot_id, profile)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_team_search_clusters_cluster
            ON {schema}.team_search_clusters(snapshot_id, profile, cluster_id)
        """,
    ]

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
        # Backfill-safe column adds for already-existing deployments.
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS unique_player_count INTEGER"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS distinct_lineup_count INTEGER"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS top_lineup_share DOUBLE PRECISION"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS lineup_entropy DOUBLE PRECISION"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS effective_lineups DOUBLE PRECISION"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS tournament_count INTEGER"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS top_lineup_player_ids BIGINT[]"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS top_lineup_player_names TEXT[]"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS roster_player_ids BIGINT[]"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS roster_player_names TEXT[]"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS roster_player_match_counts INTEGER[]"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS player_support JSONB"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS pair_support JSONB"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_embeddings "
                "ADD COLUMN IF NOT EXISTS lineup_variant_counts JSONB"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {schema}.team_search_refresh_runs "
                "ADD COLUMN IF NOT EXISTS identity_idf_cap DOUBLE PRECISION NOT NULL DEFAULT 3.0"
            )
        )
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS ix_team_search_embeddings_roster_player_ids_gin
                    ON {schema}.team_search_embeddings USING GIN (roster_player_ids)
                """
            )
        )

        vector_dim = final_vector_dim
        if vector_dim is None:
            vector_dim = _has_vector_columns(conn, schema).get("final_vector_vec_dim")
        if vector_dim:
            if not _execute_ddl_in_savepoint(
                conn, "CREATE EXTENSION IF NOT EXISTS vector"
            ):
                logger.warning(
                    "pgvector extension is not available in %s, "
                    "falling back to array embeddings.",
                    schema,
                )
            vector_info = _has_vector_columns(conn, schema)
            if vector_info["extension_enabled"]:
                vector_column_added = _execute_ddl_in_savepoint(
                    conn,
                    f"""
                    ALTER TABLE {schema}.team_search_embeddings
                        ADD COLUMN IF NOT EXISTS final_vector_vec vector({vector_dim})
                    """,
                )
                if not vector_column_added:
                    logger.warning(
                        "Unable to add pgvector columns in %s, "
                        "keeping array-only embeddings.",
                        schema,
                    )
                else:
                    vector_info = _has_vector_columns(conn, schema)
                    if vector_info["has_final_vector_vec"]:
                        if not _execute_ddl_in_savepoint(
                            conn,
                            f"""
                            CREATE INDEX IF NOT EXISTS
                            ix_team_search_embeddings_final_vector_vec_ivfflat
                            ON {schema}.team_search_embeddings
                            USING ivfflat (final_vector_vec vector_cosine_ops)
                            WITH (lists = 100)
                            """,
                        ):
                            logger.debug(
                                "Could not create ivfflat index on "
                                "final_vector_vec for %s.",
                                schema,
                            )

        # pg_trgm for fuzzy team-name matching.
        if not _execute_ddl_in_savepoint(
            conn, "CREATE EXTENSION IF NOT EXISTS pg_trgm"
        ):
            logger.info(
                "pg_trgm extension is not available, "
                "fuzzy name matching will be disabled.",
            )
        else:
            if not _execute_ddl_in_savepoint(
                conn,
                f"""
                CREATE INDEX IF NOT EXISTS
                ix_team_search_embeddings_team_name_trgm
                ON {schema}.team_search_embeddings
                USING GIN (LOWER(COALESCE(team_name, '')) gin_trgm_ops)
                """,
            ):
                logger.debug(
                    "Could not create trigram index on team_name for %s.",
                    schema,
                )


def get_vector_column_info(engine: Engine, schema: str) -> dict:
    schema = validate_identifier(schema)
    with engine.connect() as conn:
        return _has_vector_columns(conn, schema)
