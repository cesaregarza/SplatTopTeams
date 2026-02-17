from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from team_api.sql import get_vector_column_info, validate_identifier

logger = logging.getLogger(__name__)


def _is_missing_relation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "undefinedtable" in message or "does not exist" in message


def _is_missing_column_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "undefinedcolumn" in message or (
        "column" in message and "does not exist" in message
    )


def _is_access_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "permission denied" in message
        or "insufficientprivilege" in message
        or "access denied" in message
    )


_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")
_TOURNAMENT_SCORE_TIERS: tuple[tuple[float, str, str], ...] = (
    (5.0, "X", "x"),
    (10.0, "S+", "s_plus"),
    (20.0, "S", "s"),
    (40.0, "A+", "a_plus"),
    (80.0, "A", "a"),
    (160.0, "A-", "a_minus"),
)


def _normalize_query(value: str) -> str:
    return _NON_ALPHANUMERIC_RE.sub("", str(value or "").strip().lower())


def _tokenized_query(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _normalize_id_sequence(values: Any) -> List[int]:
    if values is None:
        return []

    if isinstance(values, (int, float, str)):
        values = [values]

    out: List[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, list):
            nested = _normalize_id_sequence(value)
            for team_id in nested:
                if team_id not in seen:
                    seen.add(team_id)
                    out.append(team_id)
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed <= 0:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
    return out


def _tournament_tier(value: Any) -> dict[str, str]:
    parsed = None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = None

    if parsed is None or not np.isfinite(parsed):
        return {
            "tier_id": "unscored",
            "tier_label": "Unscored",
        }

    # Smaller rank indicates stronger tournaments because rank 1 is the best
    # player. Lower values are better tiers.
    for threshold, label, tier_id in _TOURNAMENT_SCORE_TIERS:
        if parsed <= threshold:
            return {
                "tier_id": tier_id,
                "tier_label": label,
            }

    return {
        "tier_id": "a_minus",
        "tier_label": "A-",
    }


@dataclass
class EmbeddingRow:
    team_id: int
    tournament_id: Optional[int]
    team_name: str
    event_time_ms: Optional[int]
    lineup_count: int
    semantic_vector: np.ndarray
    identity_vector: np.ndarray
    final_vector: np.ndarray
    top_lineup_summary: Optional[str] = None
    unique_player_count: int = 0
    distinct_lineup_count: int = 0
    top_lineup_share: float = 0.0
    lineup_entropy: float = 0.0
    effective_lineups: float = 0.0
    top_lineup_player_ids: tuple[int, ...] = ()
    top_lineup_player_names: tuple[str, ...] = ()
    tournament_count: int = 0


class TeamSearchStore:
    _MATCH_SCORE_CANDIDATES = (
        ("team1_score", "team2_score"),
        ("team_1_score", "team_2_score"),
        ("score1", "score2"),
        ("score_1", "score_2"),
        ("team1_points", "team2_points"),
        ("team1_goals", "team2_goals"),
    )

    def __init__(self, engine: Engine, schema: str):
        self.engine = engine
        self.schema = validate_identifier(schema)
        self._vector_capability: Optional[Dict[str, Any]] = None
        self._match_columns: Optional[set[str]] = None
        self._match_score_columns_cache: Optional[tuple[str, str] | None] = None
        self._player_rankings_rank_column: Optional[str] = None
        self._player_rankings_rank_column_checked: bool = False

    @staticmethod
    def _rank_similar_teams(
        embeddings: Sequence[EmbeddingRow],
        target_team_ids: Sequence[int],
        cluster_map: Dict[int, Dict[str, object]],
        top_n: int,
        min_relevance: float,
    ) -> Dict[str, object]:
        from team_api.search_logic import rank_similar_teams

        return rank_similar_teams(
            embeddings=embeddings,
            target_team_ids=target_team_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=min_relevance,
        )

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return vec / norm

    @staticmethod
    def _weighted_centroid(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return np.array([], dtype=np.float64)
        if weights.size == 0 or float(weights.sum()) <= 0:
            return TeamSearchStore._normalize_vector(vectors.mean(axis=0))
        centroid = (vectors * weights[:, None]).sum(axis=0) / float(weights.sum())
        return TeamSearchStore._normalize_vector(centroid)

    def _match_table_columns(self) -> set[str]:
        if self._match_columns is None:
            try:
                with self.engine.connect() as conn:
                    self._match_columns = {
                        row["column_name"].lower()
                        for row in conn.execute(
                            text(
                                """
                                SELECT column_name
                                FROM information_schema.columns
                                WHERE table_schema = :schema
                                  AND table_name = 'matches'
                                """
                            ),
                            {"schema": self.schema},
                        ).mappings().all()
                    }
            except SQLAlchemyError:
                self._match_columns = set()
        return self._match_columns

    def _resolve_match_score_columns(self) -> Optional[tuple[str, str]]:
        if self._match_score_columns_cache is None:
            columns = self._match_table_columns()
            self._match_score_columns_cache = None
            for col_a, col_b in self._MATCH_SCORE_CANDIDATES:
                if col_a in columns and col_b in columns:
                    self._match_score_columns_cache = (col_a, col_b)
                    break
        return self._match_score_columns_cache

    def _fetch_player_names(self, player_ids: Sequence[int]) -> Dict[int, str]:
        if not player_ids:
            return {}

        table = validate_identifier(self.schema)
        names: Dict[int, str] = {}
        with self.engine.connect() as conn:
            for index in range(0, len(player_ids), 1000):
                chunk = player_ids[index : index + 1000]
                placeholders = ", ".join(f":player_id_{idx}" for idx in range(len(chunk)))
                params = {f"player_id_{idx}": int(player_id) for idx, player_id in enumerate(chunk)}
                sql = f"""
                    SELECT player_id, display_name
                    FROM {table}.players
                    WHERE player_id IN ({placeholders})
                """
                rows = conn.execute(text(sql), params).mappings().all()
                for row in rows:
                    names[int(row["player_id"])] = str(row["display_name"] or str(row["player_id"]))
        return names

    def _fetch_match_rosters(
        self,
        rows: Sequence[dict],
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
    ) -> Dict[tuple[int, int], Dict[str, Dict[str, object]]]:
        if not rows:
            return {}

        match_ids: list[int] = []
        match_lookup: dict[tuple[int, int], dict] = {}
        team_a_set = set(int(team_id) for team_id in team_a_ids)
        team_b_set = set(int(team_id) for team_id in team_b_ids)

        for row in rows:
            match_id = row.get("match_id")
            if match_id is None:
                continue
            mid = int(match_id)
            if mid <= 0:
                continue
            match_ids.append(mid)
            tournament_id = row.get("tournament_id")
            key = (mid, int(tournament_id) if tournament_id is not None else 0)
            match_lookup[key] = {
                "team_a": {"player_ids": [], "player_names": []},
                "team_b": {"player_ids": [], "player_names": []},
            }

        if not match_ids:
            return {}

        unique_match_ids = sorted(set(match_ids))
        all_team_ids = sorted(team_a_set | team_b_set)
        if not all_team_ids:
            return match_lookup

        query = f"""
            SELECT
                pat.match_id,
                pat.tournament_id,
                pat.team_id,
                ARRAY_AGG(DISTINCT pat.player_id ORDER BY pat.player_id) AS player_ids
            FROM {self.schema}.player_appearance_teams pat
            WHERE pat.match_id IN :match_ids
              AND pat.team_id IN :team_ids
            GROUP BY pat.match_id, pat.tournament_id, pat.team_id
        """

        rosters: Dict[tuple[int, int], Dict[str, Dict[str, object]]] = dict(match_lookup)
        player_ids_for_lookup: list[int] = []

        try:
            with self.engine.connect() as conn:
                roster_rows = conn.execute(
                    text(query).bindparams(
                        bindparam("match_ids", expanding=True),
                        bindparam("team_ids", expanding=True),
                    ),
                    {"match_ids": unique_match_ids, "team_ids": all_team_ids},
                ).mappings().all()
        except SQLAlchemyError:
            return match_lookup

        for row in roster_rows:
            match_id = int(row["match_id"])
            tournament_id = row.get("tournament_id")
            key = (match_id, int(tournament_id) if tournament_id is not None else 0)
            roster = rosters.get(key)
            if roster is None:
                continue

            team_id = int(row["team_id"])
            if team_id in team_a_set:
                target = "team_a"
            elif team_id in team_b_set:
                target = "team_b"
            else:
                continue

            ids = [int(player_id) for player_id in row.get("player_ids", []) if player_id is not None]
            roster[target]["player_ids"] = ids
            player_ids_for_lookup.extend(ids)

        if player_ids_for_lookup:
            id_to_name = self._fetch_player_names(player_ids_for_lookup)
        for match_roster in rosters.values():
            for side in ("team_a", "team_b"):
                ids = match_roster[side]["player_ids"]
                match_roster[side]["player_names"] = [
                    id_to_name.get(player_id, str(player_id)) for player_id in ids
                ]

        return rosters

    def _fetch_tournament_scores(
        self, tournament_ids: Sequence[int]
    ) -> Dict[int, float]:
        rank_field = self._player_rankings_rank_field()

        def _build_score_sql(source_table: str, source_alias: str) -> str:
            # Prefer stored player rank when available; fall back to score-derived position.
            player_rank_expr = (
                f"NULLIF(pr.{rank_field}, 0)::double precision"
                if rank_field
                else "NULL::double precision"
            )
            if rank_field is None:
                rank_filter = "pr.score IS NOT NULL"
            else:
                rank_filter = f"(pr.score IS NOT NULL OR NULLIF(pr.{rank_field}, 0) IS NOT NULL)"

            return f"""
                WITH latest_rankings AS ({latest_timestamp_sql}),
                roster AS (
                    SELECT DISTINCT
                        {source_alias}.tournament_id::bigint AS tournament_id,
                        {source_alias}.player_id::bigint AS player_id
                    FROM {self.schema}.{source_table} {source_alias}
                    WHERE {source_alias}.tournament_id IN :tournament_ids
                ),
                scored_players AS (
                    SELECT
                        r.tournament_id,
                        {player_rank_expr} AS player_rank,
                        pr.score::double precision AS score,
                        ROW_NUMBER() OVER (
                          PARTITION BY r.tournament_id
                          ORDER BY
                            CASE
                              WHEN {player_rank_expr} IS NOT NULL THEN 0
                              ELSE 1
                            END,
                            CASE
                              WHEN {player_rank_expr} IS NOT NULL THEN {player_rank_expr}
                              ELSE -pr.score
                            END ASC,
                            r.player_id ASC
                        ) AS score_rank
                    FROM roster r
                    JOIN {self.schema}.player_rankings pr
                      ON pr.player_id = r.player_id
                     AND pr.calculated_at_ms = (SELECT calculated_at_ms FROM latest_rankings)
                    WHERE {rank_filter}
                ),
                ranked_players AS (
                    SELECT
                        tournament_id,
                        COALESCE(player_rank, score_rank::double precision) AS tournament_rank,
                        score_rank,
                        COUNT(*) OVER (PARTITION BY tournament_id) AS roster_player_count
                    FROM scored_players
                )
                SELECT
                    tournament_id,
                    MAX(tournament_rank) FILTER (
                        WHERE score_rank = LEAST(10, roster_player_count)
                    ) AS tournament_score
                FROM ranked_players
                GROUP BY tournament_id
            """

        normalized_tournament_ids = sorted(set(_normalize_id_sequence(tournament_ids)))
        if not normalized_tournament_ids:
            return {}

        latest_timestamp_sql = f"""
            SELECT MAX(calculated_at_ms) AS calculated_at_ms
            FROM {self.schema}.player_rankings
        """

        roster_queries = [
            {
                "table": "roster_entries",
                "base_sql": _build_score_sql("roster_entries", "re"),
            },
            {
                "table": "player_appearance_teams",
                "base_sql": _build_score_sql("player_appearance_teams", "pat"),
            },
        ]

        tournament_scores: Dict[int, float] = {}

        def _to_float(value: Any) -> Optional[float]:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if not np.isnan(parsed) and np.isfinite(parsed) else None

        for query_info in roster_queries:
            sql = query_info["base_sql"]
            source_scores: Dict[int, float] = {}

            try:
                with self.engine.connect() as conn:
                    rows = conn.execute(
                        text(sql).bindparams(bindparam("tournament_ids", expanding=True)),
                        {"tournament_ids": normalized_tournament_ids},
                    ).mappings().all()
                for row in rows:
                    tournament_id = row.get("tournament_id")
                    score = _to_float(row.get("tournament_score"))
                    if tournament_id is not None and score is not None:
                        source_scores[int(tournament_id)] = score

                if source_scores:
                    for source_tournament_id, source_score in source_scores.items():
                        tournament_scores.setdefault(source_tournament_id, source_score)
                    logger.debug(
                        "found %d tournament score entries from %s",
                        len(source_scores),
                        query_info["table"],
                    )
                    sample_scores = list(source_scores.items())[:3]
                    if sample_scores:
                        logger.debug(
                            "sample tournament scores from %s: %s",
                            query_info["table"],
                            sample_scores,
                        )
                else:
                    logger.info(
                        "no tournament score rows found from %s for %d tournaments",
                        query_info["table"],
                        len(normalized_tournament_ids),
                    )
            except SQLAlchemyError as exc:
                if not (_is_missing_relation_error(exc) or _is_missing_column_error(exc)):
                    if not _is_access_error(exc):
                        raise
                logger.info(
                    "failing tournament score source %s for %d tournaments: %s",
                    query_info["table"],
                    len(normalized_tournament_ids),
                    exc,
                )

        if tournament_scores:
            return tournament_scores

        return {}

    def _player_rankings_has_rank_column(self) -> bool:
        return self._player_rankings_rank_field() is not None

    def _player_rankings_rank_field(self) -> str | None:
        if self._player_rankings_rank_column_checked:
            return self._player_rankings_rank_column

        self._player_rankings_rank_column_checked = True
        if self._has_column("player_rankings", "player_rank"):
            self._player_rankings_rank_column = "player_rank"
        elif self._has_column("player_rankings", "rank"):
            self._player_rankings_rank_column = "rank"
        return self._player_rankings_rank_column

    def _has_column(self, table_name: str, column_name: str) -> bool:
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                          AND table_name = :table_name
                          AND column_name = :column_name
                        LIMIT 1
                        """
                    ),
                    {
                        "schema": self.schema,
                        "table_name": table_name,
                        "column_name": column_name,
                    },
                ).fetchone()
                return rows is not None
        except SQLAlchemyError:
            return False


    def _detect_vector_support(self) -> Dict[str, Any]:
        if self._vector_capability is None:
            self._vector_capability = get_vector_column_info(self.engine, self.schema)
        return self._vector_capability

    @staticmethod
    def _vector_literal(vector: np.ndarray) -> str:
        return "[" + ",".join(f"{float(x):.10g}" for x in vector) + "]"

    @staticmethod
    def _to_db_seconds(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        iv = int(value)
        return iv // 1000 if iv > 1_000_000_000_000 else iv

    def _fetch_tournament_window(self, snapshot_id: int) -> tuple[Optional[int], Optional[int]]:
        sql = f"""
            SELECT
                source_since_ms,
                source_until_ms
            FROM {self.schema}.team_search_refresh_runs
            WHERE run_id = :snapshot_id
        """
        row = self._fetch_first_row(sql, {"snapshot_id": int(snapshot_id)})
        if not row:
            return None, None
        return (
            self._to_db_seconds(row.get("source_since_ms")),
            self._to_db_seconds(row.get("source_until_ms")),
        )

    @staticmethod
    def _seconds_expr(column_expr: str) -> str:
        return (
            "CASE\n"
            f"    WHEN {column_expr} IS NULL THEN NULL\n"
            f"    WHEN {column_expr} >= 1000000000000 THEN {column_expr} / 1000\n"
            f"    ELSE {column_expr}\n"
            "END"
        )

    def _query_tournament_counts(
        self,
        sql: str,
        team_ids: Sequence[int],
        params: Dict[str, object],
    ) -> Optional[Dict[int, int]]:
        ids = sorted({int(team_id) for team_id in team_ids})
        if not ids:
            return {}

        counts: Dict[int, int] = {}
        try:
            with self.engine.connect() as conn:
                for offset in range(0, len(ids), 500):
                    chunk = ids[offset : offset + 500]
                    rows = conn.execute(
                        text(sql).bindparams(bindparam("team_ids", expanding=True)),
                        {"team_ids": chunk, **params},
                    ).mappings().all()
                    for row in rows:
                        counts[int(row["team_id"])] = int(row["tournament_count"])
        except SQLAlchemyError as exc:
            if (
                _is_missing_relation_error(exc)
                or _is_missing_column_error(exc)
                or _is_access_error(exc)
            ):
                return None
            raise

        return counts

    def _fetch_tournament_counts(self, snapshot_id: int, team_ids: Sequence[int]) -> Dict[int, int]:
        ids = sorted({int(team_id) for team_id in team_ids})
        if not ids:
            return {}

        since_ts, until_ts = self._fetch_tournament_window(snapshot_id)
        match_last_game = self._seconds_expr("m.last_game_finished_at_ms")
        match_created = self._seconds_expr("m.created_at_ms")
        match_start = self._seconds_expr("t.start_time_ms")
        match_time_expr = f"COALESCE({match_last_game}, {match_created}, {match_start})"

        filters: list[str] = []
        params: Dict[str, object] = {}
        if since_ts is not None:
            filters.append(f"{match_time_expr} >= :source_since_ts")
            params["source_since_ts"] = int(since_ts)
        if until_ts is not None:
            filters.append(f"{match_time_expr} <= :source_until_ts")
            params["source_until_ts"] = int(until_ts)

        match_where = ""
        if filters:
            match_where = " AND " + " AND ".join(filters)

        tournament_time_expr = (
            f"COALESCE({self._seconds_expr('tt.start_time_ms')}, "
            f"{self._seconds_expr('tt.created_at_ms')})"
        )
        tournament_filters: list[str] = []
        if since_ts is not None:
            tournament_filters.append(f"{tournament_time_expr} >= :source_since_ts")
        if until_ts is not None:
            tournament_filters.append(f"{tournament_time_expr} <= :source_until_ts")
        tournament_where = ""
        if tournament_filters:
            tournament_where = " AND " + " AND ".join(tournament_filters)

        primary_sql = f"""
            SELECT
                pat.team_id,
                COUNT(DISTINCT pat.tournament_id)::int AS tournament_count
            FROM {self.schema}.player_appearance_teams pat
            JOIN {self.schema}.matches m
              ON m.match_id = pat.match_id
             AND m.tournament_id = pat.tournament_id
            JOIN {self.schema}.tournaments t
              ON t.tournament_id = pat.tournament_id
            WHERE pat.team_id IN :team_ids
              {match_where}
            GROUP BY pat.team_id
        """
        primary_unbounded_sql = f"""
            SELECT
                pat.team_id,
                COUNT(DISTINCT pat.tournament_id)::int AS tournament_count
            FROM {self.schema}.player_appearance_teams pat
            JOIN {self.schema}.matches m
              ON m.match_id = pat.match_id
             AND m.tournament_id = pat.tournament_id
            JOIN {self.schema}.tournaments t
              ON t.tournament_id = pat.tournament_id
            WHERE pat.team_id IN :team_ids
            GROUP BY pat.team_id
        """
        fallback_sql = f"""
            SELECT
                tt.team_id,
                COUNT(DISTINCT tt.tournament_id)::int AS tournament_count
            FROM {self.schema}.tournament_teams tt
            JOIN {self.schema}.tournaments tt_t
              ON tt_t.tournament_id = tt.tournament_id
            WHERE tt.team_id IN :team_ids
              {tournament_where}
            GROUP BY tt.team_id
        """
        fallback_unbounded_sql = f"""
            SELECT
                tt.team_id,
                COUNT(DISTINCT tt.tournament_id)::int AS tournament_count
            FROM {self.schema}.tournament_teams tt
            WHERE tt.team_id IN :team_ids
            GROUP BY tt.team_id
        """

        query_plan = (
            (primary_sql, params),
            (primary_unbounded_sql, {}),
            (fallback_sql, params),
            (fallback_unbounded_sql, {}),
        )

        resolved: Dict[int, int] = {}
        for query_sql, query_params in query_plan:
            missing = [team_id for team_id in ids if team_id not in resolved]
            if not missing:
                return resolved

            result = self._query_tournament_counts(
                query_sql,
                missing,
                query_params,
            )
            if result is None:
                continue
            resolved.update(result)

        return resolved

    def _hydrate_tournament_counts(
        self, snapshot_id: int, rows: List[Dict[str, Any]]
    ) -> None:
        missing = [
            int(row["team_id"])
            for row in rows
            if int(row.get("tournament_count") or 0) <= 0
        ]
        if not missing:
            return

        hydrated = self._fetch_tournament_counts(snapshot_id, missing)
        if not hydrated:
            # As a safe fallback, at least surface teams with match history as having
            # participated in one tournament when exact tournament counting can’t be resolved.
            for row in rows:
                if (
                    int(row.get("tournament_count") or 0) <= 0
                    and int(row.get("lineup_count") or 0) > 0
                ):
                    row["tournament_count"] = 1
            return

        for row in rows:
            team_id = int(row["team_id"])
            if int(row.get("tournament_count") or 0) <= 0 and team_id in hydrated:
                row["tournament_count"] = int(hydrated[team_id])

        for row in rows:
            if (
                int(row.get("tournament_count") or 0) <= 0
                and int(row.get("lineup_count") or 0) > 0
            ):
                row["tournament_count"] = 1

    def _fetch_embeddings_by_team_ids(
        self,
        snapshot_id: int,
        team_ids: Sequence[int],
    ) -> List[EmbeddingRow]:
        if not team_ids:
            return []

        unique_ids = sorted({int(tid) for tid in team_ids})
        if not unique_ids:
            return []

        sql_with_players = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                tournament_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary,
                top_lineup_player_ids,
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND team_id IN :team_ids
        """
        sql_legacy = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND team_id IN :team_ids
        """
        sql_core = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND team_id IN :team_ids
        """

        with self.engine.connect() as conn:
            try:
                rows = conn.execute(
                    text(sql_with_players).bindparams(bindparam("team_ids", expanding=True)),
                    {"snapshot_id": int(snapshot_id), "team_ids": unique_ids},
                ).mappings().all()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc):
                    logger.warning(
                        "Team-search embedding table missing in schema %s",
                        self.schema,
                    )
                    return []
                if _is_missing_column_error(exc):
                    try:
                        rows = conn.execute(
                            text(sql_legacy).bindparams(
                                bindparam("team_ids", expanding=True)
                            ),
                            {"snapshot_id": int(snapshot_id), "team_ids": unique_ids},
                        ).mappings().all()
                    except SQLAlchemyError as nested_exc:
                        if _is_missing_column_error(nested_exc):
                            rows = conn.execute(
                                text(sql_core).bindparams(
                                    bindparam("team_ids", expanding=True)
                                ),
                                {
                                    "snapshot_id": int(snapshot_id),
                                    "team_ids": unique_ids,
                                },
                            ).mappings().all()
                        else:
                            raise
                else:
                    raise

        out: List[EmbeddingRow] = []
        row_dicts = [dict(row) for row in rows]
        self._hydrate_tournament_counts(snapshot_id, row_dicts)

        for r in row_dicts:
            r.setdefault("top_lineup_player_ids", [])
            r.setdefault("top_lineup_player_names", [])
            out.append(self._row_to_embedding(r))
        return out

    def _query_vector_rank(
        self,
        snapshot_id: int,
        query_rows: List[EmbeddingRow],
        target_team_ids: List[int],
        cluster_map: Dict[int, Dict[str, object]],
        top_n: int,
        min_relevance: float,
    ) -> Optional[Dict[str, object]]:
        if not query_rows:
            return None

        vector_info = self._detect_vector_support()
        if not (
            vector_info.get("extension_enabled")
            and vector_info.get("has_final_vector_vec")
            and isinstance(vector_info.get("final_vector_vec_dim"), int)
        ):
            return None

        query_final = np.stack([row.final_vector for row in query_rows], axis=0)
        weights = np.asarray([row.lineup_count for row in query_rows], dtype=np.float64)

        query_final_centroid = self._weighted_centroid(query_final, weights)
        if query_final_centroid.size == 0:
            return None

        vector_dim = int(vector_info["final_vector_vec_dim"])
        target_dim = int(query_final_centroid.size)
        if vector_dim != target_dim:
            logger.warning(
                "Skipping ANN query in %s due vector dimension mismatch: db=%s query=%s",
                self.schema,
                vector_dim,
                target_dim,
            )
            return None

        candidate_limit = max(120, min(3000, int(top_n) * 40))
        sql = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                tournament_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary,
                top_lineup_player_ids,
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """
        sql_without_tournament_count = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary,
                top_lineup_player_ids,
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """
        sql_core = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND final_vector_vec IS NOT NULL
            ORDER BY final_vector_vec <=> CAST(:query_vector AS vector({vector_dim}))
            LIMIT :candidate_limit
        """

        with self.engine.connect() as conn:
            try:
                candidate_rows = conn.execute(
                    text(sql),
                    {
                        "snapshot_id": int(snapshot_id),
                        "query_vector": self._vector_literal(query_final_centroid),
                        "candidate_limit": int(candidate_limit),
                    },
                ).mappings().all()
            except SQLAlchemyError as exc:
                if _is_missing_column_error(exc):
                    try:
                        candidate_rows = conn.execute(
                            text(sql_without_tournament_count),
                            {
                                "snapshot_id": int(snapshot_id),
                                "query_vector": self._vector_literal(query_final_centroid),
                                "candidate_limit": int(candidate_limit),
                            },
                        ).mappings().all()
                    except SQLAlchemyError as nested_exc:
                        if _is_missing_column_error(nested_exc):
                            candidate_rows = conn.execute(
                                text(sql_core),
                                {
                                    "snapshot_id": int(snapshot_id),
                                    "query_vector": self._vector_literal(
                                        query_final_centroid
                                    ),
                                    "candidate_limit": int(candidate_limit),
                                },
                            ).mappings().all()
                        else:
                            logger.warning(
                                "Vector ANN query failed in %s; falling back to in-memory search: %s",
                                self.schema,
                                exc,
                            )
                            return None
                else:
                    logger.warning(
                        "Vector ANN query failed in %s; falling back to in-memory search: %s",
                        self.schema,
                        exc,
                    )
                    return None

        candidates: List[EmbeddingRow] = []
        seen: set[int] = set()
        candidate_dicts = [dict(row) for row in candidate_rows]
        self._hydrate_tournament_counts(snapshot_id, candidate_dicts)
        for r in candidate_dicts:
            r.setdefault("top_lineup_player_ids", [])
            r.setdefault("top_lineup_player_names", [])
            item = self._row_to_embedding(r)
            seen.add(item.team_id)
            candidates.append(item)

        for row in query_rows:
            if int(row.team_id) not in seen:
                candidates.append(row)
                seen.add(int(row.team_id))

        if not candidates:
            return None

        ranked = self._rank_similar_teams(
            embeddings=candidates,
            target_team_ids=target_team_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=min_relevance,
        )
        if ranked["results"]:
            return ranked

        # Query centroid was built from only target teams. If top_n filtering removed all results,
        # retry with a very loose threshold on the ANN candidates to avoid empty outputs from
        # aggressive defaults.
        return self._rank_similar_teams(
            embeddings=candidates,
            target_team_ids=target_team_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=0.0,
        )

    def search_similar_teams(
        self,
        snapshot_id: int,
        query: str,
        top_n: int,
        min_relevance: float,
        cluster_mode: str,
        include_clusters: bool,
        consolidate: bool = True,
        consolidate_min_overlap: float = 0.8,
    ) -> Dict[str, object]:
        target_ids = self.match_targets(
            snapshot_id=snapshot_id,
            query=query,
            limit=max(1, min(int(top_n), 60)),
        )
        cluster_map = self.load_cluster_map(snapshot_id, cluster_mode) if include_clusters else {}

        query_rows = self._fetch_embeddings_by_team_ids(snapshot_id, target_ids)
        if not query_rows:
            return {
                "query": {"matched_team_ids": [], "matched_team_names": []},
                "results": [],
            }

        target_ids = [int(row.team_id) for row in query_rows]

        ranked = self._query_vector_rank(
            snapshot_id=snapshot_id,
            query_rows=query_rows,
            target_team_ids=target_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=min_relevance,
        )

        if ranked is not None:
            if consolidate:
                from team_api.search_logic import consolidate_ranked_results

                ranked["results"] = consolidate_ranked_results(
                    ranked["results"],
                    min_overlap=consolidate_min_overlap,
                )
            return ranked

        all_rows = self.load_embeddings(snapshot_id)
        ranked = self._rank_similar_teams(
            embeddings=all_rows,
            target_team_ids=target_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=min_relevance,
        )
        if consolidate:
            from team_api.search_logic import consolidate_ranked_results

            ranked["results"] = consolidate_ranked_results(
                ranked["results"],
                min_overlap=consolidate_min_overlap,
            )
        return ranked

    def _fetch_rows(
        self,
        sql: str,
        params: Dict[str, Any],
        *,
        missing_default: Any,
    ) -> Any:
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(sql), params).mappings().all()
            return rows
        except SQLAlchemyError as exc:
            if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                return missing_default
            raise

    def _fetch_first_row(self, sql: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(sql), params).mappings().first()
            return dict(row) if row else None
        except SQLAlchemyError as exc:
            if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                return None
            raise

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        sql = f"""
            SELECT run_id, finished_at, teams_indexed
            FROM {self.schema}.team_search_refresh_runs
            WHERE status = 'completed'
            ORDER BY finished_at DESC NULLS LAST, run_id DESC
            LIMIT 1
        """
        row = self._fetch_first_row(sql, {})
        return row

    def list_completed_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        sql = f"""
            SELECT
                run_id,
                started_at,
                finished_at,
                teams_indexed,
                clusters_strict,
                clusters_explore,
                message
            FROM {self.schema}.team_search_refresh_runs
            WHERE status = 'completed'
            ORDER BY finished_at DESC NULLS LAST, run_id DESC
            LIMIT :limit
        """
        rows = self._fetch_rows(
            sql,
            {"limit": max(1, min(int(limit), 50))},
            missing_default=[],
        )
        return [dict(row) for row in rows]

    def _row_to_embedding(self, row: Dict[str, Any]) -> EmbeddingRow:
        def _coerce_vector(value: Any) -> np.ndarray:
            try:
                arr = np.asarray(value or [], dtype=np.float64)
            except (TypeError, ValueError):
                arr = np.array([], dtype=np.float64)
            return arr.reshape(-1)

        def _coerce_player_id(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        return EmbeddingRow(
            team_id=int(row["team_id"]),
            tournament_id=(
                int(row["tournament_id"])
                if row.get("tournament_id") is not None
                else None
            ),
            team_name=str(row.get("team_name") or row["team_id"]),
            event_time_ms=(
                int(row["event_time_ms"])
                if row.get("event_time_ms") is not None
                else None
            ),
            lineup_count=int(row.get("lineup_count") or 0),
            tournament_count=int(row.get("tournament_count") or 0),
            unique_player_count=int(row.get("unique_player_count") or 0),
            distinct_lineup_count=int(row.get("distinct_lineup_count") or 0),
            top_lineup_share=float(row.get("top_lineup_share") or 0.0),
            lineup_entropy=float(row.get("lineup_entropy") or 0.0),
            effective_lineups=float(row.get("effective_lineups") or 0.0),
            semantic_vector=_coerce_vector(row.get("semantic_vector")),
            identity_vector=_coerce_vector(row.get("identity_vector")),
            final_vector=_coerce_vector(row.get("final_vector")),
            top_lineup_summary=row.get("top_lineup_summary"),
            top_lineup_player_ids=tuple(
                pid
                for raw_id in (row.get("top_lineup_player_ids") or [])
                if (pid := _coerce_player_id(raw_id)) is not None
            ),
            top_lineup_player_names=tuple(
                str(player_name)
                for player_name in (row.get("top_lineup_player_names") or [])
                if player_name is not None
            ),
        )

    def load_embeddings(self, snapshot_id: int) -> List[EmbeddingRow]:
        sql_with_players = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                tournament_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary,
                top_lineup_player_ids,
                top_lineup_player_names
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
        """
        sql_legacy = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                unique_player_count,
                distinct_lineup_count,
                top_lineup_share,
                lineup_entropy,
                effective_lineups,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
        """
        sql_core = f"""
            SELECT
                team_id,
                tournament_id,
                team_name,
                event_time_ms,
                lineup_count,
                semantic_vector,
                identity_vector,
                final_vector,
                top_lineup_summary
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
        """
        with self.engine.connect() as conn:
            try:
                rows = conn.execute(
                    text(sql_with_players), {"snapshot_id": int(snapshot_id)}
                ).mappings().all()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc):
                    logger.warning(
                        "No team-search embedding table available in schema %s",
                        self.schema,
                    )
                    rows = []
                elif not _is_missing_column_error(exc):
                    raise
                else:
                    try:
                        rows = conn.execute(
                            text(sql_legacy), {"snapshot_id": int(snapshot_id)}
                        ).mappings().all()
                    except SQLAlchemyError as nested_exc:
                        if not _is_missing_column_error(nested_exc):
                            raise
                        rows = conn.execute(
                            text(sql_core), {"snapshot_id": int(snapshot_id)}
                        ).mappings().all()
        normalized_rows = []
        expected_final = None
        expected_semantic = None
        expected_identity = None

        row_dicts = [dict(row) for row in rows]
        self._hydrate_tournament_counts(snapshot_id, row_dicts)
        for r in row_dicts:
            r.setdefault("top_lineup_player_ids", [])
            r.setdefault("top_lineup_player_names", [])
            r_obj = self._row_to_embedding(r)
            final_dim = int(r_obj.final_vector.size)
            semantic_dim = int(r_obj.semantic_vector.size)
            identity_dim = int(r_obj.identity_vector.size)
            if expected_final is None:
                expected_final = final_dim
                expected_semantic = semantic_dim
                expected_identity = identity_dim
            elif (
                final_dim != expected_final
                or semantic_dim != expected_semantic
                or identity_dim != expected_identity
            ):
                logger.warning(
                    "Skipping team_id=%s due mixed embedding dimensions in snapshot %s",
                    r_obj.team_id,
                    snapshot_id,
                )
                continue

            normalized_rows.append(r_obj)
        return normalized_rows

    def match_targets(
        self, snapshot_id: int, query: str, limit: int = 25
    ) -> List[int]:
        limit = max(1, min(limit, 200))
        q = (query or "").strip().lower()
        if not q:
            return []

        normalized_q = _normalize_query(q)

        params = {
            "snapshot_id": int(snapshot_id),
            "name_like": f"%{q}%",
            "name_eq": q,
            "limit": limit,
        }
        where_clause = (
            "LOWER(team_name) = :name_eq "
            "OR LOWER(team_name) LIKE :name_like "
            "OR team_id = :team_id "
            "OR LOWER(team_id::text) = :team_id_text "
            "OR regexp_replace(LOWER(COALESCE(team_name, '')), '[^a-z0-9]+', '', 'g') = :name_norm"
        )

        team_id: Optional[int]
        try:
            team_id = int(query)
        except Exception:
            team_id = None
        params["team_id"] = team_id if team_id is not None else -1
        params["team_id_text"] = str(team_id) if team_id is not None else "__no-match__"
        params["name_norm"] = normalized_q

        sql = f"""
            SELECT team_id
            FROM {self.schema}.team_search_embeddings
            WHERE snapshot_id = :snapshot_id
              AND ({where_clause})
            ORDER BY
                CASE
                    WHEN LOWER(team_name) = :name_eq THEN 0
                    WHEN LOWER(team_id::text) = :team_id_text THEN 1
                    WHEN regexp_replace(LOWER(COALESCE(team_name, '')), '[^a-z0-9]+', '', 'g') = :name_norm THEN 2
                    ELSE 3
                END,
                lineup_count DESC
            LIMIT :limit
        """
        with self.engine.connect() as conn:
            try:
                rows = conn.execute(text(sql), params).fetchall()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                    return []
                raise
        target_ids = [int(row[0]) for row in rows]

        # Fallback path catches edge cases where normalized names are too noisy for SQL
        # matching (for example special characters or inconsistent storage values).
        if target_ids:
            return target_ids

        with self.engine.connect() as conn:
            try:
                fallback_rows = conn.execute(
                    text(
                        f"""
                        SELECT team_id, team_name
                        FROM {self.schema}.team_search_embeddings
                        WHERE snapshot_id = :snapshot_id
                        ORDER BY lineup_count DESC
                        LIMIT :fallback_limit
                    """
                    ),
                    {"snapshot_id": int(snapshot_id), "fallback_limit": limit * 10},
                ).mappings().all()
            except SQLAlchemyError as exc:
                if _is_missing_relation_error(exc) or _is_missing_column_error(exc):
                    return []
                raise

        fallback_ids: List[int] = []
        for row in fallback_rows:
            team_name = row["team_name"]
            if not team_name:
                continue
            normalized_name = _normalize_query(team_name)
            if not normalized_q:
                continue
            if len(normalized_q) <= 4:
                if normalized_q in _tokenized_query(team_name):
                    fallback_ids.append(int(row["team_id"]))
            elif normalized_q in normalized_name:
                fallback_ids.append(int(row["team_id"]))
            if len(fallback_ids) >= limit:
                break

        return fallback_ids

    def load_cluster_map(
        self, snapshot_id: int, profile: str
    ) -> Dict[int, Dict[str, Any]]:
        sql = f"""
            SELECT team_id, cluster_id, cluster_size, representative_team_name
            FROM {self.schema}.team_search_clusters
            WHERE snapshot_id = :snapshot_id
              AND profile = :profile
        """
        rows = self._fetch_rows(
            sql,
            {"snapshot_id": int(snapshot_id), "profile": profile},
            missing_default=[],
        )
        return {
            int(row["team_id"]): {
                "cluster_id": int(row["cluster_id"]),
                "cluster_size": int(row["cluster_size"]),
                "representative_team_name": row["representative_team_name"],
            }
            for row in rows
        }

    def list_clusters(
        self, snapshot_id: int, profile: str, query: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "snapshot_id": int(snapshot_id),
            "profile": profile,
        }
        where_extra = ""
        if query:
            where_extra = (
                "AND (LOWER(COALESCE(c.representative_team_name, '')) LIKE :name_like "
                "OR LOWER(COALESCE(e.team_name, '')) LIKE :name_like)"
            )
            params["name_like"] = f"%{query.lower()}%"

        sql = f"""
            SELECT
                c.cluster_id,
                c.cluster_size,
                c.representative_team_name,
                c.team_id,
                e.team_name,
                e.lineup_count,
                e.event_time_ms,
                e.tournament_id
            FROM {self.schema}.team_search_clusters c
            JOIN {self.schema}.team_search_embeddings e
              ON e.snapshot_id = c.snapshot_id
             AND e.team_id = c.team_id
            WHERE c.snapshot_id = :snapshot_id
              AND c.profile = :profile
              AND c.cluster_id >= 0
              {where_extra}
            ORDER BY c.cluster_size DESC, c.cluster_id ASC, e.lineup_count DESC
        """

        rows = self._fetch_rows(sql, params, missing_default=[])

        by_cluster: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            cid = int(row["cluster_id"])
            entry = by_cluster.setdefault(
                cid,
                {
                    "cluster_id": cid,
                    "cluster_size": int(row["cluster_size"]),
                    "representative_team_name": row["representative_team_name"],
                    "top_teams": [],
                },
            )
            top_teams = entry["top_teams"]
            if len(top_teams) < 5:
                top_teams.append(
                    {
                        "team_id": int(row["team_id"]),
                        "team_name": row["team_name"],
                        "lineup_count": int(row["lineup_count"]),
                        "tournament_id": (
                            int(row["tournament_id"])
                            if row["tournament_id"] is not None
                            else None
                        ),
                        "event_time_ms": (
                            int(row["event_time_ms"])
                            if row["event_time_ms"] is not None
                            else None
                        ),
                    }
                )

        clusters = list(by_cluster.values())
        clusters.sort(key=lambda c: (-int(c["cluster_size"]), int(c["cluster_id"])))

        for cluster in clusters:
            size = int(cluster["cluster_size"])
            cluster["stability_hint"] = (
                "stable" if size >= 8 else "rotational" if size >= 4 else "small"
            )

        return clusters[: max(1, min(limit, 200))]

    def cluster_detail(
        self,
        snapshot_id: int,
        profile: str,
        cluster_id: int,
    ) -> Optional[Dict[str, Any]]:
        sql = f"""
            SELECT
                c.cluster_id,
                c.cluster_size,
                c.representative_team_name,
                e.team_id,
                e.team_name,
                e.tournament_id,
                e.event_time_ms,
                e.lineup_count,
                e.top_lineup_summary
            FROM {self.schema}.team_search_clusters c
            JOIN {self.schema}.team_search_embeddings e
              ON e.snapshot_id = c.snapshot_id
             AND e.team_id = c.team_id
            WHERE c.snapshot_id = :snapshot_id
              AND c.profile = :profile
              AND c.cluster_id = :cluster_id
            ORDER BY e.lineup_count DESC, e.team_id ASC
        """
        rows = self._fetch_rows(
            sql,
            {
                "snapshot_id": int(snapshot_id),
                "profile": profile,
                "cluster_id": int(cluster_id),
            },
            missing_default=[],
        )

        if not rows:
            return None

        header = rows[0]
        members = []
        for row in rows:
            members.append(
                {
                    "team_id": int(row["team_id"]),
                    "team_name": row["team_name"],
                    "tournament_id": (
                        int(row["tournament_id"])
                        if row["tournament_id"] is not None
                        else None
                    ),
                    "event_time_ms": (
                        int(row["event_time_ms"])
                        if row["event_time_ms"] is not None
                        else None
                    ),
                    "lineup_count": int(row["lineup_count"]),
                    "top_lineup_summary": row["top_lineup_summary"],
                }
            )

        return {
            "cluster_id": int(header["cluster_id"]),
            "cluster_size": int(header["cluster_size"]),
            "representative_team_name": header["representative_team_name"],
            "members": members,
        }

    def get_refresh_run(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        sql = f"""
            SELECT
                run_id,
                status,
                started_at,
                finished_at,
                days_window,
                semantic_dim,
                identity_dim,
                identity_beta,
                source_since_ms,
                source_until_ms,
                teams_indexed,
                clusters_strict,
                clusters_explore,
                message
            FROM {self.schema}.team_search_refresh_runs
            WHERE run_id = :snapshot_id
            LIMIT 1
        """
        return self._fetch_first_row(sql, {"snapshot_id": int(snapshot_id)})

    def _cluster_names(self, snapshot_id: int, profile: str) -> Dict[int, str]:
        sql = f"""
            SELECT cluster_id, representative_team_name
            FROM {self.schema}.team_search_clusters
            WHERE snapshot_id = :snapshot_id
              AND profile = :profile
              AND cluster_id >= 0
        """
        rows = self._fetch_rows(
            sql,
            {"snapshot_id": int(snapshot_id), "profile": profile},
            missing_default=[],
        )
        out: Dict[int, str] = {}
        for row in rows:
            cid = int(row["cluster_id"])
            if cid not in out and row.get("representative_team_name"):
                out[cid] = str(row["representative_team_name"])
        return out

    def analytics_overview(
        self,
        *,
        snapshot_id: int,
        profile: str,
        limit_clusters: int,
        volatile_limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_overview

        embeddings = self.load_embeddings(snapshot_id)
        cluster_map = self.load_cluster_map(snapshot_id, profile)
        overview = compute_overview(
            embeddings,
            cluster_map,
            limit_clusters=limit_clusters,
            volatile_limit=volatile_limit,
        )
        run = self.get_refresh_run(snapshot_id)
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            "run": run,
            **overview,
        }

    def analytics_matchups(
        self,
        *,
        snapshot_id: int,
        profile: str,
        min_matches: int,
        limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import summarize_matchups

        sql = f"""
            SELECT
                m.match_id,
                m.team1_id,
                m.team2_id,
                m.winner_team_id,
                c1.cluster_id AS cluster_1,
                c2.cluster_id AS cluster_2
            FROM {self.schema}.matches m
            JOIN {self.schema}.team_search_clusters c1
              ON c1.snapshot_id = :snapshot_id
             AND c1.profile = :profile
             AND c1.team_id = m.team1_id
            JOIN {self.schema}.team_search_clusters c2
              ON c2.snapshot_id = :snapshot_id
             AND c2.profile = :profile
             AND c2.team_id = m.team2_id
            WHERE c1.cluster_id >= 0
              AND c2.cluster_id >= 0
              AND c1.cluster_id <> c2.cluster_id
              AND m.winner_team_id IS NOT NULL
        """
        rows = self._fetch_rows(
            sql,
            {"snapshot_id": int(snapshot_id), "profile": profile},
            missing_default=[],
        )
        cluster_names = self._cluster_names(snapshot_id, profile)
        summarized = summarize_matchups(
            rows,
            cluster_names,
            min_matches=max(1, int(min_matches)),
            limit=max(1, int(limit)),
        )
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            "total_cross_cluster_matches": len(rows),
            "matchups": summarized,
        }

    def analytics_head_to_head(
        self,
        *,
        snapshot_id: Optional[int],
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        limit: int,
    ) -> Dict[str, Any]:
        team_a = _normalize_id_sequence(team_a_ids)
        team_b = _normalize_id_sequence(team_b_ids)

        if not team_a or not team_b:
            return {
                "snapshot_id": int(snapshot_id),
                "summary": {
                    "team_a_id": None,
                    "team_b_id": None,
                    "team_a_ids": team_a,
                    "team_b_ids": team_b,
                    "team_a_name": "N/A",
                    "team_b_name": "N/A",
                    "team_a_names": [],
                    "team_b_names": [],
                    "total_matches": 0,
                    "team_a_wins": 0,
                    "team_b_wins": 0,
                    "unresolved_matches": 0,
                    "decided_matches": 0,
                    "team_a_win_rate": 0.0,
                    "team_b_win_rate": 0.0,
                    "tournaments": 0,
                    "tournament_tier_distribution": {
                        "X": 0,
                        "S+": 0,
                        "S": 0,
                        "A+": 0,
                        "A": 0,
                        "A-": 0,
                        "Unscored": 0,
                    },
                    "tournament_tier_match_distribution": {
                        "X": 0,
                        "S+": 0,
                        "S": 0,
                        "A+": 0,
                        "A": 0,
                        "A-": 0,
                        "Unscored": 0,
                    },
                },
                "matches": [],
            }

        team_a_set = set(team_a)
        team_b_set = set(team_b)
        team_a_ids_sorted = sorted(team_a_set)
        team_b_ids_sorted = sorted(team_b_set)

        if snapshot_id is not None:
            since_ts, until_ts = self._fetch_tournament_window(snapshot_id)
        else:
            since_ts, until_ts = None, None
        match_last_game = self._seconds_expr("m.last_game_finished_at_ms")
        match_created = self._seconds_expr("m.created_at_ms")
        tournament_start = self._seconds_expr("t.start_time_ms")
        match_time_expr = f"COALESCE({match_last_game}, {match_created}, {tournament_start})"
        match_time_expr_without_tournament = f"COALESCE({match_last_game}, {match_created})"

        filters_with_tournament: list[str] = []
        filters_without_tournament: list[str] = []
        params: Dict[str, object] = {
            "snapshot_id": int(snapshot_id) if snapshot_id is not None else None,
            "team_a_ids": team_a_ids_sorted,
            "team_b_ids": team_b_ids_sorted,
            "limit": max(1, int(limit)),
        }

        score_columns = self._resolve_match_score_columns()
        if score_columns is not None:
            score_col_a, score_col_b = score_columns
            score_select = f"""
                    CASE
                        WHEN m.team1_id IN :team_a_ids THEN m.{score_col_a}
                        ELSE m.{score_col_b}
                      END AS team_a_score,
                    CASE
                        WHEN m.team1_id IN :team_a_ids THEN m.{score_col_b}
                        ELSE m.{score_col_a}
                      END AS team_b_score
            """.strip()
        else:
            score_select = None

        winner_score_select = "m.winner_team_id"
        if score_select:
            winner_score_select = f"""
                {winner_score_select},
                {score_select}
            """.strip()

        def _safe_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        if since_ts is not None:
            filters_with_tournament.append(f"{match_time_expr} >= :source_since_ts")
            filters_without_tournament.append(
                f"{match_time_expr_without_tournament} >= :source_since_ts"
            )
            params["source_since_ts"] = int(since_ts)

        if until_ts is not None:
            filters_with_tournament.append(f"{match_time_expr} <= :source_until_ts")
            filters_without_tournament.append(
                f"{match_time_expr_without_tournament} <= :source_until_ts"
            )
            params["source_until_ts"] = int(until_ts)

        match_where = " AND " + " AND ".join(filters_with_tournament) if filters_with_tournament else ""
        match_where_no_tournament = (
            " AND " + " AND ".join(filters_without_tournament)
            if filters_without_tournament
            else ""
        )

        def _run_head_to_head_query(where_with_tournament: str, where_without_tournament: str) -> list[Dict[str, object]]:
            sql_with_tournament_name = f"""
                SELECT
                    m.match_id,
                    m.tournament_id,
                    t.tournament_name,
                    m.team1_id,
                    m.team2_id,
                    {winner_score_select}
                    ,
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END AS event_time_ms
                FROM {self.schema}.matches m
                LEFT JOIN {self.schema}.tournaments t
                  ON t.tournament_id = m.tournament_id
                WHERE ((m.team1_id IN :team_a_ids AND m.team2_id IN :team_b_ids)
                       OR (m.team1_id IN :team_b_ids AND m.team2_id IN :team_a_ids))
                  {where_with_tournament}
                ORDER BY
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END DESC NULLS LAST
                LIMIT :limit
            """

            sql_with_tournament_name_column = f"""
                SELECT
                    m.match_id,
                    m.tournament_id,
                    t.name AS tournament_name,
                    m.team1_id,
                    m.team2_id,
                    {winner_score_select}
                    ,
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END AS event_time_ms
                FROM {self.schema}.matches m
                LEFT JOIN {self.schema}.tournaments t
                  ON t.tournament_id = m.tournament_id
                WHERE ((m.team1_id IN :team_a_ids AND m.team2_id IN :team_b_ids)
                       OR (m.team1_id IN :team_b_ids AND m.team2_id IN :team_a_ids))
                  {where_with_tournament}
                ORDER BY
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        WHEN t.start_time_ms IS NOT NULL THEN t.start_time_ms
                        ELSE NULL
                    END DESC NULLS LAST
                LIMIT :limit
            """

            sql_without_tournament = f"""
                SELECT
                    m.match_id,
                    m.tournament_id,
                    NULL::text AS tournament_name,
                    m.team1_id,
                    m.team2_id,
                    {winner_score_select}
                    ,
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        ELSE NULL
                    END AS event_time_ms
                FROM {self.schema}.matches m
                WHERE ((m.team1_id IN :team_a_ids AND m.team2_id IN :team_b_ids)
                       OR (m.team1_id IN :team_b_ids AND m.team2_id IN :team_a_ids))
                  {where_without_tournament}
                ORDER BY
                    CASE
                        WHEN m.last_game_finished_at_ms IS NOT NULL THEN m.last_game_finished_at_ms
                        WHEN m.created_at_ms IS NOT NULL THEN m.created_at_ms
                        ELSE NULL
                    END DESC NULLS LAST
                LIMIT :limit
            """

            query_results: list[Dict[str, object]] = []
            for query in (
                sql_with_tournament_name,
                sql_with_tournament_name_column,
                sql_without_tournament,
            ):
                try:
                    with self.engine.connect() as conn:
                        query_results = [
                            dict(row)
                            for row in conn.execute(
                                text(query).bindparams(
                                    bindparam("team_a_ids", expanding=True),
                                    bindparam("team_b_ids", expanding=True),
                                ),
                                params,
                            ).mappings().all()
                        ]
                    break
                except SQLAlchemyError as exc:
                    if not (_is_missing_relation_error(exc) or _is_missing_column_error(exc)):
                        raise

            return query_results

        rows = _run_head_to_head_query(match_where, match_where_no_tournament)
        snapshot_window_enabled = bool(since_ts is not None or until_ts is not None)
        window_match_count = len(rows)
        fallback_to_all_time = False
        if not rows and snapshot_window_enabled:
            rows = _run_head_to_head_query("", "")

            if rows:
                fallback_to_all_time = True
                logger.info(
                    "head-to-head fallback to all-time matchups after snapshot window returned no matches"
                    " (snapshot_id=%s, team_a_count=%s, team_b_count=%s)",
                    snapshot_id,
                    len(team_a_ids_sorted),
                    len(team_b_ids_sorted),
                )

        match_rosters = self._fetch_match_rosters(rows, team_a_ids_sorted, team_b_ids_sorted)
        tournament_scores = self._fetch_tournament_scores(
            [
                int(row["tournament_id"])
                for row in rows
                if row.get("tournament_id") is not None
            ]
        )
        team_name_rows = []
        if snapshot_id is not None:
            try:
                with self.engine.connect() as conn:
                    team_name_rows = conn.execute(
                        text(
                            f"""
                            SELECT team_id, team_name
                              FROM {self.schema}.team_search_embeddings
                             WHERE snapshot_id = :snapshot_id
                               AND team_id IN :team_ids
                            """
                        ).bindparams(bindparam("team_ids", expanding=True)),
                        {
                            "snapshot_id": int(snapshot_id),
                            "team_ids": team_a_ids_sorted + team_b_ids_sorted,
                        },
                    ).mappings().all()
            except SQLAlchemyError as exc:
                if not (_is_missing_relation_error(exc) or _is_missing_column_error(exc)):
                    raise

        team_names = {
            int(row["team_id"]): str(row["team_name"])
            for row in team_name_rows
            if row is not None and row.get("team_id") is not None
        }

        summary = {
            "team_a_id": team_a_ids_sorted[0],
            "team_b_id": team_b_ids_sorted[0],
            "team_a_ids": team_a_ids_sorted,
            "team_b_ids": team_b_ids_sorted,
            "team_a_names": [team_names.get(team_id, f"Team {team_id}") for team_id in team_a_ids_sorted],
            "team_b_names": [team_names.get(team_id, f"Team {team_id}") for team_id in team_b_ids_sorted],
            "team_a_name": None,
            "team_b_name": None,
            "total_matches": 0,
            "team_a_wins": 0,
            "team_b_wins": 0,
            "unresolved_matches": 0,
            "decided_matches": 0,
            "team_a_win_rate": 0.0,
            "team_b_win_rate": 0.0,
            "tournaments": 0,
            "tournament_tier_distribution": {},
            "tournament_tier_match_distribution": {},
            "snapshot_window_applied": bool(snapshot_id is not None and snapshot_window_enabled),
            "matched_all_time": bool(
                snapshot_id is not None
                and snapshot_window_enabled
                and window_match_count == 0
                and fallback_to_all_time
            ),
        }
        summary["team_a_name"] = summary["team_a_names"][0] if summary["team_a_names"] else "Team n/a"
        summary["team_b_name"] = summary["team_b_names"][0] if summary["team_b_names"] else "Team n/a"

        tournaments_seen: set[int] = set()
        tournaments_by_tier: Dict[str, int] = {}
        matches_by_tier: Dict[str, int] = {}
        tournament_tier_by_id: Dict[int, dict[str, str]] = {}
        matchups = []
        for row in rows:
            tournament_id = (
                int(row["tournament_id"]) if row["tournament_id"] is not None else None
            )
            winner_team_id = (
                int(row["winner_team_id"]) if row["winner_team_id"] is not None else None
            )
            team_a_score = _safe_float(row.get("team_a_score"))
            team_b_score = _safe_float(row.get("team_b_score"))
            roster_key = (int(row["match_id"]), tournament_id or 0)
            roster = match_rosters.get(
                roster_key,
                {"team_a": {"player_ids": [], "player_names": []}, "team_b": {"player_ids": [], "player_names": []}},
            )
            team_a_roster_ids = list(roster.get("team_a", {}).get("player_ids", []))
            team_a_roster_names = list(roster.get("team_a", {}).get("player_names", []))
            team_b_roster_ids = list(roster.get("team_b", {}).get("player_ids", []))
            team_b_roster_names = list(roster.get("team_b", {}).get("player_names", []))

            side = None
            if winner_team_id in team_a_set:
                summary["team_a_wins"] += 1
                side = "team_a"
            elif winner_team_id in team_b_set:
                summary["team_b_wins"] += 1
                side = "team_b"
            elif (
                team_a_score is not None
                and team_b_score is not None
                and team_a_score != team_b_score
            ):
                if team_a_score > team_b_score:
                    summary["team_a_wins"] += 1
                    side = "team_a"
                else:
                    summary["team_b_wins"] += 1
                    side = "team_b"
            else:
                summary["unresolved_matches"] += 1

            tournament_strength = tournament_scores.get(tournament_id)
            tier = _tournament_tier(tournament_strength)
            if tournament_id is not None:
                tournaments_seen.add(tournament_id)
                previous_tier = tournament_tier_by_id.get(int(tournament_id))
                if previous_tier is None:
                    tournaments_by_tier[tier["tier_id"]] = tournaments_by_tier.get(
                        tier["tier_id"], 0
                    ) + 1
                    tournament_tier_by_id[int(tournament_id)] = tier
                elif previous_tier["tier_id"] != tier["tier_id"]:
                    tournaments_by_tier[previous_tier["tier_id"]] = max(
                        0, tournaments_by_tier.get(previous_tier["tier_id"], 0) - 1
                    )
                    tournaments_by_tier[tier["tier_id"]] = tournaments_by_tier.get(
                        tier["tier_id"], 0
                    ) + 1
                    tournament_tier_by_id[int(tournament_id)] = tier

            matches_by_tier[tier["tier_id"]] = matches_by_tier.get(tier["tier_id"], 0) + 1

            matchups.append(
                {
                    "match_id": int(row["match_id"]),
                    "tournament_id": tournament_id,
                    "tournament_name": row["tournament_name"],
                    "tournament_score": (
                        None if tournament_strength is None else round(float(tournament_strength), 4)
                    ),
                    "tournament_score_tier_id": tier["tier_id"],
                    "tournament_score_tier": tier["tier_label"],
                    "winner_team_id": winner_team_id,
                    "winner_side": side,
                    "team_a_score": team_a_score,
                    "team_b_score": team_b_score,
                    "team_a_roster": [
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                        }
                        for player_id, player_name in zip(
                            team_a_roster_ids,
                            team_a_roster_names,
                        )
                    ],
                    "team_b_roster": [
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                        }
                        for player_id, player_name in zip(
                            team_b_roster_ids,
                            team_b_roster_names,
                        )
                    ],
                    "event_time_ms": (
                        int(row["event_time_ms"]) if row["event_time_ms"] is not None else None
                    ),
                    "team_a_is_winner": bool(side == "team_a"),
                    "team_b_is_winner": bool(side == "team_b"),
                }
            )

        summary["total_matches"] = len(rows)
        summary["tournaments"] = len(tournaments_seen)
        summary["tournament_tier_distribution"] = {
            "X": tournaments_by_tier.get("x", 0),
            "S+": tournaments_by_tier.get("s_plus", 0),
            "S": tournaments_by_tier.get("s", 0),
            "A+": tournaments_by_tier.get("a_plus", 0),
            "A": tournaments_by_tier.get("a", 0),
            "A-": tournaments_by_tier.get("a_minus", 0),
            "Unscored": tournaments_by_tier.get("unscored", 0),
        }
        summary["tournament_tier_match_distribution"] = {
            "X": matches_by_tier.get("x", 0),
            "S+": matches_by_tier.get("s_plus", 0),
            "S": matches_by_tier.get("s", 0),
            "A+": matches_by_tier.get("a_plus", 0),
            "A": matches_by_tier.get("a", 0),
            "A-": matches_by_tier.get("a_minus", 0),
            "Unscored": matches_by_tier.get("unscored", 0),
        }

        if summary["total_matches"] > 0:
            decided = summary["team_a_wins"] + summary["team_b_wins"]
            summary["decided_matches"] = decided
            if decided > 0:
                summary["team_a_win_rate"] = round(summary["team_a_wins"] / decided, 4)
                summary["team_b_win_rate"] = round(summary["team_b_wins"] / decided, 4)
            else:
                summary["team_a_win_rate"] = 0.0
                summary["team_b_win_rate"] = 0.0

        return {
            "snapshot_id": snapshot_id,
            "summary": summary,
            "matches": matchups,
        }

    def analytics_roster_diversity(
        self,
        *,
        snapshot_id: int,
        profile: str,
        min_similarity: float,
        max_player_overlap: float,
        min_cluster_size: int,
        limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_roster_diversity_candidates

        embeddings = self.load_embeddings(snapshot_id)
        cluster_map = self.load_cluster_map(snapshot_id, profile)
        return compute_roster_diversity_candidates(
            embeddings=embeddings,
            cluster_map=cluster_map,
            min_similarity=float(min_similarity),
            max_player_overlap=float(max_player_overlap),
            min_cluster_size=max(2, int(min_cluster_size)),
            limit=max(1, int(limit)),
        )

    def analytics_team_lab(
        self,
        *,
        snapshot_id: int,
        profile: str,
        team_id: int,
        neighbors: int,
    ) -> Optional[Dict[str, Any]]:
        from team_api.analytics_logic import build_team_lab

        embeddings = self.load_embeddings(snapshot_id)
        cluster_map = self.load_cluster_map(snapshot_id, profile)

        sql = f"""
            SELECT
                CASE
                    WHEN m.team1_id = :team_id THEN m.team2_id
                    ELSE m.team1_id
                END AS opponent_team_id,
                CASE
                    WHEN m.winner_team_id = :team_id THEN 1
                    ELSE 0
                END AS is_win
            FROM {self.schema}.matches m
            WHERE (m.team1_id = :team_id OR m.team2_id = :team_id)
              AND m.winner_team_id IS NOT NULL
        """
        match_rows = self._fetch_rows(
            sql,
            {"team_id": int(team_id)},
            missing_default=[],
        )

        result = build_team_lab(
            team_id=int(team_id),
            embeddings=embeddings,
            cluster_map=cluster_map,
            matches=match_rows,
            neighbors=max(1, int(neighbors)),
        )
        if result is None:
            return None
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            **result,
        }

    def analytics_team_blend(
        self,
        *,
        snapshot_id: int,
        profile: str,
        team_id: int,
        semantic_weight: float,
        neighbors: int,
    ) -> Optional[Dict[str, Any]]:
        from team_api.analytics_logic import build_blended_neighbors

        embeddings = self.load_embeddings(snapshot_id)
        cluster_map = self.load_cluster_map(snapshot_id, profile)
        result = build_blended_neighbors(
            team_id=int(team_id),
            embeddings=embeddings,
            cluster_map=cluster_map,
            semantic_weight=float(semantic_weight),
            neighbors=max(1, int(neighbors)),
        )
        if result is None:
            return None
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            **result,
        }

    def analytics_outliers(
        self,
        *,
        snapshot_id: int,
        profile: str,
        limit: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_outliers

        embeddings = self.load_embeddings(snapshot_id)
        cluster_map = self.load_cluster_map(snapshot_id, profile)
        outliers = compute_outliers(
            embeddings,
            cluster_map,
            limit=max(1, int(limit)),
        )
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            "count": len(outliers),
            "outliers": outliers,
        }

    def analytics_space(
        self,
        *,
        snapshot_id: int,
        profile: str,
        max_points: int,
    ) -> Dict[str, Any]:
        from team_api.analytics_logic import compute_space_projection

        embeddings = self.load_embeddings(snapshot_id)
        cluster_map = self.load_cluster_map(snapshot_id, profile)
        projection = compute_space_projection(
            embeddings,
            cluster_map,
            max_points=max(50, min(int(max_points), 3000)),
        )
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            **projection,
        }

    def analytics_drift(
        self,
        *,
        profile: str,
        current_snapshot_id: Optional[int],
        previous_snapshot_id: Optional[int],
        top_movers: int,
    ) -> Optional[Dict[str, Any]]:
        from team_api.analytics_logic import compute_snapshot_drift

        snapshots = self.list_completed_snapshots(limit=10)
        if not snapshots:
            return None

        if current_snapshot_id is None:
            current_snapshot_id = int(snapshots[0]["run_id"])

        if previous_snapshot_id is None:
            previous_snapshot_id = None
            for snap in snapshots:
                rid = int(snap["run_id"])
                if rid != int(current_snapshot_id):
                    previous_snapshot_id = rid
                    break

        if previous_snapshot_id is None:
            return {
                "current_snapshot_id": int(current_snapshot_id),
                "previous_snapshot_id": None,
                "summary": {
                    "shared_teams": 0,
                    "new_teams": 0,
                    "dropped_teams": 0,
                    "cluster_switches": 0,
                    "newly_clustered": 0,
                    "newly_noise": 0,
                    "avg_embedding_drift": 0.0,
                    "p90_embedding_drift": 0.0,
                },
                "top_embedding_movers": [],
                "top_volatility_shifts": [],
            }

        current_embeddings = self.load_embeddings(int(current_snapshot_id))
        previous_embeddings = self.load_embeddings(int(previous_snapshot_id))
        current_cluster_map = self.load_cluster_map(
            int(current_snapshot_id), profile
        )
        previous_cluster_map = self.load_cluster_map(
            int(previous_snapshot_id), profile
        )

        drift = compute_snapshot_drift(
            current_snapshot_id=int(current_snapshot_id),
            previous_snapshot_id=int(previous_snapshot_id),
            current_embeddings=current_embeddings,
            previous_embeddings=previous_embeddings,
            current_cluster_map=current_cluster_map,
            previous_cluster_map=previous_cluster_map,
            top_movers=max(1, int(top_movers)),
        )
        return {
            "cluster_mode": profile,
            **drift,
        }
