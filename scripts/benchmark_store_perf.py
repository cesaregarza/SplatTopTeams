from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, Sequence

from sqlalchemy import create_engine

from shared_lib.db import resolve_database_url
from team_api.store import TeamSearchStore


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _measure(name: str, fn: Callable[[], object], repeats: int) -> None:
    timings: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        timings.append(_ms(start))
    print(
        f"{name:32s} "
        f"mean={statistics.mean(timings):8.1f}ms "
        f"p50={statistics.median(timings):8.1f}ms "
        f"min={min(timings):8.1f}ms "
        f"max={max(timings):8.1f}ms"
    )


def _run_search_benchmark(
    store: TeamSearchStore,
    snapshot_id: int,
    query: str,
    repeats: int,
) -> None:
    def _search() -> object:
        return store.search_similar_teams(
            snapshot_id=snapshot_id,
            query=query,
            top_n=20,
            min_relevance=0.8,
            cluster_mode="explore",
            include_clusters=True,
            consolidate=True,
            consolidate_min_overlap=0.8,
        )

    _measure(f"team-search[{query}]", _search, repeats=repeats)


def _run_analytics_benchmark(
    store: TeamSearchStore,
    snapshot_id: int,
    repeats: int,
) -> None:
    _measure(
        "analytics-overview",
        lambda: store.analytics_overview(
            snapshot_id=snapshot_id,
            profile="explore",
            limit_clusters=20,
            volatile_limit=15,
        ),
        repeats=repeats,
    )
    _measure(
        "analytics-space",
        lambda: store.analytics_space(
            snapshot_id=snapshot_id,
            profile="explore",
            max_points=900,
        ),
        repeats=repeats,
    )
    _measure(
        "analytics-outliers",
        lambda: store.analytics_outliers(
            snapshot_id=snapshot_id,
            profile="explore",
            limit=30,
        ),
        repeats=repeats,
    )
    _measure(
        "analytics-roster-overlap",
        lambda: store.analytics_roster_diversity(
            snapshot_id=snapshot_id,
            profile="explore",
            min_similarity=0.82,
            max_player_overlap=0.3,
            min_cluster_size=2,
            limit=20,
        ),
        repeats=repeats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TeamSearchStore search/analytics latency."
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://postgres:postgres@localhost:5432/rankings_db?sslmode=disable",
        help="Database URL for the benchmark target.",
    )
    parser.add_argument(
        "--schema",
        default="comp_rankings",
        help="Schema containing team_search tables.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["FTW", "Moonlight", "BlankZ"],
        help="Search queries (space-separated and/or comma-delimited).",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=3,
        help="Warm-request repeats per benchmark call.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_url = resolve_database_url(args.db_url)
    engine = create_engine(db_url, future=True)
    store = TeamSearchStore(engine=engine, schema=args.schema)

    snapshot = store.latest_snapshot()
    if not snapshot:
        raise SystemExit("No completed snapshot found in target schema.")

    snapshot_id = int(snapshot["run_id"])
    queries: Sequence[str] = []
    for raw_query in args.queries:
        queries.extend(
            q.strip() for q in str(raw_query).split(",") if q.strip()
        )
    if not queries:
        raise SystemExit("No queries provided.")

    print(f"snapshot_id={snapshot_id} schema={args.schema}")
    print("--- cold pass ---")
    for query in queries:
        _run_search_benchmark(store, snapshot_id, query, repeats=1)
    _run_analytics_benchmark(store, snapshot_id, repeats=1)

    print("--- warm pass ---")
    for query in queries:
        _run_search_benchmark(store, snapshot_id, query, repeats=args.warm_repeats)
    _run_analytics_benchmark(store, snapshot_id, repeats=args.warm_repeats)


if __name__ == "__main__":
    main()
