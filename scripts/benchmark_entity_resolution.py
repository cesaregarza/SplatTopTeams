from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import create_engine

from team_api.search_logic import (
    build_player_query_profile,
    consolidate_ranked_results,
    rescore_consolidated_results,
)
from team_api.store import EmbeddingRow, TeamSearchStore
from team_jobs.refresh_embeddings import (
    TeamPayload,
    _build_identity_vectors,
    _build_payloads,
    _build_semantic_vectors,
)


def _lineups(*lineups: Iterable[int], repeat: int = 1) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for lineup in lineups:
        normalized = tuple(int(player_id) for player_id in lineup)
        for _ in range(max(1, int(repeat))):
            out.append(normalized)
    return out


def _fixture_payloads() -> list[TeamPayload]:
    metadata: Dict[int, Dict[str, object]] = {}
    player_name_by_id: Dict[int, str] = {}
    team_lineups: Dict[int, List[tuple[int, ...]]] = {}
    lineup_counts: Dict[int, Counter[tuple[int, ...]]] = {}
    team_player_counts: Dict[int, Dict[int, int]] = {}
    team_tournament_ids: Dict[int, set[int]] = {}

    def add_team(
        team_id: int,
        team_name: str,
        event_time_ms: int,
        tournament_id: int,
        lineups: Sequence[tuple[int, ...]],
    ) -> None:
        metadata[team_id] = {
            "tournament_id": int(tournament_id),
            "team_name": team_name,
            "event_time_ms": int(event_time_ms),
        }
        team_lineups[team_id] = list(lineups)
        counter: Counter[tuple[int, ...]] = Counter()
        player_counts: Dict[int, int] = defaultdict(int)
        for lineup in lineups:
            counter[tuple(lineup)] += 1
            for player_id in lineup:
                player_counts[int(player_id)] += 1
                player_name_by_id.setdefault(int(player_id), f"Player {int(player_id)}")
        lineup_counts[team_id] = counter
        team_player_counts[team_id] = dict(player_counts)
        team_tournament_ids[team_id] = {int(tournament_id)}

    add_team(101, "Core Exact", 1_700_000_000_000, 1, _lineups((1, 2, 3, 4), repeat=12))
    add_team(102, "Core Alias", 1_700_100_000_000, 2, _lineups((1, 2, 3, 5), repeat=8) + _lineups((1, 2, 3, 4), repeat=3))
    add_team(
        201,
        "Roster Seven A",
        1_700_200_000_000,
        3,
        _lineups((10, 11, 12, 13), repeat=5)
        + _lineups((10, 11, 12, 14), repeat=4)
        + _lineups((10, 11, 15, 16), repeat=4)
        + _lineups((10, 14, 15, 16), repeat=3),
    )
    add_team(
        202,
        "Roster Seven B",
        1_700_300_000_000,
        4,
        _lineups((10, 11, 12, 13), repeat=4)
        + _lineups((10, 11, 12, 15), repeat=4)
        + _lineups((10, 11, 14, 16), repeat=4)
        + _lineups((10, 13, 15, 16), repeat=3),
    )
    add_team(301, "Generic Sub A", 1_700_400_000_000, 5, _lineups((21, 22, 23, 99), repeat=10))
    add_team(302, "Generic Sub B", 1_700_500_000_000, 6, _lineups((31, 32, 33, 99), repeat=10))
    add_team(401, "Core Four", 1_700_600_000_000, 7, _lineups((41, 42, 43, 44), repeat=10))
    add_team(402, "Star Sub", 1_700_700_000_000, 8, _lineups((41, 42, 43, 77), repeat=8) + _lineups((41, 42, 43, 44), repeat=2))
    add_team(
        501,
        "Historic A",
        1_500_000_000_000,
        9,
        _lineups((51, 52, 53, 54), repeat=8) + _lineups((51, 52, 53, 55), repeat=6),
    )
    add_team(
        502,
        "Historic B",
        1_710_000_000_000,
        10,
        _lineups((51, 52, 53, 54), repeat=8) + _lineups((51, 52, 53, 56), repeat=5),
    )
    add_team(
        701,
        "Large Ten A",
        1_700_800_000_000,
        11,
        _lineups((100, 101, 102, 103), repeat=4)
        + _lineups((100, 101, 104, 105), repeat=4)
        + _lineups((100, 106, 107, 108), repeat=4)
        + _lineups((101, 106, 108, 109), repeat=3),
    )
    add_team(
        702,
        "Large Ten B",
        1_700_900_000_000,
        12,
        _lineups((100, 101, 102, 104), repeat=4)
        + _lineups((100, 103, 105, 106), repeat=4)
        + _lineups((101, 107, 108, 109), repeat=4)
        + _lineups((102, 104, 106, 108), repeat=3),
    )
    add_team(
        801,
        "Large Fifteen A",
        1_701_000_000_000,
        13,
        _lineups((200, 201, 202, 203), repeat=4)
        + _lineups((200, 204, 205, 206), repeat=4)
        + _lineups((207, 208, 209, 210), repeat=4)
        + _lineups((211, 212, 213, 214), repeat=4)
        + _lineups((200, 207, 211, 214), repeat=3),
    )
    add_team(
        802,
        "Large Fifteen B",
        1_701_100_000_000,
        14,
        _lineups((200, 201, 204, 207), repeat=4)
        + _lineups((202, 205, 208, 211), repeat=4)
        + _lineups((203, 206, 209, 212), repeat=4)
        + _lineups((210, 213, 214, 200), repeat=4)
        + _lineups((201, 205, 209, 213), repeat=3),
    )
    add_team(601, "No Bridge A", 1_701_200_000_000, 15, _lineups((61, 62, 63, 64), repeat=8))
    add_team(602, "No Bridge B", 1_701_300_000_000, 16, _lineups((71, 72, 73, 74), repeat=8))

    semantic_vectors = _build_semantic_vectors(team_lineups, 64)
    identity_vectors = _build_identity_vectors(team_player_counts, 256, 3.0)
    return _build_payloads(
        sorted(team_lineups.keys()),
        metadata,
        player_name_by_id,
        team_lineups,
        lineup_counts,
        team_player_counts,
        team_tournament_ids,
        semantic_vectors,
        identity_vectors,
        3.0,
    )


def _payload_to_row(payload: TeamPayload) -> EmbeddingRow:
    return EmbeddingRow(
        team_id=int(payload.team_id),
        tournament_id=payload.tournament_id,
        team_name=payload.team_name,
        event_time_ms=payload.event_time_ms,
        lineup_count=int(payload.lineup_count),
        tournament_count=int(payload.tournament_count),
        unique_player_count=int(payload.unique_player_count),
        distinct_lineup_count=int(payload.distinct_lineup_count),
        top_lineup_share=float(payload.top_lineup_share),
        lineup_entropy=float(payload.lineup_entropy),
        effective_lineups=float(payload.effective_lineups),
        semantic_vector=payload.semantic_vector,
        identity_vector=payload.identity_vector,
        final_vector=payload.final_vector,
        top_lineup_summary=payload.top_lineup_summary,
        top_lineup_player_ids=tuple(payload.top_lineup_player_ids),
        top_lineup_player_names=tuple(payload.top_lineup_player_names),
        roster_player_ids=tuple(payload.roster_player_ids),
        roster_player_names=tuple(payload.roster_player_names),
        roster_player_match_counts=tuple(payload.roster_player_match_counts),
        player_support={
            int(player_id): float(support)
            for player_id, support in payload.player_support.items()
        },
        pair_support={
            tuple(int(part) for part in key.split("|", 1)): float(support)
            for key, support in payload.pair_support.items()
        },
    )


def _build_store() -> tuple[TeamSearchStore, list[EmbeddingRow]]:
    payloads = _fixture_payloads()
    rows = [_payload_to_row(payload) for payload in payloads]
    store = TeamSearchStore(create_engine("sqlite:///:memory:"), "comp_rankings")
    return store, rows


def _finalize_results(
    ranked: Dict[str, object],
    *,
    query_profile: dict[str, object] | None,
    cluster_scoring: bool,
    weights: tuple[float, float, float],
    use_pair_rerank: bool,
) -> list[dict[str, object]]:
    results = consolidate_ranked_results(
        list(ranked.get("results") or []),
        min_overlap=0.72,
    )
    if cluster_scoring and query_profile is not None:
        results = rescore_consolidated_results(
            results,
            query_profile=query_profile,
            rerank_weight_embed=weights[0],
            rerank_weight_player=weights[1],
            rerank_weight_pair=weights[2],
            use_pair_rerank=use_pair_rerank,
        )
    return results


def _top_hit_rank(results: Sequence[dict[str, object]], expected_ids: set[int]) -> int | None:
    for idx, result in enumerate(results, start=1):
        team_ids = set(int(team_id) for team_id in (result.get("consolidated_team_ids") or []))
        if not team_ids:
            team_ids = {int(result.get("team_id") or 0)}
        if team_ids & expected_ids:
            return idx
    return None


def _run_mode(
    store: TeamSearchStore,
    rows: list[EmbeddingRow],
    *,
    mode: str,
    player_ids: Sequence[int],
    top_n: int = 5,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cache_entry = store._build_snapshot_cache_entry(1, rows)
    store._get_cached_snapshot_entry = lambda snapshot_id: cache_entry  # type: ignore[assignment]
    cluster_map: dict[int, dict[str, object]] = {}

    if mode == "subset_enum_baseline":
        query_rows, query_team_weights, metadata = store._project_seed_players_to_proxy_teams(
            1,
            player_ids,
        )
        if not query_rows or not query_team_weights:
            return [], {
                "subset_enumeration_count": int(
                    metadata.get("seed_player_projection_subset_count", 0)
                )
            }
        ranked = store._rank_similar_teams(
            embeddings=cache_entry.rows,
            target_team_ids=[int(row.team_id) for row in query_rows],
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=0.0,
            cache_entry=cache_entry,
            query_team_weights=query_team_weights,
            rerank_candidate_limit=top_n,
            rerank_weight_embed=1.0,
            rerank_weight_player=0.0,
            rerank_weight_pair=0.0,
            use_pair_rerank=False,
        )
        return _finalize_results(
            ranked,
            query_profile=None,
            cluster_scoring=False,
            weights=(1.0, 0.0, 0.0),
            use_pair_rerank=False,
        ), {
            "subset_enumeration_count": int(
                metadata.get("seed_player_projection_subset_count", 0)
            )
        }

    query_profile = build_player_query_profile(
        player_ids,
        semantic_dim=cache_entry.semantic_dim,
        identity_dim=cache_entry.identity_dim,
        identity_beta=cache_entry.identity_beta,
        idf_lookup=cache_entry.idf_lookup,
    )
    if mode == "whole_set_vector_only":
        weights = (1.0, 0.0, 0.0)
        use_pair = False
        cluster_scoring = False
    elif mode == "whole_set_player_rerank":
        weights = (0.85, 0.15, 0.0)
        use_pair = False
        cluster_scoring = False
    elif mode == "whole_set_player_pair_rerank":
        weights = (0.70, 0.15, 0.15)
        use_pair = True
        cluster_scoring = False
    elif mode == "whole_set_player_pair_cluster":
        weights = (0.70, 0.15, 0.15)
        use_pair = True
        cluster_scoring = True
    else:
        raise ValueError(f"unknown mode: {mode}")

    ranked = store._rank_similar_teams(
        embeddings=cache_entry.rows,
        target_team_ids=[],
        cluster_map=cluster_map,
        top_n=top_n,
        min_relevance=0.0,
        cache_entry=cache_entry,
        query_profile=query_profile,
        rerank_candidate_limit=200,
        rerank_weight_embed=weights[0],
        rerank_weight_player=weights[1],
        rerank_weight_pair=weights[2],
        use_pair_rerank=use_pair,
    )
    return _finalize_results(
        ranked,
        query_profile=query_profile,
        cluster_scoring=cluster_scoring,
        weights=weights,
        use_pair_rerank=use_pair,
    ), {"subset_enumeration_count": 0}


def _measure_latency(
    store: TeamSearchStore,
    rows: list[EmbeddingRow],
    mode: str,
    player_ids: Sequence[int],
    *,
    repeats: int,
) -> float:
    start = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        _run_mode(store, rows, mode=mode, player_ids=player_ids, top_n=5)
    elapsed = time.perf_counter() - start
    return (elapsed * 1000.0) / float(max(1, int(repeats)))


def build_report() -> str:
    store, rows = _build_store()
    modes = [
        "subset_enum_baseline",
        "whole_set_vector_only",
        "whole_set_player_rerank",
        "whole_set_player_pair_rerank",
        "whole_set_player_pair_cluster",
    ]
    queries = [
        {"name": "4-player exact lineup", "player_ids": [1, 2, 3, 4], "expected_ids": {101}},
        {"name": "Partial 7-player roster", "player_ids": [10, 11, 12, 13, 14, 15, 16], "expected_ids": {201, 202}},
        {"name": "Generic substitute", "player_ids": [21, 22, 23, 99], "expected_ids": {301}},
        {"name": "Star substitute", "player_ids": [41, 42, 43, 77], "expected_ids": {402}},
        {"name": "Historical reunion", "player_ids": [51, 52, 53, 54, 55], "expected_ids": {501, 502}},
        {"name": "Large 10-player roster", "player_ids": list(range(100, 110)), "expected_ids": {701, 702}},
        {"name": "Large 15-player roster", "player_ids": list(range(200, 215)), "expected_ids": {801, 802}},
    ]
    no_bridge_query = {"name": "No-bridge case", "player_ids": [61, 62, 71, 72]}

    summary_rows: list[dict[str, object]] = []
    latency_queries = {
        4: [1, 2, 3, 4],
        7: [10, 11, 12, 13, 14, 15, 16],
        10: list(range(100, 110)),
        15: list(range(200, 215)),
    }

    per_query_rows: list[dict[str, object]] = []
    for mode in modes:
        ranks: list[int] = []
        subset_counts: list[int] = []
        for query in queries:
            results, metadata = _run_mode(
                store,
                rows,
                mode=mode,
                player_ids=query["player_ids"],
                top_n=5,
            )
            rank = _top_hit_rank(results, set(query["expected_ids"]))
            if rank is not None:
                ranks.append(rank)
            subset_counts.append(int(metadata.get("subset_enumeration_count", 0)))
            top_result = results[0] if results else {}
            per_query_rows.append(
                {
                    "mode": mode,
                    "query": query["name"],
                    "top_team_id": top_result.get("team_id"),
                    "top_team_name": top_result.get("team_name"),
                    "rank": rank,
                }
            )

        recall_at_1 = sum(1 for rank in ranks if rank == 1) / float(len(queries))
        recall_at_5 = sum(1 for rank in ranks if rank is not None and rank <= 5) / float(len(queries))
        mrr = sum(1.0 / rank for rank in ranks) / float(len(queries))
        latencies = {
            size: _measure_latency(store, rows, mode, query_player_ids, repeats=40)
            for size, query_player_ids in latency_queries.items()
        }
        summary_rows.append(
            {
                "mode": mode,
                "recall_at_1": recall_at_1,
                "recall_at_5": recall_at_5,
                "mrr": mrr,
                "latencies": latencies,
                "subset_count": max(subset_counts) if subset_counts else 0,
            }
        )

    no_bridge_results, _ = _run_mode(
        store,
        rows,
        mode="whole_set_player_pair_cluster",
        player_ids=no_bridge_query["player_ids"],
        top_n=5,
    )
    no_bridge_top = no_bridge_results[0] if no_bridge_results else {}

    lines = [
        "# Entity Resolution Benchmark",
        "",
        "Synthetic fixture benchmark for the deterministic team-resolution pipeline.",
        "",
        "## Summary Metrics",
        "",
        "| Mode | Recall@1 | Recall@5 | MRR | Latency 4p (ms) | Latency 7p (ms) | Latency 10p (ms) | Latency 15p (ms) | Subset Enums |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {mode} | {r1:.3f} | {r5:.3f} | {mrr:.3f} | {l4:.3f} | {l7:.3f} | {l10:.3f} | {l15:.3f} | {subset_count} |".format(
                mode=row["mode"],
                r1=row["recall_at_1"],
                r5=row["recall_at_5"],
                mrr=row["mrr"],
                l4=row["latencies"][4],
                l7=row["latencies"][7],
                l10=row["latencies"][10],
                l15=row["latencies"][15],
                subset_count=row["subset_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Query Top Result",
            "",
            "| Mode | Query | Top Team ID | Top Team Name | First Correct Rank |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for row in per_query_rows:
        lines.append(
            "| {mode} | {query} | {top_team_id} | {top_team_name} | {rank} |".format(
                mode=row["mode"],
                query=row["query"],
                top_team_id=row["top_team_id"] if row["top_team_id"] is not None else "",
                top_team_name=row["top_team_name"] if row["top_team_name"] is not None else "",
                rank=row["rank"] if row["rank"] is not None else "",
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `subset_enum_baseline` exercises the old proxy-team projection path and is the only mode with non-zero subset enumeration counts.",
            "- `whole_set_player_pair_cluster` is the new target configuration: whole-set dense retrieval, exact player+pair rerank, then post-consolidation group rescoring.",
            "- Historical reunion queries are evaluated with no temporal penalty.",
            "- No-bridge case qualitative check: `whole_set_player_pair_cluster` top result is `{name}` (team `{team_id}`) with score `{score:.3f}`; this fixture is retained to verify the pipeline does not force a global merge.".format(
                name=no_bridge_top.get("team_name", ""),
                team_id=no_bridge_top.get("team_id", ""),
                score=float(no_bridge_top.get("sim_to_query", 0.0) or 0.0),
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark deterministic entity resolution modes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    args = parser.parse_args()

    report = build_report()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
