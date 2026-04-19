from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import create_engine

from team_api.search_logic import (
    build_player_query_profile,
    consolidate_ranked_results,
    rescore_consolidated_results,
)
from team_api.store import EmbeddingRow, EmbeddingSnapshotCacheEntry, TeamSearchStore
from team_jobs.refresh_embeddings import (
    TeamPayload,
    _build_identity_vectors,
    _build_payloads,
    _build_semantic_vectors,
)

_REFERENCE_RERANK_WEIGHTS = (0.70, 0.15, 0.15)
_REFERENCE_TOP_N = 5
_REFERENCE_CONSOLIDATE_MIN_OVERLAP = 0.72


@dataclass(frozen=True)
class ReferenceQueryCase:
    name: str
    player_ids: tuple[int, ...]
    expected_ids: frozenset[int]
    expected_consolidated_ids: frozenset[int] | None = None


REFERENCE_QUERY_CASES: tuple[ReferenceQueryCase, ...] = (
    ReferenceQueryCase(
        name="4-player exact lineup",
        player_ids=(1, 2, 3, 4),
        expected_ids=frozenset({101}),
        expected_consolidated_ids=frozenset({101, 102}),
    ),
    ReferenceQueryCase(
        name="Partial 7-player roster",
        player_ids=(10, 11, 12, 13, 14, 15, 16),
        expected_ids=frozenset({201, 202}),
        expected_consolidated_ids=frozenset({201, 202}),
    ),
    ReferenceQueryCase(
        name="Generic substitute",
        player_ids=(21, 22, 23, 99),
        expected_ids=frozenset({301}),
    ),
    ReferenceQueryCase(
        name="Star substitute",
        player_ids=(41, 42, 43, 77),
        expected_ids=frozenset({402}),
    ),
    ReferenceQueryCase(
        name="Historical reunion",
        player_ids=(51, 52, 53, 54, 55),
        expected_ids=frozenset({501, 502}),
        expected_consolidated_ids=frozenset({501, 502}),
    ),
    ReferenceQueryCase(
        name="Large 10-player roster",
        player_ids=tuple(range(100, 110)),
        expected_ids=frozenset({701, 702}),
        expected_consolidated_ids=frozenset({701, 702}),
    ),
    ReferenceQueryCase(
        name="Large 15-player roster",
        player_ids=tuple(range(200, 215)),
        expected_ids=frozenset({801, 802}),
    ),
)

REFERENCE_NO_BRIDGE_CASE = ReferenceQueryCase(
    name="No-bridge case",
    player_ids=(61, 62, 71, 72),
    expected_ids=frozenset({601}),
)


def _lineups(*lineups: Iterable[int], repeat: int = 1) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for lineup in lineups:
        normalized = tuple(int(player_id) for player_id in lineup)
        for _ in range(max(1, int(repeat))):
            out.append(normalized)
    return out


def build_reference_payloads() -> list[TeamPayload]:
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
    add_team(
        102,
        "Core Alias",
        1_700_100_000_000,
        2,
        _lineups((1, 2, 3, 5), repeat=8) + _lineups((1, 2, 3, 4), repeat=3),
    )
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
    add_team(
        402,
        "Star Sub",
        1_700_700_000_000,
        8,
        _lineups((41, 42, 43, 77), repeat=8) + _lineups((41, 42, 43, 44), repeat=2),
    )
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


def payload_to_reference_row(payload: TeamPayload) -> EmbeddingRow:
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
        lineup_variant_counts={
            tuple(int(part) for part in key.split("|")): int(count)
            for key, count in payload.lineup_variant_counts.items()
        },
    )


def build_reference_rows() -> list[EmbeddingRow]:
    return [
        payload_to_reference_row(payload)
        for payload in build_reference_payloads()
    ]


def build_reference_store() -> tuple[TeamSearchStore, EmbeddingSnapshotCacheEntry]:
    rows = build_reference_rows()
    store = TeamSearchStore(create_engine("sqlite:///:memory:"), "comp_rankings")
    cache_entry = store._build_snapshot_cache_entry(1, rows)
    return store, cache_entry


def run_reference_query(
    store: TeamSearchStore,
    cache_entry: EmbeddingSnapshotCacheEntry,
    player_ids: Sequence[int],
    *,
    top_n: int = _REFERENCE_TOP_N,
) -> list[dict[str, object]]:
    candidate_top_n = store._consolidation_candidate_top_n(top_n, 200)
    query_profile = build_player_query_profile(
        player_ids,
        semantic_dim=cache_entry.semantic_dim,
        identity_dim=cache_entry.identity_dim,
        identity_beta=cache_entry.identity_beta,
        idf_lookup=cache_entry.idf_lookup,
    )
    ranked = store._rank_similar_teams(
        embeddings=cache_entry.rows,
        target_team_ids=[],
        cluster_map={},
        top_n=top_n,
        candidate_top_n=candidate_top_n,
        min_relevance=0.0,
        cache_entry=cache_entry,
        query_profile=query_profile,
        rerank_candidate_limit=200,
        rerank_weight_embed=_REFERENCE_RERANK_WEIGHTS[0],
        rerank_weight_player=_REFERENCE_RERANK_WEIGHTS[1],
        rerank_weight_pair=_REFERENCE_RERANK_WEIGHTS[2],
        use_pair_rerank=True,
    )
    results = consolidate_ranked_results(
        list(ranked.get("results") or []),
        min_overlap=_REFERENCE_CONSOLIDATE_MIN_OVERLAP,
    )
    rescored = rescore_consolidated_results(
        results,
        query_profile=query_profile,
        rerank_weight_embed=_REFERENCE_RERANK_WEIGHTS[0],
        rerank_weight_player=_REFERENCE_RERANK_WEIGHTS[1],
        rerank_weight_pair=_REFERENCE_RERANK_WEIGHTS[2],
        use_pair_rerank=True,
    )
    return rescored[:top_n]


def top_hit_rank(
    results: Sequence[dict[str, object]],
    expected_ids: frozenset[int],
) -> int | None:
    for idx, result in enumerate(results, start=1):
        team_ids = set(
            int(team_id)
            for team_id in (result.get("consolidated_team_ids") or [])
        )
        if not team_ids:
            team_ids = {int(result.get("team_id") or 0)}
        if team_ids & expected_ids:
            return idx
    return None
