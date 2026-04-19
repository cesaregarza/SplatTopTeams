from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import create_engine

from team_api.entity_resolution_reference import payload_to_reference_row
from team_api.search_logic import (
    build_player_query_profile,
    consolidate_ranked_results,
    rescore_consolidated_results,
)
from team_api.store import EmbeddingSnapshotCacheEntry, TeamSearchStore
from team_jobs.refresh_embeddings import (
    TeamPayload,
    _build_identity_vectors,
    _build_payloads,
    _build_semantic_vectors,
)

_HEALBOOK_RERANK_WEIGHTS = (0.70, 0.15, 0.15)
_HEALBOOK_CONSOLIDATE_MIN_OVERLAP = 0.72


@dataclass(frozen=True)
class HealbookQueryCase:
    name: str
    player_ids: tuple[int, ...]
    top_team_id: int
    required_top_group_ids: frozenset[int]
    forbidden_top_group_ids: frozenset[int]
    expect_consolidated: bool


HEALBOOK_QUERY_CASES: tuple[HealbookQueryCase, ...] = (
    HealbookQueryCase(
        name="core exact",
        player_ids=(277, 6048, 10881, 22614),
        top_team_id=46624,
        required_top_group_ids=frozenset({46624, 49106}),
        forbidden_top_group_ids=frozenset({47527, 55021, 65620}),
        expect_consolidated=True,
    ),
    HealbookQueryCase(
        name="core grapeyy",
        player_ids=(157, 277, 6048, 22614),
        top_team_id=54749,
        required_top_group_ids=frozenset({54634, 54749}),
        forbidden_top_group_ids=frozenset({47527, 55021, 65620}),
        expect_consolidated=True,
    ),
    HealbookQueryCase(
        name="bridgeish variant",
        player_ids=(1954, 277, 10881, 22614),
        top_team_id=31888,
        required_top_group_ids=frozenset({31888, 46624, 49106}),
        forbidden_top_group_ids=frozenset({47527, 55021, 65620}),
        expect_consolidated=True,
    ),
    HealbookQueryCase(
        name="five-player core",
        player_ids=(157, 277, 6048, 10881, 22614),
        top_team_id=54634,
        required_top_group_ids=frozenset({46624, 49106, 54634, 54749}),
        forbidden_top_group_ids=frozenset({47527, 55021, 65620}),
        expect_consolidated=True,
    ),
    HealbookQueryCase(
        name="devade branch",
        player_ids=(1954, 7898, 10403, 27966),
        top_team_id=47527,
        required_top_group_ids=frozenset({47527}),
        forbidden_top_group_ids=frozenset({31888, 46624, 49106, 54634, 54749}),
        expect_consolidated=False,
    ),
)


def _lineups(*lineups: Iterable[int], repeat: int = 1) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for lineup in lineups:
        normalized = tuple(int(player_id) for player_id in lineup)
        for _ in range(max(1, int(repeat))):
            out.append(normalized)
    return out


def build_healbook_reference_payloads() -> list[TeamPayload]:
    metadata: Dict[int, Dict[str, object]] = {}
    player_name_by_id: Dict[int, str] = {}
    team_lineups: Dict[int, List[tuple[int, ...]]] = {}
    lineup_counts: Dict[int, Counter[tuple[int, ...]]] = {}
    team_player_counts: Dict[int, Dict[int, int]] = {}
    team_tournament_ids: Dict[int, set[int]] = {}

    def add_team(
        team_id: int,
        team_name: str,
        tournament_id: int,
        lineups: Sequence[tuple[int, ...]],
    ) -> None:
        metadata[team_id] = {
            "tournament_id": int(tournament_id),
            "team_name": team_name,
            "event_time_ms": int(tournament_id),
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

    add_team(
        31888,
        "WE as in HealBook",
        1675,
        _lineups((1954, 277, 10881, 22614), repeat=3)
        + _lineups((1954, 277, 6048, 22614), repeat=2)
        + _lineups((3181, 277, 10881, 22614), repeat=1),
    )
    add_team(
        44029,
        "Healbook",
        2465,
        _lineups((277, 6048, 19981, 22614), repeat=4)
        + _lineups((277, 10881, 19981, 22614), repeat=2),
    )
    add_team(
        46624,
        "Healbook",
        2609,
        _lineups((277, 6048, 10881, 22614), repeat=6),
    )
    add_team(
        47527,
        "Healbook",
        2738,
        _lineups((1954, 7898, 10403, 27966), repeat=5),
    )
    add_team(
        49106,
        "Healbook",
        2802,
        _lineups((277, 6048, 10881, 22614), repeat=3)
        + _lineups((590, 6048, 10881, 22614), repeat=1),
    )
    add_team(
        54634,
        "Healbook",
        3061,
        _lineups((157, 277, 6048, 22614), repeat=3)
        + _lineups((157, 6048, 10881, 22614), repeat=2),
    )
    add_team(
        54749,
        "Healbook",
        3075,
        _lineups((157, 277, 6048, 22614), repeat=7),
    )
    add_team(
        55021,
        "Healbook",
        2564,
        _lineups((3181, 7794, 7898, 10403), repeat=4),
    )
    add_team(
        55478,
        "Healbook",
        3076,
        _lineups((277, 2801, 6048, 22614), repeat=6),
    )
    add_team(
        56304,
        "Healbook",
        3222,
        _lineups((157, 277, 7794, 22614), repeat=4),
    )
    add_team(
        57988,
        "Healbook",
        3224,
        _lineups((157, 277, 10881, 19981), repeat=5),
    )
    add_team(
        65620,
        "Healbook (SBR Edition)",
        3581,
        _lineups((1954, 11233, 19981, 27966), repeat=3),
    )

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


def build_healbook_reference_store() -> tuple[TeamSearchStore, EmbeddingSnapshotCacheEntry]:
    rows = [
        payload_to_reference_row(payload)
        for payload in build_healbook_reference_payloads()
    ]
    store = TeamSearchStore(create_engine("sqlite:///:memory:"), "comp_rankings")
    cache_entry = store._build_snapshot_cache_entry(1, rows)
    return store, cache_entry


def run_healbook_reference_query(
    store: TeamSearchStore,
    cache_entry: EmbeddingSnapshotCacheEntry,
    player_ids: Sequence[int],
    *,
    top_n: int = 5,
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
        rerank_weight_embed=_HEALBOOK_RERANK_WEIGHTS[0],
        rerank_weight_player=_HEALBOOK_RERANK_WEIGHTS[1],
        rerank_weight_pair=_HEALBOOK_RERANK_WEIGHTS[2],
        use_pair_rerank=True,
    )
    results = consolidate_ranked_results(
        list(ranked.get("results") or []),
        min_overlap=_HEALBOOK_CONSOLIDATE_MIN_OVERLAP,
    )
    rescored = rescore_consolidated_results(
        results,
        query_profile=query_profile,
        rerank_weight_embed=_HEALBOOK_RERANK_WEIGHTS[0],
        rerank_weight_player=_HEALBOOK_RERANK_WEIGHTS[1],
        rerank_weight_pair=_HEALBOOK_RERANK_WEIGHTS[2],
        use_pair_rerank=True,
    )
    return rescored[:top_n]
