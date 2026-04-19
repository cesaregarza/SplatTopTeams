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

_REAL_FAMILY_RERANK_WEIGHTS = (0.70, 0.15, 0.15)
_REAL_FAMILY_CONSOLIDATE_MIN_OVERLAP = 0.72


@dataclass(frozen=True)
class RealFamilyQueryCase:
    name: str
    family: str
    player_ids: tuple[int, ...]
    top_team_id: int
    required_top_group_ids: frozenset[int]
    forbidden_top_group_ids: frozenset[int]
    expect_consolidated: bool


REAL_FAMILY_IDS: dict[str, frozenset[int]] = {
    "blankz": frozenset({65217, 64208, 63586, 54583, 53652}),
    "distinkt": frozenset({64112, 60719, 59620, 56064, 54521, 54222}),
    "reload": frozenset({65225, 60352, 59687, 59584, 54619}),
    "seapunks": frozenset({23257, 23521, 24855, 22724, 24043, 26359}),
    "tidy_tidings": frozenset({23410, 23708, 25355, 33916, 42930, 45975, 60502}),
    "wave_breaker": frozenset({7911, 12445, 19829, 17204, 24859, 4807, 4413}),
    "operation_zero": frozenset({14390, 16426, 18132, 19709, 23857, 25295, 26391}),
}


def _ids_except(*families: str) -> frozenset[int]:
    skip = set(families)
    out: set[int] = set()
    for family, team_ids in REAL_FAMILY_IDS.items():
        if family in skip:
            continue
        out.update(int(team_id) for team_id in team_ids)
    return frozenset(out)


REAL_FAMILY_QUERY_CASES: tuple[RealFamilyQueryCase, ...] = (
    RealFamilyQueryCase(
        name="BlankZ core",
        family="blankz",
        player_ids=(3930, 12526, 28822, 34511),
        top_team_id=65217,
        required_top_group_ids=frozenset({65217, 64208, 63586}),
        forbidden_top_group_ids=_ids_except("blankz"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="BlankZ devil branch",
        family="blankz",
        player_ids=(3930, 12526, 28822, 29321),
        top_team_id=54583,
        required_top_group_ids=frozenset({54583}),
        forbidden_top_group_ids=_ids_except("blankz"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Distinkt core",
        family="distinkt",
        player_ids=(3545, 7007, 7440, 30220),
        top_team_id=60719,
        required_top_group_ids=frozenset({60719, 64112}),
        forbidden_top_group_ids=_ids_except("distinkt"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Distinkt jas0n branch",
        family="distinkt",
        player_ids=(894, 3545, 7007, 7440),
        top_team_id=56064,
        required_top_group_ids=frozenset({56064, 60719, 64112}),
        forbidden_top_group_ids=_ids_except("distinkt"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Reload cinnamon core",
        family="reload",
        player_ids=(8565, 17130, 28687, 32272),
        top_team_id=65225,
        required_top_group_ids=frozenset({65225, 54619}),
        forbidden_top_group_ids=_ids_except("reload"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Reload RedAccentz core",
        family="reload",
        player_ids=(8565, 17130, 20671, 28687),
        top_team_id=60352,
        required_top_group_ids=frozenset({60352, 59687, 54619}),
        forbidden_top_group_ids=_ids_except("reload"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Seapunks koa core",
        family="seapunks",
        player_ids=(9163, 13390, 30575, 37128),
        top_team_id=23257,
        required_top_group_ids=frozenset({23257, 23521}),
        forbidden_top_group_ids=_ids_except("seapunks"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Seapunks rexy core",
        family="seapunks",
        player_ids=(9163, 14605, 30575, 37128),
        top_team_id=22724,
        required_top_group_ids=frozenset({22724, 24855}),
        forbidden_top_group_ids=_ids_except("seapunks"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Seapunks eider branch",
        family="seapunks",
        player_ids=(9163, 10841, 12280, 14605),
        top_team_id=24043,
        required_top_group_ids=frozenset({24043}),
        forbidden_top_group_ids=_ids_except("seapunks"),
        expect_consolidated=False,
    ),
    RealFamilyQueryCase(
        name="Tidy Tidings core",
        family="tidy_tidings",
        player_ids=(11085, 26807, 37000, 37835),
        top_team_id=23410,
        required_top_group_ids=frozenset({23410, 33916, 42930, 45975}),
        forbidden_top_group_ids=_ids_except("tidy_tidings"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Tidy Tidings koooo branch",
        family="tidy_tidings",
        player_ids=(11085, 21063, 26807, 37000),
        top_team_id=23708,
        required_top_group_ids=frozenset({23708, 25355}),
        forbidden_top_group_ids=_ids_except("tidy_tidings"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Wave Breaker core",
        family="wave_breaker",
        player_ids=(2670, 19093, 34414, 36265),
        top_team_id=7911,
        required_top_group_ids=frozenset({7911, 12445}),
        forbidden_top_group_ids=_ids_except("wave_breaker"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Wave Breaker drag branch",
        family="wave_breaker",
        player_ids=(2670, 19093, 25168, 36265),
        top_team_id=19829,
        required_top_group_ids=frozenset({19829, 17204, 24859}),
        forbidden_top_group_ids=_ids_except("wave_breaker"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Operation Zero core",
        family="operation_zero",
        player_ids=(34461, 34511, 35458, 37068),
        top_team_id=14390,
        required_top_group_ids=frozenset({14390, 16426, 18132, 19709}),
        forbidden_top_group_ids=_ids_except("operation_zero"),
        expect_consolidated=True,
    ),
    RealFamilyQueryCase(
        name="Operation Zero cherry branch",
        family="operation_zero",
        player_ids=(28233, 34461, 34511, 35458),
        top_team_id=23857,
        required_top_group_ids=frozenset({23857, 25295}),
        forbidden_top_group_ids=_ids_except("operation_zero"),
        expect_consolidated=True,
    ),
)


def _lineups(*lineups: Iterable[int], repeat: int = 1) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for lineup in lineups:
        normalized = tuple(int(player_id) for player_id in lineup)
        for _ in range(max(1, int(repeat))):
            out.append(normalized)
    return out


def build_real_family_reference_payloads() -> list[TeamPayload]:
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

    add_team(65217, "BlankZ", 3560, _lineups((3930, 12526, 28822, 34511), repeat=6))
    add_team(64208, "BlankZ", 3542, _lineups((3930, 12526, 28822, 34511), repeat=5))
    add_team(
        63586,
        "BlankZ",
        3318,
        _lineups((3930, 12526, 28822, 34511), repeat=3)
        + _lineups((11506, 12526, 28822, 34511), repeat=2),
    )
    add_team(54583, "BlankZ", 3054, _lineups((3930, 12526, 28822, 29321), repeat=4))
    add_team(53652, "BlankZ", 3073, _lineups((18, 12526, 28822, 34511), repeat=4))

    add_team(64112, "Distinkt", 3559, _lineups((3545, 7007, 7440, 30220), repeat=4))
    add_team(60719, "Distinkt", 3356, _lineups((3545, 7007, 7440, 30220), repeat=4))
    add_team(59620, "Distinkt", 3354, _lineups((3545, 7007, 30220, 41038), repeat=3))
    add_team(56064, "Distinkt", 3195, _lineups((894, 3545, 7007, 7440), repeat=5))
    add_team(54521, "Distinkt", 3054, _lineups((3545, 7007, 7440, 25797), repeat=4))
    add_team(54222, "Distinkt", 3074, _lineups((3545, 7007, 7440, 37955), repeat=4))

    add_team(65225, "Reload", 3543, _lineups((8565, 17130, 28687, 32272), repeat=4))
    add_team(60352, "Reload", 3353, _lineups((8565, 17130, 20671, 28687), repeat=2))
    add_team(
        59687,
        "Reload",
        3314,
        _lineups((8565, 17130, 20671, 28687), repeat=3)
        + _lineups((8565, 20671, 28687, 32272), repeat=1),
    )
    add_team(59584, "Reload", 3319, _lineups((8565, 20671, 28687, 32272), repeat=5))
    add_team(
        54619,
        "Reload",
        3061,
        _lineups((8565, 17130, 28687, 32272), repeat=2)
        + _lineups((8565, 20671, 28687, 32272), repeat=2)
        + _lineups((8565, 17130, 20671, 28687), repeat=1),
    )

    add_team(23257, "Seapunks!", 1134, _lineups((9163, 13390, 30575, 37128), repeat=4))
    add_team(
        23521,
        "Seapunks!",
        1091,
        _lineups((9163, 13390, 30575, 37128), repeat=3)
        + _lineups((9163, 14605, 30575, 37128), repeat=1),
    )
    add_team(
        24855,
        "Seapunks!",
        1146,
        _lineups((9163, 14605, 30575, 37128), repeat=4)
        + _lineups((2300, 9163, 30575, 37128), repeat=1)
        + _lineups((14605, 30575, 34724, 37128), repeat=1),
    )
    add_team(
        22724,
        "Seapunks!",
        1081,
        _lineups((9163, 14605, 30575, 37128), repeat=3)
        + _lineups((9163, 10841, 30575, 37128), repeat=1),
    )
    add_team(24043, "Seapunks!", 1137, _lineups((9163, 10841, 12280, 14605), repeat=5))
    add_team(
        26359,
        "Seapunks!",
        1301,
        _lineups((7783, 9163, 14605, 30575), repeat=4)
        + _lineups((7783, 9163, 12280, 30575), repeat=1),
    )

    add_team(23410, "Tidy Tidings", 1140, _lineups((11085, 26807, 37000, 37835), repeat=6))
    add_team(
        23708,
        "Tidy Tidings",
        1145,
        _lineups((11085, 21063, 26807, 37000), repeat=6)
        + _lineups((21063, 26807, 37000, 37835), repeat=1),
    )
    add_team(
        25355,
        "Tidy Tidings",
        1161,
        _lineups((11085, 21063, 26807, 37835), repeat=4)
        + _lineups((21063, 26807, 37000, 37835), repeat=1),
    )
    add_team(33916, "Tidy Tidings", 2106, _lineups((11085, 26807, 37000, 37835), repeat=5))
    add_team(
        42930,
        "Tidy Tidings",
        2645,
        _lineups((11085, 26807, 37000, 37835), repeat=4)
        + _lineups((9342, 11085, 26807, 37000), repeat=1),
    )
    add_team(
        45975,
        "Tidy Tidings",
        2848,
        _lineups((11085, 26807, 37000, 37173), repeat=4)
        + _lineups((11085, 26807, 37000, 37835), repeat=3),
    )
    add_team(60502, "Tidy Tidings", 3348, _lineups((11085, 20154, 26807, 37000), repeat=5))

    add_team(7911, "Wave Breaker", 656, _lineups((2670, 19093, 34414, 36265), repeat=8))
    add_team(12445, "Wave Breaker", 789, _lineups((2670, 19093, 34414, 36265), repeat=5))
    add_team(19829, "Wave Breaker", 1024, _lineups((2670, 19093, 25168, 36265), repeat=8))
    add_team(17204, "Wave Breaker", 946, _lineups((2670, 25168, 36265, 41936), repeat=6))
    add_team(
        24859,
        "Wave Breaker",
        1184,
        _lineups((19093, 25168, 36265, 41936), repeat=5)
        + _lineups((19093, 25168, 26354, 41936), repeat=1),
    )
    add_team(4807, "Wave Breaker", 354, _lineups((2670, 8413, 19093, 41936), repeat=6))
    add_team(
        4413,
        "Wave Breaker",
        319,
        _lineups((2670, 8413, 19093, 41936), repeat=4)
        + _lineups((2670, 8413, 19093, 34414), repeat=1),
    )

    add_team(14390, "Operation Zero", 886, _lineups((34461, 34511, 35458, 37068), repeat=6))
    add_team(16426, "Operation Zero", 917, _lineups((34461, 34511, 35458, 37068), repeat=4))
    add_team(18132, "Operation Zero", 980, _lineups((34461, 34511, 35458, 37068), repeat=5))
    add_team(19709, "Operation Zero", 1004, _lineups((34461, 34511, 35458, 37068), repeat=5))
    add_team(23857, "Operation Zero", 1130, _lineups((28233, 34461, 34511, 35458), repeat=5))
    add_team(
        25295,
        "Operation Zero",
        1160,
        _lineups((28233, 34461, 34511, 35458), repeat=4)
        + _lineups((28233, 34461, 34511, 37068), repeat=2)
        + _lineups((34461, 34511, 35458, 37068), repeat=1),
    )
    add_team(26391, "Operation Zero", 1305, _lineups((10268, 34511, 35458, 37068), repeat=6))

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


def build_real_family_reference_store() -> tuple[TeamSearchStore, EmbeddingSnapshotCacheEntry]:
    rows = [
        payload_to_reference_row(payload)
        for payload in build_real_family_reference_payloads()
    ]
    store = TeamSearchStore(create_engine("sqlite:///:memory:"), "comp_rankings")
    cache_entry = store._build_snapshot_cache_entry(1, rows)
    return store, cache_entry


def run_real_family_reference_query(
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
        rerank_weight_embed=_REAL_FAMILY_RERANK_WEIGHTS[0],
        rerank_weight_player=_REAL_FAMILY_RERANK_WEIGHTS[1],
        rerank_weight_pair=_REAL_FAMILY_RERANK_WEIGHTS[2],
        use_pair_rerank=True,
    )
    results = consolidate_ranked_results(
        list(ranked.get("results") or []),
        min_overlap=_REAL_FAMILY_CONSOLIDATE_MIN_OVERLAP,
    )
    rescored = rescore_consolidated_results(
        results,
        query_profile=query_profile,
        rerank_weight_embed=_REAL_FAMILY_RERANK_WEIGHTS[0],
        rerank_weight_player=_REAL_FAMILY_RERANK_WEIGHTS[1],
        rerank_weight_pair=_REAL_FAMILY_RERANK_WEIGHTS[2],
        use_pair_rerank=True,
    )
    return rescored[:top_n]
