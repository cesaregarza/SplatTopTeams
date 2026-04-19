from __future__ import annotations

import math
from hashlib import blake2b
from itertools import combinations
from typing import Dict, Iterable, Sequence


def canonicalize_player_ids(values: Iterable[object]) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            player_id = int(value)
        except (TypeError, ValueError):
            continue
        if player_id <= 0 or player_id in seen:
            continue
        seen.add(player_id)
        out.append(player_id)
    out.sort()
    return tuple(out)


def hash_value(player_id: int, salt: str) -> int:
    digest = blake2b(
        f"{int(player_id)}|{salt}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def hash_index(player_id: int, dim: int, salt: str) -> int:
    if int(dim) <= 0:
        raise ValueError("dim must be positive")
    return int(hash_value(player_id, salt) % int(dim))


def hash_sign(player_id: int, salt: str) -> float:
    return 1.0 if (hash_value(player_id, salt) & 1) == 0 else -1.0


def build_identity_idf_lookup(
    team_player_groups: Iterable[Iterable[object]],
    *,
    idf_cap: float,
) -> Dict[int, float]:
    normalized_groups = [
        canonicalize_player_ids(group)
        for group in team_player_groups
    ]
    groups_with_players = [group for group in normalized_groups if group]
    n_teams = max(1, len(groups_with_players))

    player_team_count: Dict[int, int] = {}
    for group in groups_with_players:
        for player_id in group:
            player_team_count[player_id] = player_team_count.get(player_id, 0) + 1

    idf: Dict[int, float] = {}
    for player_id, team_count in player_team_count.items():
        value = math.log((n_teams + 1) / (int(team_count) + 1))
        if float(idf_cap) > 0:
            value = min(value, float(idf_cap))
        idf[int(player_id)] = float(value)
    return idf


def unordered_player_pairs(player_ids: Sequence[object]) -> tuple[tuple[int, int], ...]:
    normalized = canonicalize_player_ids(player_ids)
    return tuple(
        (int(left), int(right))
        for left, right in combinations(normalized, 2)
    )


def pair_key(left: int, right: int) -> str:
    a, b = sorted((int(left), int(right)))
    return f"{a}|{b}"


def parse_pair_key(value: object) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    left, sep, right = text.partition("|")
    if not sep:
        return None
    try:
        a = int(left)
        b = int(right)
    except (TypeError, ValueError):
        return None
    if a <= 0 or b <= 0:
        return None
    if a <= b:
        return (a, b)
    return (b, a)


def lineup_key(player_ids: Sequence[object]) -> str:
    canonical = canonicalize_player_ids(player_ids)
    return "|".join(str(player_id) for player_id in canonical)


def parse_lineup_key(value: object) -> tuple[int, ...] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split("|")]
    if not parts:
        return None
    parsed: list[int] = []
    for part in parts:
        if not part:
            return None
        try:
            player_id = int(part)
        except (TypeError, ValueError):
            return None
        if player_id <= 0:
            return None
        parsed.append(player_id)
    canonical = canonicalize_player_ids(parsed)
    if len(canonical) != len(parsed):
        return None
    return canonical
