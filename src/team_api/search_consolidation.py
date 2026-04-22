from __future__ import annotations

from typing import Any, Dict, List

from shared_lib.team_vector_utils import canonicalize_player_ids, parse_lineup_key
from team_api.store import EmbeddingRow


def _parse_player_ids(result: Dict[str, object]) -> tuple[int, ...]:
    raw = result.get("top_lineup_player_ids")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        values = raw.split(",")
        parsed = []
        for value in values:
            text = value.strip().strip('"').strip("'")
            if not text:
                continue
            try:
                parsed.append(int(text))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(set(parsed)))
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(out)))


def _parse_player_set(result: Dict[str, object]) -> set[int]:
    return set(_parse_player_ids(result))


def _normalize_player_name(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return "Unknown Player" if not text else text


def _parse_player_id_sequence(result: Dict[str, object]) -> tuple[int, ...]:
    raw = result.get("top_lineup_player_ids")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        values = raw.split(",")
        parsed: list[int] = []
        for value in values:
            text = value.strip().strip('"').strip("'")
            if not text:
                continue
            try:
                parsed.append(int(text))
            except (TypeError, ValueError):
                continue
        return tuple(dict.fromkeys(parsed))

    parsed: list[int] = []
    for value in raw:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(dict.fromkeys(parsed))


def _parse_player_name_sequence(result: Dict[str, object]) -> tuple[str, ...]:
    raw = result.get("top_lineup_player_names")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        values = raw.split(",")
        return tuple(_normalize_player_name(value) for value in values)
    return tuple(_normalize_player_name(value) for value in raw)


def _parse_top_lineup_summary_count(value: object) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    x_idx = raw.find("x")
    if x_idx <= 0:
        return None
    prefix = raw[:x_idx].strip()
    try:
        parsed = int(prefix)
    except (TypeError, ValueError):
        return None
    return parsed


def _coerce_lineup_signature(value: object) -> tuple[int, ...]:
    if isinstance(value, tuple):
        try:
            parsed = tuple(int(part) for part in value)
        except (TypeError, ValueError):
            return ()
        canonical = canonicalize_player_ids(parsed)
        return canonical if len(canonical) == len(parsed) else ()
    if isinstance(value, list):
        try:
            parsed = tuple(int(part) for part in value)
        except (TypeError, ValueError):
            return ()
        canonical = canonicalize_player_ids(parsed)
        return canonical if len(canonical) == len(parsed) else ()
    parsed_key = parse_lineup_key(value)
    return parsed_key or ()


def _estimate_top_lineup_matches_fallback(result: Dict[str, object]) -> int:
    parsed_summary = _parse_top_lineup_summary_count(result.get("top_lineup_summary"))
    if parsed_summary is not None:
        return max(0, parsed_summary)

    lineup_count = int(result.get("lineup_count") or 0)
    if lineup_count <= 0:
        return 0

    share = float(result.get("top_lineup_share") or 0.0)
    if share <= 0.0:
        return 0

    count = int(round(lineup_count * share))
    return 1 if count <= 0 else count


def _extract_lineup_variant_counts(result: Dict[str, object]) -> Dict[tuple[int, ...], int]:
    raw_row = result.get("_embedding_row")
    if isinstance(raw_row, EmbeddingRow) and raw_row.lineup_variant_counts:
        return {
            tuple(signature): int(count)
            for signature, count in raw_row.lineup_variant_counts.items()
            if signature and int(count) > 0
        }

    raw_counts = result.get("_lineup_variant_counts")
    if isinstance(raw_counts, dict):
        out: Dict[tuple[int, ...], int] = {}
        for raw_key, raw_value in raw_counts.items():
            signature = _coerce_lineup_signature(raw_key)
            if not signature:
                continue
            try:
                count = int(raw_value)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            out[signature] = out.get(signature, 0) + int(count)
        if out:
            return out

    signature = _parse_player_id_sequence(result)
    top_matches = _estimate_top_lineup_matches_fallback(result)
    if signature and top_matches > 0:
        return {signature: int(top_matches)}
    return {}


def _select_top_lineup_variant(
    lineup_variant_counts: Dict[tuple[int, ...], int],
) -> tuple[tuple[int, ...], int]:
    if not lineup_variant_counts:
        return (), 0
    signature, count = max(
        lineup_variant_counts.items(),
        key=lambda item: (int(item[1]), len(item[0]), item[0]),
    )
    return tuple(signature), int(count)


def _estimate_top_lineup_matches(result: Dict[str, object]) -> int:
    lineup_variant_counts = _extract_lineup_variant_counts(result)
    if lineup_variant_counts:
        _, count = _select_top_lineup_variant(lineup_variant_counts)
        if count > 0:
            return int(count)

    parsed_summary = _parse_top_lineup_summary_count(result.get("top_lineup_summary"))
    if parsed_summary is not None:
        return max(0, parsed_summary)

    lineup_count = int(result.get("lineup_count") or 0)
    if lineup_count <= 0:
        return 0

    share = float(result.get("top_lineup_share") or 0.0)
    if share <= 0.0:
        return 0

    count = int(round(lineup_count * share))
    return 1 if count <= 0 else count


def _group_player_name_lookup(group: List[Dict[str, object]]) -> Dict[int, str]:
    player_names: Dict[int, str] = {}
    for result in group:
        row = result.get("_embedding_row")
        if isinstance(row, EmbeddingRow):
            for idx, player_id in enumerate(row.roster_player_ids):
                if int(player_id) <= 0:
                    continue
                raw_name = (
                    row.roster_player_names[idx]
                    if idx < len(row.roster_player_names)
                    else None
                )
                name = _normalize_player_name(raw_name)
                if name != "Unknown Player":
                    player_names[int(player_id)] = name

        ids = _parse_player_id_sequence(result)
        names = _parse_player_name_sequence(result)
        for idx, player_id in enumerate(ids):
            if idx >= len(names):
                continue
            name = _normalize_player_name(names[idx])
            if name != "Unknown Player":
                player_names[int(player_id)] = name
    return player_names


def _build_core_lineup_players(
    group: list[Dict[str, object]],
) -> list[Dict[str, object]]:
    player_metrics: Dict[int, Dict[str, object]] = {}
    for result in group:
        if isinstance(result.get("core_lineup_players"), list):
            for player in result.get("core_lineup_players") or []:
                if not isinstance(player, dict):
                    continue
                player_id = player.get("player_id")
                if player_id is None:
                    continue
                try:
                    pid = int(player_id)
                except (TypeError, ValueError):
                    continue
                if pid <= 0:
                    continue
                name = _normalize_player_name(player.get("player_name"))
                matches_played = int(player.get("matches_played") or 0)

                entry = player_metrics.get(pid)
                if entry is None:
                    entry = {
                        "player_id": pid,
                        "player_name": name,
                        "matches_played": 0,
                    }
                    player_metrics[pid] = entry
                else:
                    current_name = str(entry.get("player_name") or "")
                    if current_name == "Unknown Player" and name != "Unknown Player":
                        entry["player_name"] = name

                entry["matches_played"] = int(entry["matches_played"]) + matches_played
            continue

        top_matches = _estimate_top_lineup_matches(result)
        player_ids = _parse_player_id_sequence(result)
        if not player_ids:
            continue

        player_names = _parse_player_name_sequence(result)
        for i, player_id in enumerate(player_ids):
            name = (
                player_names[i]
                if i < len(player_names)
                else f"Player {player_id}"
            )
            entry = player_metrics.get(int(player_id))
            if entry is None:
                player_metrics[int(player_id)] = {
                    "player_id": int(player_id),
                    "player_name": name,
                    "matches_played": 0,
                }
                entry = player_metrics[int(player_id)]
            else:
                current_name = str(entry.get("player_name") or "")
                if current_name == "Unknown Player" and name != "Unknown Player":
                    entry["player_name"] = name

            entry["matches_played"] = int(entry["matches_played"]) + top_matches

    output: list[Dict[str, object]] = []
    for player_id, payload in player_metrics.items():
        output.append(
            {
                "player_id": player_id,
                "player_name": str(payload.get("player_name") or "Unknown Player"),
                "matches_played": int(payload.get("matches_played") or 0),
                "sendou_url": f"https://sendou.ink/u/{player_id}",
            }
        )

    output.sort(key=lambda row: (-int(row["matches_played"]), int(row["player_id"])))
    return output


def _lineup_overlap_fraction(set_a: set[int], set_b: set[int]) -> float:
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b)) / float(len(union))


def _allowed_player_substitutions(player_count: int) -> int:
    if player_count <= 4:
        return 1
    return 1 + ((player_count - 4) // 3)


def _lineup_nearly_matches(set_a: set[int], set_b: set[int]) -> bool:
    if not set_a or not set_b:
        return False
    if len(set_a) < 3 or len(set_b) < 3:
        return False

    len_a = len(set_a)
    len_b = len(set_b)
    size_delta = abs(len_a - len_b)
    if size_delta > 2:
        return False

    min_len = min(len_a, len_b)
    allowed_substitutions = _allowed_player_substitutions(min_len)
    max_symmetric_diff = (2 * allowed_substitutions) + min(size_delta, 1)

    if len(set_a ^ set_b) > max_symmetric_diff:
        return False

    overlap = len(set_a & set_b)
    required_overlap = max(3, min_len - allowed_substitutions)
    return overlap >= required_overlap


def _lineup_signature(result: Dict[str, object]) -> tuple[int, ...]:
    return _parse_player_ids(result)


def _eligible_for_consolidation(
    a: Dict[str, object], b: Dict[str, object], *, min_overlap: float
) -> bool:
    players_a = _parse_player_set(a)
    players_b = _parse_player_set(b)
    signature_a = _lineup_signature(a)
    signature_b = _lineup_signature(b)

    if signature_a and signature_b and signature_a == signature_b:
        return True

    if _lineup_nearly_matches(players_a, players_b):
        return True

    if players_a and players_b and len(players_a) >= 3 and len(players_b) >= 3:
        overlap = _lineup_overlap_fraction(players_a, players_b)
        if overlap < min_overlap:
            return False

        count_a = int(a.get("lineup_count") or 0)
        count_b = int(b.get("lineup_count") or 0)
        count_delta = abs(count_a - count_b)
        count_threshold = max(5, int(0.35 * max(max(count_a, count_b), 1)))
        if count_delta > count_threshold:
            return False

        distinct_a = float(a.get("distinct_lineup_count") or 0.0)
        distinct_b = float(b.get("distinct_lineup_count") or 0.0)
        if abs(distinct_a - distinct_b) > 2.5:
            return False

        top_a = float(a.get("top_lineup_share") or 0.0)
        top_b = float(b.get("top_lineup_share") or 0.0)
        if abs(top_a - top_b) > 0.3:
            return False
        return True

    return False


def consolidate_ranked_results(
    results: List[Dict[str, object]],
    *,
    min_overlap: float = 0.72,
) -> List[Dict[str, object]]:
    if not results:
        return []
    if min_overlap <= 0.0:
        return results

    n_results = len(results)
    if n_results <= 1:
        return [dict(row) for row in results]

    parent = list(range(n_results))

    def find(idx: int) -> int:
        node = idx
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        parent[right_root] = left_root

    for left in range(n_results):
        base = results[left]
        for right in range(left + 1, n_results):
            candidate = results[right]
            if _eligible_for_consolidation(base, candidate, min_overlap=min_overlap):
                union(left, right)

    groups_by_root: Dict[int, List[Dict[str, object]]] = {}
    for idx, result in enumerate(results):
        root = find(idx)
        groups_by_root.setdefault(root, []).append(result)

    grouped = []
    for root in sorted(groups_by_root):
        group = groups_by_root[root]
        if len(group) == 1:
            rep = dict(group[0])
            rep["_consolidated_member_results"] = [dict(group[0])]
            grouped.append(rep)
            continue

        rep = dict(group[0])
        rep["_consolidated_member_results"] = [dict(item) for item in group]
        base_match_count = int(rep.get("match_count") or rep.get("lineup_count", 0) or 0)
        base_tournament_count = int(rep.get("tournament_count", 0) or 0)
        merged_lineup_variant_counts = _extract_lineup_variant_counts(rep)

        rep_signature = tuple(_parse_player_id_sequence(rep))
        if rep_signature:
            merged_lineup_variant_counts.setdefault(
                rep_signature,
                _estimate_top_lineup_matches_fallback(rep),
            )

        rep["is_consolidated"] = True
        rep["consolidated_team_count"] = len(group)

        total_match_count = base_match_count
        total_tournament_count = base_tournament_count
        aliases: list[Dict[str, object]] = []
        for alias in group[1:]:
            alias_match_count = int(alias.get("match_count") or alias.get("lineup_count", 0) or 0)
            alias_tournament_count = int(alias.get("tournament_count", 0) or 0)
            for signature, count in _extract_lineup_variant_counts(alias).items():
                merged_lineup_variant_counts[signature] = (
                    merged_lineup_variant_counts.get(signature, 0) + int(count)
                )
            total_match_count += alias_match_count
            total_tournament_count += alias_tournament_count
            aliases.append(
                {
                    "team_id": alias["team_id"],
                    "team_name": alias["team_name"],
                    "tournament_id": alias.get("tournament_id"),
                    "event_time_ms": alias.get("event_time_ms"),
                    "match_count": alias_match_count,
                    "tournament_count": alias_tournament_count,
                }
            )

        chosen_signature, chosen_lineup_match_count = _select_top_lineup_variant(
            merged_lineup_variant_counts
        )
        player_name_lookup = _group_player_name_lookup(group)
        rep["top_lineup_match_count"] = chosen_lineup_match_count
        rep["top_lineup_match_share"] = (
            chosen_lineup_match_count / total_match_count if total_match_count > 0 else 0.0
        )
        rep["top_lineup_share"] = rep["top_lineup_match_share"]
        rep["top_lineup_player_ids"] = list(chosen_signature)
        chosen_player_names = list(
            player_name_lookup.get(int(player_id), "Unknown Player")
            for player_id in chosen_signature
        )
        rep["top_lineup_player_names"] = chosen_player_names
        rep["top_lineup_player_count"] = len(chosen_signature)
        rep["top_lineup_summary"] = (
            f"{chosen_lineup_match_count}x:{','.join(chosen_player_names)}"
            if chosen_signature and chosen_lineup_match_count > 0
            else rep.get("top_lineup_summary", "")
        )
        rep["distinct_lineup_count"] = (
            len(merged_lineup_variant_counts)
            if merged_lineup_variant_counts
            else int(rep.get("distinct_lineup_count", 0) or 0)
        )
        rep["lineup_count"] = total_match_count
        rep["match_count"] = total_match_count
        rep["tournament_count"] = total_tournament_count
        rep["consolidated_teams"] = aliases
        rep["core_lineup_players"] = _build_core_lineup_players(group)
        rep["consolidated_team_ids"] = [rep["team_id"]] + [
            alias["team_id"] for alias in aliases
        ]
        rep["consolidated_team_names"] = [rep["team_name"]] + [
            alias["team_name"] for alias in aliases
        ]
        rep["_lineup_variant_counts"] = dict(merged_lineup_variant_counts)
        grouped.append(rep)

    return grouped
