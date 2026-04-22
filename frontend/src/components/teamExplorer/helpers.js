export const DEFAULT_CLUSTER_MODE = 'family';
export const DEFAULT_NEIGHBORS = 12;
export const DEFAULT_MATCH_LIMIT = 100;
export const DEFAULT_TEAM_SCOPE = 'family';
export const HISTORY_EVENTS_PER_PAGE = 5;
export const EMPTY_TEAM_IDS = [];

export function parseTeamIdList(value) {
  if (Array.isArray(value)) {
    return value
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item) && item > 0)
      .map((item) => Math.trunc(item));
  }
  if (Number.isFinite(Number(value)) && Number(value) > 0) {
    return [Math.trunc(Number(value))];
  }
  if (typeof value === 'string') {
    return (value.match(/\d+/g) || [])
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item) && item > 0)
      .map((item) => Math.trunc(item));
  }
  return [];
}

export function uniqueTeamIds(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    if (!Number.isFinite(value) || value <= 0 || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

export function pluralize(value, singular, plural) {
  const numeric = Number(value);
  const count = Number.isFinite(numeric) ? Math.trunc(numeric) : 0;
  let pluralForm = plural;
  if (!pluralForm) {
    if (/(s|x|z|ch|sh)$/i.test(singular)) {
      pluralForm = `${singular}es`;
    } else if (/[^aeiou]y$/i.test(singular)) {
      pluralForm = `${singular.slice(0, -1)}ies`;
    } else {
      pluralForm = `${singular}s`;
    }
  }
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

export function pct(value) {
  if (value === null || value === undefined || value === '') return 'n/a';
  const numeric = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return `${(numeric * 100).toFixed(1)}%`;
}

export function toEpochMs(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value <= 0) return null;
  if (value > 1_000_000_000_000) return Math.floor(value / 1000);
  if (value > 10_000_000_000) return Math.floor(value);
  return Math.floor(value * 1000);
}

export function fmtDate(ms) {
  const value = toEpochMs(ms);
  if (value === null) return 'n/a';
  return new Date(value).toISOString().slice(0, 10);
}

export function relativeDate(ms) {
  const value = toEpochMs(ms);
  if (value === null) return 'n/a';

  const deltaMs = Date.now() - value;
  if (deltaMs < 0) return fmtDate(ms);

  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  const week = 7 * day;
  const month = 30 * day;

  if (deltaMs < 90 * minute) {
    const count = Math.max(1, Math.round(deltaMs / minute));
    return `${count} minute${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 36 * hour) {
    const count = Math.max(1, Math.round(deltaMs / hour));
    return `${count} hour${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 14 * day) {
    const count = Math.max(1, Math.round(deltaMs / day));
    return `${count} day${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 8 * week) {
    const count = Math.max(1, Math.round(deltaMs / week));
    return `${count} week${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 10 * month) {
    const count = Math.max(1, Math.round(deltaMs / month));
    return `${count} month${count === 1 ? '' : 's'} ago`;
  }
  return fmtDate(ms);
}

export function rosterPreview(players, limit = 4) {
  if (!Array.isArray(players) || players.length === 0) return 'n/a';
  const names = players
    .map((player) => (
      typeof player === 'string'
        ? player.trim()
        : String(player?.player_name || player?.name || '').trim()
    ))
    .filter(Boolean);
  if (!names.length) return 'n/a';
  const preview = names.slice(0, limit).join(', ');
  return names.length > limit ? `${preview} +${names.length - limit}` : preview;
}

export function normalizeTimelineName(name) {
  const raw = String(name || '').trim();
  if (!raw) return 'unnamed-team';
  const normalized = typeof raw.normalize === 'function' ? raw.normalize('NFKC') : raw;
  return normalized.toLowerCase().replace(/\s+/g, ' ');
}

export function normalizePlayerRows(values) {
  if (!Array.isArray(values)) return [];

  const seen = new Set();
  const out = [];

  for (const value of values) {
    const id = Number(value?.player_id);
    const normalizedId = Number.isFinite(id) && id > 0 ? Math.trunc(id) : null;
    const name = String(value?.player_name || '').trim() || 'Unknown Player';
    const key = normalizedId === null ? `name:${name.toLowerCase()}` : `id:${normalizedId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const matchesPlayed = Number(value?.matches_played);
    out.push({
      id: normalizedId,
      name,
      matchesPlayed: Number.isFinite(matchesPlayed) ? Math.trunc(matchesPlayed) : 0,
      sendouUrl: value?.sendou_url || (normalizedId !== null ? `https://sendou.ink/u/${normalizedId}` : null),
    });
  }

  out.sort((a, b) => b.matchesPlayed - a.matchesPlayed || a.name.localeCompare(b.name));
  return out;
}

export function playerRowKey(player) {
  if (player?.id !== null && player?.id !== undefined && Number.isFinite(Number(player.id))) {
    return `id:${Math.trunc(Number(player.id))}`;
  }
  return `name:${normalizeTimelineName(player?.name)}`;
}

export function normalizeRosterPlayers(values) {
  if (!Array.isArray(values) || !values.length) return [];
  const seen = new Set();
  const out = [];

  for (const value of values) {
    const name = typeof value === 'string'
      ? value.trim()
      : String(value?.player_name || value?.name || '').trim();
    if (!name) continue;
    const key = normalizeTimelineName(name);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(name);
  }

  return out;
}

export function rowMatchesSelectedTeam(row, teamIds) {
  const rowIds = uniqueTeamIds([
    Number(row?.team_id),
    ...parseTeamIdList(row?.consolidated_team_ids),
  ]);
  return rowIds.some((teamId) => teamIds.includes(teamId));
}

export function playerUsageDescriptor(player, baselineMatches, index) {
  if (player.matchesPlayed <= 0) return 'top-lineup reference';
  if (baselineMatches <= 0) return index < 4 ? 'core' : 'sub';
  const share = player.matchesPlayed / baselineMatches;
  if (share >= 0.75) return 'core';
  if (share >= 0.4) return 'regular';
  if (player.matchesPlayed <= 1) return 'one-off';
  return 'sub';
}

export function continuityLabel(topShare, distinctLineups) {
  if (!Number.isFinite(topShare)) return 'Continuity unavailable';
  if (topShare >= 0.75) return 'Very stable core';
  if (topShare >= 0.5) return distinctLineups > 2 ? 'Stable core with rotation' : 'Mostly stable core';
  return distinctLineups > 3 ? 'Heavy rotation' : 'Rotating around the core';
}

export function buildRecentForm(matches, limit = 5) {
  if (!Array.isArray(matches) || !matches.length) return [];
  return [...matches]
    .sort((left, right) => Number(right?.event_time_ms || 0) - Number(left?.event_time_ms || 0))
    .slice(0, limit)
    .map((row) => ({
      key: row?.match_id ?? `${row?.event_time_ms ?? 'na'}-${row?.opponent_team_id ?? 'na'}`,
      label: row?.team_is_winner ? 'W' : row?.opponent_is_winner ? 'L' : '—',
      tone: row?.team_is_winner ? 'is-win' : row?.opponent_is_winner ? 'is-loss' : 'is-pending',
      title: row?.opponent_team_name || 'Unknown opponent',
    }));
}

export function matchTone(row) {
  return row?.team_is_winner ? 'is-win' : row?.opponent_is_winner ? 'is-loss' : 'is-pending';
}

export function matchResultLabel(row) {
  if (row?.team_is_winner) return 'Win';
  if (row?.opponent_is_winner) return 'Loss';
  return 'Unresolved';
}

export function matchScoreLabel(row) {
  return `${row?.team_score ?? '—'} - ${row?.opponent_score ?? '—'}`;
}

export function buildMatchEvents(matches) {
  if (!Array.isArray(matches) || !matches.length) return [];

  const groups = new Map();

  for (const row of matches) {
    const tournamentId = Number(row?.tournament_id);
    const tournamentName = String(row?.tournament_name || '').trim() || `Tournament ${row?.tournament_id ?? 'n/a'}`;
    const key = Number.isFinite(tournamentId) && tournamentId > 0
      ? `tournament:${Math.trunc(tournamentId)}`
      : `name:${tournamentName.toLowerCase()}`;
    const eventTimeMs = Number(row?.event_time_ms) || 0;

    if (!groups.has(key)) {
      groups.set(key, {
        key,
        tournamentId: Number.isFinite(tournamentId) && tournamentId > 0 ? Math.trunc(tournamentId) : null,
        tournamentName,
        tournamentTier: row?.tournament_score_tier || 'Unscored',
        latestEventTimeMs: eventTimeMs,
        wins: 0,
        losses: 0,
        rows: [],
      });
    }

    const group = groups.get(key);
    group.latestEventTimeMs = Math.max(group.latestEventTimeMs, eventTimeMs);
    if (group.tournamentTier === 'Unscored' && row?.tournament_score_tier) {
      group.tournamentTier = row.tournament_score_tier;
    }

    const tone = matchTone(row);
    if (tone === 'is-win') group.wins += 1;
    if (tone === 'is-loss') group.losses += 1;

    group.rows.push({
      key: row?.match_id ?? `${key}-${eventTimeMs}-${row?.opponent_team_id ?? 'na'}`,
      eventTimeMs,
      tone,
      resultLabel: matchResultLabel(row),
      relativeDate: relativeDate(row?.event_time_ms),
      opponentTeamId: row?.opponent_team_id ? Number(row.opponent_team_id) : null,
      opponentTeamName: row?.opponent_team_name || 'Unknown opponent',
      scoreLabel: matchScoreLabel(row),
      teamRosterLabel: rosterPreview(row?.team_roster),
      opponentRosterLabel: rosterPreview(row?.opponent_roster),
      teamRosterPlayers: normalizeRosterPlayers(row?.team_roster),
      opponentRosterPlayers: normalizeRosterPlayers(row?.opponent_roster),
    });
  }

  return Array.from(groups.values())
    .map((group) => {
      const rowsByNewest = [...group.rows].sort((left, right) => {
        const leftScore = Number(left.eventTimeMs || 0);
        const rightScore = Number(right.eventTimeMs || 0);
        return rightScore - leftScore;
      });
      const rowsByOldest = [...rowsByNewest].reverse();
      const opponentCounts = new Map();
      const teamRosterCounts = new Map();
      for (const row of rowsByNewest) {
        const opponentKey = row.opponentTeamId
          ? `id:${row.opponentTeamId}`
          : `name:${String(row.opponentTeamName || '').toLowerCase()}`;
        const existing = opponentCounts.get(opponentKey);
        if (existing) {
          existing.count += 1;
        } else {
          opponentCounts.set(opponentKey, { name: row.opponentTeamName, count: 1 });
        }
        const teamRosterKey = String(row.teamRosterLabel || '').toLowerCase();
        if (teamRosterKey) {
          teamRosterCounts.set(teamRosterKey, {
            label: row.teamRosterLabel,
            count: (teamRosterCounts.get(teamRosterKey)?.count || 0) + 1,
          });
        }
      }
      const opponentSummaryItems = Array.from(opponentCounts.values())
        .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name));
      const opponentSummary = opponentSummaryItems
        .slice(0, 3)
        .map((item) => (item.count > 1 ? `${item.name} (${item.count})` : item.name))
        .join(', ');
      const remainingOpponents = opponentSummaryItems.length - 3;
      const defaultTeamRoster = Array.from(teamRosterCounts.values())
        .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label))[0];

      return {
        ...group,
        whenLabel: group.latestEventTimeMs ? relativeDate(group.latestEventTimeMs) : 'n/a',
        opponentSummary: remainingOpponents > 0 ? `${opponentSummary} +${remainingOpponents} more` : opponentSummary,
        defaultTeamRosterLabel: defaultTeamRoster?.label || '',
        rows: rowsByOldest.map((row) => {
          const opponentKey = row.opponentTeamId
            ? `id:${row.opponentTeamId}`
            : `name:${String(row.opponentTeamName || '').toLowerCase()}`;
          return {
            ...row,
            opponentEventCount: opponentCounts.get(opponentKey)?.count || 1,
            usesDefaultTeamRoster: defaultTeamRoster?.label ? defaultTeamRoster.label === row.teamRosterLabel : false,
          };
        }),
        ribbonRows: rowsByOldest,
      };
    })
    .sort((left, right) => right.latestEventTimeMs - left.latestEventTimeMs);
}

export function buildNameTimeline(rows) {
  if (!Array.isArray(rows) || !rows.length) return [];

  const groups = new Map();

  for (const row of rows) {
    const displayName = String(row?.team_name || '').trim() || `Team ${row?.team_id ?? 'n/a'}`;
    const key = normalizeTimelineName(displayName);
    const eventTimeMs = Number(row?.event_time_ms) || 0;
    const matchCount = Number(row?.match_count) || 0;

    if (!groups.has(key)) {
      groups.set(key, {
        key,
        name: displayName,
        latestEventTimeMs: eventTimeMs,
        registrationCount: 0,
        totalMatches: 0,
      });
    }

    const group = groups.get(key);
    group.registrationCount += 1;
    group.totalMatches += matchCount;

    if (eventTimeMs >= group.latestEventTimeMs) {
      group.latestEventTimeMs = eventTimeMs;
      group.name = displayName;
    }
  }

  return Array.from(groups.values()).sort((left, right) => {
    if (right.latestEventTimeMs !== left.latestEventTimeMs) {
      return right.latestEventTimeMs - left.latestEventTimeMs;
    }
    if (right.totalMatches !== left.totalMatches) {
      return right.totalMatches - left.totalMatches;
    }
    return left.name.localeCompare(right.name);
  });
}

export function resolveTeamProfileRow(rows, teamIds) {
  if (!Array.isArray(rows) || !rows.length) return null;
  return rows.find((row) => rowMatchesSelectedTeam(row, teamIds))
    || rows.find((row) => Number(row?.team_id) === Number(teamIds[0]))
    || rows[0]
    || null;
}

export function resolveScopedTeamIds(requestedIds, matchedRow, scopeMode) {
  const normalizedRequestedIds = uniqueTeamIds(parseTeamIdList(requestedIds));
  if (scopeMode !== 'family') return normalizedRequestedIds;

  const familyIds = uniqueTeamIds([
    Number(matchedRow?.team_id),
    ...parseTeamIdList(matchedRow?.consolidated_team_ids),
  ]);
  return familyIds.length ? familyIds : normalizedRequestedIds;
}
