import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchAnalyticsTeam, fetchAnalyticsTeamMatches, fetchTeamSearch } from '../api';

const DEFAULT_CLUSTER_MODE = 'family';
const DEFAULT_NEIGHBORS = 12;
const DEFAULT_MATCH_LIMIT = 100;
const DEFAULT_TEAM_SCOPE = 'family';
const HISTORY_EVENTS_PER_PAGE = 5;

function parseTeamIdList(value) {
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

function uniqueTeamIds(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    if (!Number.isFinite(value) || value <= 0 || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

function pluralize(value, singular, plural) {
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

function pct(value) {
  if (value === null || value === undefined || value === '') return 'n/a';
  const numeric = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return `${(numeric * 100).toFixed(1)}%`;
}

function toEpochMs(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value <= 0) return null;
  if (value > 1_000_000_000_000) return Math.floor(value / 1000);
  if (value > 10_000_000_000) return Math.floor(value);
  return Math.floor(value * 1000);
}

function fmtDate(ms) {
  const value = toEpochMs(ms);
  if (value === null) return 'n/a';
  return new Date(value).toISOString().slice(0, 10);
}

function relativeDate(ms) {
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

function rosterPreview(players, limit = 4) {
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

function normalizePlayerRows(values) {
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

function playerRowKey(player) {
  if (player?.id !== null && player?.id !== undefined && Number.isFinite(Number(player.id))) {
    return `id:${Math.trunc(Number(player.id))}`;
  }
  return `name:${normalizeTimelineName(player?.name)}`;
}

function normalizeRosterPlayers(values) {
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

function rowMatchesSelectedTeam(row, teamIds) {
  const rowIds = uniqueTeamIds([
    Number(row?.team_id),
    ...parseTeamIdList(row?.consolidated_team_ids),
  ]);
  return rowIds.some((teamId) => teamIds.includes(teamId));
}

function playerUsageDescriptor(player, baselineMatches, index) {
  if (player.matchesPlayed <= 0) return 'top-lineup reference';
  if (baselineMatches <= 0) return index < 4 ? 'core' : 'sub';
  const share = player.matchesPlayed / baselineMatches;
  if (share >= 0.75) return 'core';
  if (share >= 0.4) return 'regular';
  if (player.matchesPlayed <= 1) return 'one-off';
  return 'sub';
}

function continuityLabel(topShare, distinctLineups) {
  if (!Number.isFinite(topShare)) return 'Continuity unavailable';
  if (topShare >= 0.75) return 'Very stable core';
  if (topShare >= 0.5) return distinctLineups > 2 ? 'Stable core with rotation' : 'Mostly stable core';
  return distinctLineups > 3 ? 'Heavy rotation' : 'Rotating around the core';
}

function buildRecentForm(matches, limit = 5) {
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

function matchTone(row) {
  return row?.team_is_winner ? 'is-win' : row?.opponent_is_winner ? 'is-loss' : 'is-pending';
}

function matchResultLabel(row) {
  if (row?.team_is_winner) return 'Win';
  if (row?.opponent_is_winner) return 'Loss';
  return 'Unresolved';
}

function matchScoreLabel(row) {
  return `${row?.team_score ?? '—'} - ${row?.opponent_score ?? '—'}`;
}

function buildMatchEvents(matches) {
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

function normalizeTimelineName(name) {
  const raw = String(name || '').trim();
  if (!raw) return 'unnamed-team';
  const normalized = typeof raw.normalize === 'function' ? raw.normalize('NFKC') : raw;
  return normalized.toLowerCase().replace(/\s+/g, ' ');
}

function buildNameTimeline(rows) {
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

function resolveTeamProfileRow(rows, teamIds) {
  if (!Array.isArray(rows) || !rows.length) return null;
  return rows.find((row) => rowMatchesSelectedTeam(row, teamIds))
    || rows.find((row) => Number(row?.team_id) === Number(teamIds[0]))
    || rows[0]
    || null;
}

function resolveScopedTeamIds(requestedIds, matchedRow, scopeMode) {
  const normalizedRequestedIds = uniqueTeamIds(parseTeamIdList(requestedIds));
  if (scopeMode !== 'family') return normalizedRequestedIds;

  const familyIds = uniqueTeamIds([
    Number(matchedRow?.team_id),
    ...parseTeamIdList(matchedRow?.consolidated_team_ids),
  ]);
  return familyIds.length ? familyIds : normalizedRequestedIds;
}

export default function TeamExplorer({
  selectedTeamId = '',
  selectedTeamIds = [],
  selectedSnapshotId = '',
  selectedTeamName = '',
  onOpenHeadToHead = () => {},
  onOpenTeamPage = () => {},
  onOpenPlayerLookup = () => {},
}) {
  const [teamIdsInput, setTeamIdsInput] = useState('');
  const [teamScope, setTeamScope] = useState(DEFAULT_TEAM_SCOPE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [warning, setWarning] = useState('');
  const [teamLab, setTeamLab] = useState(null);
  const [teamProfile, setTeamProfile] = useState(null);
  const [matchHistory, setMatchHistory] = useState(null);
  const [matchHistoryError, setMatchHistoryError] = useState('');
  const [resolvedTeamIds, setResolvedTeamIds] = useState([]);
  const [expandedEventKeys, setExpandedEventKeys] = useState([]);
  const [historyPage, setHistoryPage] = useState(1);
  const previousScopeRef = useRef(DEFAULT_TEAM_SCOPE);

  const selectedIds = useMemo(
    () => uniqueTeamIds([
      ...parseTeamIdList(selectedTeamIds),
      ...parseTeamIdList(selectedTeamId),
    ]),
    [selectedTeamId, selectedTeamIds],
  );
  const selectedIdsKey = selectedIds.join(',');
  const normalizedSelectedSnapshotId = selectedSnapshotId ? String(selectedSnapshotId) : '';

  const loadTeam = useCallback(async (nextTeamIds, snapshotId = normalizedSelectedSnapshotId) => {
    const requestedIds = uniqueTeamIds(parseTeamIdList(nextTeamIds));
    if (!requestedIds.length) {
      setError('Enter at least one valid team ID.');
      setWarning('');
      setTeamLab(null);
      setTeamProfile(null);
      setMatchHistory(null);
      setResolvedTeamIds([]);
      return;
    }

    const primaryTeamId = requestedIds[0];

    setLoading(true);
    setError('');
    setWarning('');
    setMatchHistoryError('');
    setHistoryPage(1);
    setExpandedEventKeys([]);
    setResolvedTeamIds(requestedIds);
    try {
      const [teamPayload, searchPayload] = await Promise.allSettled([
        fetchAnalyticsTeam({
          teamId: primaryTeamId,
          clusterMode: DEFAULT_CLUSTER_MODE,
          neighbors: DEFAULT_NEIGHBORS,
          snapshotId: snapshotId || undefined,
        }),
        fetchTeamSearch({
          q: String(primaryTeamId),
          topN: 12,
          clusterMode: DEFAULT_CLUSTER_MODE,
          minRelevance: 0,
          consolidate: true,
          consolidateMinOverlap: 0.8,
        }),
      ]);
      let matchedRow = null;
      const teamLabValue = teamPayload.status === 'fulfilled' ? teamPayload.value : null;
      setTeamLab(teamLabValue);
      if (searchPayload.status === 'fulfilled') {
        const searchRows = searchPayload.value?.results || [];
        matchedRow = resolveTeamProfileRow(searchRows, requestedIds);
        setTeamProfile(matchedRow || searchRows[0] || null);
      } else {
        setTeamProfile(null);
      }

      let partialWarning = '';
      if (!teamLabValue && searchPayload.status === 'fulfilled') {
        partialWarning = 'Team analytics are temporarily unavailable. Showing the search snapshot and any match data that loads.';
      }
      if (!teamLabValue && searchPayload.status !== 'fulfilled') {
        throw teamPayload.reason || searchPayload.reason || new Error('Failed to load team detail');
      }

      const effectiveIds = resolveScopedTeamIds(requestedIds, matchedRow, teamScope);
      setResolvedTeamIds(effectiveIds);

      try {
        const matchesPayload = await fetchAnalyticsTeamMatches({
          teamId: primaryTeamId,
          teamIds: effectiveIds,
          limit: DEFAULT_MATCH_LIMIT,
          snapshotId: snapshotId || undefined,
        });
        setMatchHistory(matchesPayload);
      } catch (_matchesError) {
        setMatchHistory(null);
        setMatchHistoryError('Recent matches are unavailable right now.');
      }
      setWarning(partialWarning);

    } catch (err) {
      setWarning('');
      setTeamLab(null);
      setTeamProfile(null);
      setMatchHistory(null);
      setMatchHistoryError('');
      setResolvedTeamIds([]);
      setError(err.message || 'Failed to load team detail');
    } finally {
      setLoading(false);
    }
  }, [normalizedSelectedSnapshotId, teamScope]);

  useEffect(() => {
    if (!selectedIds.length) return;
    setTeamIdsInput(selectedIds.join(','));
    loadTeam(selectedIds, normalizedSelectedSnapshotId);
    // Intentionally exclude loadTeam so scope toggles do not overwrite manual ID edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedSelectedSnapshotId, selectedIds, selectedIdsKey]);

  useEffect(() => {
    if (previousScopeRef.current === teamScope) return;
    previousScopeRef.current = teamScope;
    const idsToReload = uniqueTeamIds(parseTeamIdList(teamIdsInput));
    if (!idsToReload.length) return;
    loadTeam(idsToReload, normalizedSelectedSnapshotId);
  }, [loadTeam, normalizedSelectedSnapshotId, teamIdsInput, teamScope]);

  const summary = matchHistory?.summary || null;
  const team = teamLab?.team || null;
  const teamProfilePlayers = useMemo(
    () => normalizePlayerRows(teamProfile?.core_lineup_players || teamProfile?.top_lineup_players || []),
    [teamProfile],
  );
  const rosterOrderByPlayer = useMemo(() => {
    const map = new Map();
    teamProfilePlayers.forEach((player, index) => {
      map.set(playerRowKey(player), index);
    });
    return map;
  }, [teamProfilePlayers]);
  const rosterPresenceByPlayer = useMemo(() => {
    const map = new Map();
    teamProfilePlayers.forEach((player) => {
      map.set(playerRowKey(player), player.matchesPlayed);
    });
    return map;
  }, [teamProfilePlayers]);
  const topLineupPlayers = useMemo(() => {
    const players = normalizePlayerRows(teamProfile?.top_lineup_players || []);
    return players
      .map((player) => ({
        ...player,
        rosterMatchesPlayed: rosterPresenceByPlayer.get(playerRowKey(player)) ?? null,
      }))
      .sort((left, right) => {
        const leftOrder = rosterOrderByPlayer.get(playerRowKey(left));
        const rightOrder = rosterOrderByPlayer.get(playerRowKey(right));
        if (leftOrder !== undefined || rightOrder !== undefined) {
          return (leftOrder ?? Number.MAX_SAFE_INTEGER) - (rightOrder ?? Number.MAX_SAFE_INTEGER);
        }
        return left.name.localeCompare(right.name);
      });
  }, [rosterOrderByPlayer, rosterPresenceByPlayer, teamProfile]);
  const neighborsRows = teamLab?.neighbors || [];
  const matches = matchHistory?.matches || [];
  const fallbackSummary = useMemo(() => {
    const primaryTeamId = Number(teamProfile?.team_id ?? teamLab?.team?.team_id);
    const primaryTeamName = String(teamProfile?.team_name || teamLab?.team?.team_name || '').trim();
    if (!Number.isFinite(primaryTeamId) || primaryTeamId <= 0 || !primaryTeamName) return null;

    const familyIds = resolvedTeamIds.length ? resolvedTeamIds : selectedIds.length ? selectedIds : [Math.trunc(primaryTeamId)];
    const familyNames = familyIds.map((teamId) => {
      if (teamId === Math.trunc(primaryTeamId)) return primaryTeamName;
      return `Team ${teamId}`;
    });
    const fallbackMatchSummary = teamLab?.match_summary || {};
    const wins = Number(fallbackMatchSummary.wins);
    const totalMatches = Number.isFinite(Number(fallbackMatchSummary.matches))
      ? Number(fallbackMatchSummary.matches)
      : Number(teamProfile?.match_count);
    const losses = Number.isFinite(totalMatches) && Number.isFinite(wins)
      ? Math.max(0, totalMatches - wins)
      : 0;

    return {
      primary_team_id: familyIds[0],
      primary_team_name: primaryTeamName,
      team_ids: familyIds,
      team_names: familyNames,
      selected_team_count: familyIds.length,
      total_matches: Number.isFinite(totalMatches) ? totalMatches : 0,
      wins: Number.isFinite(wins) ? wins : 0,
      losses,
      unresolved_matches: 0,
      decided_matches: Number.isFinite(totalMatches) ? totalMatches : 0,
      win_rate: fallbackMatchSummary.win_rate,
      tournaments: Number(teamProfile?.tournament_count) || 0,
      tournament_tier_distribution: null,
      tournament_tier_match_distribution: null,
    };
  }, [resolvedTeamIds, selectedIds, teamLab, teamProfile]);
  const effectiveSummary = summary || fallbackSummary;
  const pageTitle = effectiveSummary?.primary_team_name
    || team?.team_name
    || String(selectedTeamName || '').trim()
    || (selectedIds[0] ? `Team ${selectedIds[0]}` : 'Teams');
  const selectedFamilyIds = useMemo(() => {
    const summaryIds = uniqueTeamIds(parseTeamIdList(effectiveSummary?.team_ids));
    if (summaryIds.length) return summaryIds;
    if (resolvedTeamIds.length) return resolvedTeamIds;
    return selectedIds;
  }, [effectiveSummary?.team_ids, resolvedTeamIds, selectedIds]);
  const playerBaselineMatches = Number(teamProfile?.match_count ?? effectiveSummary?.total_matches ?? teamLab?.match_summary?.matches ?? 0);
  const primaryMatchCount = Number(teamProfile?.match_count ?? effectiveSummary?.total_matches ?? teamLab?.match_summary?.matches ?? 0);
  const tournamentCount = Number(teamProfile?.tournament_count ?? effectiveSummary?.tournaments ?? 0);
  const trackedPlayerCount = teamProfilePlayers.length || Number(teamProfile?.unique_player_count) || 0;
  const distinctLineupCount = Number(teamProfile?.distinct_lineup_count ?? team?.distinct_lineup_count ?? 0);
  const topLineupShare = Number(teamProfile?.top_lineup_match_share ?? teamProfile?.top_lineup_share);
  const topLineupMatchCount = Number(teamProfile?.top_lineup_match_count ?? 0);
  const rotationPlayers = teamProfilePlayers.slice(4);
  const continuityHint = continuityLabel(topLineupShare, distinctLineupCount);
  const actionSnapshotId = matchHistory?.snapshot_id || normalizedSelectedSnapshotId || null;
  const registrationRows = useMemo(() => {
    const primary = teamProfile
      ? [{
          team_id: teamProfile.team_id,
          team_name: teamProfile.team_name,
          tournament_id: teamProfile.tournament_id,
          event_time_ms: teamProfile.event_time_ms,
          match_count: teamProfile.match_count,
          tournament_count: teamProfile.tournament_count,
        }]
      : [];
    const aliases = Array.isArray(teamProfile?.consolidated_teams) ? teamProfile.consolidated_teams : [];
    return [...primary, ...aliases]
      .filter((row) => row?.team_id)
      .sort((a, b) => Number(b?.event_time_ms || 0) - Number(a?.event_time_ms || 0));
  }, [teamProfile]);
  const nameTimelineRows = useMemo(() => buildNameTimeline(registrationRows), [registrationRows]);
  const comparableRows = neighborsRows.slice(0, 8);
  const comparableNameCounts = useMemo(() => {
    const counts = new Map();
    for (const row of comparableRows) {
      const key = normalizeTimelineName(row?.team_name);
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return counts;
  }, [comparableRows]);
  const matchEvents = useMemo(() => buildMatchEvents(matches), [matches]);
  const historyPageCount = useMemo(
    () => Math.max(1, Math.ceil(matchEvents.length / HISTORY_EVENTS_PER_PAGE)),
    [matchEvents.length],
  );
  const visibleMatchEvents = useMemo(() => {
    const start = (historyPage - 1) * HISTORY_EVENTS_PER_PAGE;
    return matchEvents.slice(start, start + HISTORY_EVENTS_PER_PAGE);
  }, [historyPage, matchEvents]);
  const latestMatchEventMs = useMemo(
    () => matches.reduce((latest, row) => Math.max(latest, Number(row?.event_time_ms || 0)), 0),
    [matches],
  );
  const latestRegistrationMs = registrationRows.length ? Number(registrationRows[0]?.event_time_ms || 0) : 0;
  const lastSeenMs = latestMatchEventMs || latestRegistrationMs || Number(teamProfile?.event_time_ms || 0);
  const lastSeenLabel = lastSeenMs ? relativeDate(lastSeenMs) : 'n/a';
  const lineupSupportMismatch = useMemo(() => {
    if (!(topLineupMatchCount > 0)) return false;
    return topLineupPlayers.some((player) => (
      Number.isFinite(player.rosterMatchesPlayed)
      && player.rosterMatchesPlayed > 0
      && player.rosterMatchesPlayed < topLineupMatchCount
    ));
  }, [topLineupMatchCount, topLineupPlayers]);
  const topLineupMeta = useMemo(() => {
    if (lineupSupportMismatch) {
      return 'Lineup composition is available, but its appearance count does not line up with roster presence in this snapshot.';
    }
    return [
      topLineupMatchCount > 0 ? pluralize(topLineupMatchCount, 'lineup appearance') : null,
      Number.isFinite(topLineupShare) ? `${pct(topLineupShare)} of tracked matches` : null,
    ].filter(Boolean).join(' · ') || 'Current lineup summary unavailable.';
  }, [lineupSupportMismatch, topLineupMatchCount, topLineupShare]);
  useEffect(() => {
    setHistoryPage((previousPage) => {
      if (previousPage < 1) return 1;
      if (previousPage > historyPageCount) return historyPageCount;
      return previousPage;
    });
  }, [historyPageCount]);
  useEffect(() => {
    setExpandedEventKeys((previousKeys) => {
      const availableKeys = new Set(visibleMatchEvents.map((event) => event.key));
      const retainedKeys = previousKeys.filter((key) => availableKeys.has(key));
      if (retainedKeys.length) return retainedKeys;
      return visibleMatchEvents.length ? [visibleMatchEvents[0].key] : [];
    });
  }, [visibleMatchEvents]);
  const recentForm = useMemo(() => buildRecentForm(matches), [matches]);
  const knownNameCount = nameTimelineRows.length;
  const canonicalName = String(pageTitle || '').trim();
  const heroAliases = nameTimelineRows.slice(0, 5);
  const heroKicker = useMemo(() => {
    if (teamScope === 'family') {
      const parts = ['Canonical team family'];
      if (selectedFamilyIds.length) parts.push(pluralize(selectedFamilyIds.length, 'registration'));
      if (knownNameCount > 0) parts.push(pluralize(knownNameCount, 'name'));
      return parts.join(' · ');
    }
    return `Loaded team registration · ${pluralize(selectedFamilyIds.length || 1, 'registration')}`;
  }, [knownNameCount, selectedFamilyIds.length, teamScope]);
  const performanceStats = useMemo(() => ([
    { label: 'Win Rate', value: pct(effectiveSummary?.win_rate ?? teamLab?.match_summary?.win_rate) },
    { label: 'Recorded Matches', value: primaryMatchCount || 'n/a' },
    { label: 'Events', value: tournamentCount || 'n/a' },
    { label: 'Last Seen', value: lastSeenLabel },
  ]), [effectiveSummary?.win_rate, lastSeenLabel, primaryMatchCount, teamLab?.match_summary?.win_rate, tournamentCount]);
  const rosterStats = useMemo(() => ([
    { label: 'Tracked Players', value: trackedPlayerCount || 'n/a' },
    { label: 'Distinct Lineups', value: distinctLineupCount || 'n/a' },
    { label: 'Most Common Lineup Share', value: Number.isFinite(topLineupShare) ? pct(topLineupShare) : 'n/a' },
    { label: 'Rotation Pattern', value: continuityHint },
  ]), [continuityHint, distinctLineupCount, topLineupShare, trackedPlayerCount]);
  async function onSubmit(event) {
    event.preventDefault();
    await loadTeam(teamIdsInput, normalizedSelectedSnapshotId);
  }

  function toggleEventKey(eventKey) {
    setExpandedEventKeys((previousKeys) => (
      previousKeys.includes(eventKey)
        ? previousKeys.filter((key) => key !== eventKey)
        : [...previousKeys, eventKey]
    ));
  }

  function changeHistoryPage(nextPage) {
    setHistoryPage(Math.max(1, Math.min(historyPageCount, nextPage)));
  }

  return (
    <section className="panel team-detail-panel" aria-labelledby="team-detail-title">
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Scouting board</p>
          <h2 id="team-detail-title" className="panel-title">Teams</h2>
          <p className="panel-summary">
            Roster usage, match scouting, event history, and comparable team profiles for a single squad or family.
          </p>
        </div>
      </div>

      <form className="team-explorer-toolbar" onSubmit={onSubmit}>
        <div className="field team-id-field">
          <label className="field-label" htmlFor="team-detail-ids">Team ID or IDs</label>
          <input
            id="team-detail-ids"
            className="input"
            type="text"
            inputMode="numeric"
            placeholder="e.g. 46624 or 46624,54749,49106"
            value={teamIdsInput}
            onChange={(event) => setTeamIdsInput(event.target.value)}
          />
          <span className="field-label-subtitle">
            {teamScope === 'family'
              ? 'Family mode expands the first team ID into the full resolved family when available.'
              : 'Individual mode uses exactly the IDs entered here.'}
          </span>
        </div>
        <div className="field team-scope-field">
          <label className="field-label" htmlFor="team-detail-scope-family">Scope</label>
          <div className="team-scope-segmented" role="tablist" aria-label="Team scope">
            <button
              id="team-detail-scope-family"
              type="button"
              role="tab"
              aria-selected={teamScope === 'family'}
              className={teamScope === 'family' ? 'is-on' : ''}
              onClick={() => setTeamScope('family')}
            >
              Family
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={teamScope === 'individual'}
              className={teamScope === 'individual' ? 'is-on' : ''}
              onClick={() => setTeamScope('individual')}
            >
              Individual
            </button>
          </div>
        </div>
        <button type="submit" className="button btn-pill btn-fuchsia" disabled={loading}>
          {loading ? 'Loading…' : 'Load team'}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      {!loading && !team && !effectiveSummary ? (
        <div className="empty-state team-explorer-empty" role="status">
          <p className="empty-state-title">Pick a team to start</p>
          <p className="empty-state-hint">
            Use a search result’s team-page action or enter one or more team IDs here.
          </p>
        </div>
      ) : null}

      {team || effectiveSummary ? (
        <>
          <div className="team-identity-hero">
            <div className="team-identity-head">
              <div>
                <p className="team-identity-kicker">{heroKicker}</p>
                <h3 className="team-identity-name">{canonicalName || 'Team'}</h3>
              </div>
            </div>
            {heroAliases.length ? (
              <div className="team-identity-aliases" aria-label="Known family names">
                {heroAliases.map((row, index) => (
                  <span
                    key={`hero-alias-${row.key}`}
                    className={`team-identity-alias ${index === 0 ? 'is-current' : ''}`}
                    title={`${row.name} · ${pluralize(row.registrationCount, 'registration')} · ${pluralize(row.totalMatches, 'match')}`}
                  >
                    {row.name}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
          {warning ? (
            <p className="team-missing-note team-warning-note" role="status">{warning}</p>
          ) : null}
          <div className="team-stats-stack team-summary-grid analytics-stats">
            <section className="team-stat-section" aria-label="Performance stats">
              <p className="team-stat-section-label">Performance</p>
              <div className="grid-cols-4 team-stat-grid">
                {performanceStats.map((stat) => (
                  <article key={`performance-${stat.label}`} className="analytics-card stat">
                    <span className="analytics-card-label stat-label">{stat.label}</span>
                    <span className="analytics-card-value stat-value">{stat.value}</span>
                  </article>
                ))}
              </div>
            </section>
            <section className="team-stat-section" aria-label="Roster composition stats">
              <p className="team-stat-section-label">Roster Composition</p>
              <div className="grid-cols-4 team-stat-grid">
                {rosterStats.map((stat) => (
                  <article key={`roster-${stat.label}`} className="analytics-card stat">
                    <span className="analytics-card-label stat-label">{stat.label}</span>
                    <span className="analytics-card-value stat-value">{stat.value}</span>
                  </article>
                ))}
              </div>
            </section>
          </div>

          <div className="analytics-grid team-player-grid">
            <article className="analytics-panel analytics-panel-wide">
              <div className="team-section-heading">
                <div>
                  <h3>Core Roster</h3>
                  <p className="meta">
                    Player usage ordered by match presence across the current team scope.
                  </p>
                </div>
              </div>
              {teamProfilePlayers.length ? (
                <ul className="team-roster-list">
                  {teamProfilePlayers.map((player, index) => {
                    const share = playerBaselineMatches > 0
                      ? Math.round((player.matchesPlayed / playerBaselineMatches) * 100)
                      : 0;
                    const usageDescriptor = playerUsageDescriptor(player, playerBaselineMatches, index);
                    const matchesLabel = player.matchesPlayed > 0
                      ? pluralize(player.matchesPlayed, 'match')
                      : 'top-lineup only';
                    return (
                      <li
                        key={`team-player-${player.id ?? player.name}`}
                        className={`team-roster-row ${player.id === null ? 'is-disabled' : ''}`}
                      >
                        <button
                          type="button"
                          className="team-row-primary team-roster-primary"
                          disabled={player.id === null}
                          onClick={() => onOpenPlayerLookup(player.id, player.name)}
                          aria-label={`Open player history for ${player.name}`}
                        >
                          <span className="team-roster-main">
                            <span className="team-roster-name">{player.name}</span>
                            <span className="team-roster-note">
                              {matchesLabel} · {usageDescriptor}
                            </span>
                          </span>
                          <span className="team-roster-visual">
                            <span className="team-roster-bar" aria-hidden="true">
                              <span style={{ width: `${Math.max(0, Math.min(share, 100))}%` }} />
                            </span>
                            <span className="team-roster-share">
                              {player.matchesPlayed > 0 ? `${share}%` : 'n/a'}
                            </span>
                          </span>
                        </button>
                        <div className="team-hover-actions">
                          {player.sendouUrl ? (
                            <a
                              className="team-hover-action"
                              href={player.sendouUrl}
                              target="_blank"
                              rel="noreferrer"
                              aria-label={`Open sendou.ink profile for ${player.name}`}
                            >
                              sendou
                            </a>
                          ) : null}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="team-missing-note">No player profile data available yet.</p>
              )}
            </article>

            <article className="analytics-panel">
              <div className="team-section-heading team-side-heading">
                <div>
                  <h3>Most-Used Lineup</h3>
                  <p className="meta">Sorted to match the roster usage order.</p>
                </div>
              </div>
              {topLineupPlayers.length ? (
                <div className="team-top-lineup-summary">
                  <p className="team-top-lineup-names">{topLineupPlayers.map((player) => player.name).join(' · ')}</p>
                  <p className={`meta ${lineupSupportMismatch ? 'team-lineup-meta-warning' : ''}`}>{topLineupMeta}</p>
                </div>
              ) : (
                <p className="meta">
                  {teamProfile?.top_lineup_summary || 'Current lineup summary unavailable.'}
                </p>
              )}
              <div className="team-lineup-strip" role="list" aria-label="Most-used lineup players">
                {topLineupPlayers.length ? topLineupPlayers.map((player) => (
                  <button
                    key={`top-lineup-${player.id ?? player.name}`}
                    type="button"
                    className="team-lineup-chip"
                    disabled={player.id === null}
                    role="listitem"
                    onClick={() => {
                      if (player.id !== null) onOpenPlayerLookup(player.id, player.name);
                    }}
                  >
                    {player.name}
                  </button>
                )) : (
                  <p className="meta">No top-lineup player list available.</p>
                )}
              </div>

              <div className="team-rotation-block">
                <h4>Subs</h4>
                {rotationPlayers.length ? (
                  <ul className="team-rotation-list">
                    {rotationPlayers.map((player) => (
                      <li key={`rotation-${player.id ?? player.name}`}>
                        <span>{player.name}</span>
                        <span className="meta">
                          {player.matchesPlayed > 0
                            ? `${pluralize(player.matchesPlayed, 'match')} · ${pct(playerBaselineMatches > 0 ? player.matchesPlayed / playerBaselineMatches : null)}`
                            : 'lineup only'}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="meta">No additional subs surfaced beyond the primary four.</p>
                )}
              </div>
            </article>
          </div>

          <article className="analytics-panel analytics-panel-wide">
            <div className="team-section-heading">
              <div>
                <h3>History</h3>
                <p className="meta">
                  Tournament-by-tournament match history for the currently loaded team scope.
                </p>
              </div>
            </div>
            <div className="team-history-toolbar">
              <span className="team-history-legend">Ribbons run earliest on the left and latest on the right.</span>
              {historyPageCount > 1 ? (
                <div className="team-history-pagination">
                  <button
                    type="button"
                    className="button button-secondary"
                    disabled={historyPage <= 1}
                    onClick={() => changeHistoryPage(historyPage - 1)}
                  >
                    Previous
                  </button>
                  <span className="team-history-pagination-meta">
                    Page {historyPage} of {historyPageCount} · {pluralize(matchEvents.length, 'tournament')}
                  </span>
                  <button
                    type="button"
                    className="button button-secondary"
                    disabled={historyPage >= historyPageCount}
                    onClick={() => changeHistoryPage(historyPage + 1)}
                  >
                    Next
                  </button>
                </div>
              ) : null}
            </div>
            {recentForm.length ? (
              <div className="team-form-strip" aria-label="Recent form">
                <span className="team-form-label">Recent form, newest first</span>
                <div className="team-form-pills">
                  {recentForm.map((row) => (
                    <span
                      key={row.key}
                      className={`team-form-pill ${row.tone}`}
                      title={row.title}
                    >
                      {row.label}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
            {matches.length ? (
              <div className="team-match-events">
                {visibleMatchEvents.map((event) => {
                  const isExpanded = expandedEventKeys.includes(event.key);
                  return (
                    <section key={event.key} className={`team-match-event ${isExpanded ? 'is-expanded' : ''}`}>
                      <button
                        type="button"
                        className="team-match-event-head team-match-event-toggle"
                        aria-expanded={isExpanded}
                        onClick={() => toggleEventKey(event.key)}
                      >
                        <div className="team-match-event-title">
                          <span className="team-match-event-name">{event.tournamentName}</span>
                          <span className="team-match-event-meta">
                            {event.whenLabel} · {pluralize(event.rows.length, 'match')} · {event.wins}-{event.losses}
                            {event.opponentSummary ? ` · vs ${event.opponentSummary}` : ''}
                          </span>
                        </div>
                        <span className="team-match-event-tier">{event.tournamentTier || 'Unscored'}</span>
                        <div className="team-match-event-progress">
                          <div
                            className="team-match-event-ribbon"
                            aria-label="Tournament progression ribbon, earliest match on the left and latest on the right"
                            title="Tournament progression, earliest match on the left and latest on the right."
                            style={{ gridTemplateColumns: `repeat(${Math.max(event.ribbonRows?.length || 0, 1)}, minmax(0, 1fr))` }}
                          >
                            {(event.ribbonRows || []).map((row) => (
                              <span
                                key={`ribbon-${row.key}`}
                                className={`team-match-ribbon-cell ${row.tone}`}
                                title={`${row.opponentTeamName}: ${row.scoreLabel}`}
                              />
                            ))}
                          </div>
                        </div>
                        <span className="team-match-event-chevron" aria-hidden="true">{isExpanded ? '▾' : '▸'}</span>
                      </button>
                      {isExpanded ? (
                        <ul className="team-match-rows">
                          {event.rows.map((row) => (
                        <li
                          key={row.key}
                          className={`team-match-row ${row.opponentTeamId ? '' : 'is-disabled'}`}
                        >
                          <button
                            type="button"
                            className="team-row-primary team-match-primary"
                            disabled={!row.opponentTeamId}
                            onClick={() => onOpenTeamPage(
                              row.opponentTeamId ? [row.opponentTeamId] : [],
                              row.opponentTeamName,
                              actionSnapshotId,
                            )}
                            aria-label={`Open team page for ${row.opponentTeamName}`}
                          >
                            <span className="team-match-row-top">
                              <span className="team-match-date">{row.relativeDate}</span>
                              <span className="team-match-opponent-block">
                                <span className="team-match-opp">{row.opponentTeamName}</span>
                                {row.opponentEventCount > 1 ? (
                                  <span className="team-match-opponent-note">
                                    {pluralize(row.opponentEventCount, 'meeting')} in this event
                                  </span>
                                ) : null}
                              </span>
                              <span className={`team-result-pill ${row.tone}`}>{row.resultLabel}</span>
                              <span className="team-match-score">{row.scoreLabel}</span>
                            </span>
                            <span className="team-match-row-bottom">
                              <span className="team-match-roster-group">
                                <span className="team-match-roster-label">vs</span>
                                <span className="team-match-roster-chips">
                                  {row.opponentRosterPlayers.length ? row.opponentRosterPlayers.map((name) => (
                                    <span key={`${row.key}-opp-${name}`} className="team-match-roster-chip">{name}</span>
                                  )) : (
                                    <span className="team-match-roster-chip is-muted">Roster unavailable</span>
                                  )}
                                </span>
                              </span>
                              {!row.usesDefaultTeamRoster ? (
                                <span className="team-match-roster-group is-owned">
                                  <span className="team-match-roster-label">Tracked</span>
                                  <span className="team-match-roster-chips">
                                    {row.teamRosterPlayers.length ? row.teamRosterPlayers.map((name) => (
                                      <span key={`${row.key}-team-${name}`} className="team-match-roster-chip is-owned">{name}</span>
                                    )) : (
                                      <span className="team-match-roster-chip is-muted">Roster unavailable</span>
                                    )}
                                  </span>
                                </span>
                              ) : null}
                            </span>
                          </button>
                          <div className="team-hover-actions">
                            <button
                              type="button"
                              className="team-hover-action"
                              disabled={!row.opponentTeamId}
                              onClick={() => onOpenHeadToHead(
                                selectedFamilyIds,
                                row.opponentTeamId ? [row.opponentTeamId] : [],
                                actionSnapshotId,
                              )}
                            >
                              Compare
                            </button>
                          </div>
                        </li>
                          ))}
                        </ul>
                      ) : null}
                    </section>
                  );
                })}
              </div>
            ) : (
              <p className="team-missing-note">
                {matchHistoryError || 'No match log is available for this team yet.'}
              </p>
            )}
          </article>

          <article className="analytics-panel analytics-panel-wide">
            <div className="team-section-heading">
              <div>
                <h3>Comparable Teams</h3>
                <p className="meta">Similarity and tracked lineup volume from the current search snapshot.</p>
              </div>
            </div>
            {comparableRows.length ? (
              <ul className="team-compare-list">
                {comparableRows.map((row) => {
                  const similarity = Math.max(0, Math.min(Number(row.sim_to_query || 0) * 100, 100));
                  const duplicateNameCount = comparableNameCounts.get(normalizeTimelineName(row.team_name)) || 0;
                  const comparableMeta = [
                    `${pct(row.sim_to_query)} similar`,
                    Number.isFinite(Number(row.lineup_count))
                      ? `${pluralize(row.lineup_count, 'tracked lineup')}`
                      : null,
                    duplicateNameCount > 1 ? 'same-name variant' : null,
                  ].filter(Boolean).join(' · ');

                  return (
                    <li key={`team-neighbor-${row.team_id}`} className="team-compare-row">
                      <button
                        type="button"
                        className="team-row-primary team-compare-primary"
                        onClick={() => onOpenTeamPage([row.team_id], row.team_name, actionSnapshotId)}
                        aria-label={`Open team page for ${row.team_name}`}
                        title={`Team ID ${row.team_id}`}
                      >
                        <span className="team-compare-main">
                          <span className="team-compare-name">{row.team_name}</span>
                          <span className="team-compare-meta">{comparableMeta || 'Comparable roster profile'}</span>
                        </span>
                        <span className="team-compare-visual">
                          <span className="team-compare-match" aria-hidden="true">
                            <span style={{ width: `${similarity}%` }} />
                          </span>
                          <span className="team-compare-pct">{pct(row.sim_to_query)}</span>
                        </span>
                      </button>
                      <div className="team-hover-actions">
                        <button
                          type="button"
                          className="team-hover-action"
                          onClick={() => onOpenHeadToHead(selectedFamilyIds, [row.team_id], actionSnapshotId)}
                        >
                          Compare
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="team-missing-note">No comparable teams loaded yet.</p>
            )}
          </article>

          {nameTimelineRows.length ? (
            <div className="team-alias-strip">
              <div className="team-alias-head">
                <p className="team-alias-kicker">
                  Name history{teamScope === 'family' ? ` · family of ${selectedFamilyIds.length}` : ''}
                </p>
                <span className="team-alias-meta">Newest first · registrations · matches</span>
              </div>
              <div className="team-alias-rail">
                {nameTimelineRows.map((row, index) => {
                  const isCurrent = index === 0;
                  return (
                    <div
                      key={`team-alias-${row.key}`}
                      className={`team-alias-item ${isCurrent ? 'is-current' : ''}`}
                      title={`${row.name} · ${pluralize(row.registrationCount, 'registration')} · ${pluralize(row.totalMatches, 'match')}`}
                    >
                      <div className="team-alias-name">{row.name}</div>
                      <div className="team-alias-date">
                        {row.latestEventTimeMs ? relativeDate(row.latestEventTimeMs) : 'n/a'}
                      </div>
                      <div className="team-alias-matches">
                        {pluralize(row.registrationCount, 'registration')} · {pluralize(row.totalMatches, 'match')}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
