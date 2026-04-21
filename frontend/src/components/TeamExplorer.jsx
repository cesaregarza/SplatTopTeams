import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchAnalyticsTeam, fetchAnalyticsTeamMatches, fetchTeamSearch } from '../api';

const DEFAULT_CLUSTER_MODE = 'family';
const DEFAULT_NEIGHBORS = 12;
const DEFAULT_MATCH_LIMIT = 25;
const DEFAULT_TEAM_SCOPE = 'family';

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

function sameTeamIds(left, right) {
  const normalizedLeft = uniqueTeamIds(parseTeamIdList(left));
  const normalizedRight = uniqueTeamIds(parseTeamIdList(right));
  if (normalizedLeft.length !== normalizedRight.length) return false;
  const rightSet = new Set(normalizedRight);
  return normalizedLeft.every((value) => rightSet.has(value));
}

function pct(value) {
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
    .map((player) => String(player?.player_name || '').trim())
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

function rowMatchesSelectedTeam(row, teamIds) {
  const rowIds = uniqueTeamIds([
    Number(row?.team_id),
    ...parseTeamIdList(row?.consolidated_team_ids),
  ]);
  return rowIds.some((teamId) => teamIds.includes(teamId));
}

function playerRole(player, baselineMatches, index) {
  if (player.matchesPlayed <= 0) return 'Top lineup';
  if (baselineMatches <= 0) return index < 4 ? 'Core' : 'Rotation';
  const share = player.matchesPlayed / baselineMatches;
  if (share >= 0.75) return 'Core';
  if (share >= 0.4) return 'Regular';
  return 'Rotation';
}

function continuityLabel(topShare, distinctLineups) {
  if (!Number.isFinite(topShare)) return 'Continuity unavailable';
  if (topShare >= 0.75) return 'Very stable core';
  if (topShare >= 0.5) return distinctLineups > 2 ? 'Stable core with rotation' : 'Mostly stable core';
  return distinctLineups > 3 ? 'Heavy rotation' : 'Rotating around the core';
}

function buildRecentForm(matches, limit = 5) {
  if (!Array.isArray(matches) || !matches.length) return [];
  return matches.slice(0, limit).map((row) => ({
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
    });
  }

  return Array.from(groups.values())
    .map((group) => ({
      ...group,
      whenLabel: group.latestEventTimeMs ? relativeDate(group.latestEventTimeMs) : 'n/a',
      rows: group.rows.sort((left, right) => {
        const leftScore = Number(left.eventTimeMs || 0);
        const rightScore = Number(right.eventTimeMs || 0);
        return rightScore - leftScore;
      }),
    }))
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
  const [teamLab, setTeamLab] = useState(null);
  const [teamProfile, setTeamProfile] = useState(null);
  const [matchHistory, setMatchHistory] = useState(null);
  const [matchHistoryError, setMatchHistoryError] = useState('');
  const [resolvedTeamIds, setResolvedTeamIds] = useState([]);
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
      setTeamLab(null);
      setTeamProfile(null);
      setMatchHistory(null);
      setResolvedTeamIds([]);
      return;
    }

    const primaryTeamId = requestedIds[0];

    setLoading(true);
    setError('');
    setMatchHistoryError('');
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
      if (teamPayload.status !== 'fulfilled') {
        throw teamPayload.reason;
      }

      setTeamLab(teamPayload.value);

      let matchedRow = null;
      if (searchPayload.status === 'fulfilled') {
        const searchRows = searchPayload.value?.results || [];
        matchedRow = resolveTeamProfileRow(searchRows, requestedIds);
        setTeamProfile(matchedRow || searchRows[0] || null);
      } else {
        setTeamProfile(null);
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

    } catch (err) {
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
  const enteredIds = useMemo(
    () => uniqueTeamIds(parseTeamIdList(teamIdsInput)),
    [teamIdsInput],
  );
  const teamProfilePlayers = useMemo(
    () => normalizePlayerRows(teamProfile?.core_lineup_players || teamProfile?.top_lineup_players || []),
    [teamProfile],
  );
  const topLineupPlayers = useMemo(
    () => normalizePlayerRows(teamProfile?.top_lineup_players || []),
    [teamProfile],
  );
  const neighborsRows = teamLab?.neighbors || [];
  const matches = matchHistory?.matches || [];
  const fallbackSummary = useMemo(() => {
    if (!teamLab?.team) return null;
    const familyIds = resolvedTeamIds.length ? resolvedTeamIds : selectedIds.length ? selectedIds : [teamLab.team.team_id];
    const familyNames = familyIds.map((teamId) => {
      if (teamId === teamLab.team.team_id) return teamLab.team.team_name;
      return `Team ${teamId}`;
    });
    const fallbackMatchSummary = teamLab?.match_summary || {};
    const wins = Number(fallbackMatchSummary.wins);
    const totalMatches = Number(fallbackMatchSummary.matches);
    const losses = Number.isFinite(totalMatches) && Number.isFinite(wins)
      ? Math.max(0, totalMatches - wins)
      : 0;

    return {
      primary_team_id: familyIds[0],
      primary_team_name: teamLab.team.team_name,
      team_ids: familyIds,
      team_names: familyNames,
      selected_team_count: familyIds.length,
      total_matches: Number.isFinite(totalMatches) ? totalMatches : 0,
      wins: Number.isFinite(wins) ? wins : 0,
      losses,
      unresolved_matches: 0,
      decided_matches: Number.isFinite(totalMatches) ? totalMatches : 0,
      win_rate: Number(fallbackMatchSummary.win_rate) || 0,
      tournaments: 0,
      tournament_tier_distribution: null,
      tournament_tier_match_distribution: null,
    };
  }, [resolvedTeamIds, selectedIds, teamLab]);
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
  const uniquePlayerCount = Number(teamProfile?.unique_player_count ?? teamProfilePlayers.length ?? 0);
  const distinctLineupCount = Number(teamProfile?.distinct_lineup_count ?? team?.distinct_lineup_count ?? 0);
  const topLineupShare = Number(teamProfile?.top_lineup_match_share ?? teamProfile?.top_lineup_share);
  const rotationPlayers = teamProfilePlayers.slice(4);
  const continuityHint = continuityLabel(topLineupShare, distinctLineupCount);
  const lastActiveLabel = teamProfile?.event_time_ms ? relativeDate(teamProfile.event_time_ms) : 'n/a';
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
  const matchEvents = useMemo(() => buildMatchEvents(matches), [matches]);
  const teamSummaryLine = useMemo(() => {
    if (!pageTitle) return '';
    const bits = [
      primaryMatchCount > 0 ? `${pageTitle} has ${primaryMatchCount} recorded matches` : null,
      tournamentCount > 0 ? `across ${tournamentCount} event${tournamentCount === 1 ? '' : 's'}` : null,
      uniquePlayerCount > 0 ? `with ${uniquePlayerCount} player${uniquePlayerCount === 1 ? '' : 's'} used` : null,
    ].filter(Boolean);
    if (!bits.length && uniquePlayerCount > 0) {
      bits.push(`${pageTitle} has ${uniquePlayerCount} player${uniquePlayerCount === 1 ? '' : 's'} in the current roster profile`);
    }
    if (!bits.length && tournamentCount > 0) {
      bits.push(`${pageTitle} appears in ${tournamentCount} tracked event${tournamentCount === 1 ? '' : 's'}`);
    }
    if (!bits.length) return '';
    const continuity = Number.isFinite(topLineupShare)
      ? `Top lineup share is ${pct(topLineupShare)}, which points to ${continuityHint.toLowerCase()}.`
      : '';
    return `${bits.join(' ')}. ${continuity}`.trim();
  }, [continuityHint, pageTitle, primaryMatchCount, topLineupShare, tournamentCount, uniquePlayerCount]);
  const recentForm = useMemo(() => buildRecentForm(matches), [matches]);
  const scopeSummary = useMemo(() => {
    if (!selectedFamilyIds.length) return '';

    if (teamScope === 'family') {
      if (enteredIds.length && !sameTeamIds(enteredIds, selectedFamilyIds)) {
        return `Family mode expanded ${enteredIds[0]} into ${selectedFamilyIds.length} related registrations.`;
      }
      return `Family mode is using ${selectedFamilyIds.length} registration${selectedFamilyIds.length === 1 ? '' : 's'} together.`;
    }

    return `Individual mode is using exactly ${selectedFamilyIds.length} ID${selectedFamilyIds.length === 1 ? '' : 's'}.`;
  }, [enteredIds, selectedFamilyIds, teamScope]);
  const scopeExpanded = teamScope === 'family'
    && enteredIds.length
    && selectedFamilyIds.length
    && !sameTeamIds(enteredIds, selectedFamilyIds);

  async function onSubmit(event) {
    event.preventDefault();
    await loadTeam(teamIdsInput, normalizedSelectedSnapshotId);
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
          {scopeSummary ? (
            <div className="team-scope-banner" role="status">
              <span className="team-scope-icon">{teamScope === 'family' ? 'F' : '1'}</span>
              <span>
                {scopeExpanded
                  ? `Family mode expanded ${enteredIds[0]} into ${selectedFamilyIds.length} related registrations.`
                  : scopeSummary}
              </span>
            </div>
          ) : null}
          {teamSummaryLine ? (
            <p className="team-analyst-summary">{teamSummaryLine}</p>
          ) : null}
          {nameTimelineRows.length ? (
            <div className="team-alias-strip">
              <div className="team-alias-head">
                <p className="team-alias-kicker">
                  Name history{teamScope === 'family' ? ` · family of ${selectedFamilyIds.length}` : ''}
                </p>
                <span className="team-alias-meta">newest unique names → oldest</span>
              </div>
              <div className="team-alias-rail">
                {nameTimelineRows.map((row, index) => {
                  const isCurrent = index === 0;
                  return (
                    <div
                      key={`team-alias-${row.key}`}
                      className={`team-alias-item ${isCurrent ? 'is-current' : ''}`}
                      title={`${row.name} · ${row.registrationCount} registration${row.registrationCount === 1 ? '' : 's'} · ${row.totalMatches} matches`}
                    >
                      <div className="team-alias-name">{row.name}</div>
                      <div className="team-alias-date">
                        {row.latestEventTimeMs ? relativeDate(row.latestEventTimeMs) : 'n/a'}
                      </div>
                      <div className="team-alias-matches">
                        {row.registrationCount} registration{row.registrationCount === 1 ? '' : 's'} · {row.totalMatches} matches
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          <div className="analytics-cards team-summary-grid analytics-stats">
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Win rate</span>
              <span className="analytics-card-value stat-value">{pct(effectiveSummary?.win_rate ?? teamLab?.match_summary?.win_rate)}</span>
            </article>
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Last active</span>
              <span className="analytics-card-value stat-value">{lastActiveLabel}</span>
            </article>
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Matches</span>
              <span className="analytics-card-value stat-value">{primaryMatchCount}</span>
            </article>
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Events</span>
              <span className="analytics-card-value stat-value">{tournamentCount || 'n/a'}</span>
            </article>
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Players used</span>
              <span className="analytics-card-value stat-value">{uniquePlayerCount || 'n/a'}</span>
            </article>
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Lineups used</span>
              <span className="analytics-card-value stat-value">{distinctLineupCount || 'n/a'}</span>
            </article>
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Core lineup share</span>
              <span className="analytics-card-value stat-value">{Number.isFinite(topLineupShare) ? pct(topLineupShare) : 'n/a'}</span>
            </article>
            <article className="analytics-card stat">
              <span className="analytics-card-label stat-label">Continuity</span>
              <span className="analytics-card-value stat-value">{continuityHint}</span>
            </article>
          </div>

          <div className="analytics-grid team-player-grid">
            <article className="analytics-panel analytics-panel-wide">
              <div className="team-section-heading">
                <div>
                  <h3>Core Roster</h3>
                  <p className="meta">
                    Player usage ordered by match presence across the current roster family.
                  </p>
                </div>
              </div>
              {teamProfilePlayers.length ? (
                <ul className="team-roster-list">
                  {teamProfilePlayers.map((player, index) => {
                    const share = playerBaselineMatches > 0
                      ? Math.round((player.matchesPlayed / playerBaselineMatches) * 100)
                      : 0;
                    const role = playerRole(player, playerBaselineMatches, index);
                    const roleKey = role.toLowerCase().replace(/\s+/g, '-');
                    return (
                      <li key={`team-player-${player.id ?? player.name}`} className="team-roster-row">
                        <div className="team-roster-name">
                          <span>{player.name}</span>
                          <span className={`team-roster-role is-${roleKey}`}>{role}</span>
                        </div>
                        <span className="team-roster-matches">
                          {player.matchesPlayed > 0 ? player.matchesPlayed : '—'}
                        </span>
                        <div className="team-roster-bar" aria-hidden="true">
                          <span style={{ width: `${Math.max(0, Math.min(share, 100))}%` }} />
                        </div>
                        <span className="team-roster-share">
                          {player.matchesPlayed > 0 ? `${share}%` : 'n/a'}
                        </span>
                        <div className="team-compact-actions team-roster-actions">
                          <button
                            type="button"
                            className="button button-secondary team-compact-button"
                            disabled={player.id === null}
                            onClick={() => onOpenPlayerLookup(player.id, player.name)}
                            aria-label={`Open player history for ${player.name}`}
                          >
                            History
                          </button>
                          {player.sendouUrl ? (
                            <a
                              className="button button-secondary team-compact-button"
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
              <h3>Top Lineup</h3>
              <p className="meta">
                {teamProfile?.top_lineup_summary || 'Current lineup summary unavailable.'}
              </p>
              <div className="team-player-chip-grid">
                {topLineupPlayers.length ? topLineupPlayers.map((player) => (
                  <button
                    key={`top-lineup-${player.id ?? player.name}`}
                    type="button"
                    className="team-player-chip"
                    disabled={player.id === null}
                    onClick={() => {
                      if (player.id !== null) onOpenPlayerLookup(player.id, player.name);
                    }}
                  >
                    <span className="team-player-chip-name">{player.name}</span>
                    <span className="team-player-chip-meta">
                      {player.matchesPlayed > 0 ? `${player.matchesPlayed} matches` : 'Open player history'}
                    </span>
                  </button>
                )) : (
                  <p className="meta">No top-lineup player list available.</p>
                )}
              </div>

              <div className="team-rotation-block">
                <h4>Supporting Players</h4>
                {rotationPlayers.length ? (
                  <ul className="team-rotation-list">
                    {rotationPlayers.map((player) => (
                      <li key={`rotation-${player.id ?? player.name}`}>
                        <span>{player.name}</span>
                        <span className="meta">
                          {player.matchesPlayed > 0 ? `${player.matchesPlayed} matches` : 'lineup only'}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="meta">No supporting players surfaced beyond the core four.</p>
                )}
              </div>
            </article>
          </div>

          <article className="analytics-panel analytics-panel-wide">
            <div className="team-section-heading">
              <div>
                <h3>Recent Matches</h3>
                <p className="meta">
                  Match-by-match scouting log for this team family.
                </p>
              </div>
            </div>
            {recentForm.length ? (
              <div className="team-form-strip" aria-label="Recent form">
                <span className="team-form-label">Recent form</span>
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
                {matchEvents.map((event) => (
                  <section key={event.key} className="team-match-event">
                    <header className="team-match-event-head">
                      <div className="team-match-event-title">
                        <span className="team-match-event-name">{event.tournamentName}</span>
                        <span className="team-match-event-meta">
                          {event.whenLabel} · {event.rows.length} match{event.rows.length === 1 ? '' : 'es'} · {event.wins}-{event.losses}
                        </span>
                      </div>
                      <span className="team-match-event-tier">{event.tournamentTier || 'Unscored'}</span>
                      <div className="team-match-event-ribbon" aria-label="Results ribbon">
                        {event.rows.map((row) => (
                          <span
                            key={`ribbon-${row.key}`}
                            className={`team-match-ribbon-cell ${row.tone}`}
                            title={`${row.opponentTeamName}: ${row.scoreLabel}`}
                          />
                        ))}
                      </div>
                    </header>
                    <div className="team-match-rows">
                      {event.rows.map((row) => (
                        <div key={row.key} className="team-match-row">
                          <span className="team-match-date">{row.relativeDate}</span>
                          <span className="team-match-opp">{row.opponentTeamName}</span>
                          <span className="team-match-score">{row.scoreLabel}</span>
                          <span className="team-match-roster">
                            {row.teamRosterLabel}
                            <br />
                            <span className="meta">vs {row.opponentRosterLabel}</span>
                          </span>
                          <span className={`team-result-pill ${row.tone}`}>{row.resultLabel}</span>
                          <div className="team-row-actions">
                            <button
                              type="button"
                              className="button button-secondary"
                              disabled={!row.opponentTeamId}
                              onClick={() => onOpenHeadToHead(
                                selectedFamilyIds,
                                row.opponentTeamId ? [row.opponentTeamId] : [],
                                actionSnapshotId,
                              )}
                            >
                              Compare
                            </button>
                            <button
                              type="button"
                              className="button button-secondary"
                              disabled={!row.opponentTeamId}
                              onClick={() => onOpenTeamPage(
                                row.opponentTeamId ? [row.opponentTeamId] : [],
                                row.opponentTeamName,
                                actionSnapshotId,
                              )}
                            >
                              Open
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            ) : (
              <p className="team-missing-note">
                {matchHistoryError || 'No match log is available for this team yet.'}
              </p>
            )}
          </article>

          <div className="analytics-grid">
            <article className="analytics-panel">
              <div className="team-section-heading">
                <div>
                  <h3>Event Timeline</h3>
                  <p className="meta">
                    Consolidated names used by this roster family, ordered by last appearance.
                  </p>
                </div>
              </div>
              <div className="table-wrap">
                <table className="analytics-table">
                  <thead>
                    <tr>
                      <th>Last seen</th>
                      <th>Name</th>
                      <th>Registrations</th>
                      <th>Matches</th>
                    </tr>
                  </thead>
                  <tbody>
                    {nameTimelineRows.length ? nameTimelineRows.map((row) => (
                      <tr key={`registration-${row.key}`}>
                        <td title={fmtDate(row.latestEventTimeMs)}>{row.latestEventTimeMs ? relativeDate(row.latestEventTimeMs) : 'n/a'}</td>
                        <td className="analytics-team-name">{row.name}</td>
                        <td>{row.registrationCount}</td>
                        <td>{row.totalMatches}</td>
                      </tr>
                    )) : (
                      <tr>
                        <td colSpan={4} className="table-empty-cell">No registration history available yet.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </article>

            <article className="analytics-panel">
              <div className="team-section-heading">
                <div>
                  <h3>Comparable Teams</h3>
                  <p className="meta">Teams with similar roster construction and lineup usage. Volume reflects indexed observations, not distinct lineups.</p>
                </div>
              </div>
              {comparableRows.length ? (
                <ul className="team-compare-list">
                  {comparableRows.map((row) => (
                    <li key={`team-neighbor-${row.team_id}`} className="team-compare-row">
                      <span className="team-compare-name">{row.team_name}</span>
                      <div className="team-compare-match" aria-hidden="true">
                        <span style={{ width: `${Math.max(0, Math.min(Number(row.sim_to_query || 0) * 100, 100))}%` }} />
                      </div>
                      <span className="team-compare-pct">{pct(row.sim_to_query)}</span>
                      <span className="team-compare-lineups">{row.lineup_count ?? 'n/a'} observations</span>
                      <div className="team-compact-actions">
                        <button
                          type="button"
                          className="button button-secondary team-compact-button"
                          onClick={() => onOpenHeadToHead(selectedFamilyIds, [row.team_id], actionSnapshotId)}
                        >
                          Compare
                        </button>
                        <button
                          type="button"
                          className="button button-secondary team-compact-button"
                          onClick={() => onOpenTeamPage([row.team_id], row.team_name, actionSnapshotId)}
                        >
                          Open
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="team-missing-note">No comparable teams loaded yet.</p>
              )}
            </article>
          </div>
        </>
      ) : null}
    </section>
  );
}
