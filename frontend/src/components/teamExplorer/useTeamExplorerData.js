import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchAnalyticsTeam,
  fetchAnalyticsTeamMatches,
  fetchTeamSearch,
} from '../../api';
import {
  buildMatchEvents,
  buildNameTimeline,
  buildRecentForm,
  continuityLabel,
  DEFAULT_CLUSTER_MODE,
  DEFAULT_MATCH_LIMIT,
  DEFAULT_NEIGHBORS,
  DEFAULT_TEAM_SCOPE,
  HISTORY_EVENTS_PER_PAGE,
  normalizePlayerRows,
  parseTeamIdList,
  playerRowKey,
  pluralize,
  relativeDate,
  resolveScopedTeamIds,
  resolveTeamProfileRow,
  uniqueTeamIds,
  pct,
} from './helpers';

const INITIAL_DETAIL_STATE = {
  loading: false,
  error: '',
  warning: '',
  teamLab: null,
  teamProfile: null,
  matchHistory: null,
  matchHistoryError: '',
  resolvedTeamIds: [],
};

const INITIAL_HISTORY_UI = {
  expandedEventKeys: [],
  page: 1,
};

export function useTeamExplorerData({
  selectedTeamId,
  selectedTeamIds,
  selectedSnapshotId,
  selectedTeamName,
}) {
  const [teamIdsInput, setTeamIdsInput] = useState('');
  const [teamScope, setTeamScope] = useState(DEFAULT_TEAM_SCOPE);
  const [detailState, setDetailState] = useState(INITIAL_DETAIL_STATE);
  const [historyUi, setHistoryUi] = useState(INITIAL_HISTORY_UI);
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
      setDetailState({
        ...INITIAL_DETAIL_STATE,
        error: 'Enter at least one valid team ID.',
      });
      return;
    }

    const primaryTeamId = requestedIds[0];

    setDetailState({
      ...INITIAL_DETAIL_STATE,
      loading: true,
      resolvedTeamIds: requestedIds,
    });
    setHistoryUi(INITIAL_HISTORY_UI);

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
      let teamProfileValue = null;
      if (searchPayload.status === 'fulfilled') {
        const searchRows = searchPayload.value?.results || [];
        matchedRow = resolveTeamProfileRow(searchRows, requestedIds);
        teamProfileValue = matchedRow || searchRows[0] || null;
      }

      let partialWarning = '';
      if (!teamLabValue && searchPayload.status === 'fulfilled') {
        partialWarning = 'Team analytics are temporarily unavailable. Showing the search snapshot and any match data that loads.';
      }
      if (!teamLabValue && searchPayload.status !== 'fulfilled') {
        throw teamPayload.reason || searchPayload.reason || new Error('Failed to load team detail');
      }

      const effectiveIds = resolveScopedTeamIds(requestedIds, matchedRow, teamScope);
      let matchHistoryValue = null;
      let matchHistoryError = '';

      try {
        matchHistoryValue = await fetchAnalyticsTeamMatches({
          teamId: primaryTeamId,
          teamIds: effectiveIds,
          limit: DEFAULT_MATCH_LIMIT,
          snapshotId: snapshotId || undefined,
        });
      } catch (_matchesError) {
        matchHistoryError = 'Recent matches are unavailable right now.';
      }

      setDetailState({
        loading: false,
        error: '',
        warning: partialWarning,
        teamLab: teamLabValue,
        teamProfile: teamProfileValue,
        matchHistory: matchHistoryValue,
        matchHistoryError,
        resolvedTeamIds: effectiveIds,
      });
    } catch (err) {
      setDetailState({
        ...INITIAL_DETAIL_STATE,
        error: err.message || 'Failed to load team detail',
      });
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

  const summary = detailState.matchHistory?.summary || null;
  const team = detailState.teamLab?.team || null;
  const teamProfilePlayers = useMemo(
    () => normalizePlayerRows(detailState.teamProfile?.core_lineup_players || detailState.teamProfile?.top_lineup_players || []),
    [detailState.teamProfile],
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
    const players = normalizePlayerRows(detailState.teamProfile?.top_lineup_players || []);
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
  }, [detailState.teamProfile, rosterOrderByPlayer, rosterPresenceByPlayer]);

  const neighborsRows = detailState.teamLab?.neighbors || [];
  const matches = detailState.matchHistory?.matches || [];
  const fallbackSummary = useMemo(() => {
    const primaryTeamId = Number(detailState.teamProfile?.team_id ?? detailState.teamLab?.team?.team_id);
    const primaryTeamName = String(detailState.teamProfile?.team_name || detailState.teamLab?.team?.team_name || '').trim();
    if (!Number.isFinite(primaryTeamId) || primaryTeamId <= 0 || !primaryTeamName) return null;

    const familyIds = detailState.resolvedTeamIds.length
      ? detailState.resolvedTeamIds
      : selectedIds.length
        ? selectedIds
        : [Math.trunc(primaryTeamId)];
    const familyNames = familyIds.map((teamId) => (
      teamId === Math.trunc(primaryTeamId) ? primaryTeamName : `Team ${teamId}`
    ));
    const fallbackMatchSummary = detailState.teamLab?.match_summary || {};
    const wins = Number(fallbackMatchSummary.wins);
    const totalMatches = Number.isFinite(Number(fallbackMatchSummary.matches))
      ? Number(fallbackMatchSummary.matches)
      : Number(detailState.teamProfile?.match_count);
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
      tournaments: Number(detailState.teamProfile?.tournament_count) || 0,
      tournament_tier_distribution: null,
      tournament_tier_match_distribution: null,
    };
  }, [detailState.resolvedTeamIds, detailState.teamLab, detailState.teamProfile, selectedIds]);
  const effectiveSummary = summary || fallbackSummary;
  const pageTitle = effectiveSummary?.primary_team_name
    || team?.team_name
    || String(selectedTeamName || '').trim()
    || (selectedIds[0] ? `Team ${selectedIds[0]}` : 'Teams');
  const selectedFamilyIds = useMemo(() => {
    const summaryIds = uniqueTeamIds(parseTeamIdList(effectiveSummary?.team_ids));
    if (summaryIds.length) return summaryIds;
    if (detailState.resolvedTeamIds.length) return detailState.resolvedTeamIds;
    return selectedIds;
  }, [detailState.resolvedTeamIds, effectiveSummary?.team_ids, selectedIds]);
  const playerBaselineMatches = Number(
    detailState.teamProfile?.match_count
      ?? effectiveSummary?.total_matches
      ?? detailState.teamLab?.match_summary?.matches
      ?? 0,
  );
  const primaryMatchCount = Number(
    detailState.teamProfile?.match_count
      ?? effectiveSummary?.total_matches
      ?? detailState.teamLab?.match_summary?.matches
      ?? 0,
  );
  const tournamentCount = Number(detailState.teamProfile?.tournament_count ?? effectiveSummary?.tournaments ?? 0);
  const trackedPlayerCount = teamProfilePlayers.length || Number(detailState.teamProfile?.unique_player_count) || 0;
  const distinctLineupCount = Number(detailState.teamProfile?.distinct_lineup_count ?? team?.distinct_lineup_count ?? 0);
  const topLineupShare = Number(detailState.teamProfile?.top_lineup_match_share ?? detailState.teamProfile?.top_lineup_share);
  const topLineupMatchCount = Number(detailState.teamProfile?.top_lineup_match_count ?? 0);
  const rotationPlayers = teamProfilePlayers.slice(4);
  const continuityHint = continuityLabel(topLineupShare, distinctLineupCount);
  const actionSnapshotId = detailState.matchHistory?.snapshot_id || normalizedSelectedSnapshotId || null;
  const registrationRows = useMemo(() => {
    const primary = detailState.teamProfile
      ? [{
          team_id: detailState.teamProfile.team_id,
          team_name: detailState.teamProfile.team_name,
          tournament_id: detailState.teamProfile.tournament_id,
          event_time_ms: detailState.teamProfile.event_time_ms,
          match_count: detailState.teamProfile.match_count,
          tournament_count: detailState.teamProfile.tournament_count,
        }]
      : [];
    const aliases = Array.isArray(detailState.teamProfile?.consolidated_teams) ? detailState.teamProfile.consolidated_teams : [];
    return [...primary, ...aliases]
      .filter((row) => row?.team_id)
      .sort((a, b) => Number(b?.event_time_ms || 0) - Number(a?.event_time_ms || 0));
  }, [detailState.teamProfile]);
  const nameTimelineRows = useMemo(() => buildNameTimeline(registrationRows), [registrationRows]);
  const comparableRows = neighborsRows.slice(0, 8);
  const comparableNameCounts = useMemo(() => {
    const counts = new Map();
    for (const row of comparableRows) {
      const key = String(row?.team_name || '').trim().toLowerCase();
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
    const start = (historyUi.page - 1) * HISTORY_EVENTS_PER_PAGE;
    return matchEvents.slice(start, start + HISTORY_EVENTS_PER_PAGE);
  }, [historyUi.page, matchEvents]);
  const latestMatchEventMs = useMemo(
    () => matches.reduce((latest, row) => Math.max(latest, Number(row?.event_time_ms || 0)), 0),
    [matches],
  );
  const latestRegistrationMs = registrationRows.length ? Number(registrationRows[0]?.event_time_ms || 0) : 0;
  const lastSeenMs = latestMatchEventMs || latestRegistrationMs || Number(detailState.teamProfile?.event_time_ms || 0);
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
    setHistoryUi((previousUi) => {
      const boundedPage = Math.max(1, Math.min(previousUi.page, historyPageCount));
      if (previousUi.page === boundedPage) return previousUi;
      return { ...previousUi, page: boundedPage };
    });
  }, [historyPageCount]);

  useEffect(() => {
    setHistoryUi((previousUi) => {
      const availableKeys = new Set(visibleMatchEvents.map((event) => event.key));
      const retainedKeys = previousUi.expandedEventKeys.filter((key) => availableKeys.has(key));
      if (retainedKeys.length) {
        return previousUi.expandedEventKeys === retainedKeys
          ? previousUi
          : { ...previousUi, expandedEventKeys: retainedKeys };
      }
      return {
        ...previousUi,
        expandedEventKeys: visibleMatchEvents.length ? [visibleMatchEvents[0].key] : [],
      };
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
    { label: 'Win Rate', value: pct(effectiveSummary?.win_rate ?? detailState.teamLab?.match_summary?.win_rate) },
    { label: 'Recorded Matches', value: primaryMatchCount || 'n/a' },
    { label: 'Events', value: tournamentCount || 'n/a' },
    { label: 'Last Seen', value: lastSeenLabel },
  ]), [detailState.teamLab?.match_summary?.win_rate, effectiveSummary?.win_rate, lastSeenLabel, primaryMatchCount, tournamentCount]);
  const rosterStats = useMemo(() => ([
    { label: 'Tracked Players', value: trackedPlayerCount || 'n/a' },
    { label: 'Distinct Lineups', value: distinctLineupCount || 'n/a' },
    { label: 'Most Common Lineup Share', value: Number.isFinite(topLineupShare) ? pct(topLineupShare) : 'n/a' },
    { label: 'Rotation Pattern', value: continuityHint },
  ]), [continuityHint, distinctLineupCount, topLineupShare, trackedPlayerCount]);

  const submitTeam = useCallback(async (event) => {
    event.preventDefault();
    await loadTeam(teamIdsInput, normalizedSelectedSnapshotId);
  }, [loadTeam, normalizedSelectedSnapshotId, teamIdsInput]);

  const toggleEventKey = useCallback((eventKey) => {
    setHistoryUi((previousUi) => ({
      ...previousUi,
      expandedEventKeys: previousUi.expandedEventKeys.includes(eventKey)
        ? previousUi.expandedEventKeys.filter((key) => key !== eventKey)
        : [...previousUi.expandedEventKeys, eventKey],
    }));
  }, []);

  const changeHistoryPage = useCallback((nextPage) => {
    setHistoryUi((previousUi) => ({
      ...previousUi,
      page: Math.max(1, Math.min(historyPageCount, nextPage)),
    }));
  }, [historyPageCount]);

  return {
    teamIdsInput,
    setTeamIdsInput,
    teamScope,
    setTeamScope,
    submitTeam,
    loadTeam,
    selectedIds,
    selectedIdsKey,
    normalizedSelectedSnapshotId,
    ...detailState,
    summary,
    effectiveSummary,
    team,
    matches,
    pageTitle,
    selectedFamilyIds,
    playerBaselineMatches,
    primaryMatchCount,
    tournamentCount,
    trackedPlayerCount,
    distinctLineupCount,
    topLineupShare,
    topLineupPlayers,
    topLineupMeta,
    lineupSupportMismatch,
    registrationRows,
    nameTimelineRows,
    comparableRows,
    comparableNameCounts,
    matchEvents,
    historyPage: historyUi.page,
    expandedEventKeys: historyUi.expandedEventKeys,
    historyPageCount,
    visibleMatchEvents,
    lastSeenLabel,
    recentForm,
    canonicalName,
    heroAliases,
    heroKicker,
    performanceStats,
    rosterStats,
    teamProfilePlayers,
    teamProfile: detailState.teamProfile,
    rotationPlayers,
    matchHistoryError: detailState.matchHistoryError,
    actionSnapshotId,
    toggleEventKey,
    changeHistoryPage,
  };
}
