import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchAnalyticsHeadToHead, fetchTeamSearch } from '../api';

function toEpochMs(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value <= 0) return null;

  // Upstream timestamps are mixed (seconds, milliseconds, microseconds).
  if (value > 1_000_000_000_000) return Math.floor(value / 1000);
  if (value > 10_000_000_000) return Math.floor(value);

  const milliseconds = Math.floor(value * 1000);
  return milliseconds < 946684800000 ? null : milliseconds;
}

function relativeTime(ms) {
  const value = toEpochMs(ms);
  if (value === null) return '—';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  const now = Date.now();
  const delta = now - value;
  if (delta <= 0) return 'just now';

  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  const week = 7 * day;
  const month = 30 * day;
  const year = 365 * day;

  if (delta < 45 * 1000) return 'just now';
  if (delta < 90 * minute) {
    const count = Math.round(delta / minute);
    return `${count} minute${count === 1 ? '' : 's'} ago`;
  }
  if (delta < 36 * hour) {
    const count = Math.round(delta / hour);
    return `${count} hour${count === 1 ? '' : 's'} ago`;
  }
  if (delta < 14 * day) {
    const count = Math.round(delta / day);
    return `${count} day${count === 1 ? '' : 's'} ago`;
  }
  if (delta < 8 * week) {
    const count = Math.round(delta / week);
    return `${count} week${count === 1 ? '' : 's'} ago`;
  }
  if (delta < 10 * month) {
    const count = Math.round(delta / month);
    return `${count} month${count === 1 ? '' : 's'} ago`;
  }
  if (delta < 2 * year) {
    const count = Math.round(delta / month);
    return `${count} month${count === 1 ? '' : 's'} ago`;
  }

  return `${Math.round(delta / year)} year${Math.round(delta / year) === 1 ? '' : 's'} ago`;
}

function toAbsoluteDate(ms) {
  const value = toEpochMs(ms);
  if (value === null) return '—';
  return new Date(value).toLocaleDateString();
}

function formatMatchTime(ms) {
  const abs = toAbsoluteDate(ms);
  if (abs === '—') return '—';
  return `${abs} (${relativeTime(ms)})`;
}

function safeInt(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function safeIntOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function safeFloat(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function percent(value) {
  return value === null ? 'n/a' : `${(value * 100).toFixed(0)}%`;
}

function formatMatchScore(value) {
  const parsed = safeFloat(value);
  if (parsed === null) return 'n/a';
  const asFixed = parsed.toFixed(0);
  if (Math.abs(parsed - Number(asFixed)) < 1e-9) {
    return asFixed;
  }
  return parsed.toString();
}

const TOURNAMENT_TIER_META = [
  { id: 'x', label: 'X', max: 5.0 },
  { id: 's_plus', label: 'S+', max: 10.0 },
  { id: 's', label: 'S', max: 20.0 },
  { id: 'a_plus', label: 'A+', max: 40.0 },
  { id: 'a', label: 'A', max: 80.0 },
  { id: 'a_minus', label: 'A-', max: 160.0 },
  { id: 'unscored', label: 'Unscored' },
];

function classifyTournamentTier(value) {
  const parsed = safeFloat(value);
  if (parsed === null) {
    return TOURNAMENT_TIER_META.find((tier) => tier.id === 'unscored');
  }

  for (const tier of TOURNAMENT_TIER_META) {
    if (tier.id === 'unscored') continue;
    if (parsed <= tier.max) {
      return tier;
    }
  }

  return TOURNAMENT_TIER_META.find((tier) => tier.id === 'unscored');
}

function resolveTournamentTier(score, tierId) {
  if (typeof tierId === 'string') {
    const found = TOURNAMENT_TIER_META.find((tier) => tier.id === tierId);
    if (found) {
      return found;
    }
  }

  return classifyTournamentTier(score);
}

function formatScorePair(sideA, sideB) {
  return `${formatMatchScore(sideA)} – ${formatMatchScore(sideB)}`;
}

function normalizeText(value) {
  const text = String(value || '').trim();
  return text || null;
}

function isStageLikeLabel(value) {
  const normalized = normalizeText(value)?.toLowerCase();
  if (!normalized) return false;

  return [
    'group',
    'group stage',
    'qualifier',
    'qualifiers',
    'top cut',
    'winners',
    'losers',
    'final',
    'semi',
    'upper bracket',
    'lower bracket',
    'bracket',
    'single elimination',
    'double elimination',
    'swiss',
    'robin',
    'round',
    'elimination',
  ].some((token) => normalized.includes(token));
}

function normalizeMapMode(value, mapsCount = null) {
  const normalized = normalizeText(value);
  if (!normalized) return null;

  const lowered = normalized.toLowerCase().replace(/[_-]+/g, ' ').trim();
  if (isStageLikeLabel(lowered)) return null;

  if (
    lowered.includes('best of') ||
    lowered === 'bestof' ||
    /^bo\s*\d+$/i.test(lowered)
  ) {
    return mapsCount ? `Best of ${mapsCount}` : 'Best of';
  }

  if (
    lowered.includes('play all') ||
    lowered === 'playall'
  ) {
    return mapsCount ? `Play all ${mapsCount}` : 'Play all';
  }

  return null;
}

const MODE_LABEL_BY_TOKEN = {
  SZ: 'Splat Zones',
  TC: 'Tower Control',
  RM: 'Rainmaker',
  CB: 'Clam Blitz',
  TW: 'Turf War',
};
const SHOW_MAP_RULESET_COLUMN = false;

function normalizeTagList(value) {
  if (Array.isArray(value)) {
    return value
      .map((item) => normalizeText(item))
      .filter((item) => Boolean(item));
  }

  const normalized = normalizeText(value);
  if (!normalized) return [];

  if (normalized.startsWith('[') && normalized.endsWith(']')) {
    try {
      const parsed = JSON.parse(normalized);
      if (Array.isArray(parsed)) {
        return parsed
          .map((item) => normalizeText(item))
          .filter((item) => Boolean(item));
      }
    } catch {
      return [];
    }
  }

  return normalized
    .split(/[\s,]+/)
    .map((item) => normalizeText(item))
    .filter((item) => Boolean(item));
}

function detectModeToken(value) {
  const normalized = normalizeText(value);
  if (!normalized) return null;

  const upper = normalized.toUpperCase();
  if (upper.includes('SPLAT ZONES')) return 'SZ';
  if (upper.includes('TOWER CONTROL')) return 'TC';
  if (upper.includes('RAINMAKER')) return 'RM';
  if (upper.includes('CLAM BLITZ')) return 'CB';
  if (upper.includes('TURF WAR')) return 'TW';

  const parts = upper.split(/[^A-Z0-9]+/).filter(Boolean);
  for (const part of parts) {
    if (MODE_LABEL_BY_TOKEN[part]) {
      return part;
    }
  }
  return null;
}

function resolveGameModeLabel(mapMode, fallbackMode, fallbackStyle, tournamentTags) {
  const candidates = [
    mapMode,
    fallbackMode,
    fallbackStyle,
    ...normalizeTagList(tournamentTags),
  ];

  for (const candidate of candidates) {
    const token = detectModeToken(candidate);
    if (token && MODE_LABEL_BY_TOKEN[token]) {
      return MODE_LABEL_BY_TOKEN[token];
    }
  }

  return null;
}

const MATCH_SORT_FIELDS = [
  { value: 'time|desc', label: 'Match time · newest first' },
  { value: 'time|asc', label: 'Match time · oldest first' },
  { value: 'match_id|desc', label: 'Match ID · high to low' },
  { value: 'match_id|asc', label: 'Match ID · low to high' },
  { value: 'round|asc', label: 'Round · low to high' },
  { value: 'round|desc', label: 'Round · high to low' },
  { value: 'tournament|asc', label: 'Tournament · A to Z' },
  { value: 'tournament|desc', label: 'Tournament · Z to A' },
  { value: 'tier|desc', label: 'Tournament tier · strongest first' },
  { value: 'tier|asc', label: 'Tournament tier · weakest first' },
  { value: 'mode|asc', label: 'Map Slot / Ruleset · A to Z' },
  { value: 'mode|desc', label: 'Map Slot / Ruleset · Z to A' },
  { value: 'winner|asc', label: 'Winner · A to Z' },
  { value: 'winner|desc', label: 'Winner · Z to A' },
  { value: 'score|desc', label: 'Match score · A higher' },
  { value: 'score|asc', label: 'Match score · A lower' },
];
const EMPTY_TEAM_IDS = [];

function getTierOrderLabel(score, tierId) {
  const tier = resolveTournamentTier(score, tierId);
  return tier && typeof tier.id === 'string'
    ? tier.id
    : 'unscored';
}

function tierSortPriority(tierId) {
  if (tierId === 'x') return 0;
  if (tierId === 's_plus') return 1;
  if (tierId === 's') return 2;
  if (tierId === 'a_plus') return 3;
  if (tierId === 'a') return 4;
  if (tierId === 'a_minus') return 5;
  return 6;
}

function formatRoundMapMode(mapName, mapMode, fallbackMode, fallbackStyle, tournamentTags, mapIndex, mapCount) {
  const normalizedMap = normalizeText(mapName);
  const filteredMapName = isStageLikeLabel(normalizedMap) ? null : normalizedMap;
  const setFormat = normalizeMapMode(mapMode, mapCount)
    || normalizeMapMode(fallbackMode, mapCount)
    || normalizeMapMode(fallbackStyle, mapCount);
  const gameMode = resolveGameModeLabel(
    mapMode,
    fallbackMode,
    fallbackStyle,
    tournamentTags,
  );

  const details = [gameMode, setFormat].filter((value) => Boolean(value)).join(' / ');

  if (filteredMapName && details) {
    return mapIndex
      ? `Map ${mapIndex}: ${filteredMapName} / ${details}`
      : `${filteredMapName} / ${details}`;
  }
  if (details && mapIndex) {
    return `Map ${mapIndex} / ${details}`;
  }
  if (mapIndex) {
    return filteredMapName
      ? `${filteredMapName} (${mapIndex})`
      : `Map ${mapIndex}`;
  }
  return filteredMapName || details || 'Unknown';
}

function buildRosterList(raw) {
  if (!Array.isArray(raw) || !raw.length) return [];
  const rows = [];
  const seen = new Set();

  for (const row of raw) {
    if (!row) continue;
    const playerId = safeIntOrNull(row.player_id);
    const playerName = playerNameFn(row.player_name);
    const key = playerId === null ? `name:${playerName.toLowerCase()}` : `id:${playerId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({
      playerId,
      playerName,
      sendouUrl: playerId ? `https://sendou.ink/u/${playerId}` : null,
    });
  }

  return rows;
}

function playerNameFn(value) {
  const text = String(value || '').trim();
  if (!text) return 'Unknown Player';
  if (/^\d+$/.test(text)) return 'Unknown Player';
  return text;
}

function toLabelSummary(label) {
  if (!label) return 'Team';
  return String(label).split(' (')[0];
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function uniquePreserveOrder(values) {
  const out = [];
  const seen = new Set();
  for (const value of values) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) continue;
    const normalized = Math.trunc(numeric);
    if (!seen.has(normalized)) {
      seen.add(normalized);
      out.push(normalized);
    }
  }
  return out;
}

function parseTeamIdList(value) {
  if (Array.isArray(value)) {
    return uniquePreserveOrder(value);
  }
  if (value === null || value === undefined) return [];
  const tokens = String(value).match(/\d+/g) || [];
  return uniquePreserveOrder(tokens.map((token) => Number(token)));
}

function idsToText(ids) {
  return uniquePreserveOrder(ids).join(', ');
}

function summarizeTeamSet(ids, nameById, fallbackLabel) {
  const uniqueIds = uniquePreserveOrder(ids);
  if (!uniqueIds.length) {
    return {
      label: fallbackLabel,
      title: fallbackLabel,
      count: 0,
      groups: [],
    };
  }

  const nameToIds = new Map();
  for (const teamId of uniqueIds) {
    const name = nameById.get(teamId) || `Team ${teamId}`;
    if (!nameToIds.has(name)) {
      nameToIds.set(name, []);
    }
    nameToIds.get(name).push(teamId);
  }

  const groups = Array.from(nameToIds.entries()).map(([name, groupIds]) => ({
    name,
    count: groupIds.length,
    ids: groupIds,
  }));

  const label = groups.length === 1
    ? `${groups[0].name} (${groups[0].count} variants)`
    : groups.length === 2
      ? `${groups[0].name} · ${groups[1].name} (+1)`
      : `${groups[0].name} · ${groups[1].name} (+${groups.length - 2})`;

  const title = groups
    .map((group) => `${group.name} (${group.ids.join(', ')})`)
    .join(' · ');

  return {
    label,
    title,
    count: uniqueIds.length,
    groups,
  };
}

function clusterProfileCopy(mode) {
  if (mode === 'family') return 'Family';
  if (mode === 'strict') return 'Strict';
  if (mode === 'explore') return 'Explore';
  return mode;
}

function compareTeamRows(rows, sortBy, direction) {
  const sorted = [...rows];
  const sign = direction === 'desc' ? -1 : 1;
  sorted.sort((left, right) => {
    const leftMatchups = safeInt(left.match_count ?? left.lineup_count);
    const rightMatchups = safeInt(right.match_count ?? right.lineup_count);
    const leftRelevance = safeFloat(left.sim_to_query) ?? 0;
    const rightRelevance = safeFloat(right.sim_to_query) ?? 0;
    const leftName = String(left.team_name || `Team ${left.team_id || ''}`).toLowerCase();
    const rightName = String(right.team_name || `Team ${right.team_id || ''}`).toLowerCase();

    if (sortBy === 'matchups') {
      if (leftMatchups !== rightMatchups) return (leftMatchups - rightMatchups) * sign;
      return leftName.localeCompare(rightName);
    }

    if (sortBy === 'name') {
      return leftName.localeCompare(rightName) * sign;
    }

    return (leftRelevance - rightRelevance) * sign;
  });
  return sorted;
}

function compareMatchRows(rows, sortBy, direction) {
  const sorted = [...rows];
  const sign = direction === 'desc' ? -1 : 1;

  sorted.sort((left, right) => {
    if (sortBy === 'time') {
      const leftTime = toEpochMs(left.event_time_ms) || 0;
      const rightTime = toEpochMs(right.event_time_ms) || 0;
      if (leftTime !== rightTime) return (leftTime - rightTime) * sign;
      return compareMatchIds(left, right);
    }

    if (sortBy === 'match_id') {
      return compareMatchIds(left, right);
    }

    if (sortBy === 'round') {
      const leftRound = safeIntOrNull(left.round_no);
      const rightRound = safeIntOrNull(right.round_no);
      if (leftRound !== rightRound) {
        if (leftRound === null) return 1;
        if (rightRound === null) return -1;
        return (leftRound - rightRound) * sign;
      }
      const leftMapIndex = safeIntOrNull(left.map_index);
      const rightMapIndex = safeIntOrNull(right.map_index);
      if (leftMapIndex !== rightMapIndex) {
        if (leftMapIndex === null) return 1;
        if (rightMapIndex === null) return -1;
        return (leftMapIndex - rightMapIndex) * sign;
      }
      return compareMatchIds(left, right);
    }

    if (sortBy === 'tournament') {
      const leftTournament = normalizeText(
        left.tournament_name
          || (left.tournament_id !== null && left.tournament_id !== undefined
            ? `Tournament ${safeInt(left.tournament_id)}`
            : ''),
      );
      const rightTournament = normalizeText(
        right.tournament_name
          || (right.tournament_id !== null && right.tournament_id !== undefined
            ? `Tournament ${safeInt(right.tournament_id)}`
            : ''),
      );
      if (leftTournament !== rightTournament) {
        if (!leftTournament) return 1;
        if (!rightTournament) return -1;
        return leftTournament.localeCompare(rightTournament) * sign;
      }
      return compareMatchIds(left, right);
    }

    if (sortBy === 'tier') {
      const leftTier = getTierOrderLabel(safeFloat(left.tournament_score), left.tournament_score_tier_id);
      const rightTier = getTierOrderLabel(safeFloat(right.tournament_score), right.tournament_score_tier_id);
      if (leftTier !== rightTier) {
        return (tierSortPriority(leftTier) - tierSortPriority(rightTier)) * sign;
      }
      return compareMatchIds(left, right);
    }

        if (sortBy === 'mode') {
          const leftMode = normalizeText(
            formatRoundMapMode(
              left.round_map_name,
              left.round_map_mode,
              left.tournament_mode,
              left.map_picking_style,
              left.tournament_tags,
              safeIntOrNull(left.map_index),
              safeIntOrNull(left.round_maps_count),
            ),
          );
          const rightMode = normalizeText(
            formatRoundMapMode(
              right.round_map_name,
              right.round_map_mode,
              right.tournament_mode,
              right.map_picking_style,
              right.tournament_tags,
              safeIntOrNull(right.map_index),
              safeIntOrNull(right.round_maps_count),
            ),
          );
      if (leftMode !== rightMode) {
        if (!leftMode) return 1;
        if (!rightMode) return -1;
        return leftMode.localeCompare(rightMode) * sign;
      }
      return compareMatchIds(left, right);
    }

    if (sortBy === 'winner') {
      const leftWinner = normalizeText(
        left.winner_side === 'team_a'
          ? 'Team A'
          : left.winner_side === 'team_b'
            ? 'Team B'
            : 'Unresolved',
      );
      const rightWinner = normalizeText(
        right.winner_side === 'team_a'
          ? 'Team A'
          : right.winner_side === 'team_b'
            ? 'Team B'
            : 'Unresolved',
      );
      if (leftWinner !== rightWinner) {
        return leftWinner.localeCompare(rightWinner) * sign;
      }
      const leftTime = toEpochMs(left.event_time_ms) || 0;
      const rightTime = toEpochMs(right.event_time_ms) || 0;
      return (leftTime - rightTime) * sign;
    }

    if (sortBy === 'score') {
      const leftA = safeFloat(left.team_a_score);
      const leftB = safeFloat(left.team_b_score);
      const rightA = safeFloat(right.team_a_score);
      const rightB = safeFloat(right.team_b_score);

      const leftTotal = (leftA ?? 0) + (leftB ?? 0);
      const rightTotal = (rightA ?? 0) + (rightB ?? 0);
      if (leftTotal !== rightTotal) return (leftTotal - rightTotal) * sign;

      if (leftA !== rightA) {
        if (leftA === null) return 1 * sign;
        if (rightA === null) return -1 * sign;
        return (leftA - rightA) * sign;
      }
      return compareMatchIds(left, right);
    }

    return compareMatchIds(left, right);
  });

  function compareMatchIds(leftMatch, rightMatch) {
    const leftMatchId = safeInt(leftMatch.match_id);
    const rightMatchId = safeInt(rightMatch.match_id);
    if (leftMatchId !== rightMatchId) return (leftMatchId - rightMatchId) * sign;
    const leftTime = toEpochMs(leftMatch.event_time_ms) || 0;
    const rightTime = toEpochMs(rightMatch.event_time_ms) || 0;
    return (leftTime - rightTime) * sign;
  }

  return sorted;
}

export default function HeadToHead({
  selectedTeamAId = '',
  selectedTeamBId = '',
  selectedTeamAIds = EMPTY_TEAM_IDS,
  selectedTeamBIds = EMPTY_TEAM_IDS,
  selectedSnapshotId = '',
  onStateChange = () => {},
  onSelectTeamA = () => {},
  onSelectTeamB = () => {},
}) {
  const [query, setQuery] = useState('');
  const [clusterMode, setClusterMode] = useState('family');
  const [topN, setTopN] = useState(20);
  const [minRelevance, setMinRelevance] = useState(0.8);
  const [consolidate, setConsolidate] = useState(true);
  const [consolidateMinOverlap, setConsolidateMinOverlap] = useState(0.8);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [searchPayload, setSearchPayload] = useState(null);

  const [teamAInput, setTeamAInput] = useState('');
  const [teamBInput, setTeamBInput] = useState('');
  const [teamAAddInput, setTeamAAddInput] = useState('');
  const [teamBAddInput, setTeamBAddInput] = useState('');
  const [teamAAddError, setTeamAAddError] = useState('');
  const [teamBAddError, setTeamBAddError] = useState('');

  const [headToHeadLimit, setHeadToHeadLimit] = useState(200);
  const [headToHeadLoading, setHeadToHeadLoading] = useState(false);
  const [headToHeadError, setHeadToHeadError] = useState('');
  const [headToHeadPayload, setHeadToHeadPayload] = useState(null);
  const [headToHeadSnapshotFilter, setHeadToHeadSnapshotFilter] = useState('');
  const [resultsSortBy, setResultsSortBy] = useState('relevance');
  const [resultsSortDir, setResultsSortDir] = useState('desc');
  const [matchSortBy, setMatchSortBy] = useState('time');
  const [matchSortDir, setMatchSortDir] = useState('desc');
  const [lastHeadToHeadSignature, setLastHeadToHeadSignature] = useState('');
  const lastAutoLoadedRouteKeyRef = useRef('');
  const visibleMatchSortFields = useMemo(
    () => (
      SHOW_MAP_RULESET_COLUMN
        ? MATCH_SORT_FIELDS
        : MATCH_SORT_FIELDS.filter((option) => !option.value.startsWith('mode|'))
    ),
    [],
  );

  useEffect(() => {
    const next = uniquePreserveOrder([
      ...parseTeamIdList(selectedTeamAIds),
      ...parseTeamIdList(selectedTeamAId),
    ]);
    setTeamAInput(idsToText(next));
  }, [selectedTeamAId, selectedTeamAIds]);

  useEffect(() => {
    const next = uniquePreserveOrder([
      ...parseTeamIdList(selectedTeamBIds),
      ...parseTeamIdList(selectedTeamBId),
    ]);
    setTeamBInput(idsToText(next));
  }, [selectedTeamBId, selectedTeamBIds]);

  useEffect(() => {
    const snapshot = safeIntOrNull(selectedSnapshotId);
    setHeadToHeadSnapshotFilter(snapshot ? String(snapshot) : '');
  }, [selectedSnapshotId]);

  useEffect(() => {
    if (!SHOW_MAP_RULESET_COLUMN && matchSortBy === 'mode') {
      setMatchSortBy('time');
      setMatchSortDir('desc');
    }
  }, [matchSortBy]);

  const teamAIds = useMemo(() => uniquePreserveOrder(parseTeamIdList(teamAInput)), [teamAInput]);
  const teamBIds = useMemo(() => uniquePreserveOrder(parseTeamIdList(teamBInput)), [teamBInput]);
  const teamASet = useMemo(() => new Set(teamAIds), [teamAIds]);
  const teamBSet = useMemo(() => new Set(teamBIds), [teamBIds]);
  const routeTeamAIds = useMemo(
    () => uniquePreserveOrder([
      ...parseTeamIdList(selectedTeamAIds),
      ...parseTeamIdList(selectedTeamAId),
    ]),
    [selectedTeamAId, selectedTeamAIds],
  );
  const routeTeamBIds = useMemo(
    () => uniquePreserveOrder([
      ...parseTeamIdList(selectedTeamBIds),
      ...parseTeamIdList(selectedTeamBId),
    ]),
    [selectedTeamBId, selectedTeamBIds],
  );
  const routeSnapshotId = useMemo(
    () => safeIntOrNull(selectedSnapshotId),
    [selectedSnapshotId],
  );
  const isSelectionOverlap = useMemo(
    () => teamAIds.some((teamId) => teamBSet.has(teamId)),
    [teamAIds, teamBSet],
  );
  const canRunHeadToHead = teamAIds.length > 0 && teamBIds.length > 0 && !isSelectionOverlap;

  const teamNameById = useMemo(() => {
    const out = new Map();
    for (const row of searchPayload?.results || []) {
      const teamId = safeInt(row.team_id);
      if (teamId > 0) {
        out.set(teamId, row.team_name || `Team ${teamId}`);
      }
    }

    if (headToHeadPayload?.summary) {
      const summary = headToHeadPayload.summary;
      (summary.team_a_ids || []).forEach((teamId, index) => {
        if (!Number.isFinite(teamId)) return;
        out.set(teamId, summary.team_a_names?.[index] || `Team ${teamId}`);
      });
      (summary.team_b_ids || []).forEach((teamId, index) => {
        if (!Number.isFinite(teamId)) return;
        out.set(teamId, summary.team_b_names?.[index] || `Team ${teamId}`);
      });
    }
    return out;
  }, [searchPayload?.results, headToHeadPayload?.summary]);

  const teamASummary = useMemo(
    () => summarizeTeamSet(teamAIds, teamNameById, 'Select Team A'),
    [teamAIds, teamNameById],
  );
  const teamBSummary = useMemo(
    () => summarizeTeamSet(teamBIds, teamNameById, 'Select Team B'),
    [teamBIds, teamNameById],
  );

  const searchResults = useMemo(() => {
    const rows = searchPayload?.results || [];
    if (!rows.length) return [];
    return compareTeamRows(rows, resultsSortBy, resultsSortDir);
  }, [searchPayload?.results, resultsSortBy, resultsSortDir]);

  const comparisonSignature = useMemo(() => {
    const a = idsToText(teamAIds);
    const b = idsToText(teamBIds);
    const snapshot = safeIntOrNull(headToHeadSnapshotFilter) || 'latest';
    return `A:${a}|B:${b}|S:${snapshot}|N:${headToHeadLimit}`;
  }, [teamAIds, teamBIds, headToHeadSnapshotFilter, headToHeadLimit]);
  const isHeadToHeadStale = !!headToHeadPayload && lastHeadToHeadSignature !== comparisonSignature;

  const stepState = {
    searchDone: !!searchPayload,
    selectedDone: canRunHeadToHead || (teamAIds.length > 0 || teamBIds.length > 0),
    compareDone: !!headToHeadPayload && !isHeadToHeadStale,
  };

  function applyTeamSelection(role, rawTeamIds, snapshotId = null) {
    const nextIds = uniquePreserveOrder(parseTeamIdList(rawTeamIds));
    const nextSnapshot = safeIntOrNull(snapshotId);
    const nextSnapshotText = nextSnapshot ? String(nextSnapshot) : '';
    const nextText = idsToText(nextIds);

    if (role === 'A') {
      setTeamAInput(nextText);
      setTeamAAddInput('');
      setTeamAAddError('');
      if (nextSnapshotText) {
        setHeadToHeadSnapshotFilter(nextSnapshotText);
      }
      onSelectTeamA(nextIds, nextSnapshot);
      if (nextIds.some((teamId) => teamBSet.has(teamId))) {
        setTeamBInput('');
        onSelectTeamB([], nextSnapshot);
      }
      return;
    }

    setTeamBInput(nextText);
    setTeamBAddInput('');
    setTeamBAddError('');
    if (nextSnapshotText) {
      setHeadToHeadSnapshotFilter(nextSnapshotText);
    }
    onSelectTeamB(nextIds, nextSnapshot);
    if (nextIds.some((teamId) => teamASet.has(teamId))) {
      setTeamAInput('');
      onSelectTeamA([], nextSnapshot);
    }
  }

  function clearTeam(role) {
    if (role === 'A') {
      setTeamAInput('');
      setTeamAAddInput('');
      setTeamAAddError('');
      onSelectTeamA([]);
      return;
    }

    setTeamBInput('');
    setTeamBAddInput('');
    setTeamBAddError('');
    onSelectTeamB([]);
  }

  async function copyTeamIds(role) {
    const teamIds = role === 'A' ? teamAIds : teamBIds;
    if (!teamIds.length || !navigator.clipboard?.writeText) {
      return;
    }

    try {
      await navigator.clipboard.writeText(idsToText(teamIds));
    } catch (error) {
      // Best effort only; keep flow non-blocking if copy permissions are unavailable.
      console.error('Failed to copy team IDs', error);
    }
  }

  function swapTeams() {
    const nextA = [...teamAIds];
    const nextB = [...teamBIds];
    setTeamAInput(idsToText(nextB));
    setTeamBInput(idsToText(nextA));
    onSelectTeamA(nextB);
    onSelectTeamB(nextA);
    setHeadToHeadSnapshotFilter('');
  }

  function addTeamIds(role, rawValue) {
    const values = parseTeamIdList(rawValue);
    if (!values.length) {
      if (role === 'A') setTeamAAddError('Paste one or more numeric IDs.');
      else setTeamBAddError('Paste one or more numeric IDs.');
      return;
    }
    if (role === 'A') {
      const next = uniquePreserveOrder([...teamAIds, ...values]);
      setTeamAInput(idsToText(next));
      setTeamAAddInput('');
      setTeamAAddError('');
      onSelectTeamA(next);
      if (values.some((teamId) => teamBSet.has(teamId))) {
        setTeamBInput(idsToText(teamBIds.filter((teamId) => !values.includes(teamId))));
        onSelectTeamB(teamBIds.filter((teamId) => !values.includes(teamId)));
      }
      return;
    }

    const next = uniquePreserveOrder([...teamBIds, ...values]);
    setTeamBInput(idsToText(next));
    setTeamBAddInput('');
    setTeamBAddError('');
    onSelectTeamB(next);
    if (values.some((teamId) => teamASet.has(teamId))) {
      setTeamAInput(idsToText(teamAIds.filter((teamId) => !values.includes(teamId))));
      onSelectTeamA(teamAIds.filter((teamId) => !values.includes(teamId)));
    }
  }

  function removeTeamId(role, teamId) {
    const normalizedId = Math.trunc(Number(teamId));
    if (!Number.isFinite(normalizedId) || normalizedId <= 0) return;

    if (role === 'A') {
      const next = teamAIds.filter((id) => id !== normalizedId);
      setTeamAInput(idsToText(next));
      onSelectTeamA(next);
      return;
    }

    const next = teamBIds.filter((id) => id !== normalizedId);
    setTeamBInput(idsToText(next));
    onSelectTeamB(next);
  }

  function handleTeamAddKeyDown(role, event) {
    if (event.key !== 'Enter') return;
    event.preventDefault();

    if (role === 'A') {
      addTeamIds('A', teamAAddInput);
      return;
    }
    addTeamIds('B', teamBAddInput);
  }

  async function onSearchSubmit(event) {
    event.preventDefault();
    if (!query.trim()) return;

    setSearchLoading(true);
    setSearchError('');
    setHeadToHeadPayload(null);
    setLastHeadToHeadSignature('');
    try {
      const payload = await fetchTeamSearch({
        q: query.trim(),
        topN,
        clusterMode,
        minRelevance,
        consolidate,
        consolidateMinOverlap,
      });
      setSearchPayload(payload);
    } catch (error) {
      setSearchError(error.message || 'Failed to load teams');
    } finally {
      setSearchLoading(false);
    }
  }

  async function onRunHeadToHead(event) {
    event.preventDefault();
    await runHeadToHead({ syncRoute: true });
  }

  const runHeadToHead = useCallback(async ({
    teamAIdsOverride = teamAIds,
    teamBIdsOverride = teamBIds,
    snapshotIdOverride = headToHeadSnapshotFilter,
    syncRoute = false,
  } = {}) => {
    const nextTeamAIds = uniquePreserveOrder(parseTeamIdList(teamAIdsOverride));
    const nextTeamBIds = uniquePreserveOrder(parseTeamIdList(teamBIdsOverride));
    const overlap = nextTeamAIds.some((teamId) => nextTeamBIds.includes(teamId));
    if (!nextTeamAIds.length || !nextTeamBIds.length || overlap) {
      return false;
    }

    const snapshotId = safeIntOrNull(snapshotIdOverride);
    const routeKey = `A:${idsToText(nextTeamAIds)}|B:${idsToText(nextTeamBIds)}|S:${snapshotId || 'latest'}`;
    const nextSignature = `A:${idsToText(nextTeamAIds)}|B:${idsToText(nextTeamBIds)}|S:${snapshotId || 'latest'}|N:${headToHeadLimit}`;

    if (syncRoute) {
      lastAutoLoadedRouteKeyRef.current = routeKey;
      onStateChange({
        teamAIds: nextTeamAIds,
        teamBIds: nextTeamBIds,
        snapshotId,
      });
    }

    setHeadToHeadLoading(true);
    setHeadToHeadError('');
    try {
      const payload = await fetchAnalyticsHeadToHead({
        teamAIds: nextTeamAIds,
        teamBIds: nextTeamBIds,
        snapshotId,
        limit: headToHeadLimit,
      });
      setHeadToHeadPayload(payload);
      setLastHeadToHeadSignature(nextSignature);
      return true;
    } catch (error) {
      setHeadToHeadError(error.message || 'Failed to load head-to-head');
      setHeadToHeadPayload(null);
      setLastHeadToHeadSignature('');
      return false;
    } finally {
      setHeadToHeadLoading(false);
    }
  }, [
    headToHeadLimit,
    headToHeadSnapshotFilter,
    onStateChange,
    teamAIds,
    teamBIds,
  ]);

  useEffect(() => {
    const routeKey = `A:${idsToText(routeTeamAIds)}|B:${idsToText(routeTeamBIds)}|S:${routeSnapshotId || 'latest'}`;
    const overlap = routeTeamAIds.some((teamId) => routeTeamBIds.includes(teamId));

    if (!routeTeamAIds.length || !routeTeamBIds.length || overlap) {
      return;
    }

    if (lastAutoLoadedRouteKeyRef.current === routeKey) {
      return;
    }

    lastAutoLoadedRouteKeyRef.current = routeKey;
    runHeadToHead({
      teamAIdsOverride: routeTeamAIds,
      teamBIdsOverride: routeTeamBIds,
      snapshotIdOverride: routeSnapshotId,
      syncRoute: false,
    });
  }, [routeSnapshotId, routeTeamAIds, routeTeamBIds, runHeadToHead]);

  const summary = headToHeadPayload?.summary || {};
  const matches = headToHeadPayload?.matches || [];
  const expandedMatchRows = useMemo(() => {
    const out = [];
    for (const match of matches) {
      const rounds = Array.isArray(match.match_rounds) ? match.match_rounds : [];
      const rowRounds = rounds.length
        ? rounds
        : [
          {
            round_no: null,
            map_name: null,
            map_mode: null,
            team_a_score: match.team_a_score,
            team_b_score: match.team_b_score,
            winner_team_id: match.winner_team_id,
            winner_side: match.winner_side,
            is_synthetic_round: true,
          },
        ];

      for (const [index, round] of rowRounds.entries()) {
        out.push({
          ...match,
          round_no: round.round_no ?? null,
          round_maps_count: round.maps_count ?? null,
          round_map_name: round.map_name || null,
          round_map_mode: round.map_mode || null,
          team_a_score: safeFloat(round.team_a_score),
          team_b_score: safeFloat(round.team_b_score),
          winner_team_id: round.winner_team_id ?? match.winner_team_id ?? null,
          winner_side: round.winner_side ?? match.winner_side ?? null,
          row_id: `${match.match_id}-${round.round_id || round.round_no || `syn-${index}`}`,
          is_synthetic_round: Boolean(round.is_synthetic_round),
        });
      }
    }
    return out;
  }, [matches]);
  const sortedMatches = useMemo(() => {
    if (!expandedMatchRows.length) return [];
    return compareMatchRows(expandedMatchRows, matchSortBy, matchSortDir);
  }, [expandedMatchRows, matchSortBy, matchSortDir]);
  const snapshotIdLabel = safeIntOrNull(headToHeadPayload?.snapshot_id) || safeIntOrNull(headToHeadSnapshotFilter);
  const snapshotLabel = snapshotIdLabel !== null ? snapshotIdLabel : 'n/a';
  const teamAWins = safeInt(summary.team_a_wins);
  const teamBWins = safeInt(summary.team_b_wins);
  const unresolved = safeInt(summary.unresolved_matches);
  const totalMatches = safeInt(summary.total_matches);
  const decidedMatches = safeInt(summary.decided_matches);
  const tournamentTierDistribution = useMemo(() => {
    if (summary.tournament_tier_match_distribution) {
      return summary.tournament_tier_match_distribution;
    }
    if (summary.tournament_tier_distribution) {
      return summary.tournament_tier_distribution;
    }

    const derived = {};
    for (const match of matches) {
      const tournamentScore = safeFloat(match.tournament_score);
      const tier = resolveTournamentTier(
        tournamentScore,
        match.tournament_score_tier_id,
      );
      if (!tier) continue;
      derived[tier.label] = (derived[tier.label] || 0) + 1;
    }

    return derived;
  }, [summary, matches]);
  const tournamentTierUnit = summary.tournament_tier_match_distribution ? 'matches' : 'tournaments';
  const aWinRate = summary.team_a_win_rate;
  const bWinRate = summary.team_b_win_rate;
  const teamADisplayName = toLabelSummary(summary.team_a_name) || teamASummary.label;
  const teamBDisplayName = toLabelSummary(summary.team_b_name) || teamBSummary.label;
  const teamARepresentativeGroup = teamASummary.groups[0]?.name || teamADisplayName || 'Team A';
  const teamBRepresentativeGroup = teamBSummary.groups[0]?.name || teamBDisplayName || 'Team B';
  const teamAWinRateText = percent(aWinRate);
  const teamBWinRateText = percent(bWinRate);
  const matchTableColSpan = SHOW_MAP_RULESET_COLUMN ? 9 : 8;

  const renderMatchRoster = (players, heading) => {
    const roster = buildRosterList(players);
    const displayRoster = roster.slice(0, 6);
    return (
      <div className="h2h-roster-group">
        <span className="h2h-roster-label">{heading}</span>
        <span className="h2h-roster-items">
          {displayRoster.length ? (
            displayRoster.map((player) => (
              player.sendouUrl ? (
                <a
                  key={`${heading}-${player.playerId}-${player.playerName}`}
                  className="h2h-roster-item"
                  href={player.sendouUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={player.playerName}
                >
                  {player.playerName}
                </a>
              ) : (
                <span
                  key={`${heading}-${player.playerName}`}
                  className="h2h-roster-item"
                  title={player.playerName}
                >
                  {player.playerName}
                </span>
              )
            ))
          ) : (
            <span className="h2h-roster-empty">no roster</span>
          )}
          {roster.length > 6 ? (
            <span className="h2h-roster-overflow" title={`${roster.length - 6} more`}>
              +{roster.length - 6} more
            </span>
          ) : null}
        </span>
      </div>
    );
  };

  const tournamentTierItems = TOURNAMENT_TIER_META.map((tier) => ({
    ...tier,
    matchCount: safeInt(tournamentTierDistribution[tier.label]),
  })).filter((tier) => tier.matchCount > 0);

  return (
    <section className="panel team-search-panel" aria-labelledby="head-to-head-title">
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Matchup review</p>
          <h2 id="head-to-head-title" className="panel-title">Head-to-Head</h2>
          <p className="panel-summary">
            Search teams, pick Team A and Team B, then load matchup history.
          </p>
        </div>
      </div>

      <ol className="h2h-stepper" aria-label="Head-to-head flow">
        <li className={stepState.searchDone ? 'is-complete' : 'is-current'}>1 Search</li>
        <li className={stepState.selectedDone ? 'is-complete' : 'is-current'}>2 Select Teams</li>
        <li className={stepState.compareDone ? 'is-complete' : 'is-current'}>3 Compare</li>
      </ol>

      <form className="form-grid search-form" onSubmit={onSearchSubmit}>
        <div className="h2h-search-row">
          <div className="h2h-search-input-wrap">
            <label htmlFor="head-to-head-query" className="field-label">Team lookup</label>
            <div className="h2h-search-with-action">
              <input
                id="head-to-head-query"
                className="input"
                type="search"
                placeholder="e.g. FTW, Moonlight, 57202"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                required
              />
              <button className="button btn-pill btn-fuchsia" type="submit" disabled={searchLoading}>
                {searchLoading ? 'Searching…' : 'Search teams'}
              </button>
            </div>
            <p className="field-hint">Search by team name or numeric IDs (partial name supported).</p>
          </div>
          <button
            type="button"
            className="result-select-btn btn-pill btn-fuchsia-outline"
            onClick={() => setShowAdvancedFilters((value) => !value)}
            aria-expanded={showAdvancedFilters}
            aria-controls="head-to-head-advanced-controls"
          >
            {showAdvancedFilters ? 'Hide advanced' : 'Advanced filters'}
          </button>
        </div>

        {showAdvancedFilters ? (
          <div className="form-row row fields-row" id="head-to-head-advanced-controls">
            <div className="field">
              <label htmlFor="head-to-head-cluster-mode" className="field-label">
                Cluster profile
                <span className="field-label-subtitle">Controls how near-duplicate names are merged.</span>
              </label>
              <select
                id="head-to-head-cluster-mode"
                className="input"
                value={clusterMode}
                onChange={(event) => setClusterMode(event.target.value)}
              >
                <option value="family">Family</option>
                <option value="strict">Strict</option>
                <option value="explore">Explore</option>
              </select>
              <p className="field-hint">
                Profile mode: {clusterProfileCopy(clusterMode)}
              </p>
            </div>

            <div className="field">
              <label className="checkbox-field">
                <input
                  id="head-to-head-consolidate"
                  type="checkbox"
                  checked={consolidate}
                  onChange={(event) => setConsolidate(event.target.checked)}
                />
                Consolidate near-duplicate teams
              </label>
              <p className="field-hint">
                Example: “Moonlight” and “Moonlight.” merged when confidence is high.
              </p>
            </div>

            <div className="field">
              <label htmlFor="head-to-head-top-n" className="field-label">Max results</label>
              <input
                id="head-to-head-top-n"
                className="input"
                type="number"
                min={1}
                max={200}
                value={topN}
                onChange={(event) => setTopN(clamp(Number(event.target.value), 1, 200))}
              />
            </div>

            <div className="field">
              <label htmlFor="head-to-head-min-relevance" className="field-label slider-label">
                Minimum relevance
                <span className="field-label-subtitle">
                  Threshold for including search results.
                </span>
              </label>
              <div className="slider-wrap">
                <output
                  className="slider-value"
                  style={{ left: `calc(${Math.round(minRelevance * 100)}% - 1.4rem)` }}
                  aria-hidden="true"
                >
                  {Math.round(minRelevance * 100)}%
                </output>
                <input
                  id="head-to-head-min-relevance"
                  className="input slider-input"
                  type="range"
                  min={0}
                  max={100}
                  step={10}
                  value={Math.round(minRelevance * 100)}
                  onChange={(event) => setMinRelevance(Number(event.target.value) / 100)}
                  list="head-to-head-relevance-marks"
                  aria-describedby="min-relevance-value"
                />
                <span className="slider-number-wrap">
                  <input
                    type="number"
                    inputMode="decimal"
                    className="input slider-number"
                    min={0}
                    max={100}
                    step={10}
                    value={Math.round(minRelevance * 100)}
                    onChange={(event) => {
                      const value = clamp(Number(event.target.value), 0, 100);
                      setMinRelevance(value / 100);
                    }}
                    aria-label="Minimum relevance percent"
                  />
                </span>
              </div>
              <datalist id="head-to-head-relevance-marks">
                <option value="0" />
                <option value="10" />
                <option value="20" />
                <option value="30" />
                <option value="40" />
                <option value="50" />
                <option value="60" />
                <option value="70" />
                <option value="80" />
                <option value="90" />
                <option value="100" />
              </datalist>
              <p className="field-hint">Value increments in 10% steps.</p>
            </div>
            <div className="field">
              <label htmlFor="head-to-head-consolidate-overlap" className="field-label slider-label">
                Consolidation overlap
              </label>
              <div className="slider-wrap">
                <output
                  className="slider-value"
                  style={{ left: `calc(${Math.round(consolidateMinOverlap * 100)}% - 1.4rem)` }}
                  aria-hidden="true"
                >
                  {Math.round(consolidateMinOverlap * 100)}%
                </output>
                <input
                  id="head-to-head-consolidate-overlap"
                  className="input slider-input"
                  type="range"
                  min={0}
                  max={100}
                  step={10}
                  value={Math.round(consolidateMinOverlap * 100)}
                  onChange={(event) => setConsolidateMinOverlap(Number(event.target.value) / 100)}
                  aria-describedby="consolidate-overlap-value"
                  disabled={!consolidate}
                />
              </div>
            </div>
          </div>
        ) : null}
      </form>

      {searchError ? <p className="error" role="status">{searchError}</p> : null}

      {searchPayload ? (
        <section className="head-to-head-panel" aria-label="Compact team search results">
          <div className="results-head">
            <h3 className="results-title">Search results</h3>
            <span className="results-count">
              {searchPayload.result_count ?? 0} groups · snapshot {searchPayload.snapshot_id ?? 'n/a'} · profile {clusterProfileCopy(clusterMode)}
            </span>
          </div>
          <div className="head-to-head-result-meta">
            <div className="result-meta-chips">
              <span className="result-meta-chip chip">
                {searchPayload.result_count ?? 0} result group{(searchPayload.result_count ?? 0) === 1 ? '' : 's'}
              </span>
              <span className="result-meta-chip chip">
                Snapshot {searchPayload.snapshot_id ?? 'n/a'}
              </span>
              <span className="result-meta-chip chip">
                Min relevance {(Math.round(minRelevance * 100))}%
              </span>
              <span className="result-meta-chip chip">
                Profile {clusterProfileCopy(clusterMode)}
              </span>
              <span className="result-meta-chip chip">
                Consolidation {consolidate ? 'on' : 'off'}
              </span>
              <span className="result-meta-chip chip chip-accent">
                Top results {topN}
              </span>
            </div>
            <div className="head-to-head-sort-controls">
              <label htmlFor="head-to-head-sort-by" className="field-label">Sort</label>
              <select
                id="head-to-head-sort-by"
                className="input"
                value={`${resultsSortBy}|${resultsSortDir}`}
                onChange={(event) => {
                  const [field, direction] = event.target.value.split('|');
                  setResultsSortBy(field);
                  setResultsSortDir(direction);
                }}
              >
                <option value="relevance|desc">Relevance · high to low</option>
                <option value="relevance|asc">Relevance · low to high</option>
                <option value="matchups|desc">Matchups · high to low</option>
                <option value="matchups|asc">Matchups · low to high</option>
                <option value="name|asc">Team name · A to Z</option>
                <option value="name|desc">Team name · Z to A</option>
              </select>
            </div>
          </div>

          <div className="table-wrap">
              <table className="head-to-head-table data-table">
                <thead>
                  <tr>
                    <th>Team</th>
                    <th title="Potential matchup count for this team profile.">Matchups</th>
                    <th title="Similarity score against your query.">Search Score</th>
                    <th title="How many raw team identities are represented in this result row.">Variants</th>
                    <th>Actions</th>
                  </tr>
                </thead>
              <tbody>
                {searchResults.length ? (
                  searchResults.map((row) => {
                    const rowTeamId = safeInt(row.team_id);
                    const isTeamA = rowTeamId > 0 && teamASet.has(rowTeamId);
                    const isTeamB = rowTeamId > 0 && teamBSet.has(rowTeamId);
                    const teamName = row.team_name || `Team ${rowTeamId}`;
                    const relevance = percent(safeFloat(row.sim_to_query));
                    const matchups = safeInt(row.match_count ?? row.lineup_count);
                    const consolidatedCount = Number(row.consolidated_team_count) || 1;
                    return (
                      <tr key={rowTeamId}>
                        <td>
                          <strong>{teamName}</strong>
                          <div className="meta-subtext">
                            ID {rowTeamId}
                            {consolidatedCount > 1 ? ` · ${consolidatedCount} variants` : ''}
                          </div>
                        </td>
                        <td>{matchups.toLocaleString()}</td>
                        <td>{relevance}</td>
                        <td>{consolidatedCount} team{consolidatedCount === 1 ? '' : 's'}</td>
                        <td>
                          <div className="result-quick-actions">
                            <button
                              type="button"
                              className={`result-select-btn ${isTeamA ? 'is-selected' : ''}`}
                              onClick={() => applyTeamSelection('A', row.consolidated_team_ids || [rowTeamId], searchPayload.snapshot_id)}
                              aria-pressed={isTeamA}
                              title={isTeamA ? 'Selected as Team A' : 'Set as Team A'}
                            >
                              {isTeamA ? 'Selected as A' : 'Set as A'}
                            </button>
                            <button
                              type="button"
                              className={`result-select-btn ${isTeamB ? 'is-selected' : ''}`}
                              onClick={() => applyTeamSelection('B', row.consolidated_team_ids || [rowTeamId], searchPayload.snapshot_id)}
                              aria-pressed={isTeamB}
                              title={isTeamB ? 'Selected as Team B' : 'Set as Team B'}
                            >
                              {isTeamB ? 'Selected as B' : 'Set as B'}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={5}>No teams found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="head-to-head-panel" aria-labelledby="head-to-head-run-title">
        <h3 id="head-to-head-run-title" className="meta-label">
          Selected teams & matchup
        </h3>

        <div className="h2h-team-pair">
          <section className="h2h-team-card">
            <header>
              <h4>Team A</h4>
              <span className="team-header-actions">
                <button
                  type="button"
                  className="result-select-btn"
                  onClick={() => copyTeamIds('A')}
                  disabled={!teamAIds.length}
                >
                  Copy IDs
                </button>
                <button
                  type="button"
                  className="result-select-btn"
                  onClick={() => clearTeam('A')}
                  disabled={!teamAIds.length}
                >
                  Clear
                </button>
              </span>
            </header>
            <p className="h2h-team-summary" title={teamASummary.title}>
              {teamASummary.label}
            </p>
            <div className="h2h-id-chips">
              {teamASummary.groups.length
                ? teamASummary.groups.map((group) => (
                  <span key={`a-${group.name}`} className="id-chip">
                    {group.name} ({group.count})
                  </span>
                ))
                : null}
            </div>
            <div className="h2h-id-list">
              {teamAIds.map((teamId) => (
                <span key={`a-id-${teamId}`} className="id-chip id-chip--compact">
                  <span>#{teamId}</span>
                  <button type="button" onClick={() => removeTeamId('A', teamId)} aria-label={`Remove ${teamId}`}>×</button>
                </span>
              ))}
            </div>
            <div className="h2h-add-row">
              <label htmlFor="team-a-add" className="sr-only">Add Team A IDs</label>
              <input
                id="team-a-add"
                className="input"
                type="text"
                value={teamAAddInput}
                onChange={(event) => {
                  setTeamAAddInput(event.target.value);
                  if (teamAAddError) setTeamAAddError('');
                }}
                onKeyDown={(event) => handleTeamAddKeyDown('A', event)}
                placeholder="Add ID(s), Enter"
              />
              <button
                type="button"
                className="result-select-btn"
                onClick={() => addTeamIds('A', teamAAddInput)}
              >
                Add
              </button>
            </div>
            {teamAAddError ? <p className="error">{teamAAddError}</p> : null}
          </section>

          <div className="h2h-team-actions">
            <button type="button" className="result-select-btn" onClick={swapTeams}>
              Swap A/B
            </button>
          </div>

          <section className="h2h-team-card">
            <header>
              <h4>Team B</h4>
              <span className="team-header-actions">
                <button
                  type="button"
                  className="result-select-btn"
                  onClick={() => copyTeamIds('B')}
                  disabled={!teamBIds.length}
                >
                  Copy IDs
                </button>
                <button
                  type="button"
                  className="result-select-btn"
                  onClick={() => clearTeam('B')}
                  disabled={!teamBIds.length}
                >
                  Clear
                </button>
              </span>
            </header>
            <p className="h2h-team-summary" title={teamBSummary.title}>
              {teamBSummary.label}
            </p>
            <div className="h2h-id-chips">
              {teamBSummary.groups.length
                ? teamBSummary.groups.map((group) => (
                  <span key={`b-${group.name}`} className="id-chip">
                    {group.name} ({group.count})
                  </span>
                ))
                : null}
            </div>
            <div className="h2h-id-list">
              {teamBIds.map((teamId) => (
                <span key={`b-id-${teamId}`} className="id-chip id-chip--compact">
                  <span>#{teamId}</span>
                  <button type="button" onClick={() => removeTeamId('B', teamId)} aria-label={`Remove ${teamId}`}>×</button>
                </span>
              ))}
            </div>
            <div className="h2h-add-row">
              <label htmlFor="team-b-add" className="sr-only">Add Team B IDs</label>
              <input
                id="team-b-add"
                className="input"
                type="text"
                value={teamBAddInput}
                onChange={(event) => {
                  setTeamBAddInput(event.target.value);
                  if (teamBAddError) setTeamBAddError('');
                }}
                onKeyDown={(event) => handleTeamAddKeyDown('B', event)}
                placeholder="Add ID(s), Enter"
              />
              <button
                type="button"
                className="result-select-btn"
                onClick={() => addTeamIds('B', teamBAddInput)}
              >
                Add
              </button>
            </div>
            {teamBAddError ? <p className="error">{teamBAddError}</p> : null}
          </section>
        </div>

        <form className="head-to-head-controls" onSubmit={onRunHeadToHead}>
          <div className="head-to-head-field">
            <label htmlFor="head-to-head-snapshot" className="field-label">Snapshot</label>
            <input
              id="head-to-head-snapshot"
              className="input"
              type="text"
              inputMode="numeric"
              value={headToHeadSnapshotFilter}
              onChange={(event) => setHeadToHeadSnapshotFilter(event.target.value)}
              placeholder="latest"
            />
          </div>
          <div className="head-to-head-field">
            <label htmlFor="head-to-head-limit" className="field-label">Max matches to load</label>
            <input
              id="head-to-head-limit"
              className="input"
              type="number"
              min={1}
              max={1000}
              value={headToHeadLimit}
              onChange={(event) => setHeadToHeadLimit(clamp(Number(event.target.value), 1, 1000))}
            />
          </div>
          <button
            className="button btn-pill btn-fuchsia"
            type="submit"
            disabled={!canRunHeadToHead || headToHeadLoading}
          >
            {headToHeadLoading ? 'Loading matchups…' : 'Load head-to-head'}
          </button>
          {teamAIds.length === 0 || teamBIds.length === 0 || isSelectionOverlap ? (
            <p className="error">
              {isSelectionOverlap
                ? 'Teams must be different.'
                : 'Select at least one ID for Team A and Team B.'}
            </p>
          ) : null}
          {isHeadToHeadStale ? (
            <p className="status">
              Selections changed since last run. Reload for updated results.
            </p>
          ) : null}
        </form>
      </section>

      {headToHeadError ? <p className="error" role="alert">{headToHeadError}</p> : null}

      {headToHeadPayload ? (
        <section className="head-to-head-panel head-to-head-results" aria-label="Head-to-head results">
          <div className="grid-cols-4 h2h-summary-stats">
            <div className="stat">
              <span className="stat-label">Matched matches</span>
              <span className="stat-value">{totalMatches.toLocaleString()}</span>
            </div>
            <div className="stat">
              <span className="stat-label">{teamARepresentativeGroup}</span>
              <span className="stat-value">{teamAWinRateText}</span>
            </div>
            <div className="stat">
              <span className="stat-label">{teamBRepresentativeGroup}</span>
              <span className="stat-value">{teamBWinRateText}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Unresolved</span>
              <span className="stat-value">{unresolved.toLocaleString()}</span>
            </div>
          </div>

          <div className="h2h-score-grid result-meta-grid result-meta-subtle">
            <div className="meta-item">
              <span>Snapshot</span>
              <strong>{snapshotLabel}</strong>
            </div>
            <div className="meta-item">
              <span>Matched matches</span>
              <strong>{totalMatches.toLocaleString()}</strong>
            </div>
            <div className="meta-item">
              <span>Tournaments</span>
              <strong>{safeInt(summary.tournaments).toLocaleString()}</strong>
            </div>
            <div className="meta-item">
              <span>Unresolved</span>
              <strong>{unresolved.toLocaleString()}</strong>
            </div>
          </div>

          {tournamentTierItems.length ? (
            <div className="h2h-score-grid result-meta-grid result-meta-subtle">
              <div className="meta-item meta-item-full">
                <span>{`Tournament tier mix (${tournamentTierUnit})`}</span>
                <div className="h2h-tier-breakdown">
                  {tournamentTierItems.map((tier) => (
                    <span
                      key={`tier-${tier.id}`}
                      className={`h2h-tier-pill h2h-tier-pill--${tier.id}`}
                      title={`${tier.label} tier matches`}
                    >
                      {tier.label}: {safeInt(tier.matchCount).toLocaleString()}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : null}

          <div className="h2h-score-grid result-meta-grid">
            <div className="meta-item meta-item-hero meta-item-primary">
              <span>Team A</span>
              <strong title={teamASummary.title}>{teamARepresentativeGroup}</strong>
              <span className="meta-subtext">
                {teamAWins} wins · {teamAWinRateText}
              </span>
            </div>
            <div className="meta-item meta-item-hero meta-item-primary">
              <span>Head-to-head</span>
              <strong>{teamAWins}–{teamBWins}</strong>
              <span className="meta-subtext">
                {decidedMatches} decided · {totalMatches - decidedMatches} unresolved · {teamAWins + teamBWins} total
              </span>
            </div>
            <div className="meta-item meta-item-hero meta-item-primary">
              <span>Team B</span>
              <strong title={teamBSummary.title}>{teamBRepresentativeGroup}</strong>
              <span className="meta-subtext">
                {teamBWins} wins · {teamBWinRateText}
              </span>
            </div>
          </div>

          <div className="table-wrap">
            <div className="h2h-match-sort">
              <label htmlFor="head-to-head-match-sort" className="field-label">Sort matches</label>
              <select
                id="head-to-head-match-sort"
                className="input"
                value={`${matchSortBy}|${matchSortDir}`}
                onChange={(event) => {
                  const [sortBy, direction] = event.target.value.split('|');
                  setMatchSortBy(sortBy);
                  setMatchSortDir(direction);
                }}
              >
                {visibleMatchSortFields.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <table className="head-to-head-table h2h-match-table data-table">
              <thead>
                      <tr>
                        <th>Match</th>
                        <th>Round</th>
                        <th>Tournament</th>
                        {SHOW_MAP_RULESET_COLUMN ? <th>Map Slot / Ruleset</th> : null}
                        <th>Tournament Tier</th>
                        <th>Score</th>
                        <th>Winner</th>
                        <th>Rosters</th>
                        <th>Time</th>
                    </tr>
              </thead>
              <tbody>
                {sortedMatches.length ? (
                    sortedMatches.map((match) => {
                      const winnerSide = match.winner_side;
                    const tournamentScore = safeFloat(match.tournament_score);
                    const tournamentTier = resolveTournamentTier(
                      tournamentScore,
                      match.tournament_score_tier_id,
                    );
                    const teamAScore = safeFloat(match.team_a_score);
                    const teamBScore = safeFloat(match.team_b_score);
                    const matchMode = SHOW_MAP_RULESET_COLUMN
                      ? formatRoundMapMode(
                        match.round_map_name,
                        match.round_map_mode,
                        match.tournament_mode,
                        match.map_picking_style,
                        match.tournament_tags,
                        safeIntOrNull(match.map_index),
                        safeIntOrNull(match.round_maps_count),
                      )
                      : null;
                    const roundNo = safeIntOrNull(match.round_no);
                    const mapIndex = safeIntOrNull(match.map_index);
                    const roundLabel = roundNo === null
                      ? (mapIndex === null ? '—' : `#${mapIndex}`)
                      : mapIndex === null
                        ? `#${roundNo}`
                        : `#${roundNo}.${mapIndex}`;
                    const winnerLabel = winnerSide === 'team_a'
                      ? 'Team A'
                      : winnerSide === 'team_b'
                        ? 'Team B'
                        : 'Unresolved';
                    const winnerClass = winnerSide === 'team_a'
                      ? 'head-to-head-winner team-a'
                      : winnerSide === 'team_b'
                        ? 'head-to-head-winner team-b'
                        : 'head-to-head-winner unresolved';
                    const winnerTeamLabel = winnerSide === 'team_a'
                      ? teamADisplayName
                      : winnerSide === 'team_b'
                        ? teamBDisplayName
                        : 'Unresolved';
                    const winnerMeta = winnerSide === 'team_a'
                      ? `${teamADisplayName} ({teamASummary.label})`
                      : winnerSide === 'team_b'
                        ? `${teamBDisplayName} (${teamBSummary.label})`
                        : 'Winner unavailable';
                    const rowKey = `${match.match_id}-${match.row_id || match.round_no || '0'}${mapIndex !== null ? `.${mapIndex}` : ''}-${roundNo === null ? 'r' : roundNo}`;
                    return (
                      <tr
                        key={rowKey}
                        className={match.is_synthetic_round ? 'h2h-match-row h2h-match-row--synthetic' : 'h2h-match-row'}
                      >
                        <td>{match.match_id !== null && match.match_id !== undefined ? String(match.match_id) : '—'}</td>
                        <td>{roundLabel}</td>
                        <td>
                          {match.tournament_name || (match.tournament_id !== null ? `Tournament ${safeInt(match.tournament_id)}` : '—')}
                        </td>
                        {SHOW_MAP_RULESET_COLUMN ? <td>{matchMode}</td> : null}
                        <td>
                          <span
                            className={`h2h-tier-pill h2h-tier-pill--${tournamentTier.id}`}
                          >
                            {tournamentTier.label}
                          </span>
                        </td>
                        <td>{formatScorePair(teamAScore, teamBScore)}</td>
                        <td>
                          <span className="head-to-head-winner-wrap">
                            <span className={winnerClass} title={winnerMeta}>
                              {winnerLabel}
                            </span>
                            <span className="head-to-head-winner-name" title={winnerMeta}>
                              {winnerSide === 'team_a' ? `${winnerTeamLabel} won` : winnerSide === 'team_b' ? `${winnerTeamLabel} won` : winnerTeamLabel}
                              {' '}
                              ({formatScorePair(teamAScore, teamBScore)})
                            </span>
                          </span>
                        </td>
                        <td>
                          <div className="h2h-roster-wrap">
                            {renderMatchRoster(match.team_a_roster, teamASummary.label)}
                            {renderMatchRoster(match.team_b_roster, teamBSummary.label)}
                          </div>
                        </td>
                        <td title={toAbsoluteDate(match.event_time_ms)}>{formatMatchTime(match.event_time_ms)}</td>
                      </tr>
                    );
                  })
                ) : (
                    <tr>
                    <td colSpan={matchTableColSpan}>No matchup rows found for current Team A and Team B.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </section>
  );
}
