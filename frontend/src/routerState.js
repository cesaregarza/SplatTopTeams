const CLUSTER_MODES = new Set(['family', 'strict', 'explore']);
const TEAM_SCOPES = new Set(['family', 'individual']);

export const DEFAULT_SEARCH_ROUTE_STATE = {
  q: '',
  clusterMode: 'family',
  topN: 20,
  minRelevance: 0,
  consolidate: true,
  consolidateMinOverlap: 0.8,
  recencyWeight: 0,
  tournamentId: '',
  seedPlayerIds: [],
  seedTeamId: null,
  seedTeamName: '',
  teamAIds: [],
  teamBIds: [],
  snapshotId: '',
};

function cleanText(value) {
  return String(value || '').trim();
}

function parsePositiveInt(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return Math.trunc(parsed);
}

function clamp(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function parseBoolean(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  if (typeof value === 'boolean') return value;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'true' || normalized === '1') return true;
  if (normalized === 'false' || normalized === '0') return false;
  return fallback;
}

export function parseIdListParam(value) {
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

export function uniqueIds(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) continue;
    const normalized = Math.trunc(parsed);
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

export function idListToParam(values) {
  return uniqueIds(values).join(',');
}

export function normalizeClusterMode(value, fallback = 'family') {
  const normalized = cleanText(value).toLowerCase();
  return CLUSTER_MODES.has(normalized) ? normalized : fallback;
}

export function normalizeTeamScope(value, fallback = 'family') {
  const normalized = cleanText(value).toLowerCase();
  return TEAM_SCOPES.has(normalized) ? normalized : fallback;
}

export function normalizeSearchRouteState(state = {}) {
  return {
    q: cleanText(state.q),
    clusterMode: normalizeClusterMode(state.clusterMode, DEFAULT_SEARCH_ROUTE_STATE.clusterMode),
    topN: Math.trunc(clamp(state.topN, 1, 200, DEFAULT_SEARCH_ROUTE_STATE.topN)),
    minRelevance: clamp(state.minRelevance, 0, 1, DEFAULT_SEARCH_ROUTE_STATE.minRelevance),
    consolidate: parseBoolean(state.consolidate, DEFAULT_SEARCH_ROUTE_STATE.consolidate),
    consolidateMinOverlap: clamp(
      state.consolidateMinOverlap,
      0,
      1,
      DEFAULT_SEARCH_ROUTE_STATE.consolidateMinOverlap,
    ),
    recencyWeight: clamp(state.recencyWeight, 0, 1, DEFAULT_SEARCH_ROUTE_STATE.recencyWeight),
    tournamentId: parsePositiveInt(state.tournamentId)
      ? String(parsePositiveInt(state.tournamentId))
      : '',
    seedPlayerIds: uniqueIds(state.seedPlayerIds || parseIdListParam(state.seedPlayers)),
    seedTeamId: parsePositiveInt(state.seedTeamId),
    seedTeamName: cleanText(state.seedTeamName),
  };
}

export function searchRouteSignature(state = {}) {
  return JSON.stringify(normalizeSearchRouteState(state));
}

export function parseSearchRouteState(searchParams) {
  const normalized = normalizeSearchRouteState({
    q: searchParams.get('q'),
    clusterMode: searchParams.get('clusterMode'),
    topN: searchParams.get('topN'),
    minRelevance: searchParams.get('minRelevance'),
    consolidate: searchParams.get('consolidate'),
    consolidateMinOverlap: searchParams.get('consolidateMinOverlap'),
    recencyWeight: searchParams.get('recencyWeight'),
    tournamentId: searchParams.get('tournamentId'),
    seedPlayers: searchParams.get('seedPlayers'),
    seedTeamId: searchParams.get('seedTeamId'),
    seedTeamName: searchParams.get('seedTeamName'),
  });
  return {
    ...normalized,
    teamAIds: uniqueIds(parseIdListParam(searchParams.get('teamA'))),
    teamBIds: uniqueIds(parseIdListParam(searchParams.get('teamB'))),
    snapshotId: parsePositiveInt(searchParams.get('snapshot'))
      ? String(parsePositiveInt(searchParams.get('snapshot')))
      : '',
  };
}

function buildHref(pathname, params) {
  const search = params.toString();
  return search ? `${pathname}?${search}` : pathname;
}

export function buildSearchHref(state = {}) {
  const normalized = normalizeSearchRouteState(state);
  const params = new URLSearchParams();
  if (normalized.q) params.set('q', normalized.q);
  if (normalized.clusterMode !== DEFAULT_SEARCH_ROUTE_STATE.clusterMode) {
    params.set('clusterMode', normalized.clusterMode);
  }
  if (normalized.topN !== DEFAULT_SEARCH_ROUTE_STATE.topN) {
    params.set('topN', String(normalized.topN));
  }
  if (normalized.minRelevance !== DEFAULT_SEARCH_ROUTE_STATE.minRelevance) {
    params.set('minRelevance', String(normalized.minRelevance));
  }
  if (normalized.consolidate !== DEFAULT_SEARCH_ROUTE_STATE.consolidate) {
    params.set('consolidate', String(normalized.consolidate));
  }
  if (normalized.consolidateMinOverlap !== DEFAULT_SEARCH_ROUTE_STATE.consolidateMinOverlap) {
    params.set('consolidateMinOverlap', String(normalized.consolidateMinOverlap));
  }
  if (normalized.recencyWeight !== DEFAULT_SEARCH_ROUTE_STATE.recencyWeight) {
    params.set('recencyWeight', String(normalized.recencyWeight));
  }
  if (normalized.tournamentId) params.set('tournamentId', normalized.tournamentId);
  if (normalized.seedPlayerIds.length) params.set('seedPlayers', idListToParam(normalized.seedPlayerIds));
  if (normalized.seedTeamId) params.set('seedTeamId', String(normalized.seedTeamId));
  if (normalized.seedTeamName) params.set('seedTeamName', normalized.seedTeamName);
  const teamAIds = uniqueIds(state.teamAIds);
  const teamBIds = uniqueIds(state.teamBIds);
  if (teamAIds.length) params.set('teamA', idListToParam(teamAIds));
  if (teamBIds.length) params.set('teamB', idListToParam(teamBIds));
  const snapshotId = parsePositiveInt(state.snapshotId);
  if (snapshotId) params.set('snapshot', String(snapshotId));
  return buildHref('/search', params);
}

export function parseTeamRouteState(teamIdParam, searchParams) {
  const ids = uniqueIds([
    ...parseIdListParam(teamIdParam),
    ...parseIdListParam(searchParams.get('teamIds')),
  ]);
  return {
    teamId: ids[0] ? String(ids[0]) : '',
    teamIds: ids,
    scope: normalizeTeamScope(searchParams.get('scope'), 'family'),
    snapshotId: parsePositiveInt(searchParams.get('snapshot'))
      ? String(parsePositiveInt(searchParams.get('snapshot')))
      : '',
  };
}

export function buildTeamHref({
  teamId,
  teamIds = [],
  scope = 'family',
  snapshotId = '',
} = {}) {
  const ids = uniqueIds([
    ...parseIdListParam(teamId),
    ...parseIdListParam(teamIds),
  ]);
  const params = new URLSearchParams();
  if (ids.length > 1) params.set('teamIds', idListToParam(ids));
  const normalizedScope = normalizeTeamScope(scope, 'family');
  if (normalizedScope !== 'family') params.set('scope', normalizedScope);
  const snapshot = parsePositiveInt(snapshotId);
  if (snapshot) params.set('snapshot', String(snapshot));
  if (!ids.length) return buildHref('/teams', params);
  return buildHref(`/teams/${ids[0]}`, params);
}

export function parsePlayerRouteState(playerIdParam, searchParams) {
  const playerId = parsePositiveInt(playerIdParam);
  return {
    playerId: playerId ? String(playerId) : '',
    playerName: cleanText(searchParams.get('name')),
  };
}

export function buildPlayerHref({ playerId, playerName = '' } = {}) {
  const parsed = parsePositiveInt(playerId);
  if (!parsed) return '/players';
  const params = new URLSearchParams();
  const normalizedName = cleanText(playerName);
  if (normalizedName && normalizedName !== `Player ${parsed}`) {
    params.set('name', normalizedName);
  }
  return buildHref(`/players/${parsed}`, params);
}

export function parseHeadToHeadRouteState(searchParams) {
  const teamAIds = uniqueIds(parseIdListParam(searchParams.get('teamA')));
  const teamBIds = uniqueIds(parseIdListParam(searchParams.get('teamB')));
  const snapshot = parsePositiveInt(searchParams.get('snapshot'));
  return {
    teamAId: teamAIds[0] ? String(teamAIds[0]) : '',
    teamBId: teamBIds[0] ? String(teamBIds[0]) : '',
    teamAIds,
    teamBIds,
    snapshotId: snapshot ? String(snapshot) : '',
  };
}

export function buildHeadToHeadHref({
  teamAIds = [],
  teamBIds = [],
  snapshotId = '',
} = {}) {
  const params = new URLSearchParams();
  const normalizedA = uniqueIds(teamAIds);
  const normalizedB = uniqueIds(teamBIds);
  if (normalizedA.length) params.set('teamA', idListToParam(normalizedA));
  if (normalizedB.length) params.set('teamB', idListToParam(normalizedB));
  const snapshot = parsePositiveInt(snapshotId);
  if (snapshot) params.set('snapshot', String(snapshot));
  return buildHref('/head-to-head', params);
}

export function parseClusterRouteState(clusterIdParam, searchParams) {
  const clusterId = parsePositiveInt(clusterIdParam);
  return {
    clusterId: clusterId ? String(clusterId) : '',
    query: cleanText(searchParams.get('q')),
    clusterMode: normalizeClusterMode(searchParams.get('clusterMode'), 'explore'),
    limit: Math.trunc(clamp(searchParams.get('limit'), 1, 200, 40)),
  };
}

export function buildClusterHref({
  clusterId = '',
  query = '',
  clusterMode = 'explore',
  limit = 40,
} = {}) {
  const params = new URLSearchParams();
  const normalizedQuery = cleanText(query);
  if (normalizedQuery) params.set('q', normalizedQuery);
  const normalizedMode = normalizeClusterMode(clusterMode, 'explore');
  if (normalizedMode !== 'explore') params.set('clusterMode', normalizedMode);
  const normalizedLimit = Math.trunc(clamp(limit, 1, 200, 40));
  if (normalizedLimit !== 40) params.set('limit', String(normalizedLimit));
  const parsedClusterId = parsePositiveInt(clusterId);
  return buildHref(parsedClusterId ? `/clusters/${parsedClusterId}` : '/clusters', params);
}
