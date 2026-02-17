const API_BASE = import.meta.env.VITE_API_BASE || '';

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return response.json();
}

export function fetchHealth() {
  return requestJson('/api/health');
}

export function fetchTeamSearch({ q, topN, clusterMode, minRelevance, consolidate = true, consolidateMinOverlap = 0.8 }) {
  const params = new URLSearchParams({
    q,
    top_n: String(topN),
    cluster_mode: clusterMode,
  });
  if (typeof minRelevance === 'number') {
    params.set('min_relevance', String(minRelevance));
  }
  if (typeof consolidateMinOverlap === 'number') {
    params.set('consolidate_min_overlap', String(consolidateMinOverlap));
  }
  if (typeof consolidate === 'boolean') {
    params.set('consolidate', String(consolidate));
  }
  return requestJson(`/api/team-search?${params.toString()}`);
}

export function fetchClusters({ q, clusterMode, limit }) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    limit: String(limit),
  });
  if (q) {
    params.set('q', q);
  }
  return requestJson(`/api/clusters?${params.toString()}`);
}

export function fetchClusterDetail(clusterId, clusterMode) {
  const params = new URLSearchParams({ cluster_mode: clusterMode });
  return requestJson(`/api/clusters/${clusterId}?${params.toString()}`);
}

export function fetchAnalyticsOverview({ clusterMode, limitClusters = 20, volatileLimit = 15 }) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    limit_clusters: String(limitClusters),
    volatile_limit: String(volatileLimit),
  });
  return requestJson(`/api/analytics/overview?${params.toString()}`);
}

export function fetchAnalyticsMatchups({ clusterMode, minMatches = 3, limit = 30 }) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    min_matches: String(minMatches),
    limit: String(limit),
  });
  return requestJson(`/api/analytics/matchups?${params.toString()}`);
}

export function fetchAnalyticsHeadToHead({
  teamAId,
  teamBId,
  teamAIds,
  teamBIds,
  snapshotId,
  limit = 200,
}) {
  const params = new URLSearchParams({
    limit: String(limit),
  });

  const toIdList = (value) => {
    if (Array.isArray(value)) {
      return value
        .map((item) => Number(item))
        .filter((item) => Number.isFinite(item) && item > 0);
    }
    if (Number.isFinite(Number(value))) {
      return [Number(value)];
    }
    if (typeof value === 'string') {
      return (value.match(/\d+/g) || [])
        .map((item) => Number(item))
        .filter((item) => Number.isFinite(item) && item > 0);
    }
    return [];
  };

  const teamAList = toIdList(teamAIds).map((item) => String(Math.trunc(item)));
  const teamBList = toIdList(teamBIds).map((item) => String(Math.trunc(item)));

  if (teamAList.length > 0) {
    if (teamAList.length > 1 || !teamAId) {
      params.set('team_a_ids', teamAList.join(','));
    } else {
      params.set('team_a_id', teamAList[0]);
    }
  } else if (teamAId !== null && teamAId !== undefined) {
    params.set('team_a_id', String(teamAId));
  }

  if (teamBList.length > 0) {
    if (teamBList.length > 1 || !teamBId) {
      params.set('team_b_ids', teamBList.join(','));
    } else {
      params.set('team_b_id', teamBList[0]);
    }
  } else if (teamBId !== null && teamBId !== undefined) {
    params.set('team_b_id', String(teamBId));
  }

  if (snapshotId) {
    params.set('snapshot_id', String(snapshotId));
  }

  if (!params.get('team_a_id') && !params.get('team_a_ids')) {
    throw new Error('Team A id(s) are required for head-to-head');
  }
  if (!params.get('team_b_id') && !params.get('team_b_ids')) {
    throw new Error('Team B id(s) are required for head-to-head');
  }

  return requestJson(`/api/analytics/head-to-head?${params.toString()}`);
}

export function fetchAnalyticsRosterOverlap({
  clusterMode,
  minSimilarity = 0.8,
  maxPlayerOverlap = 0.3,
  minClusterSize = 2,
  limit = 30,
}) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    min_similarity: String(minSimilarity),
    max_player_overlap: String(maxPlayerOverlap),
    min_cluster_size: String(minClusterSize),
    limit: String(limit),
  });
  return requestJson(`/api/analytics/roster-overlap?${params.toString()}`);
}

export function fetchAnalyticsTeam({ teamId, clusterMode, neighbors = 12 }) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    neighbors: String(neighbors),
  });
  return requestJson(`/api/analytics/team/${teamId}?${params.toString()}`);
}

export function fetchAnalyticsTeamBlend({
  teamId,
  clusterMode,
  neighbors = 12,
  semanticWeight = 0.5,
}) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    neighbors: String(neighbors),
    semantic_weight: String(semanticWeight),
  });
  return requestJson(`/api/analytics/team/${teamId}/blend?${params.toString()}`);
}

export function fetchAnalyticsOutliers({ clusterMode, limit = 30 }) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    limit: String(limit),
  });
  return requestJson(`/api/analytics/outliers?${params.toString()}`);
}

export function fetchAnalyticsSpace({ clusterMode, maxPoints = 800 }) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    max_points: String(maxPoints),
  });
  return requestJson(`/api/analytics/space?${params.toString()}`);
}

export function fetchAnalyticsDrift({
  clusterMode,
  currentSnapshotId,
  previousSnapshotId,
  topMovers = 20,
}) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    top_movers: String(topMovers),
  });
  if (currentSnapshotId) {
    params.set('current_snapshot_id', String(currentSnapshotId));
  }
  if (previousSnapshotId) {
    params.set('previous_snapshot_id', String(previousSnapshotId));
  }
  return requestJson(`/api/analytics/drift?${params.toString()}`);
}
