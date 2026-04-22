import {
  getApiBase,
  getTeamMatchesFallbackBase,
  requestJson,
  requestJsonWithBase,
} from './client';
import { toPositiveIntList } from './utils';

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

  const teamAList = toPositiveIntList(teamAIds).map((item) => String(Math.trunc(item)));
  const teamBList = toPositiveIntList(teamBIds).map((item) => String(Math.trunc(item)));

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

export function fetchAnalyticsTeam({ teamId, clusterMode, neighbors = 12, snapshotId } = {}) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    neighbors: String(neighbors),
  });
  if (snapshotId) {
    params.set('snapshot_id', String(snapshotId));
  }
  return requestJson(`/api/analytics/team/${teamId}?${params.toString()}`);
}

export function fetchAnalyticsTeamBlend({
  teamId,
  clusterMode,
  neighbors = 12,
  semanticWeight = 0.5,
  snapshotId,
} = {}) {
  const params = new URLSearchParams({
    cluster_mode: clusterMode,
    neighbors: String(neighbors),
    semantic_weight: String(semanticWeight),
  });
  if (snapshotId) {
    params.set('snapshot_id', String(snapshotId));
  }
  return requestJson(`/api/analytics/team/${teamId}/blend?${params.toString()}`);
}

export function fetchAnalyticsTeamMatches({
  teamId,
  teamIds,
  limit = 25,
  snapshotId,
} = {}) {
  const params = new URLSearchParams({
    limit: String(limit),
  });

  const normalizedTeamIds = toPositiveIntList(teamIds);

  if (normalizedTeamIds.length > 0) {
    params.set('team_ids', normalizedTeamIds.join(','));
  }
  if (snapshotId) {
    params.set('snapshot_id', String(snapshotId));
  }

  const path = `/api/analytics/team/${teamId}/matches?${params.toString()}`;

  return requestJson(path).catch((error) => {
    const shouldRetryWithFallback = error?.status === 404
      && getTeamMatchesFallbackBase()
      && getTeamMatchesFallbackBase() !== getApiBase();

    if (!shouldRetryWithFallback) {
      throw error;
    }

    return requestJsonWithBase(path, getTeamMatchesFallbackBase());
  });
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
