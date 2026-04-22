import { requestJson } from './client';

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
