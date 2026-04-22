import { requestJson } from './client';

export function fetchTeamSearch({
  q,
  topN,
  clusterMode,
  minRelevance,
  tournamentId,
  seedPlayerIds,
  consolidate = true,
  consolidateMinOverlap = 0.8,
  recencyWeight = 0,
}) {
  const params = new URLSearchParams({
    q,
    top_n: String(topN),
    cluster_mode: clusterMode,
  });
  if (Number.isFinite(Number(tournamentId)) && Number(tournamentId) > 0) {
    params.set('tournament_id', String(Math.trunc(Number(tournamentId))));
  }
  if (typeof minRelevance === 'number') {
    params.set('min_relevance', String(minRelevance));
  }
  if (typeof consolidateMinOverlap === 'number') {
    params.set('consolidate_min_overlap', String(consolidateMinOverlap));
  }
  if (typeof consolidate === 'boolean') {
    params.set('consolidate', String(consolidate));
  }
  if (typeof recencyWeight === 'number' && recencyWeight > 0) {
    params.set('recency_weight', String(recencyWeight));
  }
  if (Array.isArray(seedPlayerIds) && seedPlayerIds.length > 0) {
    for (const value of seedPlayerIds) {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed <= 0) continue;
      params.append('seed_player_ids', String(Math.trunc(parsed)));
    }
  }
  return requestJson(`/api/team-search?${params.toString()}`);
}

export function fetchTeamSuggestions({ q, limit = 8, signal } = {}) {
  const params = new URLSearchParams({ q, limit: String(limit) });
  return requestJson(`/api/team-search/suggest?${params.toString()}`, { signal });
}

export function fetchTournamentTeams({ tournamentId, q = '', limit = 200, signal } = {}) {
  const tid = Number(tournamentId);
  if (!Number.isFinite(tid) || tid <= 0) {
    throw new Error('Tournament ID must be a positive integer');
  }

  const params = new URLSearchParams({
    limit: String(Math.max(1, Math.min(700, Number(limit) || 200))),
  });
  if (q && String(q).trim()) {
    params.set('q', String(q).trim());
  }
  return requestJson(`/api/tournaments/${Math.trunc(tid)}/teams?${params.toString()}`, { signal });
}
