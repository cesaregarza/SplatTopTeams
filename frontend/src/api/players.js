import { requestJson } from './client';

export function fetchPlayerSuggestions({ q, limit = 10, signal } = {}) {
  const params = new URLSearchParams({ q, limit: String(limit) });
  return requestJson(`/api/players/suggest?${params.toString()}`, { signal });
}

export function fetchPlayerTeams({ playerId, limit = 50, signal } = {}) {
  const pid = Number(playerId);
  if (!Number.isFinite(pid) || pid <= 0) {
    throw new Error('Player ID must be a positive integer');
  }
  const params = new URLSearchParams({ limit: String(Math.max(1, Math.min(200, Number(limit) || 50))) });
  return requestJson(`/api/players/${Math.trunc(pid)}/teams?${params.toString()}`, { signal });
}
