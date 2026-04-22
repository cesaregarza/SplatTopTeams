import { requestJson } from './client';

export function fetchHealth() {
  return requestJson('/api/health');
}
