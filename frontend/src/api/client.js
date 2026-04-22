const API_BASE = import.meta.env.VITE_API_BASE || '';
const TEAM_MATCHES_FALLBACK_BASE = import.meta.env.VITE_MATCHES_API_BASE
  || (import.meta.env.DEV ? '/__splat_top_api' : 'https://splat.top');
const TRANSIENT_GATEWAY_STATUSES = new Set([502, 503, 504]);
const MAX_TRANSIENT_RETRIES = 2;

function replaceFamilyClusterMode(path) {
  const queryStart = path.indexOf('?');
  if (queryStart < 0) return null;

  const pathname = path.slice(0, queryStart);
  const search = path.slice(queryStart + 1);
  const params = new URLSearchParams(search);
  if (params.get('cluster_mode') !== 'family') return null;

  params.set('cluster_mode', 'explore');
  return `${pathname}?${params.toString()}`;
}

function isUnsupportedFamilyClusterModeError(status, text) {
  if (status !== 422 || !text) return false;

  try {
    const payload = JSON.parse(text);
    const details = Array.isArray(payload?.detail) ? payload.detail : [];
    return details.some((detail) => (
      detail?.type === 'literal_error'
      && Array.isArray(detail?.loc)
      && detail.loc[0] === 'query'
      && detail.loc[1] === 'cluster_mode'
      && typeof detail?.ctx?.expected === 'string'
      && detail.ctx.expected.includes("'strict'")
      && detail.ctx.expected.includes("'explore'")
    ));
  } catch {
    return false;
  }
}

function buildRequestUrl(path, base = API_BASE) {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  return `${base}${path}`;
}

function isSafeRetryMethod(method) {
  if (!method) return true;
  return String(method).toUpperCase() === 'GET';
}

function looksLikeHtmlGatewayPage(text) {
  if (typeof text !== 'string' || !text.trim()) return false;
  const normalized = text.toLowerCase();
  return normalized.includes('<html')
    && (normalized.includes('bad gateway')
      || normalized.includes('nginx')
      || normalized.includes('<body'));
}

function friendlyErrorMessage(path, status, text) {
  if (TRANSIENT_GATEWAY_STATUSES.has(status) && looksLikeHtmlGatewayPage(text)) {
    if (path.startsWith('/api/team-search')) {
      return 'Team search is temporarily unavailable. Try again in a moment.';
    }
    if (path.startsWith('/api/analytics/team/')) {
      if (path.includes('/matches')) {
        return 'Recent match data is temporarily unavailable. Try again in a moment.';
      }
      return 'Team detail is temporarily unavailable. Try again in a moment.';
    }
    return 'The API is temporarily unavailable. Try again in a moment.';
  }
  return text || `Request failed: ${status}`;
}

function wait(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export function getApiBase() {
  return API_BASE;
}

export function getTeamMatchesFallbackBase() {
  return TEAM_MATCHES_FALLBACK_BASE;
}

export async function requestJson(path, options = {}) {
  return requestJsonWithBase(path, API_BASE, options);
}

export async function requestJsonWithBase(path, base = API_BASE, options = {}, attempt = 0) {
  const response = await fetch(buildRequestUrl(path, base), {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    const fallbackPath = replaceFamilyClusterMode(path);
    if (fallbackPath && isUnsupportedFamilyClusterModeError(response.status, text)) {
      return requestJsonWithBase(fallbackPath, base, options, attempt);
    }

    if (attempt < MAX_TRANSIENT_RETRIES
      && isSafeRetryMethod(options.method)
      && TRANSIENT_GATEWAY_STATUSES.has(response.status)) {
      await wait(250 * (attempt + 1));
      return requestJsonWithBase(path, base, options, attempt + 1);
    }

    const error = new Error(friendlyErrorMessage(path, response.status, text));
    error.status = response.status;
    error.responseText = text;
    error.url = buildRequestUrl(path, base);
    throw error;
  }

  return response.json();
}
