import React, { useEffect, useMemo, useState } from 'react';
import { fetchTeamSearch } from '../api';

function fmtDate(ms) {
  const value = toEpochMs(ms);
  if (value === null) return 'n/a';
  return new Date(value).toISOString().slice(0, 10);
}

function percent(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(0)}%` : 'n/a';
}

function toEpochMs(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value <= 0) return null;
  const minReasonableEpochMs = 946684800000;

  // Upstream sources vary between seconds, milliseconds, and microseconds.
  if (value > 1_000_000_000_000) return Math.floor(value / 1000);
  if (value > 10_000_000_000) return Math.floor(value);
  const milliseconds = Math.floor(value * 1000);
  return milliseconds < minReasonableEpochMs ? null : milliseconds;
}

function relativeOrMissingDate(ms) {
  const value = toEpochMs(ms);
  if (value === null) return '—';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';

  const now = Date.now();
  const deltaMs = now - value;
  if (deltaMs < 0) return date.toISOString().slice(0, 10);

  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  const week = 7 * day;
  const month = 30 * day;

  if (deltaMs < 45 * 1000) return 'just now';
  if (deltaMs < 90 * 60 * 1000) {
    const count = Math.round(deltaMs / minute);
    return `${count} minute${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 36 * hour) {
    const count = Math.round(deltaMs / hour);
    return `${count} hour${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 14 * day) {
    const count = Math.round(deltaMs / day);
    return `${count} day${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 8 * week) {
    const count = Math.round(deltaMs / week);
    return `${count} week${count === 1 ? '' : 's'} ago`;
  }
  if (deltaMs < 10 * month) {
    const count = Math.round(deltaMs / month);
    return `${count} month${count === 1 ? '' : 's'} ago`;
  }

  return date.toISOString().slice(0, 10);
}

function clamp01(value) {
  if (!Number.isFinite(value)) return null;
  return Math.max(0, Math.min(1, value));
}

function confidenceBand(value) {
  const score = clamp01(value);
  if (score === null) return { label: 'n/a', tone: 'neutral' };
  if (score >= 0.95) return { label: 'High', tone: 'high' };
  if (score >= 0.85) return { label: 'Medium', tone: 'medium' };
  return { label: 'Low', tone: 'low' };
}

function describeDelta(current, baseline) {
  const currentValue = safeInt(current);
  const baselineValue = safeInt(baseline);
  if (!Number.isFinite(currentValue) || !Number.isFinite(baselineValue)) return null;
  if (currentValue === baselineValue) {
    return {
      text: '0',
      direction: 'neutral',
      sign: '',
    };
  }

  const diff = currentValue - baselineValue;
  return {
    text: `${diff > 0 ? '+' : ''}${diff}`,
    direction: diff > 0 ? 'up' : 'down',
    sign: diff > 0 ? '+' : '-',
  };
}

function safeInt(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function safeIntOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function safeNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function safePlayerId(value) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? Math.trunc(parsed) : null;
}

function playerInitials(name) {
  const trimmed = String(name || '').trim();
  if (!trimmed) return '?';

  const parts = trimmed
    .replace(/[^a-zA-Z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
  if (!parts.length) return '?';
  if (parts.length === 1) {
    return parts[0].slice(0, 2).toUpperCase();
  }
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
}

function playerSortKeyId(value) {
  return value === null ? Number.MAX_SAFE_INTEGER : value;
}

function normalizeCorePlayers(values) {
  if (!Array.isArray(values)) return [];

  const out = [];
  const seen = new Set();

  for (const value of values) {
    const id = safePlayerId(value?.player_id);
    const name = playerName(value?.player_name);
    const sendouUrl = value?.sendou_url || (id !== null ? `https://sendou.ink/u/${id}` : null);
    const matchesPlayed = safeInt(value?.matches_played);
    const key = id === null ? `name:${name.toLowerCase()}` : `id:${id}`;
    if (seen.has(key)) continue;
    seen.add(key);

    out.push({
      id,
      name,
      sendouUrl,
      matchesPlayed,
    });
  }

  out.sort((a, b) => {
    const matchesDiff = b.matchesPlayed - a.matchesPlayed;
    if (matchesDiff !== 0) return matchesDiff;
    return playerSortKeyId(a.id) - playerSortKeyId(b.id);
  });

  return out;
}

function formatMatchesPlayed(value) {
  const parsed = safeInt(value);
  return `${parsed} ${parsed === 1 ? 'match' : 'matches'}`;
}

function lineupFrequency(summary) {
  if (!summary) return null;
  const withBracket = /^(\d+)x:/.exec(summary);
  if (withBracket) return Number(withBracket[1]);

  const bare = /^(\d+)x/.exec(summary);
  return bare ? Number(bare[1]) : null;
}

function playerName(value) {
  const text = String(value || '').trim();
  if (!text) return 'Unknown Player';
  if (/^\d+$/.test(text)) return 'Unknown Player';
  return text;
}

function buildTopLineupRoster(playerIds, playerNames) {
  if (!Array.isArray(playerNames) || !playerNames.length) return [];
  const normalizedNames = playerNames.map(playerName);
  const normalizedIds = Array.isArray(playerIds)
    ? playerIds.map(safePlayerId)
    : [];

  const seen = new Set();
  const roster = [];
  const maxLen = Math.max(normalizedNames.length, normalizedIds.length);

  for (let idx = 0; idx < maxLen; idx += 1) {
    const id = normalizedIds[idx] ?? null;
    const name = normalizedNames[idx] || 'Unknown Player';
    const key = id === null ? `name:${name.toLowerCase()}` : `id:${id}`;
    if (seen.has(key)) continue;
    seen.add(key);

    roster.push({
      id,
      name,
      sendouUrl: id === null ? null : `https://sendou.ink/u/${id}`,
    });
  }

  return roster;
}

function pluralizeCount(value, singular) {
  const parsed = safeInt(value);
  return `${parsed} ${singular}${parsed === 1 ? '' : 's'}`;
}

const CONSOLIDATED_BUCKETS = [
  { label: '1', min: 1, max: 1 },
  { label: '2–3', min: 2, max: 3 },
  { label: '4–7', min: 4, max: 7 },
  { label: '8–15', min: 8, max: 15 },
  { label: '16–30', min: 16, max: 30 },
  { label: '31–60', min: 31, max: 60 },
  { label: '61–100', min: 61, max: 100 },
  { label: '101+', min: 101, max: Number.POSITIVE_INFINITY },
];

function buildConsolidatedBuckets(values) {
  const counts = CONSOLIDATED_BUCKETS.map(() => 0);

  for (const value of values) {
    const parsed = safeInt(value);
    if (!Number.isFinite(parsed) || parsed < 1) continue;

    const bucketIndex = CONSOLIDATED_BUCKETS.findIndex((bucket) => parsed >= bucket.min && parsed <= bucket.max);
    if (bucketIndex === -1) continue;
    counts[bucketIndex] += 1;
  }

  return CONSOLIDATED_BUCKETS.map((bucket, index) => ({
    label: bucket.label,
    count: counts[index],
  }));
}

export default function TeamSearch({
  selectedTeamAId = '',
  selectedTeamBId = '',
  selectedTeamAIds = [],
  selectedTeamBIds = [],
  onOpenHeadToHead = () => {},
}) {
  const [query, setQuery] = useState('');
  const [clusterMode, setClusterMode] = useState('explore');
  const [topN, setTopN] = useState(20);
  const [minRelevance, setMinRelevance] = useState(0.8);
  const [consolidate, setConsolidate] = useState(true);
  const [consolidateMinOverlap, setConsolidateMinOverlap] = useState(0.8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [payload, setPayload] = useState(null);
  const [expandedCards, setExpandedCards] = useState(() => new Set());
  const [expandedPlayers, setExpandedPlayers] = useState(() => new Set());
  const [expandedAliases, setExpandedAliases] = useState(() => new Set());

  const resultLabel = useMemo(() => {
    if (!payload) return 'No query yet.';
    const clustered = (payload.results || []).filter((row) => row.is_clustered).length;
    const consolidatedGroups = (payload.results || []).filter((row) => row.is_consolidated).length;
    const deduped = (payload.results || []).reduce(
      (acc, row) => acc + (Number(row.consolidated_team_count) || 1),
      0,
    );
    return `${payload.result_count} result groups (${deduped} teams) from snapshot ${payload.snapshot_id} · ${clustered} clustered · ${consolidatedGroups} consolidated (${payload.cluster_mode})`;
  }, [payload]);

  const results = useMemo(() => payload?.results || [], [payload]);
  const baselineResult = useMemo(() => (results.length > 0 ? results[0] : null), [results]);
  const teamAValue = safeIntOrNull(selectedTeamAId);
  const teamBValue = safeIntOrNull(selectedTeamBId);
  const selectedTeamAIdSet = useMemo(() => {
    const ids = [...selectedTeamAIds, teamAValue].filter((value) => Number.isFinite(value) && value > 0);
    return new Set(ids);
  }, [selectedTeamAIds, teamAValue]);
  const selectedTeamBIdSet = useMemo(() => {
    const ids = [...selectedTeamBIds, teamBValue].filter((value) => Number.isFinite(value) && value > 0);
    return new Set(ids);
  }, [selectedTeamBIds, teamBValue]);

  useEffect(() => {
    const nextExpandedCards = new Set();
    const firstTeamId = results[0]?.team_id;
    if (firstTeamId !== undefined && firstTeamId !== null) {
      nextExpandedCards.add(String(firstTeamId));
    }
    setExpandedCards(nextExpandedCards);
    setExpandedPlayers(new Set());
    setExpandedAliases(new Set());
  }, [payload, results]);

  function toggleCardExpansion(teamId) {
    const key = String(teamId);
    setExpandedCards((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  function togglePlayerExpansion(teamId) {
    const key = String(teamId);
    setExpandedPlayers((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  function toggleAliasExpansion(teamId) {
    const key = String(teamId);
    setExpandedAliases((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    try {
      const data = await fetchTeamSearch({
        q: query.trim(),
        topN,
        clusterMode,
        minRelevance,
        consolidate,
        consolidateMinOverlap,
      });
      setPayload(data);
    } catch (err) {
      setError(err.message || 'Failed to load results');
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel team-search-panel" aria-labelledby="team-search-title">
      <h2 id="team-search-title" className="panel-title">Team Search</h2>
      <p className="panel-subtitle">
        Search by team name and rank similar teams using identity-anchored embeddings.
      </p>

      <form className="search-form" onSubmit={onSubmit}>
        <label htmlFor="team-query" className="field-label">Team query</label>
        <input
          id="team-query"
          className="input"
          type="search"
          placeholder="e.g. FTW, Moonlight, Hypernova"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          required
        />

        <div className="row fields-row">
          <div className="field">
            <label htmlFor="cluster-mode" className="field-label">Cluster profile</label>
            <select
              id="cluster-mode"
              className="input"
              value={clusterMode}
              onChange={(e) => setClusterMode(e.target.value)}
            >
              <option value="strict">strict</option>
              <option value="explore">explore</option>
            </select>
          </div>

          <div className="field">
            <label className="checkbox-field">
              <input
                id="consolidate-results"
                type="checkbox"
                checked={consolidate}
                onChange={(e) => setConsolidate(e.target.checked)}
              />
              Consolidate near-duplicate teams
            </label>
          </div>

          <div className="field">
            <label htmlFor="top-n" className="field-label">Top N</label>
            <input
              id="top-n"
              className="input"
              type="number"
              min={1}
              max={100}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value) || 20)}
            />
          </div>

          <div className="field">
            <label htmlFor="consolidate-min-overlap" className="field-label slider-label">
              Consolidation overlap
            </label>
            <input
              id="consolidate-min-overlap"
              className="input slider-input"
              type="range"
              min={0}
              max={100}
              step={10}
              value={Math.round(consolidateMinOverlap * 100)}
              onChange={(e) => setConsolidateMinOverlap(Number(e.target.value) / 100)}
              aria-describedby="consolidate-overlap-value"
              disabled={!consolidate}
            />
            <output id="consolidate-overlap-value" className="slider-output">
              {Math.round(consolidateMinOverlap * 100)}%
            </output>
          </div>

          <div className="field">
            <label htmlFor="min-relevance" className="field-label slider-label">
              Minimum relevance
            </label>
            <input
              id="min-relevance"
              className="input slider-input"
              type="range"
              min={0}
              max={100}
              step={10}
              value={Math.round(minRelevance * 100)}
              onChange={(e) => setMinRelevance(Number(e.target.value) / 100)}
              list="relevance-marks"
              aria-describedby="min-relevance-value"
            />
            <output id="min-relevance-value" className="slider-output">
              {Math.round(minRelevance * 100)}%
            </output>
            <datalist id="relevance-marks">
              <option value="0" label="0%" />
              <option value="10" label="10%" />
              <option value="20" label="20%" />
              <option value="30" label="30%" />
              <option value="40" label="40%" />
              <option value="50" label="50%" />
              <option value="60" label="60%" />
              <option value="70" label="70%" />
              <option value="80" label="80%" />
              <option value="90" label="90%" />
              <option value="100" label="100%" />
            </datalist>
          </div>
        </div>

        <button className="button" type="submit" disabled={loading}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      <p className="status" role="status" aria-live="polite">{resultLabel}</p>
      {error ? <p className="error">{error}</p> : null}


      <div className="card-grid">
        {results.map((row, index) => {
          const rowTeamId = safeInt(row.team_id);
          const teamIdText = String(row.team_id || 'n/a');
          const teamDisplayName = row.team_name || `Team ${teamIdText}`;
          const isTeamASelected = selectedTeamAIdSet.has(rowTeamId);
          const isTeamBSelected = selectedTeamBIdSet.has(rowTeamId);
          const fallbackPlayers = normalizeCorePlayers(
            (row.top_lineup_players || []).map((player) => ({
              player_id: player?.player_id,
              player_name: player?.player_name || playerName(player?.player_name),
              matches_played: 0,
              sendou_url: player?.sendou_url,
            })),
          );
          const corePlayers = normalizeCorePlayers(
            row.core_lineup_players && row.core_lineup_players.length
              ? row.core_lineup_players
              : fallbackPlayers,
          );
          const displayPlayers = corePlayers.length ? corePlayers : [
            { id: null, name: 'Unknown Player', sendouUrl: null, matchesPlayed: 0 },
          ];
          const isExpanded = expandedPlayers.has(teamIdText);
          const maxVisiblePlayers = 4;
          const visiblePlayers = isExpanded ? displayPlayers : displayPlayers.slice(0, maxVisiblePlayers);
          const canExpandPlayers = displayPlayers.length > maxVisiblePlayers;
          const visiblePlayerSummary = isExpanded
            ? `Showing all ${displayPlayers.length} players`
            : `Showing top ${Math.min(maxVisiblePlayers, displayPlayers.length)} of ${displayPlayers.length} players`;
          const playerListId = `team-core-players-${teamIdText}`;
          const topLineupCount = lineupFrequency(row.top_lineup_summary);
          const matchCount = safeInt(row.match_count ?? row.lineup_count);
          const tournamentCount = safeInt(row.tournament_count);
          const consolidatedCount = Number(row.consolidated_team_count) || 1;
          const normalizedConsolidatedCount = Math.max(1, consolidatedCount);
          const consolidatedTeams = row.consolidated_teams || [];
          const consolidatedEntries = normalizedConsolidatedCount > 1
                ? [
                {
                team_id: teamIdText,
                team_name: row.team_name,
                tournament_id: row.tournament_id,
                event_time_ms: row.event_time_ms,
                match_count: matchCount,
                tournament_count: tournamentCount,
              },
              ...consolidatedTeams,
            ]
            : [];
          const consolidatedEntriesSorted = consolidatedEntries
            .slice()
            .sort((a, b) => safeInt(b.match_count) - safeInt(a.match_count) || safeInt(b.tournament_count) - safeInt(a.tournament_count));
          const consolidatedMatchValues = consolidatedEntriesSorted.map((team) => safeInt(team.match_count));
          const consolidatedMatchRangeMin = consolidatedMatchValues.length ? Math.min(...consolidatedMatchValues) : 0;
          const consolidatedMatchRangeMax = consolidatedMatchValues.length ? Math.max(...consolidatedMatchValues) : 0;
          const consolidatedMatchAverage = consolidatedMatchValues.length
            ? consolidatedMatchValues.reduce((acc, count) => acc + count, 0) / consolidatedMatchValues.length
            : 0;
          const consolidatedMatchBuckets = buildConsolidatedBuckets(consolidatedMatchValues);
          const maxBucketCount = consolidatedMatchBuckets.reduce((acc, bucket) => Math.max(acc, bucket.count), 0);
          const consolidatedTournamentIds = new Set(
            consolidatedEntriesSorted
              .map((entry) => entry.tournament_id)
              .filter((value) => value !== null && value !== undefined)
              .map((value) => String(value)),
          );
          const consolidatedTournamentsCovered = consolidatedTournamentIds.size || tournamentCount;
          const maxVisibleAliases = 3;
          const aliasesExpanded = expandedAliases.has(teamIdText);
          const visibleAliases = aliasesExpanded
            ? consolidatedEntriesSorted
            : consolidatedEntriesSorted.slice(0, maxVisibleAliases);
          const canExpandAliases = consolidatedEntriesSorted.length > maxVisibleAliases;
          const aliasListId = `team-aliases-${teamIdText}`;
          const distinctLineups = safeInt(row.distinct_lineup_count ?? 0);
          const uniquePlayers = safeInt(row.unique_player_count);
          const topLineupPlayers = buildTopLineupRoster(
            row.top_lineup_player_ids,
            row.top_lineup_player_names,
          );
          const topLineupPlayerCount = safeInt(
            row.top_lineup_player_count ?? topLineupPlayers.length,
          );
          const isClustered = row.cluster_id != null;
          const clusterSize = row.cluster_size != null ? Number(row.cluster_size) : null;
          const clusterSizeText = clusterSize == null ? 'n/a' : `${clusterSize} team${clusterSize === 1 ? '' : 's'}`;
          const clusterIdText = safeInt(row.cluster_id);
          const isConsolidatedProfile = normalizedConsolidatedCount > 1;
          const clusterLabel = isClustered
            ? `cluster ${clusterIdText} · size ${clusterSizeText}`
            : 'unclustered';
          const clusterTitle = isClustered
            ? `Team belongs to cluster ${clusterIdText} with ${clusterSizeText} in total`
            : 'Team was not assigned to a cluster';
          const cardSubtitle = isConsolidatedProfile
            ? `Consolidated profile (${normalizedConsolidatedCount} teams) — Unclustered seed`
            : isClustered
              ? `Cluster ${clusterIdText} (${clusterSizeText}) — Candidate cluster`
              : 'Unclustered profile (single team) — No cluster assignment';
          const idLabel = isClustered
            ? `Cluster member ID: ${teamIdText}`
            : `Entity ID: ${teamIdText}`;
          const topLineupMatchPayload = safeNumber(row.top_lineup_match_count);
          const topLineupSharePayload = safeNumber(row.top_lineup_match_share);
          const topLineupMatches = topLineupMatchPayload !== null
            ? safeInt(topLineupMatchPayload)
            : (Number.isFinite(topLineupCount)
              ? safeInt(topLineupCount)
              : (matchCount > 0 && topLineupSharePayload !== null
                ? Math.round(matchCount * topLineupSharePayload)
                : 0));
          const topLineupPlayerShare = matchCount > 0
            ? topLineupMatches / matchCount
            : 0;
          const topLineupDistinctShareCount = topLineupMatches > 0 && distinctLineups > 0 ? 1 : 0;
          const topLineupDistinctShare = distinctLineups > 0
            ? topLineupDistinctShareCount / distinctLineups
            : 0;
          const topLineupUsageLabel = matchCount > 0
            ? `${topLineupMatches} / ${matchCount} matches (${percent(topLineupPlayerShare)})`
            : `${topLineupMatches} matches (n/a)`;
          const topLineupShareLabel = distinctLineups > 0
            ? `${topLineupDistinctShareCount} / ${distinctLineups} lineups (${percent(topLineupDistinctShare)})`
            : 'No lineup-variant metadata';
          const consolidatedAliasCount = consolidatedEntriesSorted.length;
          const visibleAliasCount = visibleAliases.length;
          const compareToBaseline = index > 0 && baselineResult;
          const baselineMatchCount = compareToBaseline ? safeInt(baselineResult.match_count ?? baselineResult.lineup_count) : null;
          const baselineTournamentCount = compareToBaseline ? safeInt(baselineResult.tournament_count) : null;
          const baselineDistinctLineups = compareToBaseline ? safeInt(baselineResult.distinct_lineup_count ?? 0) : null;
          const baselineUniquePlayers = compareToBaseline ? safeInt(baselineResult.unique_player_count) : null;
          const deltaMatchCount = compareToBaseline ? describeDelta(matchCount, baselineMatchCount) : null;
          const deltaTournamentCount = compareToBaseline ? describeDelta(tournamentCount, baselineTournamentCount) : null;
          const deltaDistinctLineups = compareToBaseline ? describeDelta(distinctLineups, baselineDistinctLineups) : null;
          const deltaUniquePlayers = compareToBaseline ? describeDelta(uniquePlayers, baselineUniquePlayers) : null;
          const playerMetricLabel = isConsolidatedProfile
            ? 'Players (core list)'
            : 'Players (overall)';
          const playerMetricCount = isConsolidatedProfile
            ? displayPlayers.length
            : uniquePlayers;
          const playerMetricDelta = isConsolidatedProfile ? null : deltaUniquePlayers;
          const maxCoreMatches = displayPlayers.length
            ? Math.max(...displayPlayers.map((player) => player.matchesPlayed))
            : 0;
          const coreLeadShare = matchCount > 0 ? Math.round((maxCoreMatches / matchCount) * 100) : 0;
          const coreMatchLabel =
            maxCoreMatches > 0
              ? `Highest player contribution: ${maxCoreMatches} match${maxCoreMatches === 1 ? '' : 'es'} (${coreLeadShare}% of matches)`
              : null;
          const confidenceOverallRaw = clamp01(Number(row.sim_to_query));
          const confidenceSemantic = clamp01(Number(row.sim_semantic_to_query));
          const confidenceIdentity = clamp01(Number(row.sim_identity_to_query));
          const confidenceOverall = confidenceOverallRaw === null
            ? (() => {
              const chunks = [];
              if (confidenceSemantic !== null) chunks.push(confidenceSemantic);
              if (confidenceIdentity !== null) chunks.push(confidenceIdentity);
              if (!chunks.length) return null;
              return chunks.reduce((acc, value) => acc + value, 0) / chunks.length;
            })()
            : confidenceOverallRaw;
          const confidenceOverallLabel = confidenceOverall === null
            ? 'n/a'
            : `${(confidenceOverall * 100).toFixed(1)}%`;
          const confidenceTone = confidenceBand(confidenceOverall);
          const representativeTeamName = row.representative_team_name || '—';
          const isResultExpanded = expandedCards.has(teamIdText);

          return (
              <article className="result-card" key={teamIdText}>
              <header className="result-head">
                <div className="result-head-top">
                  <h3 className="result-team-title">{teamDisplayName}</h3>
                  <p className="result-head-subtitle">{cardSubtitle}</p>
                  <p className="result-purpose">
                    {isConsolidatedProfile
                      ? 'Compare consolidated identity profile versus one cluster instance.'
                      : 'Inspect team structure and compare against nearby identities.'}
                  </p>
                  <div className="result-head-badges">
                    <span className="badge">{idLabel}</span>
                    <span
                      className={`badge ${isClustered ? 'badge-cluster' : 'badge-muted'}`}
                      title={clusterTitle}
                      aria-label={clusterTitle}
                    >
                      {clusterLabel}
                    </span>
                    {normalizedConsolidatedCount > 1 ? (
                      <span
                        className="badge badge-consolidated"
                        title={`Merged from ${normalizedConsolidatedCount} teams using lineup overlap + metadata similarity.`}
                        aria-label={`Merged from ${normalizedConsolidatedCount} teams using lineup overlap + metadata similarity.`}
                      >
                        consolidated · {normalizedConsolidatedCount} teams
                      </span>
                    ) : null}
                  </div>
                </div>
              </header>
              <div className="result-quick-row">
                <p className="result-quick-metrics">
                  <span>{safeInt(matchCount).toLocaleString()} matches</span>
                  <span>{safeInt(tournamentCount).toLocaleString()} tournaments</span>
                  <span>{safeInt(distinctLineups)} lineups</span>
                  <span>Relevance {confidenceOverallLabel}</span>
                </p>
                <div className="result-quick-actions">
                  <button
                    type="button"
                    className={`result-select-btn ${isTeamASelected ? 'is-selected' : ''}`}
                      onClick={() => onOpenHeadToHead('A', row.consolidated_team_ids || [row.team_id], payload?.snapshot_id)}
                    aria-pressed={isTeamASelected}
                    title={isTeamASelected ? 'This is Team A' : 'Select this result as Team A'}
                  >
                    {isTeamASelected ? 'Selected as Team A' : 'Select as Team A'}
                  </button>
                  <button
                    type="button"
                    className={`result-select-btn ${isTeamBSelected ? 'is-selected' : ''}`}
                      onClick={() => onOpenHeadToHead('B', row.consolidated_team_ids || [row.team_id], payload?.snapshot_id)}
                    aria-pressed={isTeamBSelected}
                    title={isTeamBSelected ? 'This is Team B' : 'Select this result as Team B'}
                  >
                    {isTeamBSelected ? 'Selected as Team B' : 'Select as Team B'}
                  </button>
                  <button
                    type="button"
                    className="result-expand-toggle"
                    onClick={() => toggleCardExpansion(teamIdText)}
                    aria-expanded={isResultExpanded}
                  >
                    {isResultExpanded ? 'Hide details' : 'Show details'}
                  </button>
                </div>
              </div>

              {isResultExpanded ? (
                <>
                <section className="player-core" aria-label="Top lineup players">
                <div className="player-core-header">
                  <div>
                    <p className="meta-label">Core lineup players</p>
                    <p className="player-core-summary">{visiblePlayerSummary}</p>
                  </div>
                  {canExpandPlayers ? (
                    <button
                      type="button"
                      className="player-core-toggle"
                      onClick={() => togglePlayerExpansion(teamIdText)}
                      aria-expanded={isExpanded}
                      aria-controls={playerListId}
                    >
                      {isExpanded
                        ? `Show top ${maxVisiblePlayers}`
                        : `Show all ${displayPlayers.length}`
                      }
                    </button>
                  ) : null}
                </div>
                <div className="player-chip-grid" id={playerListId}>
                  {visiblePlayers.map((player) => {
                    const contribution = maxCoreMatches > 0
                      ? Math.round((player.matchesPlayed / maxCoreMatches) * 100)
                      : 0;
                    const playerShare = matchCount > 0
                      ? Math.round((player.matchesPlayed / matchCount) * 100)
                      : 0;
                    const chipContent = (
                      <>
                        <span className="player-chip-avatar" aria-hidden="true">
                          {playerInitials(player.name)}
                        </span>
                        <span className="player-chip-name">{player.name}</span>
                        {player.matchesPlayed > 0 ? (
                          <span className="player-chip-meta">
                            {formatMatchesPlayed(player.matchesPlayed)} • {playerShare}% of matches
                          </span>
                        ) : null}
                        <span
                          className="player-chip-fill"
                            style={{
                              width: `${maxCoreMatches > 0
                              ? Math.max(10, contribution)
                              : 0}%`,
                            }}
                          aria-hidden="true"
                        />
                      </>
                    );
                    return player.sendouUrl ? (
                      <a
                        key={`${teamIdText}-${player.id ?? player.name}`}
                        className="player-chip player-chip-link"
                        href={player.sendouUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label={`${player.name} profile on sendou.ink`}
                      >
                        {chipContent}
                      </a>
                    ) : (
                      <span
                          key={`${teamIdText}-${player.id ?? player.name}`}
                        className="player-chip"
                      >
                        {chipContent}
                      </span>
                    );
                  })}
                </div>
                {coreMatchLabel ? <p className="player-core-subtext">{coreMatchLabel}</p> : null}
              </section>

                <div className="result-meta-grid">
                <div className="meta-item">
                  <span>Matches</span>
                  <strong className="meta-item-hero meta-item-primary">
                    <span>{safeInt(matchCount).toLocaleString()}</span>
                    {deltaMatchCount ? (
                      <span className={`meta-delta meta-delta-${deltaMatchCount.direction}`}>
                        {deltaMatchCount.text}
                      </span>
                    ) : null}
                  </strong>
                </div>
                <div className="meta-item">
                  <span>Tournaments</span>
                  <strong>
                    <span>{safeInt(tournamentCount).toLocaleString()}</span>
                    {deltaTournamentCount ? (
                      <span className={`meta-delta meta-delta-${deltaTournamentCount.direction}`}>
                        {deltaTournamentCount.text}
                      </span>
                    ) : null}
                  </strong>
                </div>
                <div className="meta-item">
                  <span>Distinct lineups</span>
                  <strong>
                    <span>{safeInt(distinctLineups).toLocaleString()}</span>
                    {deltaDistinctLineups ? (
                      <span className={`meta-delta meta-delta-${deltaDistinctLineups.direction}`}>
                        {deltaDistinctLineups.text}
                      </span>
                    ) : null}
                  </strong>
                </div>
                <div className="meta-item meta-item-full">
                  <span>Top lineup</span>
                  <strong>By match count</strong>
                  <span className="meta-subtext">Criterion: match count</span>
                  <div className="top-lineup-metrics">
                    <p>
                      <span>Usage (matches)</span>
                      <strong>{topLineupUsageLabel}</strong>
                    </p>
                    <p>
                      <span>Share (lineup variants)</span>
                      <strong>{topLineupShareLabel}</strong>
                    </p>
                  </div>
                  <span className="meta-subtext">
                      {`Top lineup includes ${topLineupPlayerCount} players.`}
                  </span>
                  {topLineupPlayers.length ? (
                    <ul className="top-lineup-list top-lineup-compact">
                      {topLineupPlayers.map((player) => (
                        <li key={`${teamIdText}-${player.id ?? player.name}`} className="top-lineup-chip">
                          {player.sendouUrl ? (
                            <a
                              href={player.sendouUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="top-lineup-link"
                              aria-label={`${player.name} profile on sendou.ink`}
                            >
                              {player.name}
                            </a>
                          ) : (
                            <span className="top-lineup-name">{player.name}</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <span className="meta-subtext">n/a</span>
                  )}
                </div>
              </div>

                <div className="result-meta-grid result-meta-subtle">
                  <div className="meta-item">
                    <span>Last match</span>
                    <strong title={fmtDate(row.event_time_ms)}>{relativeOrMissingDate(row.event_time_ms)}</strong>
                  </div>
                  <div className="meta-item">
                    <span>{playerMetricLabel}</span>
                    <strong>
                      <span>{safeInt(playerMetricCount).toLocaleString()}</span>
                      {playerMetricDelta ? (
                        <span className={`meta-delta meta-delta-${playerMetricDelta.direction}`}>
                          {playerMetricDelta.text}
                        </span>
                      ) : null}
                    </strong>
                  </div>
                  <div className="meta-item">
                    <span>Cluster rep</span>
                    <strong>{representativeTeamName}</strong>
                  </div>
                </div>

                <div className="confidence-panel">
                  <div className="confidence-panel-head">
                    <p className="meta-label">Match confidence</p>
                    <span className={`confidence-badge confidence-${confidenceTone.tone}`}>
                      {confidenceTone.label}
                    </span>
                  </div>
                  <p className="confidence-score" aria-live="polite">
                    {confidenceOverallLabel}
                  </p>
                  <div className="confidence-track" role="img" aria-label={`Confidence meter ${confidenceOverallLabel}`}>
                    <span
                      className={`confidence-fill confidence-${confidenceTone.tone}`}
                      style={{ width: `${confidenceOverall === null ? 0 : confidenceOverall * 100}%` }}
                    />
                  </div>
                  <details className="confidence-details">
                    <summary>Details</summary>
                    <div className="confidence-metric-row">
                      <span>Overall</span>
                      <span>{confidenceOverall === null ? 'n/a' : `${(confidenceOverall * 100).toFixed(1)}%`}</span>
                    </div>
                    <div className="confidence-metric-row">
                      <span>Semantic</span>
                      <span>{confidenceSemantic === null ? 'n/a' : `${(confidenceSemantic * 100).toFixed(1)}%`}</span>
                    </div>
                    <div className="confidence-metric-row">
                      <span>Identity</span>
                      <span>{confidenceIdentity === null ? 'n/a' : `${(confidenceIdentity * 100).toFixed(1)}%`}</span>
                    </div>
                  </details>
                </div>

              {normalizedConsolidatedCount > 1 ? (
                <section className="consolidated-section">
                  <div className="consolidated-header">
                    <div>
                      <p className="meta-label">Consolidated history</p>
                    <p className="consolidated-summary">
                        {pluralizeCount(consolidatedAliasCount, 'alias record')}, {safeInt(matchCount).toLocaleString()} matches across {consolidatedTournamentsCovered}
                        tournament{consolidatedTournamentsCovered === 1 ? '' : 's'}.
                        Match-count range {consolidatedMatchRangeMin}–{consolidatedMatchRangeMax}, avg
                        {` ${consolidatedMatchAverage.toFixed(1)}.`}
                      </p>
                    </div>
                    {canExpandAliases ? (
                      <button
                        type="button"
                        className="player-core-toggle"
                        onClick={() => toggleAliasExpansion(teamIdText)}
                        aria-expanded={aliasesExpanded}
                        aria-controls={aliasListId}
                      >
                        {aliasesExpanded ? 'View fewer aliases' : `View all ${normalizedConsolidatedCount} aliases`}
                      </button>
                    ) : null}
                  </div>
                  {consolidatedMatchBuckets.length ? (
                    <div className="consolidated-distribution">
                      {consolidatedMatchBuckets.map((bucket) => {
                        const width = maxBucketCount > 0
                          ? Math.round((bucket.count / maxBucketCount) * 100)
                          : 0;
                        return (
                          <div className="consolidated-bucket" key={`${teamIdText}-bucket-${bucket.label}`}>
                            <div className="consolidated-bucket-label">
                              <span>{bucket.label}</span>
                              <span>{pluralizeCount(bucket.count, 'alias')}</span>
                            </div>
                            <div className="consolidated-bucket-track">
                              <span
                                className="consolidated-bucket-fill"
                                style={{ width: `${width}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                  <div className="table-wrap">
                    <p className="meta-label">
                      {`Top ${Math.min(visibleAliasCount, consolidatedAliasCount)} aliases by matches`}
                    </p>
                    <table className="consolidated-table" id={aliasListId}>
                      <thead>
                        <tr>
                          <th>Team ID</th>
                          <th>Alias team</th>
                          <th>Matches</th>
                          <th>Tournaments</th>
                          <th>Last match</th>
                        </tr>
                      </thead>
                      <tbody>
                        {visibleAliases.map((team, index) => (
                          <tr
                            key={`${teamIdText}-${team.team_id}-${team.team_name || 'team'}-${index}`}
                          >
                            <td>{safeInt(team.team_id).toLocaleString()}</td>
                            <td>{team.team_name || `Team ${team.team_id}`}</td>
                            <td>{safeInt(team.match_count).toLocaleString()}</td>
                            <td>{safeInt(team.tournament_count).toLocaleString()}</td>
                            <td title={fmtDate(team.event_time_ms)}>{relativeOrMissingDate(team.event_time_ms)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              ) : null}
                </>
              ) : null}

            </article>
          );
        })}
      </div>
    </section>
  );
}
