import React, { useEffect, useMemo, useRef, useState } from 'react';
import { fetchTeamSearch, fetchTeamSuggestions, fetchTournamentTeams } from '../api';

const EMPTY_TEAM_IDS = [];

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

function tournamentTeamDisplayName(team) {
  const display = String(team?.display_name || '').trim();
  if (display) return display;

  const name = String(team?.team_name || '').trim();
  if (name) return name;

  const parsedId = safeIntOrNull(team?.team_id);
  if (parsedId && parsedId > 0) return `Team ${parsedId}`;
  return 'Untitled team';
}

function tournamentTeamMeta(team) {
  const members = Array.isArray(team?.member_names)
    ? team.member_names.flatMap((value) => {
        const normalized = String(value || '').trim();
        return normalized ? [normalized] : [];
      })
    : [];
  const id = safeIntOrNull(team?.team_id);
  const pieces = [];
  if (id && id > 0) {
    pieces.push(`ID ${id}`);
  }
  if (members.length > 0) {
    const preview = members.slice(0, 3).join(', ');
    pieces.push(members.length > 3 ? `${preview} +${members.length - 3}` : preview);
  }
  return pieces.join(' · ');
}

export default function TeamSearch({
  selectedTeamAId = '',
  selectedTeamBId = '',
  selectedTeamAIds = EMPTY_TEAM_IDS,
  selectedTeamBIds = EMPTY_TEAM_IDS,
  onOpenTeamPage = () => {},
  onOpenHeadToHead = () => {},
}) {
  const [query, setQuery] = useState('');
  const [clusterMode, setClusterMode] = useState('family');
  const [topN, setTopN] = useState(20);
  const [minRelevance, setMinRelevance] = useState(0.8);
  const [consolidate, setConsolidate] = useState(true);
  const [consolidateMinOverlap, setConsolidateMinOverlap] = useState(0.8);
  const [recencyWeight, setRecencyWeight] = useState(0);
  const [tournamentId, setTournamentId] = useState('');
  const [tournamentTeamLookup, setTournamentTeamLookup] = useState('');
  const [selectedTournamentSeedMeta, setSelectedTournamentSeedMeta] = useState(null);
  const [selectedTournamentSeedPlayerIds, setSelectedTournamentSeedPlayerIds] = useState([]);
  const [tournamentTeams, setTournamentTeams] = useState([]);
  const [tournamentTeamsSource, setTournamentTeamsSource] = useState('none');
  const [tournamentTeamsLoading, setTournamentTeamsLoading] = useState(false);
  const [tournamentTeamsError, setTournamentTeamsError] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [payload, setPayload] = useState(null);
  const [expandedCards, setExpandedCards] = useState(() => new Set());
  const [expandedPlayers, setExpandedPlayers] = useState(() => new Set());
  const [expandedAliases, setExpandedAliases] = useState(() => new Set());
  const [sortBy, setSortBy] = useState('relevance');
  const [visibleCount, setVisibleCount] = useState(20);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const queryInputRef = useRef(null);
  const suppressSuggestionsRef = useRef(false);

  const resultLabel = useMemo(() => {
    if (!payload) return 'No query yet.';
    const clustered = (payload.results || []).filter((row) => row.is_clustered).length;
    const consolidatedGroups = (payload.results || []).filter((row) => row.is_consolidated).length;
    const deduped = (payload.results || []).reduce(
      (acc, row) => acc + (Number(row.consolidated_team_count) || 1),
      0,
    );
    const tournamentScope = payload?.query_context?.tournament_id
      ? ` · tournament ${payload.query_context.tournament_id} (${payload?.query_context?.tournament_source || 'dataset'})`
      : '';
    const seededPlayers = Number(payload?.query_context?.seed_player_id_count || 0);
    const seedScope = seededPlayers > 0 ? ` · seeded from ${seededPlayers} player ID${seededPlayers === 1 ? '' : 's'}` : '';
    return `${payload.result_count} result groups (${deduped} teams) from snapshot ${payload.snapshot_id}${tournamentScope}${seedScope} · ${clustered} clustered · ${consolidatedGroups} consolidated (${payload.cluster_mode})`;
  }, [payload]);

  const results = useMemo(() => {
    const raw = payload?.results || [];
    if (sortBy === 'relevance' || !raw.length) return raw;
    const sorted = [...raw];
    if (sortBy === 'matches') {
      sorted.sort(
        (a, b) =>
          safeInt(b.match_count ?? b.lineup_count) -
          safeInt(a.match_count ?? a.lineup_count),
      );
    } else if (sortBy === 'recency') {
      sorted.sort(
        (a, b) =>
          (toEpochMs(b.event_time_ms) ?? 0) -
          (toEpochMs(a.event_time_ms) ?? 0),
      );
    }
    return sorted;
  }, [payload, sortBy]);
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
  const parsedTournamentId = useMemo(() => {
    const parsed = Number(tournamentId);
    if (!Number.isFinite(parsed) || parsed <= 0) return null;
    return Math.trunc(parsed);
  }, [tournamentId]);

  useEffect(() => {
    setSelectedTournamentSeedMeta(null);
    setSelectedTournamentSeedPlayerIds([]);
    if (parsedTournamentId !== null) return;
    setTournamentTeamLookup('');
    setTournamentTeams([]);
    setTournamentTeamsSource('none');
    setTournamentTeamsError('');
    setTournamentTeamsLoading(false);
  }, [parsedTournamentId]);

  useEffect(() => {
    if (parsedTournamentId === null) return;

    const requestQuery = tournamentTeamLookup.trim();
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setTournamentTeamsLoading(true);
      setTournamentTeamsError('');
      try {
        const response = await fetchTournamentTeams({
          tournamentId: parsedTournamentId,
          q: requestQuery,
          limit: 700,
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        setTournamentTeams(Array.isArray(response?.teams) ? response.teams : []);
        setTournamentTeamsSource(response?.source || 'none');
      } catch (err) {
        if (controller.signal.aborted) return;
        setTournamentTeams([]);
        setTournamentTeamsSource('none');
        setTournamentTeamsError(err.message || 'Failed to load tournament teams');
      } finally {
        if (!controller.signal.aborted) {
          setTournamentTeamsLoading(false);
        }
      }
    }, 220);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [parsedTournamentId, tournamentTeamLookup]);

  useEffect(() => {
    setExpandedCards(new Set());
    setExpandedPlayers(new Set());
    setExpandedAliases(new Set());
    setVisibleCount(20);
  }, [payload, results]);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setSuggestions([]);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const data = await fetchTeamSuggestions({
          q: trimmed,
          limit: 8,
          signal: controller.signal,
        });
        if (!controller.signal.aborted) {
          const nextSuggestions = data.suggestions || [];
          setSuggestions(nextSuggestions);
          setShowSuggestions(
            !suppressSuggestionsRef.current
              && document.activeElement === queryInputRef.current
              && nextSuggestions.length > 0,
          );
        }
      } catch {
        if (!controller.signal.aborted) {
          setSuggestions([]);
          setShowSuggestions(false);
        }
      }
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

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

  function onSelectTournamentTeam(team) {
    const normalized = tournamentTeamDisplayName(team);
    if (!normalized) return;
    const normalizedTeamId = safeIntOrNull(team?.team_id);
    const seedIds = Array.isArray(team?.member_user_ids)
      ? team.member_user_ids
          .map((value) => Number(value))
          .filter((value) => Number.isFinite(value) && value > 0)
          .map((value) => Math.trunc(value))
      : [];
    setQuery(normalized);
    setTournamentTeamLookup(normalized);
    setSelectedTournamentSeedMeta({
      teamName: normalized,
      teamId: Number.isFinite(normalizedTeamId) && normalizedTeamId > 0 ? Math.trunc(normalizedTeamId) : null,
    });
    setSelectedTournamentSeedPlayerIds(seedIds);
  }

  function clearSeedSelection() {
    setSelectedTournamentSeedMeta(null);
    setSelectedTournamentSeedPlayerIds([]);
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!query.trim()) return;
    suppressSuggestionsRef.current = true;
    setShowSuggestions(false);
    setLoading(true);
    setError('');
    try {
      const data = await fetchTeamSearch({
        q: query.trim(),
        topN,
        clusterMode,
        minRelevance,
        tournamentId: parsedTournamentId ?? undefined,
        seedPlayerIds: selectedTournamentSeedPlayerIds,
        consolidate,
        consolidateMinOverlap,
        recencyWeight,
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
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Similarity search</p>
          <h2 id="team-search-title" className="panel-title">Team Search</h2>
          <p className="panel-summary">
            Search by team name, optional tournament scope, and snapshot-aware identity signals.
          </p>
        </div>
      </div>

      <form className="form-grid search-form" onSubmit={onSubmit}>
        <label htmlFor="team-query" className="field-label">Team query</label>
        <div
          className="search-input-wrap"
          onBlur={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget)) {
              suppressSuggestionsRef.current = false;
              setTimeout(() => setShowSuggestions(false), 150);
            }
          }}
        >
          <input
            id="team-query"
            ref={queryInputRef}
            className="input"
            type="search"
            placeholder="e.g. FTW, Moonlight, Hypernova"
            value={query}
            onChange={(e) => {
              suppressSuggestionsRef.current = false;
              setQuery(e.target.value);
              clearSeedSelection();
            }}
            onFocus={() => {
              if (suggestions.length && !suppressSuggestionsRef.current) setShowSuggestions(true);
            }}
            autoComplete="off"
            role="combobox"
            aria-expanded={showSuggestions && suggestions.length > 0}
            aria-controls="team-suggestions"
            aria-autocomplete="list"
            required
          />
          {showSuggestions && suggestions.length > 0 ? (
            <ul id="team-suggestions" className="suggestion-dropdown" role="listbox">
              {suggestions.map((s) => (
                <li key={s.team_id} role="option" aria-selected={false}>
                  <button
                    type="button"
                    className="suggestion-option"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => {
                      suppressSuggestionsRef.current = true;
                      setQuery(s.team_name);
                      clearSeedSelection();
                      setShowSuggestions(false);
                    }}
                  >
                    <span className="suggestion-name">{s.team_name}</span>
                    <span className="suggestion-meta">{s.lineup_count} matches</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        {selectedTournamentSeedPlayerIds.length > 0 ? (
          <div className="seed-affordance" role="status" aria-live="polite">
            <span className="seed-affordance-label">Embedding seed active</span>
            <span className="seed-affordance-name">
              {selectedTournamentSeedMeta?.teamName || 'Tournament team'}
            </span>
            <span className="seed-affordance-meta">
              {selectedTournamentSeedPlayerIds.length} player ID{selectedTournamentSeedPlayerIds.length === 1 ? '' : 's'}
            </span>
            <button
              type="button"
              className="seed-affordance-clear"
              onClick={clearSeedSelection}
            >
              Clear seed
            </button>
          </div>
        ) : null}

        <div className="form-row row fields-row tournament-row">
          <div className="field">
            <label htmlFor="tournament-id" className="field-label">Tournament ID (optional)</label>
            <input
              id="tournament-id"
              className="input"
              type="number"
              min={1}
              step={1}
              placeholder="e.g. 3192"
              value={tournamentId}
              onChange={(e) => setTournamentId(e.target.value)}
            />
            <span className="field-label-subtitle">
              Scope query rows to this tournament. If missing in snapshot, teams are pulled from sendou.ink.
            </span>
          </div>
          <div className="field tournament-picker-field">
            <label htmlFor="tournament-team-search" className="field-label">Tournament team pull-down</label>
            <input
              id="tournament-team-search"
              className="input"
              type="search"
              placeholder={parsedTournamentId === null ? 'Set tournament ID first' : 'Type team or player name'}
              value={tournamentTeamLookup}
              onChange={(event) => setTournamentTeamLookup(event.target.value)}
              disabled={parsedTournamentId === null}
            />
            <span className="field-label-subtitle">
              Search teams inside the tournament and pick one to fill Team query. Supports unnamed teams via player names.
            </span>
            {parsedTournamentId !== null ? (
              <div className="tournament-team-picker" aria-live="polite">
                <p className="meta">
                  {tournamentTeamsLoading
                    ? 'Loading teams…'
                    : `${tournamentTeams.length} team${tournamentTeams.length === 1 ? '' : 's'} loaded from ${tournamentTeamsSource}.`}
                </p>
                {!tournamentTeamsLoading && tournamentTeams.length ? (
                  <ul className="tournament-team-dropdown" role="listbox" aria-label="Tournament teams">
                    {tournamentTeams.slice(0, 80).map((team, index) => {
                      const label = tournamentTeamDisplayName(team);
                      const optionTeamId = safeIntOrNull(team?.team_id);
                      const isSelectedTeam = selectedTournamentSeedPlayerIds.length > 0
                        && (
                          (
                            Number.isFinite(optionTeamId)
                            && optionTeamId > 0
                            && Number.isFinite(selectedTournamentSeedMeta?.teamId)
                            && Math.trunc(optionTeamId) === Math.trunc(selectedTournamentSeedMeta.teamId)
                          )
                          || (
                            !Number.isFinite(selectedTournamentSeedMeta?.teamId)
                            && selectedTournamentSeedMeta?.teamName === label
                          )
                        );
                      const baseMeta = tournamentTeamMeta(team);
                      const meta = isSelectedTeam
                        ? `${baseMeta ? `${baseMeta} · ` : ''}seeding active`
                        : baseMeta;
                      return (
                        <li
                          key={`${team.team_id ?? 'ext'}-${label}-${index}`}
                          role="option"
                          aria-selected={isSelectedTeam}
                        >
                          <button
                            type="button"
                            className={`tournament-team-option ${isSelectedTeam ? 'is-selected' : ''}`}
                            onClick={() => onSelectTournamentTeam(team)}
                            title={meta || label}
                          >
                            <span className="tournament-team-option-name">{label}</span>
                            {meta ? <span className="tournament-team-option-meta">{meta}</span> : null}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                ) : null}
                {!tournamentTeamsLoading && !tournamentTeams.length && tournamentTeamLookup.trim() ? (
                  <p className="meta">No teams matched this search.</p>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>

        {tournamentTeamsError ? <p className="error">{tournamentTeamsError}</p> : null}

        <div className="form-row row fields-row">
          <div className="field">
            <label htmlFor="cluster-mode" className="field-label">Cluster profile</label>
            <select
              id="cluster-mode"
              className="input"
              value={clusterMode}
              onChange={(e) => setClusterMode(e.target.value)}
            >
              <option value="family">family</option>
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

          <div className="field">
            <label htmlFor="recency-weight" className="field-label slider-label">
              Recency bias
            </label>
            <input
              id="recency-weight"
              className="input slider-input"
              type="range"
              min={0}
              max={100}
              step={10}
              value={Math.round(recencyWeight * 100)}
              onChange={(e) => setRecencyWeight(Number(e.target.value) / 100)}
              aria-describedby="recency-weight-value"
            />
            <output id="recency-weight-value" className="slider-output">
              {Math.round(recencyWeight * 100)}%
            </output>
          </div>
        </div>

        <button className="button btn-pill btn-fuchsia" type="submit" disabled={loading}>
          {loading ? 'Searching…' : 'Search teams'}
        </button>
      </form>

      <p className="status" role="status" aria-live="polite">{resultLabel}</p>
      {error ? <p className="error">{error}</p> : null}

      {payload && results.length === 0 && !loading && !error ? (
        <div className="empty-state" role="status">
          <p className="empty-state-title">No teams found for &ldquo;{query}&rdquo;</p>
          <p className="empty-state-hint">
            Try a different spelling or a shorter query. You can also search by
            team ID or player name.
            {Number(minRelevance) > 0.5
              ? ' Lowering the minimum relevance slider may reveal more results.'
              : null}
          </p>
        </div>
      ) : null}

      {results.length > 0 ? (
        <div className="results-head">
          <div>
            <h3 className="results-title">Results</h3>
            <p className="results-subtitle">Bento profile view with roster, lineup, and confidence evidence.</p>
          </div>
          <div className="results-toolbar">
            <span className="results-count">
              {results.length} groups · snapshot {payload?.snapshot_id ?? 'n/a'} · profile {clusterMode}
            </span>
            <label htmlFor="sort-by" className="field-label">Sort by</label>
            <select
              id="sort-by"
              className="input input-compact"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="relevance">Relevance</option>
              <option value="matches">Match count</option>
              <option value="recency">Most recent</option>
            </select>
          </div>
        </div>
      ) : null}

      <div className="card-grid">
        {results.slice(0, visibleCount).map((row, index) => {
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
          const isConsolidatedProfile = normalizedConsolidatedCount > 1;
          const cardSubtitle = isConsolidatedProfile
            ? `Consolidated profile (${normalizedConsolidatedCount} teams)`
            : 'Single team profile';
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
          const isResultExpanded = expandedCards.has(teamIdText);
          const bentoRosterPlayers = displayPlayers.slice(0, 5);
          const exactLastMatch = fmtDate(row.event_time_ms);
          const freshnessValue = matchCount > 0
            ? Math.max(0, Math.min(100, 100 - Math.min(100, Math.round((Date.now() - (toEpochMs(row.event_time_ms) || Date.now())) / (1000 * 60 * 60 * 24 * 3)))))
            : 0;
          const similarityRows = [
            { label: 'overall', value: confidenceOverall },
            { label: 'semantic', value: confidenceSemantic },
            { label: 'identity', value: confidenceIdentity },
          ];

          return (
            <article className={`result-card bento-card-shell ${isResultExpanded ? 'is-expanded' : ''}`} key={teamIdText}>
              <div className="bento-card">
                <header className="bento-tile bento-hero">
                  <div className="bento-hero-rank">#{String(index + 1).padStart(2, '0')}</div>
                  <div className="bento-hero-main">
                    <p className="bento-k">team profile</p>
                    <h3 className="bento-name">{teamDisplayName}</h3>
                    <p className="bento-meta">id {teamIdText} · {cardSubtitle.toLowerCase()}</p>
                    <p className="result-purpose">
                      {isConsolidatedProfile
                        ? 'Merged roster history across related registrations.'
                        : 'Roster and lineup evidence for the closest current identity.'}
                    </p>
                    <div className="result-quick-metrics">
                      <span className="chip">{safeInt(matchCount).toLocaleString()} matches</span>
                      <span className="chip">{safeInt(tournamentCount).toLocaleString()} tournaments</span>
                      <span className="chip">lineups {safeInt(distinctLineups)}</span>
                      <span className="chip chip-accent">players {safeInt(playerMetricCount)}</span>
                    </div>
                  </div>
                  <div className="bento-hero-side">
                    <div className="sim-score sim-score-strong">
                      <span className="sim-score-num">{confidenceOverall === null ? 'n/a' : (confidenceOverall * 100).toFixed(1)}</span>
                      {confidenceOverall === null ? null : <span className="sim-score-unit">%</span>}
                    </div>
                    <span className={`confidence-badge confidence-${confidenceTone.tone}`}>
                      {confidenceTone.label}
                    </span>
                  </div>
                  <div className="bento-actions">
                    <button
                      type="button"
                      className="result-select-btn"
                      onClick={() => onOpenTeamPage(row.consolidated_team_ids || [row.team_id], teamDisplayName, payload?.snapshot_id)}
                      title="Open this result on the Teams page"
                    >
                      Open team page
                    </button>
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
                </header>

                <section className="bento-tile bento-sim" aria-label="Search signal breakdown">
                  <p className="bento-k">search breakdown</p>
                  {similarityRows.map((item) => {
                    const pctValue = item.value === null ? 0 : Math.max(0, Math.min(100, item.value * 100));
                    return (
                      <div key={`${teamIdText}-${item.label}`} className="bento-sim-row">
                        <span className="bento-sim-label">{item.label}</span>
                        <span className="vm-bar">
                          <span className="vm-bar-fill" style={{ width: `${pctValue}%` }} />
                        </span>
                        <span className="bento-sim-v">
                          {item.value === null ? 'n/a' : `${pctValue.toFixed(0)}%`}
                        </span>
                      </div>
                    );
                  })}
                </section>

                <section className="bento-tile bento-matches">
                  <p className="bento-k">matches</p>
                  <p className="bento-big">{safeInt(matchCount).toLocaleString()}</p>
                  <p className="bento-sub">
                    {deltaMatchCount ? `${deltaMatchCount.text} vs top result` : 'Top result baseline'}
                  </p>
                </section>

                <section className="bento-tile bento-tourneys">
                  <p className="bento-k">tournaments</p>
                  <p className="bento-big">{safeInt(tournamentCount).toLocaleString()}</p>
                  <p className="bento-sub">
                    {safeInt(distinctLineups).toLocaleString()} lineups · {safeInt(playerMetricCount).toLocaleString()} players
                  </p>
                </section>

                <section className="bento-tile bento-fresh">
                  <p className="bento-k">last match</p>
                  <p className="bento-big bento-big-date">{relativeOrMissingDate(row.event_time_ms)}</p>
                  <p className="bento-sub">{exactLastMatch}</p>
                  <div className="bento-fresh-bar">
                    <span style={{ width: `${freshnessValue}%` }} />
                  </div>
                </section>

                <section className="bento-tile bento-lineup">
                  <p className="bento-k">top lineup</p>
                  <p className="bento-big">{safeInt(topLineupMatches).toLocaleString()}</p>
                  <p className="bento-sub">{topLineupUsageLabel}</p>
                  {topLineupPlayers.length ? (
                    <div className="bento-chip-row">
                      {topLineupPlayers.map((player) => (
                        <span key={`${teamIdText}-lineup-${player.id ?? player.name}`} className="top-lineup-chip">
                          {player.name}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </section>

                <section className="bento-tile bento-roster" aria-label="Core roster">
                  <div className="bento-roster-head">
                    <div>
                      <p className="bento-k">core roster</p>
                      <p className="player-core-summary">{visiblePlayerSummary}</p>
                    </div>
                    {normalizedConsolidatedCount > 1 ? (
                      <span className="chip chip-accent">
                        consolidated · {normalizedConsolidatedCount}
                      </span>
                    ) : null}
                  </div>
                  <ul className="bento-roster-list">
                    {bentoRosterPlayers.map((player) => {
                      const playerShare = matchCount > 0
                        ? Math.round((player.matchesPlayed / matchCount) * 100)
                        : 0;
                      return (
                        <li key={`${teamIdText}-bento-${player.id ?? player.name}`}>
                          <span className="bento-r-role">{player.id ?? '—'}</span>
                          <span className="bento-r-name">{player.name}</span>
                          <span className="bento-r-sr">{player.matchesPlayed > 0 ? `${playerShare}%` : 'n/a'}</span>
                          <span className="bento-r-bar">
                            <span style={{ width: `${Math.max(0, Math.min(100, playerShare))}%` }} />
                          </span>
                          <span className="bento-r-mp">{player.matchesPlayed > 0 ? `${player.matchesPlayed}m` : '—'}</span>
                        </li>
                      );
                    })}
                  </ul>
                  {displayPlayers.length > bentoRosterPlayers.length ? (
                    <p className="bento-merged">
                      {displayPlayers.length - bentoRosterPlayers.length} more players in the expanded view
                    </p>
                  ) : null}
                </section>
              </div>

              {isResultExpanded ? (
                <div className="bento-details">
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
                              style={{ width: `${Math.max(0, Math.min(100, playerShare))}%` }}
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
                            {pluralizeCount(consolidatedAliasCount, 'alias')} merged into this profile.
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
                      <div className="table-wrap">
                        <p className="meta-label">
                          {`Team aliases (${Math.min(visibleAliasCount, consolidatedAliasCount)} of ${consolidatedAliasCount})`}
                        </p>
                        <table className="consolidated-table" id={aliasListId}>
                          <thead>
                            <tr>
                              <th>Team ID</th>
                              <th>Team alias</th>
                              <th>Matches</th>
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
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </section>
                  ) : null}
                </div>
              ) : null}

            </article>
          );
        })}
      </div>

      {results.length > visibleCount ? (
        <div className="load-more-wrap">
          <button
            type="button"
            className="button button-secondary"
            onClick={() =>
              setVisibleCount((prev) => Math.min(prev + 20, results.length))
            }
          >
            Show more ({results.length - visibleCount} remaining)
          </button>
        </div>
      ) : null}
    </section>
  );
}
