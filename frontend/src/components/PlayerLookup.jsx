import React, { useEffect, useMemo, useRef, useState } from 'react';
import { fetchPlayerSuggestions, fetchPlayerTeams } from '../api';

export default function PlayerLookup({
  selectedPlayerId = '',
  selectedPlayerName = '',
  onOpenPlayerPage = () => {},
  onOpenTeamSearch,
}) {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sortBy, setSortBy] = useState('matches');
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);

  // Debounced player name autocomplete
  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetchPlayerSuggestions({ q: trimmed, limit: 10, signal: controller.signal })
        .then((data) => {
          setSuggestions(data.suggestions || []);
          setShowSuggestions(true);
        })
        .catch((err) => {
          if (err.name !== 'AbortError') setSuggestions([]);
        });
    }, 250);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target) &&
        inputRef.current &&
        !inputRef.current.contains(e.target)
      ) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  function selectPlayer(player, { syncRoute = true } = {}) {
    setSelectedPlayer(player);
    setQuery(player.display_name);
    setShowSuggestions(false);
    setSuggestions([]);
    setLoading(true);
    setError('');
    setTeams([]);
    if (syncRoute) {
      onOpenPlayerPage(player.player_id, player.display_name);
    }

    fetchPlayerTeams({ playerId: player.player_id })
      .then((data) => {
        setTeams(data.teams || []);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load player teams');
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    const playerId = Number(selectedPlayerId);
    if (!Number.isFinite(playerId) || playerId <= 0) return;

    const nextName = String(selectedPlayerName || '').trim() || `Player ${Math.trunc(playerId)}`;
    if (
      selectedPlayer
      && Number(selectedPlayer.player_id) === Math.trunc(playerId)
      && String(selectedPlayer.display_name || '').trim() === nextName
    ) {
      return;
    }

    selectPlayer({
      player_id: Math.trunc(playerId),
      display_name: nextName,
    }, { syncRoute: false });
  }, [onOpenPlayerPage, selectedPlayer, selectedPlayerId, selectedPlayerName]);

  const sortedTeams = useMemo(() => {
    const list = [...teams];
    if (sortBy === 'matches') {
      list.sort((a, b) => b.player_match_count - a.player_match_count || b.lineup_count - a.lineup_count);
    } else if (sortBy === 'lineups') {
      list.sort((a, b) => b.lineup_count - a.lineup_count || b.player_match_count - a.player_match_count);
    } else if (sortBy === 'recency') {
      list.sort((a, b) => (b.event_time_ms || 0) - (a.event_time_ms || 0));
    }
    return list;
  }, [teams, sortBy]);

  function formatDate(ms) {
    if (!ms) return '--';
    return new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  }

  return (
    <section className="panel player-lookup" aria-labelledby="player-lookup-title">
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Player profile</p>
          <h2 id="player-lookup-title" className="panel-title">Player Lookup</h2>
          <p className="panel-summary">Search for a player to see which teams they are most associated with.</p>
        </div>
      </div>

      <div className="form-grid">
        <div className="search-input-wrap player-search-wrap">
          <input
            ref={inputRef}
            type="text"
            className="input"
            placeholder="Type a player name..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              if (selectedPlayer) setSelectedPlayer(null);
            }}
            onFocus={() => {
              if (suggestions.length > 0) setShowSuggestions(true);
            }}
            role="combobox"
            aria-expanded={showSuggestions}
            aria-autocomplete="list"
            aria-controls="player-suggestions"
          />
          {showSuggestions && suggestions.length > 0 && (
            <ul
              ref={dropdownRef}
              id="player-suggestions"
              className="suggestion-dropdown"
              role="listbox"
            >
              {suggestions.map((s) => (
                <li
                  key={s.player_id}
                  className="suggestion-option"
                  role="option"
                  aria-selected={Number(selectedPlayer?.player_id) === Number(s.player_id)}
                  tabIndex={0}
                  onClick={() => selectPlayer(s)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') selectPlayer(s);
                  }}
                >
                  <span className="suggestion-name">{s.display_name}</span>
                  <span className="suggestion-meta">
                    {s.team_count} team{s.team_count !== 1 ? 's' : ''}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {error ? <p className="error">{error}</p> : null}
      {loading ? <p className="status">Loading teams…</p> : null}

      {selectedPlayer && !loading && teams.length === 0 && !error ? (
        <div className="empty-state">
          <p className="empty-state-title">No teams found for {selectedPlayer.display_name}</p>
          <p className="empty-state-hint">This player may not appear in any indexed team rosters.</p>
        </div>
      ) : null}

      {selectedPlayer && teams.length > 0 ? (
        <>
          <div className="results-head">
            <div className="player-info-card">
              <div>
                <p className="panel-kicker">Player record</p>
                <h3>{selectedPlayer.display_name}</h3>
              </div>
              <span className="player-info-meta">
                {teams.length} team{teams.length !== 1 ? 's' : ''}
              </span>
            </div>

            <div className="results-toolbar">
              <span className="results-count">{teams.length} recent entries</span>
              <label className="toolbar-label">
                Sort by{' '}
                <select className="input-compact" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                  <option value="matches">Match count</option>
                  <option value="lineups">Lineup count</option>
                  <option value="recency">Most recent</option>
                </select>
              </label>
            </div>
          </div>

          <div className="table-wrap">
            <table className="player-teams-table data-table">
              <thead>
                <tr>
                  <th>Team Name</th>
                  <th>Player Matches</th>
                  <th>Lineup Count</th>
                  <th>Last Active</th>
                  <th>Roster</th>
                  {onOpenTeamSearch && <th></th>}
                </tr>
              </thead>
              <tbody>
                {sortedTeams.map((t) => (
                  <tr key={t.team_id}>
                    <td className="cell-team-name">
                      {onOpenTeamSearch ? (
                        <button
                          type="button"
                          className="link link-button"
                          onClick={() => onOpenTeamSearch(t.team_name || `Team ${t.team_id}`)}
                        >
                          {t.team_name || `Team ${t.team_id}`}
                        </button>
                      ) : (
                        <span>{t.team_name || `Team ${t.team_id}`}</span>
                      )}
                    </td>
                    <td className="cell-numeric cell-num">{t.player_match_count}</td>
                    <td className="cell-numeric cell-num">{t.lineup_count}</td>
                    <td className="cell-date cell-mute">{formatDate(t.event_time_ms)}</td>
                    <td className="cell-roster cell-mute">{(t.roster_player_names || []).join(', ')}</td>
                    {onOpenTeamSearch && (
                      <td>
                        <button
                          className="button-secondary button-sm btn-pill btn-fuchsia-outline"
                          onClick={() => onOpenTeamSearch(t.team_name || `Team ${t.team_id}`)}
                        >
                          Search similar
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </section>
  );
}
