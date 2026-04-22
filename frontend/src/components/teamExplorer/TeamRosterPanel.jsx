import React from 'react';
import {
  pct,
  playerUsageDescriptor,
  pluralize,
} from './helpers';

export function TeamRosterPanel({
  teamProfilePlayers,
  playerBaselineMatches,
  onOpenPlayerLookup,
  topLineupPlayers,
  topLineupMeta,
  lineupSupportMismatch,
  lineupSummaryFallback,
  rotationPlayers,
}) {
  return (
    <div className="analytics-grid team-player-grid">
      <article className="analytics-panel analytics-panel-wide">
        <div className="team-section-heading">
          <div>
            <h3>Core Roster</h3>
            <p className="meta">
              Player usage ordered by match presence across the current team scope.
            </p>
          </div>
        </div>
        {teamProfilePlayers.length ? (
          <ul className="team-roster-list">
            {teamProfilePlayers.map((player, index) => {
              const share = playerBaselineMatches > 0
                ? Math.round((player.matchesPlayed / playerBaselineMatches) * 100)
                : 0;
              const usageDescriptor = playerUsageDescriptor(player, playerBaselineMatches, index);
              const matchesLabel = player.matchesPlayed > 0
                ? pluralize(player.matchesPlayed, 'match')
                : 'top-lineup only';
              return (
                <li
                  key={`team-player-${player.id ?? player.name}`}
                  className={`team-roster-row ${player.id === null ? 'is-disabled' : ''}`}
                >
                  <button
                    type="button"
                    className="team-row-primary team-roster-primary"
                    disabled={player.id === null}
                    onClick={() => onOpenPlayerLookup(player.id, player.name)}
                    aria-label={`Open player history for ${player.name}`}
                  >
                    <span className="team-roster-main">
                      <span className="team-roster-name">{player.name}</span>
                      <span className="team-roster-note">
                        {matchesLabel} · {usageDescriptor}
                      </span>
                    </span>
                    <span className="team-roster-visual">
                      <span className="team-roster-bar" aria-hidden="true">
                        <span style={{ width: `${Math.max(0, Math.min(share, 100))}%` }} />
                      </span>
                      <span className="team-roster-share">
                        {player.matchesPlayed > 0 ? `${share}%` : 'n/a'}
                      </span>
                    </span>
                  </button>
                  <div className="team-hover-actions">
                    {player.sendouUrl ? (
                      <a
                        className="team-hover-action"
                        href={player.sendouUrl}
                        target="_blank"
                        rel="noreferrer"
                        aria-label={`Open sendou.ink profile for ${player.name}`}
                      >
                        sendou
                      </a>
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="team-missing-note">No player profile data available yet.</p>
        )}
      </article>

      <article className="analytics-panel">
        <div className="team-section-heading team-side-heading">
          <div>
            <h3>Most-Used Lineup</h3>
            <p className="meta">Sorted to match the roster usage order.</p>
          </div>
        </div>
        {topLineupPlayers.length ? (
          <div className="team-top-lineup-summary">
            <p className="team-top-lineup-names">{topLineupPlayers.map((player) => player.name).join(' · ')}</p>
            <p className={`meta ${lineupSupportMismatch ? 'team-lineup-meta-warning' : ''}`}>{topLineupMeta}</p>
          </div>
        ) : (
          <p className="meta">
            {lineupSummaryFallback || 'Current lineup summary unavailable.'}
          </p>
        )}
        <div className="team-lineup-strip" role="list" aria-label="Most-used lineup players">
          {topLineupPlayers.length ? topLineupPlayers.map((player) => (
            <button
              key={`top-lineup-${player.id ?? player.name}`}
              type="button"
              className="team-lineup-chip"
              disabled={player.id === null}
              role="listitem"
              onClick={() => {
                if (player.id !== null) onOpenPlayerLookup(player.id, player.name);
              }}
            >
              {player.name}
            </button>
          )) : (
            <p className="meta">No top-lineup player list available.</p>
          )}
        </div>

        <div className="team-rotation-block">
          <h4>Subs</h4>
          {rotationPlayers.length ? (
            <ul className="team-rotation-list">
              {rotationPlayers.map((player) => (
                <li key={`rotation-${player.id ?? player.name}`}>
                  <span>{player.name}</span>
                  <span className="meta">
                    {player.matchesPlayed > 0
                      ? `${pluralize(player.matchesPlayed, 'match')} · ${pct(playerBaselineMatches > 0 ? player.matchesPlayed / playerBaselineMatches : null)}`
                      : 'lineup only'}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="meta">No additional subs surfaced beyond the primary four.</p>
          )}
        </div>
      </article>
    </div>
  );
}
