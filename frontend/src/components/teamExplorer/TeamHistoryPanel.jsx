import React from 'react';
import { pluralize } from './helpers';

export function TeamHistoryPanel({
  historyPageCount,
  historyPage,
  changeHistoryPage,
  matchEventsCount,
  recentForm,
  matches,
  visibleMatchEvents,
  expandedEventKeys,
  toggleEventKey,
  onOpenTeamPage,
  onOpenHeadToHead,
  selectedFamilyIds,
  actionSnapshotId,
  matchHistoryError,
}) {
  return (
    <article className="analytics-panel analytics-panel-wide">
      <div className="team-section-heading">
        <div>
          <h3>History</h3>
          <p className="meta">
            Tournament-by-tournament match history for the currently loaded team scope.
          </p>
        </div>
      </div>
      <div className="team-history-toolbar">
        <span className="team-history-legend">Ribbons run earliest on the left and latest on the right.</span>
        {historyPageCount > 1 ? (
          <div className="team-history-pagination">
            <button
              type="button"
              className="button button-secondary"
              disabled={historyPage <= 1}
              onClick={() => changeHistoryPage(historyPage - 1)}
            >
              Previous
            </button>
            <span className="team-history-pagination-meta">
              Page {historyPage} of {historyPageCount} · {pluralize(matchEventsCount, 'tournament')}
            </span>
            <button
              type="button"
              className="button button-secondary"
              disabled={historyPage >= historyPageCount}
              onClick={() => changeHistoryPage(historyPage + 1)}
            >
              Next
            </button>
          </div>
        ) : null}
      </div>
      {recentForm.length ? (
        <div className="team-form-strip" aria-label="Recent form">
          <span className="team-form-label">Recent form, newest first</span>
          <div className="team-form-pills">
            {recentForm.map((row) => (
              <span
                key={row.key}
                className={`team-form-pill ${row.tone}`}
                title={row.title}
              >
                {row.label}
              </span>
            ))}
          </div>
        </div>
      ) : null}
      {matches.length ? (
        <div className="team-match-events">
          {visibleMatchEvents.map((event) => {
            const isExpanded = expandedEventKeys.includes(event.key);
            return (
              <section key={event.key} className={`team-match-event ${isExpanded ? 'is-expanded' : ''}`}>
                <button
                  type="button"
                  className="team-match-event-head team-match-event-toggle"
                  aria-expanded={isExpanded}
                  onClick={() => toggleEventKey(event.key)}
                >
                  <div className="team-match-event-title">
                    <span className="team-match-event-name">{event.tournamentName}</span>
                    <span className="team-match-event-meta">
                      {event.whenLabel} · {pluralize(event.rows.length, 'match')} · {event.wins}-{event.losses}
                      {event.opponentSummary ? ` · vs ${event.opponentSummary}` : ''}
                    </span>
                  </div>
                  <span className="team-match-event-tier">{event.tournamentTier || 'Unscored'}</span>
                  <div className="team-match-event-progress">
                    <div
                      className="team-match-event-ribbon"
                      aria-label="Tournament progression ribbon, earliest match on the left and latest on the right"
                      title="Tournament progression, earliest match on the left and latest on the right."
                      style={{ gridTemplateColumns: `repeat(${Math.max(event.ribbonRows?.length || 0, 1)}, minmax(0, 1fr))` }}
                    >
                      {(event.ribbonRows || []).map((row) => (
                        <span
                          key={`ribbon-${row.key}`}
                          className={`team-match-ribbon-cell ${row.tone}`}
                          title={`${row.opponentTeamName}: ${row.scoreLabel}`}
                        />
                      ))}
                    </div>
                  </div>
                  <span className="team-match-event-chevron" aria-hidden="true">{isExpanded ? '▾' : '▸'}</span>
                </button>
                {isExpanded ? (
                  <ul className="team-match-rows">
                    {event.rows.map((row) => (
                      <li
                        key={row.key}
                        className={`team-match-row ${row.opponentTeamId ? '' : 'is-disabled'}`}
                      >
                        <button
                          type="button"
                          className="team-row-primary team-match-primary"
                          disabled={!row.opponentTeamId}
                          onClick={() => onOpenTeamPage(
                            row.opponentTeamId ? [row.opponentTeamId] : [],
                            row.opponentTeamName,
                            actionSnapshotId,
                          )}
                          aria-label={`Open team page for ${row.opponentTeamName}`}
                        >
                          <span className="team-match-row-top">
                            <span className="team-match-date">{row.relativeDate}</span>
                            <span className="team-match-opponent-block">
                              <span className="team-match-opp">{row.opponentTeamName}</span>
                              {row.opponentEventCount > 1 ? (
                                <span className="team-match-opponent-note">
                                  {pluralize(row.opponentEventCount, 'meeting')} in this event
                                </span>
                              ) : null}
                            </span>
                            <span className={`team-result-pill ${row.tone}`}>{row.resultLabel}</span>
                            <span className="team-match-score">{row.scoreLabel}</span>
                          </span>
                          <span className="team-match-row-bottom">
                            <span className="team-match-roster-group">
                              <span className="team-match-roster-label">vs</span>
                              <span className="team-match-roster-chips">
                                {row.opponentRosterPlayers.length ? row.opponentRosterPlayers.map((name) => (
                                  <span key={`${row.key}-opp-${name}`} className="team-match-roster-chip">{name}</span>
                                )) : (
                                  <span className="team-match-roster-chip is-muted">Roster unavailable</span>
                                )}
                              </span>
                            </span>
                            {!row.usesDefaultTeamRoster ? (
                              <span className="team-match-roster-group is-owned">
                                <span className="team-match-roster-label">Tracked</span>
                                <span className="team-match-roster-chips">
                                  {row.teamRosterPlayers.length ? row.teamRosterPlayers.map((name) => (
                                    <span key={`${row.key}-team-${name}`} className="team-match-roster-chip is-owned">{name}</span>
                                  )) : (
                                    <span className="team-match-roster-chip is-muted">Roster unavailable</span>
                                  )}
                                </span>
                              </span>
                            ) : null}
                          </span>
                        </button>
                        <div className="team-hover-actions">
                          <button
                            type="button"
                            className="team-hover-action"
                            disabled={!row.opponentTeamId}
                            onClick={() => onOpenHeadToHead(
                              selectedFamilyIds,
                              row.opponentTeamId ? [row.opponentTeamId] : [],
                              actionSnapshotId,
                            )}
                          >
                            Compare
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </section>
            );
          })}
        </div>
      ) : (
        <p className="team-missing-note">
          {matchHistoryError || 'No match log is available for this team yet.'}
        </p>
      )}
    </article>
  );
}
