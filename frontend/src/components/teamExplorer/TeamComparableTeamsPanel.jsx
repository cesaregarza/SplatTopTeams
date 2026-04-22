import React from 'react';
import {
  normalizeTimelineName,
  pct,
  pluralize,
} from './helpers';

export function TeamComparableTeamsPanel({
  comparableRows,
  comparableNameCounts,
  onOpenTeamPage,
  onOpenHeadToHead,
  selectedFamilyIds,
  actionSnapshotId,
}) {
  return (
    <article className="analytics-panel analytics-panel-wide">
      <div className="team-section-heading">
        <div>
          <h3>Comparable Teams</h3>
          <p className="meta">Similarity and tracked lineup volume from the current search snapshot.</p>
        </div>
      </div>
      {comparableRows.length ? (
        <ul className="team-compare-list">
          {comparableRows.map((row) => {
            const similarity = Math.max(0, Math.min(Number(row.sim_to_query || 0) * 100, 100));
            const duplicateNameCount = comparableNameCounts.get(normalizeTimelineName(row.team_name)) || 0;
            const comparableMeta = [
              `${pct(row.sim_to_query)} similar`,
              Number.isFinite(Number(row.lineup_count))
                ? `${pluralize(row.lineup_count, 'tracked lineup')}`
                : null,
              duplicateNameCount > 1 ? 'same-name variant' : null,
            ].filter(Boolean).join(' · ');

            return (
              <li key={`team-neighbor-${row.team_id}`} className="team-compare-row">
                <button
                  type="button"
                  className="team-row-primary team-compare-primary"
                  onClick={() => onOpenTeamPage([row.team_id], row.team_name, actionSnapshotId)}
                  aria-label={`Open team page for ${row.team_name}`}
                  title={`Team ID ${row.team_id}`}
                >
                  <span className="team-compare-main">
                    <span className="team-compare-name">{row.team_name}</span>
                    <span className="team-compare-meta">{comparableMeta || 'Comparable roster profile'}</span>
                  </span>
                  <span className="team-compare-visual">
                    <span className="team-compare-match" aria-hidden="true">
                      <span style={{ width: `${similarity}%` }} />
                    </span>
                    <span className="team-compare-pct">{pct(row.sim_to_query)}</span>
                  </span>
                </button>
                <div className="team-hover-actions">
                  <button
                    type="button"
                    className="team-hover-action"
                    onClick={() => onOpenHeadToHead(selectedFamilyIds, [row.team_id], actionSnapshotId)}
                  >
                    Compare
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="team-missing-note">No comparable teams loaded yet.</p>
      )}
    </article>
  );
}
