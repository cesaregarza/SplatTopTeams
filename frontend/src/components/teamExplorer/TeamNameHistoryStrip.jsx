import React from 'react';
import { pluralize, relativeDate } from './helpers';

export function TeamNameHistoryStrip({
  nameTimelineRows,
  teamScope,
  selectedFamilyIds,
}) {
  if (!nameTimelineRows.length) return null;

  return (
    <div className="team-alias-strip">
      <div className="team-alias-head">
        <p className="team-alias-kicker">
          Name history{teamScope === 'family' ? ` · family of ${selectedFamilyIds.length}` : ''}
        </p>
        <span className="team-alias-meta">Newest first · registrations · matches</span>
      </div>
      <div className="team-alias-rail">
        {nameTimelineRows.map((row, index) => {
          const isCurrent = index === 0;
          return (
            <div
              key={`team-alias-${row.key}`}
              className={`team-alias-item ${isCurrent ? 'is-current' : ''}`}
              title={`${row.name} · ${pluralize(row.registrationCount, 'registration')} · ${pluralize(row.totalMatches, 'match')}`}
            >
              <div className="team-alias-name">{row.name}</div>
              <div className="team-alias-date">
                {row.latestEventTimeMs ? relativeDate(row.latestEventTimeMs) : 'n/a'}
              </div>
              <div className="team-alias-matches">
                {pluralize(row.registrationCount, 'registration')} · {pluralize(row.totalMatches, 'match')}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
