import React from 'react';
import { pluralize } from './helpers';

export function TeamIdentityHero({
  heroKicker,
  canonicalName,
  heroAliases,
  warning,
}) {
  return (
    <>
      <div className="team-identity-hero">
        <div className="team-identity-head">
          <div>
            <p className="team-identity-kicker">{heroKicker}</p>
            <h3 className="team-identity-name">{canonicalName || 'Team'}</h3>
          </div>
        </div>
        {heroAliases.length ? (
          <div className="team-identity-aliases" aria-label="Known family names">
            {heroAliases.map((row, index) => (
              <span
                key={`hero-alias-${row.key}`}
                className={`team-identity-alias ${index === 0 ? 'is-current' : ''}`}
                title={`${row.name} · ${pluralize(row.registrationCount, 'registration')} · ${pluralize(row.totalMatches, 'match')}`}
              >
                {row.name}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      {warning ? (
        <p className="team-missing-note team-warning-note" role="status">{warning}</p>
      ) : null}
    </>
  );
}
