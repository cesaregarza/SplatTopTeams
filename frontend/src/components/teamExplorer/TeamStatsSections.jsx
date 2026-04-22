import React from 'react';

export function TeamStatsSections({ performanceStats, rosterStats }) {
  return (
    <div className="team-stats-stack team-summary-grid analytics-stats">
      <section className="team-stat-section" aria-label="Performance stats">
        <p className="team-stat-section-label">Performance</p>
        <div className="grid-cols-4 team-stat-grid">
          {performanceStats.map((stat) => (
            <article key={`performance-${stat.label}`} className="analytics-card stat">
              <span className="analytics-card-label stat-label">{stat.label}</span>
              <span className="analytics-card-value stat-value">{stat.value}</span>
            </article>
          ))}
        </div>
      </section>
      <section className="team-stat-section" aria-label="Roster composition stats">
        <p className="team-stat-section-label">Roster Composition</p>
        <div className="grid-cols-4 team-stat-grid">
          {rosterStats.map((stat) => (
            <article key={`roster-${stat.label}`} className="analytics-card stat">
              <span className="analytics-card-label stat-label">{stat.label}</span>
              <span className="analytics-card-value stat-value">{stat.value}</span>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
