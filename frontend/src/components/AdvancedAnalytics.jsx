import React, { useEffect, useMemo, useState } from 'react';
import {
  fetchAnalyticsDrift,
  fetchAnalyticsMatchups,
  fetchAnalyticsOutliers,
  fetchAnalyticsOverview,
  fetchAnalyticsRosterOverlap,
  fetchAnalyticsSpace,
  fetchAnalyticsTeam,
  fetchAnalyticsTeamBlend,
} from '../api';

function pct(value) {
  const numeric = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return `${(numeric * 100).toFixed(1)}%`;
}

function pct1(value) {
  const numeric = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return `${numeric.toFixed(1)}%`;
}

function dec(value, digits = 3) {
  const numeric = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return numeric.toFixed(digits);
}

function clusterColor(clusterId) {
  if (clusterId == null || clusterId < 0) {
    return 'rgba(148, 163, 184, 0.7)';
  }
  const hue = (Number(clusterId) * 47) % 360;
  return `hsl(${hue} 78% 62%)`;
}

function renderSpacePoint(point, idx) {
  const cx = ((point.x + 1) / 2) * 100;
  const cy = ((1 - (point.y + 1) / 2) * 100);
  const radius = Math.max(1.3, Math.min(4.8, 1.2 + Math.log10((point.lineup_count || 1) + 1)));
  return (
    <circle
      key={`${point.team_id}-${idx}`}
      cx={cx}
      cy={cy}
      r={radius}
      fill={clusterColor(point.cluster_id)}
      fillOpacity={0.74}
      stroke="rgba(255,255,255,0.35)"
      strokeWidth="0.2"
    >
      <title>{`${point.team_name} | team_id=${point.team_id} | cluster=${point.cluster_id ?? 'n/a'} | lineups=${point.lineup_count}`}</title>
    </circle>
  );
}

export default function AdvancedAnalytics() {
  const [clusterMode, setClusterMode] = useState('explore');

  const [overview, setOverview] = useState(null);
  const [matchups, setMatchups] = useState(null);
  const [drift, setDrift] = useState(null);
  const [space, setSpace] = useState(null);
  const [outliers, setOutliers] = useState(null);
  const [rosterOverlap, setRosterOverlap] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [teamIdInput, setTeamIdInput] = useState('');
  const [neighbors, setNeighbors] = useState(12);
  const [teamLab, setTeamLab] = useState(null);
  const [teamLabLoading, setTeamLabLoading] = useState(false);
  const [teamLabError, setTeamLabError] = useState('');

  const [semanticWeight, setSemanticWeight] = useState(0.5);
  const [blendResult, setBlendResult] = useState(null);
  const [blendLoading, setBlendLoading] = useState(false);

  const [rosterMinSimilarity, setRosterMinSimilarity] = useState(0.82);
  const [rosterMaxOverlap, setRosterMaxOverlap] = useState(0.3);
  const [rosterLimit, setRosterLimit] = useState(20);

  async function refreshAll() {
    setError('');
    setLoading(true);
    try {
      const [
        overviewPayload,
        matchupPayload,
        driftPayload,
        spacePayload,
        outlierPayload,
        rosterPayload,
      ] = await Promise.all([
        fetchAnalyticsOverview({ clusterMode, limitClusters: 20, volatileLimit: 15 }),
        fetchAnalyticsMatchups({ clusterMode, minMatches: 3, limit: 30 }),
        fetchAnalyticsDrift({ clusterMode, topMovers: 20 }),
        fetchAnalyticsSpace({ clusterMode, maxPoints: 900 }),
        fetchAnalyticsOutliers({ clusterMode, limit: 30 }),
        fetchAnalyticsRosterOverlap({
          clusterMode,
          minSimilarity: rosterMinSimilarity,
          maxPlayerOverlap: rosterMaxOverlap,
          minClusterSize: 2,
          limit: rosterLimit,
        }),
      ]);
      setOverview(overviewPayload);
      setMatchups(matchupPayload);
      setDrift(driftPayload);
      setSpace(spacePayload);
      setOutliers(outlierPayload);
      setRosterOverlap(rosterPayload);
    } catch (err) {
      setError(err.message || 'Failed to load analytics');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusterMode]);

  const summaryCards = useMemo(() => {
    const summary = overview?.summary;
    if (!summary) return [];
    const coverageValue = Number(summary.coverage_pct);
    return [
      { label: 'Teams Indexed', value: summary.teams_indexed },
      { label: 'Clustered Teams', value: summary.clustered_teams },
      { label: 'Noise Teams', value: summary.noise_teams },
      { label: 'Clusters', value: summary.cluster_count },
      {
        label: 'Coverage',
        value: `${Number.isFinite(coverageValue) ? coverageValue.toFixed(1) : '0.0'}%`,
      },
      { label: 'Avg Lineups/Team', value: summary.avg_lineups_per_team },
      { label: 'Median Lineups/Team', value: summary.median_lineups_per_team },
    ];
  }, [overview]);

  const driftCards = useMemo(() => {
    const summary = drift?.summary;
    if (!summary) return [];
    return [
      { label: 'Shared Teams', value: summary.shared_teams },
      { label: 'New Teams', value: summary.new_teams },
      { label: 'Dropped Teams', value: summary.dropped_teams },
      { label: 'Cluster Switches', value: summary.cluster_switches },
      { label: 'Newly Clustered', value: summary.newly_clustered },
      { label: 'Newly Noise', value: summary.newly_noise },
      { label: 'Avg Drift', value: dec(summary.avg_embedding_drift) },
      { label: 'P90 Drift', value: dec(summary.p90_embedding_drift) },
    ];
  }, [drift]);

  async function onTeamLabSubmit(event) {
    event.preventDefault();
    const teamId = Number(teamIdInput);
    if (!Number.isFinite(teamId) || teamId <= 0) return;

    setTeamLabLoading(true);
    setTeamLabError('');
    setBlendResult(null);
    try {
      const payload = await fetchAnalyticsTeam({
        teamId,
        clusterMode,
        neighbors,
      });
      setTeamLab(payload);
    } catch (err) {
      setTeamLab(null);
      setTeamLabError(err.message || 'Failed to load team analytics');
    } finally {
      setTeamLabLoading(false);
    }
  }

  async function runBlendNeighbors() {
    const teamId = Number(teamIdInput);
    if (!Number.isFinite(teamId) || teamId <= 0) return;

    setBlendLoading(true);
    try {
      const payload = await fetchAnalyticsTeamBlend({
        teamId,
        clusterMode,
        neighbors,
        semanticWeight,
      });
      setBlendResult(payload);
      setTeamLabError('');
    } catch (err) {
      setBlendResult(null);
      setTeamLabError(err.message || 'Failed to compute blend neighbors');
    } finally {
      setBlendLoading(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="analytics-title">
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Research console</p>
          <h2 id="analytics-title" className="panel-title">Advanced Analytics</h2>
          <p className="panel-summary">
            Drift intelligence, embedding-space geometry, outlier forensics, rivalry mapping, and what-if similarity blending.
          </p>
        </div>
      </div>

      <div className="analytics-controls">
        <div className="field">
          <label className="field-label" htmlFor="analytics-cluster-mode">Cluster profile</label>
          <select
            id="analytics-cluster-mode"
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
          <label className="field-label slider-label" htmlFor="analytics-roster-similarity">
            Min similarity ({Math.round(rosterMinSimilarity * 100)}%)
          </label>
          <input
            id="analytics-roster-similarity"
            className="input slider-input"
            type="range"
            min={50}
            max={99}
            step={1}
            value={Math.round(rosterMinSimilarity * 100)}
            onChange={(e) => setRosterMinSimilarity(Number(e.target.value) / 100)}
          />
        </div>
        <div className="field">
          <label className="field-label slider-label" htmlFor="analytics-roster-overlap">
            Max overlap ({Math.round(rosterMaxOverlap * 100)}%)
          </label>
          <input
            id="analytics-roster-overlap"
            className="input slider-input"
            type="range"
            min={0}
            max={70}
            step={1}
            value={Math.round(rosterMaxOverlap * 100)}
            onChange={(e) => setRosterMaxOverlap(Number(e.target.value) / 100)}
          />
        </div>
        <div className="field">
          <label className="field-label" htmlFor="analytics-roster-limit">Pairs/Cases</label>
          <input
            id="analytics-roster-limit"
            className="input"
            type="number"
            min={5}
            max={80}
            value={rosterLimit}
            onChange={(e) => setRosterLimit(Number(e.target.value) || 20)}
          />
        </div>
        <button className="button btn-pill btn-fuchsia" type="button" onClick={refreshAll}>
          {loading ? 'Refreshing…' : 'Refresh analytics'}
        </button>
      </div>

      {error ? <p className="error">{error}</p> : null}

      <div className="grid-cols-4 analytics-cards analytics-stats">
        {summaryCards.map((card) => (
          <article key={card.label} className="analytics-card stat">
            <span className="analytics-card-label stat-label">{card.label}</span>
            <span className="analytics-card-value stat-value">{card.value}</span>
          </article>
        ))}
      </div>

      <article className="analytics-panel analytics-panel-wide">
        <h3>Snapshot Drift Report</h3>
        <p className="meta">
          Current snapshot {drift?.current_snapshot_id ?? 'n/a'} vs baseline {drift?.previous_snapshot_id ?? 'n/a'}.
        </p>
        <div className="grid-cols-4 analytics-cards drift-cards analytics-stats">
          {driftCards.map((card) => (
            <article key={card.label} className="analytics-card stat">
              <span className="analytics-card-label stat-label">{card.label}</span>
              <span className="analytics-card-value stat-value">{card.value}</span>
            </article>
          ))}
        </div>
        <div className="analytics-grid">
          <div className="analytics-panel">
            <h4>Top Embedding Movers</h4>
            <div className="table-wrap">
              <table className="analytics-table">
                <thead>
                  <tr>
                    <th>Team</th>
                    <th>Drift</th>
                    <th>Prev C</th>
                    <th>Curr C</th>
                    <th>Lineup Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {(drift?.top_embedding_movers || []).slice(0, 12).map((row) => (
                    <tr key={row.team_id}>
                      <td className="analytics-team-name">{row.team_name}</td>
                      <td>{dec(row.drift)}</td>
                      <td>{row.prev_cluster_id ?? 'n/a'}</td>
                      <td>{row.current_cluster_id ?? 'n/a'}</td>
                      <td>{row.lineup_delta}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="analytics-panel">
            <h4>Top Volatility Shifts</h4>
            <div className="table-wrap">
              <table className="analytics-table">
                <thead>
                  <tr>
                    <th>Team</th>
                    <th>Prev</th>
                    <th>Curr</th>
                    <th>Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {(drift?.top_volatility_shifts || []).slice(0, 12).map((row) => (
                    <tr key={row.team_id}>
                      <td className="analytics-team-name">{row.team_name}</td>
                      <td>{dec(row.volatility_prev)}</td>
                      <td>{dec(row.volatility_current)}</td>
                      <td>{dec(row.volatility_delta)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </article>

      <div className="analytics-grid">
        <article className="analytics-panel">
          <h3>Cluster Health</h3>
          <p className="meta">Separation gap compares cluster cohesion against nearest neighboring cluster centroid.</p>
          <div className="table-wrap">
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Size</th>
                  <th>Cohesion</th>
                  <th>P10</th>
                  <th>Nearest</th>
                  <th>Gap</th>
                </tr>
              </thead>
              <tbody>
                {(overview?.cluster_health || []).slice(0, 12).map((row) => (
                  <tr key={row.cluster_id}>
                    <td>{row.cluster_id}</td>
                    <td>{row.cluster_size}</td>
                    <td>{dec(row.cohesion_mean)}</td>
                    <td>{dec(row.cohesion_p10)}</td>
                    <td>{dec(row.nearest_cluster_similarity)}</td>
                    <td>{dec(row.separation_gap)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="analytics-panel">
          <h3>Volatility Leaders</h3>
          <p className="meta">Higher volatility implies more rotational lineups and less dominant top roster.</p>
          <div className="table-wrap">
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Team</th>
                  <th>Lineups</th>
                  <th>Distinct</th>
                  <th>Top Share</th>
                  <th>Entropy</th>
                  <th>Vol Score</th>
                </tr>
              </thead>
              <tbody>
                {(overview?.volatility_leaders || []).slice(0, 12).map((row) => (
                <tr key={row.team_id}>
                  <td className="analytics-team-name">{row.team_name}</td>
                  <td>{row.lineup_count}</td>
                  <td>{row.distinct_lineup_count}</td>
                  <td>{pct1(row.top_lineup_share_pct)}</td>
                  <td>{dec(row.lineup_entropy)}</td>
                  <td>{dec(row.volatility_score)}</td>
                </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <article className="analytics-panel analytics-panel-wide">
        <h3>Embedding Space Map</h3>
        <p className="meta">2D projection of high-dimensional team vectors. Point size tracks lineup volume; color tracks cluster id.</p>
        <div className="space-map-wrap">
          <svg viewBox="0 0 100 100" className="space-map" role="img" aria-label="Embedding space map">
            <rect x="0" y="0" width="100" height="100" fill="rgba(15,23,42,0.7)" />
            <line x1="50" y1="0" x2="50" y2="100" stroke="rgba(255,255,255,0.12)" strokeWidth="0.25" />
            <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(255,255,255,0.12)" strokeWidth="0.25" />
            {(space?.points || []).map((point, idx) => renderSpacePoint(point, idx))}
            {(space?.centroids || []).slice(0, 80).map((centroid) => {
              const cx = ((centroid.x + 1) / 2) * 100;
              const cy = ((1 - (centroid.y + 1) / 2) * 100);
              return (
                <g key={`centroid-${centroid.cluster_id}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={1.8}
                    fill={clusterColor(centroid.cluster_id)}
                    stroke="white"
                    strokeWidth="0.3"
                  />
                </g>
              );
            })}
          </svg>
        </div>
      </article>

      <article className="analytics-panel analytics-panel-wide">
        <h3>Cluster Outliers</h3>
        <p className="meta">Teams furthest from their own cluster centroid; useful for detecting roster pivots and boundary members.</p>
        <div className="table-wrap">
          <table className="analytics-table">
            <thead>
              <tr>
                <th>Team</th>
                <th>Cluster</th>
                <th>Similarity</th>
                <th>Cluster Mean</th>
                <th>Z</th>
                <th>Outlier Score</th>
                <th>Top Share</th>
              </tr>
            </thead>
            <tbody>
              {(outliers?.outliers || []).slice(0, 15).map((row) => (
                  <tr key={row.team_id}>
                  <td className="analytics-team-name">{row.team_name}</td>
                  <td>{row.cluster_id}</td>
                  <td>{dec(row.cluster_similarity)}</td>
                  <td>{dec(row.cluster_mean_similarity)}</td>
                  <td>{dec(row.outlier_z)}</td>
                  <td>{dec(row.outlier_score)}</td>
                  <td>{pct1(row.top_lineup_share_pct)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </article>

      <article className="analytics-panel analytics-panel-wide">
        <h3>Roster Divergence in Similar Clusters</h3>
        <p className="meta">
          Same-cluster teams with high embedding similarity but low player overlap. “Potential Squad” is the unique
          top-lineup player count across that cohort (the full player pool you could assemble for one team conceptually).
        </p>
        <p className="meta">
          {rosterOverlap?.clusters_considered ?? 0} clusters scanned · {rosterOverlap?.total_pairs_found ?? 0} qualifying pairs.
        </p>
        <div className="analytics-grid">
          <div className="analytics-panel">
            <h4>Top Cohorts</h4>
            <div className="table-wrap">
              <table className="analytics-table">
                <thead>
                  <tr>
                    <th>Teams</th>
                    <th>Cluster</th>
                    <th>Tourneys</th>
                    <th>Team Count</th>
                    <th>Pairs</th>
                    <th>Potential Squad</th>
                    <th>Sim (avg / min / max)</th>
                    <th>Overlap (min / max)</th>
                  </tr>
                </thead>
                <tbody>
                  {(rosterOverlap?.cohorts || []).map((row) => (
                    <tr key={`${row.cluster_id}-${row.team_ids.join('-')}`}>
                      <td>
                        <div>{row.team_names?.slice(0, 5).join(' · ') || row.team_ids.join(', ')}</div>
                        {row.team_ids?.length > 5 ? (
                          <div className="meta-item" style={{ marginTop: '0.32rem', padding: '0.18rem 0.35rem' }}>
                            and {row.team_ids.length - 5} more
                          </div>
                        ) : null}
                      </td>
                      <td>{row.cluster_id}</td>
                      <td>{Object.values(row.team_tournament_counts || {}).reduce((acc, value) => acc + Number(value), 0)}</td>
                      <td>{row.team_count}</td>
                      <td>{row.pair_count}</td>
                      <td>{row.roster_pool_size}</td>
                      <td>
                        {dec(row.pair_similarity_avg)} / {dec(row.pair_similarity_min)} / {dec(row.pair_similarity_max)}
                      </td>
                      <td>
                        {dec(row.pair_overlap_min)} / {dec(row.pair_overlap_max)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="analytics-panel">
            <h4>Top Qualifying Pairs</h4>
            <div className="table-wrap">
              <table className="analytics-table">
                <thead>
                  <tr>
                    <th>Team A</th>
                    <th>Team B</th>
                    <th>A Tourn</th>
                    <th>B Tourn</th>
                    <th>Sim</th>
                    <th>Overlap</th>
                    <th>Shared</th>
                    <th>Potential Squad</th>
                  </tr>
                </thead>
                <tbody>
                  {(rosterOverlap?.pairs || []).map((row) => (
                    <tr key={`${row.team_a_id}-${row.team_b_id}`}>
                      <td>{row.team_a_name}</td>
                      <td>{row.team_b_name}</td>
                      <td>{row.team_a_tournament_count || 0}</td>
                      <td>{row.team_b_tournament_count || 0}</td>
                      <td>{dec(row.similarity)}</td>
                      <td>{dec(row.overlap_fraction)}</td>
                      <td>{row.overlap_count}</td>
                      <td>{row.roster_pool_size}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </article>

      <article className="analytics-panel analytics-panel-wide">
        <h3>Cross-Cluster Rivalries</h3>
        <p className="meta">
          Rivalry score blends matchup volume and balance; high score means frequent and competitive cluster clashes.
        </p>
        <div className="table-wrap">
          <table className="analytics-table">
            <thead>
              <tr>
                <th>Cluster A</th>
                <th>Cluster B</th>
                <th>Matches</th>
                <th>A Wins</th>
                <th>B Wins</th>
                <th>A Win%</th>
                <th>Close</th>
                <th>Rivalry</th>
              </tr>
            </thead>
            <tbody>
              {(matchups?.matchups || []).map((row) => (
                <tr key={`${row.cluster_a}-${row.cluster_b}`}>
                  <td>{row.cluster_a_name} ({row.cluster_a})</td>
                  <td>{row.cluster_b_name} ({row.cluster_b})</td>
                  <td>{row.matches}</td>
                  <td>{row.wins_a}</td>
                  <td>{row.wins_b}</td>
                  <td>{pct(row.win_rate_a)}</td>
                  <td>{dec(row.close_factor)}</td>
                  <td>{dec(row.rivalry_score, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

      <article className="analytics-panel analytics-panel-wide">
        <h3>Team Lab + What-If Blend</h3>
        <p className="meta">Inspect a single team’s neighborhood geometry and then recompute nearest teams by semantic-vs-identity blend.</p>

        <form className="team-lab-form" onSubmit={onTeamLabSubmit}>
          <div className="field">
            <label className="field-label" htmlFor="team-lab-id">Team ID</label>
            <input
              id="team-lab-id"
              className="input"
              type="number"
              min={1}
              value={teamIdInput}
              onChange={(e) => setTeamIdInput(e.target.value)}
              required
            />
          </div>
          <div className="field">
            <label className="field-label" htmlFor="team-lab-neighbors">Neighbors</label>
            <input
              id="team-lab-neighbors"
              className="input"
              type="number"
              min={3}
              max={40}
              value={neighbors}
              onChange={(e) => setNeighbors(Number(e.target.value) || 12)}
            />
          </div>
          <button type="submit" className="button" disabled={teamLabLoading}>
            {teamLabLoading ? 'Loading…' : 'Run team lab'}
          </button>
        </form>

        {teamLabError ? <p className="error">{teamLabError}</p> : null}

        {teamLab ? (
          <div className="team-lab-output">
            <div className="analytics-cards team-lab-cards">
              <article className="analytics-card">
                <p className="analytics-card-label">Team</p>
                <p className="analytics-card-value">{teamLab.team.team_name}</p>
              </article>
              <article className="analytics-card">
                <p className="analytics-card-label">Volatility</p>
                <p className="analytics-card-value">{dec(teamLab.team.volatility_score)}</p>
              </article>
              <article className="analytics-card">
                <p className="analytics-card-label">Uniqueness</p>
                <p className="analytics-card-value">{dec(teamLab.team.uniqueness_score)}</p>
              </article>
              <article className="analytics-card">
                <p className="analytics-card-label">Win Rate</p>
                <p className="analytics-card-value">{pct(teamLab.match_summary.win_rate)}</p>
              </article>
            </div>

            <div className="blend-controls">
              <div className="field">
                <label className="field-label" htmlFor="semantic-weight-range">
                  Semantic Weight ({Math.round(semanticWeight * 100)}%)
                </label>
                <input
                  id="semantic-weight-range"
                  className="input"
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(semanticWeight * 100)}
                  onChange={(e) => setSemanticWeight(Number(e.target.value) / 100)}
                />
              </div>
              <button type="button" className="button" onClick={runBlendNeighbors} disabled={blendLoading}>
                {blendLoading ? 'Computing…' : 'Recompute blend neighbors'}
              </button>
            </div>

            <div className="analytics-grid">
              <div className="analytics-panel">
                <h4>Nearest Teams (Default)</h4>
                <div className="table-wrap">
                  <table className="analytics-table">
                    <thead>
                      <tr>
                        <th>Team</th>
                        <th>Overall</th>
                        <th>Semantic</th>
                        <th>Identity</th>
                        <th>Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {teamLab.neighbors.map((row) => (
                        <tr key={`default-${row.team_id}`}>
                          <td className="analytics-team-name">{row.team_name}</td>
                          <td>{dec(row.sim_to_query)}</td>
                          <td>{dec(row.sim_semantic)}</td>
                          <td>{dec(row.sim_identity)}</td>
                          <td>{dec(row.identity_delta)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="analytics-panel">
                <h4>Nearest Teams (What-If Blend)</h4>
                <p className="meta">Semantic {Math.round((blendResult?.semantic_weight ?? semanticWeight) * 100)}% / Identity {Math.round((blendResult?.identity_weight ?? (1 - semanticWeight)) * 100)}%</p>
                <div className="table-wrap">
                  <table className="analytics-table">
                    <thead>
                      <tr>
                        <th>Team</th>
                        <th>Overall</th>
                        <th>Semantic</th>
                        <th>Identity</th>
                        <th>Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(blendResult?.neighbors || []).map((row) => (
                        <tr key={`blend-${row.team_id}`}>
                          <td className="analytics-team-name">{row.team_name}</td>
                          <td>{dec(row.sim_to_query)}</td>
                          <td>{dec(row.sim_semantic)}</td>
                          <td>{dec(row.sim_identity)}</td>
                          <td>{dec(row.identity_delta)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="analytics-panel">
              <h4>Opponent Cluster Breakdown</h4>
              <div className="table-wrap">
                <table className="analytics-table">
                  <thead>
                    <tr>
                      <th>Cluster</th>
                      <th>Matches</th>
                      <th>Wins</th>
                      <th>Win%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {teamLab.opponent_cluster_breakdown.map((row) => (
                      <tr key={row.cluster_id}>
                        <td>{row.cluster_id}</td>
                        <td>{row.matches}</td>
                        <td>{row.wins}</td>
                        <td>{pct(row.win_rate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : null}
      </article>
    </section>
  );
}
