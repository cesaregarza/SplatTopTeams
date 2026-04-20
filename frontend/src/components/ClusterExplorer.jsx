import React, { useEffect, useState } from 'react';
import { fetchClusterDetail, fetchClusters } from '../api';

function fmtDate(ms) {
  if (!ms) return 'n/a';
  const value = Number(ms);
  const seconds = value > 1_000_000_000_000 ? Math.floor(value / 1000) : value;
  return new Date(seconds * 1000).toISOString().slice(0, 10);
}

export default function ClusterExplorer() {
  const [query, setQuery] = useState('');
  const [clusterMode, setClusterMode] = useState('explore');
  const [limit, setLimit] = useState(40);
  const [clusters, setClusters] = useState([]);
  const [selected, setSelected] = useState(null);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function loadClusters() {
    setLoading(true);
    setError('');
    try {
      const data = await fetchClusters({ q: query.trim(), clusterMode, limit });
      setClusters(data.clusters || []);
      if (data.clusters?.length) {
        setSelected((current) => current ?? data.clusters[0].cluster_id);
      } else {
        setSelected(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to load clusters');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadClusters();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusterMode]);

  useEffect(() => {
    async function loadDetail() {
      if (selected == null) {
        setSelectedDetail(null);
        return;
      }
      try {
        const detail = await fetchClusterDetail(selected, clusterMode);
        setSelectedDetail(detail);
      } catch (err) {
        setSelectedDetail(null);
        setError(err.message || 'Failed to load cluster detail');
      }
    }
    loadDetail();
  }, [selected, clusterMode]);

  return (
    <section className="panel" aria-labelledby="cluster-title">
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Latent groups</p>
          <h2 id="cluster-title" className="panel-title">Cluster Explorer</h2>
          <p className="panel-summary">
            Browse latent team clusters and inspect member iterations.
          </p>
        </div>
      </div>

      <div className="form-grid search-form">
        <label htmlFor="cluster-filter" className="field-label">Cluster/team filter</label>
        <input
          id="cluster-filter"
          className="input"
          type="search"
          placeholder="Filter by representative or team name"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="form-row row fields-row">
          <div className="field">
            <label htmlFor="cluster-mode-2" className="field-label">Profile</label>
            <select
              id="cluster-mode-2"
              className="input"
              value={clusterMode}
              onChange={(e) => setClusterMode(e.target.value)}
            >
              <option value="strict">strict</option>
              <option value="explore">explore</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="cluster-limit" className="field-label">Limit</label>
            <input
              id="cluster-limit"
              className="input"
              type="number"
              min={1}
              max={200}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value) || 40)}
            />
          </div>
        </div>

        <button className="button btn-pill btn-fuchsia" type="button" onClick={loadClusters} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh clusters'}
        </button>
      </div>

      {error ? <p className="error">{error}</p> : null}

      <div className="results-head">
        <h3 className="results-title">Clusters</h3>
        <span className="results-count">{clusters.length} clusters · profile {clusterMode}</span>
      </div>

      <div className="cluster-layout">
        <div className="cluster-list" role="listbox" aria-label="Cluster list">
          {clusters.map((cluster) => (
            <button
              key={cluster.cluster_id}
              type="button"
              className={`cluster-item ${selected === cluster.cluster_id ? 'is-active' : ''}`}
              onClick={() => setSelected(cluster.cluster_id)}
              aria-selected={selected === cluster.cluster_id}
            >
              <span className="cluster-item-head">
                <strong>#{cluster.cluster_id}</strong>
                <span className="badge">{cluster.cluster_size}</span>
              </span>
              <span className="cluster-name">{cluster.representative_team_name || 'Unnamed'}</span>
              <span className="cluster-hint">{cluster.stability_hint}</span>
            </button>
          ))}
        </div>

        <div className="cluster-detail" aria-live="polite">
          {selectedDetail ? (
            <>
              <div className="panel-head cluster-detail-head">
                <div>
                  <p className="panel-kicker">Cluster detail</p>
                  <h3 className="panel-title">
                    Cluster #{selectedDetail.cluster_id} ({selectedDetail.cluster_size})
                  </h3>
                  <p className="panel-summary">representative {selectedDetail.representative_team_name || 'n/a'}</p>
                </div>
              </div>

              <div className="member-grid">
                {(selectedDetail.members || []).map((member) => (
                  <article className="member-card" key={member.team_id}>
                    <h4>{member.team_name || `Team ${member.team_id}`}</h4>
                    <p className="meta">team_id={member.team_id}</p>
                    <p className="meta">event={fmtDate(member.event_time_ms)} lineups={member.lineup_count ?? 'n/a'}</p>
                    {member.top_lineup_summary ? (
                      <p className="lineup">{member.top_lineup_summary}</p>
                    ) : null}
                  </article>
                ))}
              </div>
            </>
          ) : (
            <p className="meta">Select a cluster to inspect members.</p>
          )}
        </div>
      </div>
    </section>
  );
}
