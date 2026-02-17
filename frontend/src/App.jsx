import React, { useEffect, useState } from 'react';
import AdvancedAnalytics from './components/AdvancedAnalytics';
import { fetchHealth } from './api';
import ClusterExplorer from './components/ClusterExplorer';
import HeadToHead from './components/HeadToHead';
import TeamSearch from './components/TeamSearch';

function parseTeamIds(value) {
  if (Array.isArray(value)) {
    return value
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item) && item > 0)
      .map((item) => Math.trunc(item));
  }
  const parsed = Number(value);
  if (Number.isFinite(parsed) && parsed > 0) {
    return [Math.trunc(parsed)];
  }
  if (typeof value === 'string') {
    return value
      .split(',')
      .map((item) => item.trim())
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item) && item > 0)
      .map((item) => Math.trunc(item));
  }
  return [];
}

function uniqueTeamIds(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    if (value > 0 && !seen.has(value)) {
      seen.add(value);
      out.push(value);
    }
  }
  return out;
}

export default function App() {
  const [tab, setTab] = useState('search');
  const [health, setHealth] = useState(null);
  const [headToHeadTeamAId, setHeadToHeadTeamAId] = useState('');
  const [headToHeadTeamBId, setHeadToHeadTeamBId] = useState('');
  const [headToHeadTeamAIds, setHeadToHeadTeamAIds] = useState([]);
  const [headToHeadTeamBIds, setHeadToHeadTeamBIds] = useState([]);
  const [headToHeadSnapshotId, setHeadToHeadSnapshotId] = useState('');

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  function pickHeadToHeadTeam(role, teamIds, snapshotId = null) {
    const nextIds = uniqueTeamIds(parseTeamIds(teamIds));
    const nextId = nextIds[0] ? String(nextIds[0]) : '';
    const nextSet = new Set(nextIds);
    if (snapshotId !== null && snapshotId !== undefined && String(snapshotId).trim() !== '') {
      setHeadToHeadSnapshotId(String(snapshotId).trim());
    }

    if (role === 'A') {
      setHeadToHeadTeamAId(nextId);
      setHeadToHeadTeamAIds(nextIds);
      if (nextId && nextId === headToHeadTeamBId) {
        setHeadToHeadTeamBId('');
        setHeadToHeadTeamBIds([]);
      }
      if (nextSet.size && headToHeadTeamBIds.some((teamId) => nextSet.has(teamId))) {
        setHeadToHeadTeamBId('');
        setHeadToHeadTeamBIds([]);
      }
      return;
    }

    setHeadToHeadTeamBId(nextId);
    setHeadToHeadTeamBIds(nextIds);
    if (nextId && nextId === headToHeadTeamAId) {
      setHeadToHeadTeamAId('');
      setHeadToHeadTeamAIds([]);
    }
    if (nextSet.size && headToHeadTeamAIds.some((teamId) => nextSet.has(teamId))) {
      setHeadToHeadTeamAId('');
      setHeadToHeadTeamAIds([]);
    }
  }

  return (
    <div className="page-shell">
      <header className="site-header">
        <div className="container site-header__inner">
          <a href="#" className="site-title">
            <span className="site-title__mark" />
            <span className="site-title__text">SplatTopTeams</span>
          </a>
          <div className="site-header__actions" role="tablist" aria-label="Primary views">
            <button
              role="tab"
              aria-selected={tab === 'search'}
              className={`tab ${tab === 'search' ? 'is-active' : ''}`}
              onClick={() => setTab('search')}
            >
              Team Search
            </button>
            <button
              role="tab"
              aria-selected={tab === 'head-to-head'}
              className={`tab ${tab === 'head-to-head' ? 'is-active' : ''}`}
              onClick={() => setTab('head-to-head')}
            >
              Head-to-Head
            </button>
            <button
              role="tab"
              aria-selected={tab === 'clusters'}
              className={`tab ${tab === 'clusters' ? 'is-active' : ''}`}
              onClick={() => setTab('clusters')}
            >
              Cluster Explorer
            </button>
            <button
              role="tab"
              aria-selected={tab === 'analytics'}
              className={`tab ${tab === 'analytics' ? 'is-active' : ''}`}
              onClick={() => setTab('analytics')}
            >
              Advanced Analytics
            </button>
          </div>
        </div>
      </header>

      <main className="site-main container">
        <section className="page-hero">
          <h1 className="page-title">Tournament Team Discovery</h1>
          <p className="page-intro">
            Public read-only search over Postgres-backed team embeddings with strict and exploratory cluster profiles.
          </p>
          <p className="meta">
            snapshot {health?.latest_snapshot?.run_id ?? 'n/a'}
            {' '}
            teams indexed {health?.latest_snapshot?.teams_indexed ?? 'n/a'}
          </p>
        </section>

        {tab === 'search' ? (
          <TeamSearch
            selectedTeamAId={headToHeadTeamAId}
            selectedTeamBId={headToHeadTeamBId}
            selectedTeamAIds={headToHeadTeamAIds}
            selectedTeamBIds={headToHeadTeamBIds}
            onOpenHeadToHead={(role, teamId, snapshotId) => {
              pickHeadToHeadTeam(role, teamId, snapshotId);
              setTab('head-to-head');
            }}
          />
        ) : null}
        {tab === 'head-to-head' ? (
          <HeadToHead
            selectedTeamAId={headToHeadTeamAId}
            selectedTeamBId={headToHeadTeamBId}
            selectedTeamAIds={headToHeadTeamAIds}
            selectedTeamBIds={headToHeadTeamBIds}
            selectedSnapshotId={headToHeadSnapshotId}
            onSelectTeamA={(teamId, snapshotId) => pickHeadToHeadTeam('A', teamId, snapshotId)}
            onSelectTeamB={(teamId, snapshotId) => pickHeadToHeadTeam('B', teamId, snapshotId)}
          />
        ) : null}
        {tab === 'clusters' ? <ClusterExplorer /> : null}
        {tab === 'analytics' ? <AdvancedAnalytics /> : null}
      </main>
    </div>
  );
}
