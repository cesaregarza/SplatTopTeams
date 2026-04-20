import React, { useEffect, useMemo, useState } from 'react';
import AdvancedAnalytics from './components/AdvancedAnalytics';
import { fetchHealth } from './api';
import ClusterExplorer from './components/ClusterExplorer';
import HeadToHead from './components/HeadToHead';
import PlayerLookup from './components/PlayerLookup';
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

function formatSnapshotStamp(value) {
  if (!value) return 'No completed snapshot';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
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

  const tabs = useMemo(() => ([
    { id: 'search', label: 'Team Search' },
    { id: 'players', label: 'Player Lookup' },
    { id: 'head-to-head', label: 'Head-to-Head' },
    { id: 'clusters', label: 'Cluster Explorer' },
    { id: 'analytics', label: 'Advanced Analytics' },
  ]), []);

  const latestSnapshot = health?.latest_snapshot || null;
  const healthTone = health?.status === 'ok' ? 'pill-chip-admin' : 'pill-chip-stale';
  const indexedTeams = Number(latestSnapshot?.teams_indexed);
  const indexedTeamsLabel = Number.isFinite(indexedTeams)
    ? indexedTeams.toLocaleString()
    : 'n/a';

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
    <div className="page-shell shell">
      <header className="site-header">
        <div className="container header-inner">
          <div className="header-top">
            <div className="header-copy">
              <div className="crumb">
                <a href="#" className="brand-pill">comp.splat.top</a>
                <span className="crumb-divider" aria-hidden="true" />
                <span className="crumb-parent">Discovery</span>
                <span className="crumb-sep" aria-hidden="true">/</span>
                <span className="crumb-current">Teams</span>
              </div>
              <h1 className="hero-title">SplatTopTeams</h1>
              <p className="hero-sub">
                Team search, player history, cluster exploration, matchup review, and snapshot analytics over the live team index.
              </p>
            </div>

            <div className="header-right">
              <span className="pill-chip">
                api
                {' '}
                <code>/api/health</code>
              </span>
              <span className={`pill-chip ${healthTone}`}>
                {health?.status === 'ok' ? 'ready' : 'degraded'}
              </span>
              <span className="pill-chip">
                snapshot
                {' '}
                <code>{latestSnapshot?.run_id ?? 'n/a'}</code>
              </span>
            </div>
          </div>
        </div>
      </header>

      <section className="status-bar" aria-label="Snapshot status">
        <div className="container status-inner">
          <div className="snapshot-group">
            <div className="snapshot-item">
              <p className="status-label">Latest Snapshot</p>
              <span className="status-value mono">#{latestSnapshot?.run_id ?? 'n/a'}</span>
            </div>
            <span className="snapshot-sep" aria-hidden="true" />
            <div className="snapshot-item">
              <p className="status-label">Indexed Teams</p>
              <span className="status-value">{indexedTeamsLabel}</span>
            </div>
            <span className="snapshot-sep" aria-hidden="true" />
            <div className="snapshot-item">
              <p className="status-label">Completed</p>
              <span className="status-value">{formatSnapshotStamp(latestSnapshot?.finished_at)}</span>
            </div>
          </div>
        </div>
      </section>

      <div className="subnav-wrap">
        <div className="container">
          <div className="subnav-inner" role="tablist" aria-label="Primary views">
            {tabs.map((item) => (
              <button
                key={item.id}
                role="tab"
                aria-selected={tab === item.id}
                className={`subnav-link ${tab === item.id ? 'is-active' : ''}`}
                onClick={() => setTab(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <main className="site-main container">

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
        {tab === 'players' ? (
          <PlayerLookup
            onOpenTeamSearch={(teamName) => {
              setTab('search');
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
