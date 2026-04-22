import React, { useEffect, useMemo, useState } from 'react';
import {
  Navigate,
  NavLink,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from 'react-router-dom';
import AdvancedAnalytics from './components/AdvancedAnalytics';
import { fetchHealth } from './api';
import ClusterExplorer from './components/ClusterExplorer';
import HeadToHead from './components/HeadToHead';
import PlayerLookup from './components/PlayerLookup';
import TeamExplorer from './components/TeamExplorer';
import TeamSearch from './components/TeamSearch';
import {
  buildClusterHref,
  buildHeadToHeadHref,
  buildPlayerHref,
  buildSearchHref,
  buildTeamHref,
  parseClusterRouteState,
  parseHeadToHeadRouteState,
  parsePlayerRouteState,
  parseSearchRouteState,
  parseTeamRouteState,
} from './routerState';

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

function routeLabel(pathname) {
  if (pathname.startsWith('/teams')) return 'Teams';
  if (pathname.startsWith('/players')) return 'Players';
  if (pathname.startsWith('/head-to-head')) return 'Head-to-Head';
  if (pathname.startsWith('/clusters')) return 'Clusters';
  if (pathname.startsWith('/analytics')) return 'Analytics';
  return 'Search';
}

function SearchRoute() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const routeState = useMemo(() => parseSearchRouteState(searchParams), [searchParams]);

  return (
    <TeamSearch
      selectedTeamAIds={routeState.teamAIds}
      selectedTeamBIds={routeState.teamBIds}
      selectedSnapshotId={routeState.snapshotId}
      initialRouteState={routeState}
      onSearchStateChange={(nextState) => {
        navigate(buildSearchHref({
          ...routeState,
          ...nextState,
        }));
      }}
      onOpenTeamPage={(teamIds, _teamName, snapshotId) => {
        navigate(buildTeamHref({ teamIds, snapshotId }));
      }}
      onOpenHeadToHead={(role, teamIds, snapshotId) => {
        const nextIds = Array.isArray(teamIds) ? teamIds : [];
        const nextTeamAIds = role === 'A' ? nextIds : routeState.teamAIds;
        const nextTeamBIds = role === 'B' ? nextIds : routeState.teamBIds;
        const sanitizedTeamAIds = nextTeamAIds.filter((teamId) => !nextTeamBIds.includes(teamId));
        const sanitizedTeamBIds = nextTeamBIds.filter((teamId) => !sanitizedTeamAIds.includes(teamId));
        navigate(buildSearchHref({
          ...routeState,
          teamAIds: sanitizedTeamAIds,
          teamBIds: sanitizedTeamBIds,
          snapshotId: snapshotId || routeState.snapshotId,
        }));
      }}
      onOpenHeadToHeadPage={(teamAIds, teamBIds, snapshotId) => {
        navigate(buildHeadToHeadHref({
          teamAIds,
          teamBIds,
          snapshotId,
        }));
      }}
    />
  );
}

function TeamsRoute() {
  const navigate = useNavigate();
  const location = useLocation();
  const params = useParams();
  const [searchParams] = useSearchParams();
  const routeState = useMemo(
    () => parseTeamRouteState(params.teamId, searchParams),
    [params.teamId, searchParams],
  );

  if (!params.teamId && routeState.teamIds.length) {
    return <Navigate to={buildTeamHref(routeState)} replace />;
  }

  return (
    <TeamExplorer
      key={`${location.pathname}${location.search}`}
      selectedTeamId={routeState.teamId}
      selectedTeamIds={routeState.teamIds}
      selectedSnapshotId={routeState.snapshotId}
      initialTeamScope={routeState.scope}
      onStateChange={(nextState) => {
        navigate(buildTeamHref(nextState));
      }}
      onOpenHeadToHead={(teamAIds, teamBIds, snapshotId) => {
        navigate(buildHeadToHeadHref({ teamAIds, teamBIds, snapshotId }));
      }}
      onOpenTeamPage={(teamIds, _teamName, snapshotId) => {
        navigate(buildTeamHref({ teamIds, snapshotId }));
      }}
      onOpenPlayerLookup={(playerId, playerName) => {
        navigate(buildPlayerHref({ playerId, playerName }));
      }}
    />
  );
}

function PlayersRoute() {
  const navigate = useNavigate();
  const params = useParams();
  const [searchParams] = useSearchParams();
  const routeState = useMemo(
    () => parsePlayerRouteState(params.playerId, searchParams),
    [params.playerId, searchParams],
  );

  return (
    <PlayerLookup
      selectedPlayerId={routeState.playerId}
      selectedPlayerName={routeState.playerName}
      onOpenPlayerPage={(playerId, playerName) => {
        navigate(buildPlayerHref({ playerId, playerName }));
      }}
      onOpenTeamSearch={(teamName) => {
        navigate(buildSearchHref({ q: teamName }));
      }}
    />
  );
}

function HeadToHeadRoute() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const routeState = useMemo(
    () => parseHeadToHeadRouteState(searchParams),
    [searchParams],
  );

  return (
    <HeadToHead
      selectedTeamAId={routeState.teamAId}
      selectedTeamBId={routeState.teamBId}
      selectedTeamAIds={routeState.teamAIds}
      selectedTeamBIds={routeState.teamBIds}
      selectedSnapshotId={routeState.snapshotId}
      onStateChange={(nextState) => {
        navigate(buildHeadToHeadHref(nextState));
      }}
      onSelectTeamA={(teamIds, snapshotId) => {
        navigate(buildHeadToHeadHref({
          teamAIds: teamIds,
          teamBIds: routeState.teamBIds,
          snapshotId: snapshotId || routeState.snapshotId,
        }));
      }}
      onSelectTeamB={(teamIds, snapshotId) => {
        navigate(buildHeadToHeadHref({
          teamAIds: routeState.teamAIds,
          teamBIds: teamIds,
          snapshotId: snapshotId || routeState.snapshotId,
        }));
      }}
    />
  );
}

function ClustersRoute() {
  const navigate = useNavigate();
  const params = useParams();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const routeState = useMemo(
    () => parseClusterRouteState(params.clusterId, searchParams),
    [params.clusterId, searchParams],
  );

  if (!params.clusterId && routeState.clusterId) {
    return <Navigate to={buildClusterHref(routeState)} replace />;
  }

  return (
    <ClusterExplorer
      key={`${location.pathname}${location.search}`}
      initialQuery={routeState.query}
      initialClusterMode={routeState.clusterMode}
      initialLimit={routeState.limit}
      initialSelectedClusterId={routeState.clusterId}
      onStateChange={(nextState) => {
        navigate(buildClusterHref(nextState));
      }}
    />
  );
}

export default function App() {
  const location = useLocation();
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const tabs = useMemo(() => ([
    { id: 'search', label: 'Team Search', to: '/search' },
    { id: 'teams', label: 'Teams', to: '/teams' },
    { id: 'players', label: 'Player Lookup', to: '/players' },
    { id: 'head-to-head', label: 'Head-to-Head', to: '/head-to-head' },
    { id: 'clusters', label: 'Cluster Explorer', to: '/clusters' },
    { id: 'analytics', label: 'Advanced Analytics', to: '/analytics' },
  ]), []);

  const latestSnapshot = health?.latest_snapshot || null;
  const healthTone = health?.status === 'ok' ? 'pill-chip-admin' : 'pill-chip-stale';
  const indexedTeams = Number(latestSnapshot?.teams_indexed);
  const indexedTeamsLabel = Number.isFinite(indexedTeams)
    ? indexedTeams.toLocaleString()
    : 'n/a';
  const currentViewLabel = routeLabel(location.pathname);

  return (
    <div className="page-shell shell">
      <header className="site-header">
        <div className="container header-inner">
          <div className="header-top">
            <div className="header-copy">
              <div className="crumb">
                <a href="https://comp.splat.top" className="brand-pill" target="_blank" rel="noreferrer">comp.splat.top</a>
                <span className="crumb-divider" aria-hidden="true" />
                <span className="crumb-parent">Discovery</span>
                <span className="crumb-sep" aria-hidden="true">/</span>
                <span className="crumb-current">{currentViewLabel}</span>
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
          <nav className="subnav-inner" aria-label="Primary views">
            {tabs.map((item) => (
              <NavLink
                key={item.id}
                to={item.to}
                className={({ isActive }) => `subnav-link ${isActive ? 'is-active' : ''}`}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>

      <main className="site-main container">
        <Routes>
          <Route path="/" element={<Navigate to="/search" replace />} />
          <Route path="/search" element={<SearchRoute />} />
          <Route path="/teams" element={<TeamsRoute />} />
          <Route path="/teams/:teamId" element={<TeamsRoute />} />
          <Route path="/players" element={<PlayersRoute />} />
          <Route path="/players/:playerId" element={<PlayersRoute />} />
          <Route path="/head-to-head" element={<HeadToHeadRoute />} />
          <Route path="/clusters" element={<ClustersRoute />} />
          <Route path="/clusters/:clusterId" element={<ClustersRoute />} />
          <Route path="/analytics" element={<AdvancedAnalytics />} />
          <Route path="*" element={<Navigate to="/search" replace />} />
        </Routes>
      </main>
    </div>
  );
}
