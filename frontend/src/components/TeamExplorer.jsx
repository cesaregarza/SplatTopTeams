import React from 'react';
import { TeamComparableTeamsPanel } from './teamExplorer/TeamComparableTeamsPanel';
import { TeamHistoryPanel } from './teamExplorer/TeamHistoryPanel';
import { TeamIdentityHero } from './teamExplorer/TeamIdentityHero';
import { TeamNameHistoryStrip } from './teamExplorer/TeamNameHistoryStrip';
import { TeamRosterPanel } from './teamExplorer/TeamRosterPanel';
import { TeamStatsSections } from './teamExplorer/TeamStatsSections';
import {
  DEFAULT_TEAM_SCOPE,
  EMPTY_TEAM_IDS,
  parseTeamIdList,
} from './teamExplorer/helpers';
import { useTeamExplorerData } from './teamExplorer/useTeamExplorerData';

export default function TeamExplorer({
  selectedTeamId = '',
  selectedTeamIds = EMPTY_TEAM_IDS,
  selectedSnapshotId = '',
  selectedTeamName = '',
  initialTeamScope = DEFAULT_TEAM_SCOPE,
  onStateChange = () => {},
  onOpenHeadToHead = () => {},
  onOpenTeamPage = () => {},
  onOpenPlayerLookup = () => {},
}) {
  const data = useTeamExplorerData({
    selectedTeamId,
    selectedTeamIds,
    selectedSnapshotId,
    selectedTeamName,
    initialTeamScope,
  });

  function updateRoute(nextScope = data.teamScope) {
    const nextTeamIds = parseTeamIdList(data.teamIdsInput);
    if (!nextTeamIds.length) return;
    onStateChange({
      teamIds: nextTeamIds,
      scope: nextScope,
      snapshotId: data.normalizedSelectedSnapshotId,
    });
  }

  async function handleSubmit(event) {
    updateRoute();
    await data.submitTeam(event);
  }

  return (
    <section className="panel team-detail-panel" aria-labelledby="team-detail-title">
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Scouting board</p>
          <h2 id="team-detail-title" className="panel-title">Teams</h2>
          <p className="panel-summary">
            Roster usage, match scouting, event history, and comparable team profiles for a single squad or family.
          </p>
        </div>
      </div>

      <form className="team-explorer-toolbar" onSubmit={handleSubmit}>
        <div className="field team-id-field">
          <label className="field-label" htmlFor="team-detail-ids">Team ID or IDs</label>
          <input
            id="team-detail-ids"
            className="input"
            type="text"
            inputMode="numeric"
            placeholder="e.g. 46624 or 46624,54749,49106"
            value={data.teamIdsInput}
            onChange={(event) => data.setTeamIdsInput(event.target.value)}
          />
          <span className="field-label-subtitle">
            {data.teamScope === DEFAULT_TEAM_SCOPE
              ? 'Family mode expands the first team ID into the full resolved family when available.'
              : 'Individual mode uses exactly the IDs entered here.'}
          </span>
        </div>
        <div className="field team-scope-field">
          <label className="field-label" htmlFor="team-detail-scope-family">Scope</label>
          <div className="team-scope-segmented" role="tablist" aria-label="Team scope">
            <button
              id="team-detail-scope-family"
              type="button"
              role="tab"
              aria-selected={data.teamScope === 'family'}
              className={data.teamScope === 'family' ? 'is-on' : ''}
              onClick={() => {
                data.setTeamScope('family');
                updateRoute('family');
              }}
            >
              Family
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={data.teamScope === 'individual'}
              className={data.teamScope === 'individual' ? 'is-on' : ''}
              onClick={() => {
                data.setTeamScope('individual');
                updateRoute('individual');
              }}
            >
              Individual
            </button>
          </div>
        </div>
        <button type="submit" className="button btn-pill btn-fuchsia" disabled={data.loading}>
          {data.loading ? 'Loading…' : 'Load team'}
        </button>
      </form>

      {data.error ? <p className="error">{data.error}</p> : null}

      {!data.loading && !data.team && !data.effectiveSummary ? (
        <div className="empty-state team-explorer-empty" role="status">
          <p className="empty-state-title">Pick a team to start</p>
          <p className="empty-state-hint">
            Use a search result’s team-page action or enter one or more team IDs here.
          </p>
        </div>
      ) : null}

      {data.team || data.effectiveSummary ? (
        <>
          <TeamIdentityHero
            heroKicker={data.heroKicker}
            canonicalName={data.canonicalName}
            heroAliases={data.heroAliases}
            warning={data.warning}
          />
          <TeamStatsSections
            performanceStats={data.performanceStats}
            rosterStats={data.rosterStats}
          />
          <TeamRosterPanel
            teamProfilePlayers={data.teamProfilePlayers}
            playerBaselineMatches={data.playerBaselineMatches}
            onOpenPlayerLookup={onOpenPlayerLookup}
            topLineupPlayers={data.topLineupPlayers}
            topLineupMeta={data.topLineupMeta}
            lineupSupportMismatch={data.lineupSupportMismatch}
            lineupSummaryFallback={data.teamProfile?.top_lineup_summary}
            rotationPlayers={data.rotationPlayers}
          />
          <TeamHistoryPanel
            historyPageCount={data.historyPageCount}
            historyPage={data.historyPage}
            changeHistoryPage={data.changeHistoryPage}
            matchEventsCount={data.matchEvents.length}
            recentForm={data.recentForm}
            matches={data.matches}
            visibleMatchEvents={data.visibleMatchEvents}
            expandedEventKeys={data.expandedEventKeys}
            toggleEventKey={data.toggleEventKey}
            onOpenTeamPage={onOpenTeamPage}
            onOpenHeadToHead={onOpenHeadToHead}
            selectedFamilyIds={data.selectedFamilyIds}
            actionSnapshotId={data.actionSnapshotId}
            matchHistoryError={data.matchHistoryError}
          />
          <TeamComparableTeamsPanel
            comparableRows={data.comparableRows}
            comparableNameCounts={data.comparableNameCounts}
            onOpenTeamPage={onOpenTeamPage}
            onOpenHeadToHead={onOpenHeadToHead}
            selectedFamilyIds={data.selectedFamilyIds}
            actionSnapshotId={data.actionSnapshotId}
          />
          <TeamNameHistoryStrip
            nameTimelineRows={data.nameTimelineRows}
            teamScope={data.teamScope}
            selectedFamilyIds={data.selectedFamilyIds}
          />
        </>
      ) : null}
    </section>
  );
}
