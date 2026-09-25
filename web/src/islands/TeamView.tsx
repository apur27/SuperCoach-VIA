import { useEffect, useState } from 'react';
import type { TeamIndex, TeamSeason } from '../lib/contracts';
import { parseUrlState, serializeUrlState, type StateSpec } from '../lib/filters';
import { encodeId, isSafeKey, withBase } from '../lib/ids';
import { formatDateOnly, formatStat } from '../lib/format';
import { DataState } from './common/DataState';
import { siteBase, useHeading, useResource, useUrlSearch } from './common/runtime';
import { NoScriptNotice } from './common/NoScript';
import { StatTable } from './common/StatTable';
import { LadderTable } from './common/Ladder';
import { MatchRowsTable } from './common/MatchRows';
import { BarChart } from './common/Charts';

const spec = { id: { kind: 'token' }, season: { kind: 'int', min: 1850, max: 2100 } } as const satisfies StateSpec;

export default function TeamView({ downloadHref }: { downloadHref: string }) {
  const [search, setSearch] = useUrlSearch();
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  const state = parseUrlState(search, spec);
  const id = state.id && isSafeKey(state.id) ? state.id : null;
  const index = useResource('team_index', ready && id ? 'teams/index.json' : null);
  const base = siteBase();
  if (!ready) return <NoScriptNotice what="team view" href={downloadHref} linkText="browse downloads" />;
  if (!id) return <div className="banner banner-info"><h2>Team not found</h2><p>No valid team was chosen. <a href={withBase(base, 'teams/')}>See all teams</a>.</p></div>;
  return (
    <DataState state={index.state} retry={index.retry} what="team list">
      {(idx) => <TeamSeasonPicker idx={idx} id={id} season={state.season} onSeason={(s) => setSearch(serializeUrlState({ id, season: s }, spec))} />}
    </DataState>
  );
}

function TeamSeasonPicker({ idx, id, season, onSeason }: { idx: TeamIndex; id: string; season: number | undefined; onSeason: (s: number) => void }) {
  const base = siteBase();
  const team = idx.teams.find((t) => t.club_id === id);
  const seasons = [...(team?.seasons ?? [])].sort((a, b) => b - a);
  const chosen = season !== undefined && seasons.includes(season) ? season : seasons[0];
  const data = useResource('team_season', team && chosen !== undefined ? `teams/${id}/${chosen}.json` : null);
  useHeading(team ? `${team.name}${chosen ? ` ${chosen}` : ''}` : null);
  if (!team) return <div className="banner banner-info" data-state="notfound"><h2>Team not found</h2><p>No club with ID <code>{id}</code> is in this release. <a href={withBase(base, 'teams/')}>See all teams</a>.</p></div>;
  return (
    <div className="stack">
      <p className="muted">{team.active ? 'Current club' : 'Historical club (no longer competing)'} · seasons {team.first_season ?? '?'}–{team.last_season ?? '?'} · lineage <code>{team.lineage_id}</code></p>
      {season !== undefined && !seasons.includes(season) ? <p className="banner banner-stale">No data for {team.name} in {season}; showing {chosen}.</p> : null}
      {seasons.length ? (
        <div className="field">
          <label htmlFor="team-season">Season</label>
          <select id="team-season" value={chosen} onChange={(e) => onSeason(Number(e.target.value))}>{seasons.map((s) => <option key={s} value={s}>{s}</option>)}</select>
        </div>
      ) : <p className="muted">No seasons with data for this club.</p>}
      <DataState state={data.state} retry={data.retry} what="team season">
        {(ts) => <TeamSeasonBody ts={ts} />}
      </DataState>
    </div>
  );
}

function TeamSeasonBody({ ts }: { ts: TeamSeason }) {
  const base = siteBase();
  return (
    <div className="stack">
      <section aria-labelledby="ladder-h">
        <h2 id="ladder-h">Ladder</h2>
        <p>{ts.position !== null ? <>Finished position: <strong>{ts.position}</strong>.</> : 'Ladder position not recorded.'} {ts.ladder_note}</p>
        <LadderTable rows={ts.ladder} caption={`${ts.season} ladder (regular season only; finals excluded)`} highlight={ts.club.club_id} />
      </section>
      <section aria-labelledby="form-h">
        <h2 id="form-h">Form (last {ts.form_window})</h2>
        {ts.form.length ? (
          <ol className="cluster list-plain" aria-label="Recent results, oldest first">
            {ts.form.map((f) => <li key={f.match_id} className="card"><strong>{f.result === 'W' ? 'Win' : f.result === 'L' ? 'Loss' : 'Draw'}</strong> v {f.opponent} by {Math.abs(f.margin)} <span className="muted">({formatDateOnly(f.match_date)})</span></li>)}
          </ol>
        ) : <p className="muted">No completed regular-season matches recorded.</p>}
      </section>
      <section aria-labelledby="fixtures-h">
        <h2 id="fixtures-h">Fixtures and results</h2>
        {ts.fixtures.length ? <MatchRowsTable matches={ts.fixtures} caption={`${ts.club.name} ${ts.season} fixtures and results`} /> : <p className="muted">No fixtures recorded.</p>}
      </section>
      <section aria-labelledby="stats-h">
        <h2 id="stats-h">Team statistics</h2>
        <StatTable caption={`${ts.club.name} ${ts.season} team statistics`} stats={ts.team_stats} />
      </section>
      <section aria-labelledby="leaders-h">
        <h2 id="leaders-h">Club leaders</h2>
        {ts.leaders.length ? (
          <>
            <BarChart id="leaders-chart" title={`${ts.club.name} ${ts.season} leaders`} description="Leading players by recorded statistic" categoryLabel="Player" valueLabel="Value" data={ts.leaders.map((l) => ({ label: `${l.name} (${l.stat})`, value: l.value }))} />
            <ul>{ts.leaders.map((l) => <li key={`${l.player_id}-${l.stat}`}><a href={withBase(base, `player/?id=${encodeId(l.player_id)}`)}>{l.name}</a>: {formatStat(l.value, 1)} {l.stat} ({l.observed_games} games observed)</li>)}</ul>
          </>
        ) : <p className="muted">No leaders recorded.</p>}
      </section>
      <section aria-labelledby="five-h">
        <h2 id="five-h">Five-year view</h2>
        <LadderTable rows={ts.five_year} caption={`${ts.club.name}: ladder finishes over up to five seasons`} highlight={ts.club.club_id} />
      </section>
      <section aria-labelledby="heur-h">
        <h2 id="heur-h">Heuristics</h2>
        {ts.heuristics.length ? (
          <ul>{ts.heuristics.map((h) => <li key={h.label}><strong>Heuristic — {h.label}:</strong> {h.text} <span className="muted">Method: {h.method}. This is a rule of thumb, not a model forecast.</span></li>)}</ul>
        ) : <p className="muted">No heuristics for this season.</p>}
      </section>
    </div>
  );
}
