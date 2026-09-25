import { useEffect, useState } from 'react';
import type { PlayerDetail, PlayerSeasonGames, PredictionRow } from '../lib/contracts';
import { parsePlayerIdParam, withBase } from '../lib/ids';
import { formatDateOnly, formatStat } from '../lib/format';
import { DataState } from './common/DataState';
import { siteBase, useHeading, useResource, useUrlSearch } from './common/runtime';
import { StatTable } from './common/StatTable';
import { Stat } from './common/Stat';
import { LineChart } from './common/Charts';
import { Instant } from './common/Instant';
import { WatchButton } from './common/watch';
import { NoScriptNotice } from './common/NoScript';

const QUALITY: Record<PlayerDetail['birth_date_quality'], string> = {
  source: 'from source', legacy_filename: 'from legacy file name (unverified)', conflicting: 'sources conflict', unknown: 'unknown',
};

export default function PlayerView({ downloadHref }: { downloadHref: string }) {
  const [search] = useUrlSearch();
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  const raw = new URLSearchParams(search).get('id');
  const parsed = parsePlayerIdParam(raw);
  const detail = useResource('player_detail', ready && parsed ? `players/${parsed.key}.json` : null);
  useHeading(detail.state.status === 'success' ? detail.state.data.name : null);
  const base = siteBase();
  if (!ready) return <NoScriptNotice what="player profile" href={downloadHref} linkText="download all player data (CSV)" />;
  if (!parsed) {
    return (
      <div className="banner banner-info" data-state="invalid-id">
        <h2>Player not found</h2>
        <p>{raw ? 'That is not a valid player ID.' : 'No player was chosen.'} <a href={withBase(base, 'players/')}>Search all players</a>.</p>
      </div>
    );
  }
  return (
    <DataState
      state={detail.state} retry={detail.retry} what="player profile"
      notFound={<div className="banner banner-info" data-state="notfound"><h2>Player not found</h2><p>No player with ID <code>{parsed.id}</code> is in this release. It may have been merged or renamed. <a href={withBase(base, 'players/')}>Search all players</a>.</p></div>}
    >
      {(p) => <PlayerBody p={p} />}
    </DataState>
  );
}

function Forecast({ f }: { f: PredictionRow | null }) {
  const base = siteBase();
  return (
    <section aria-labelledby="fc-h" data-testid="player-forecast">
      <h2 id="fc-h">Current forecast</h2>
      {f ? (
        <div className="card">
          <p><strong>{formatStat(f.predicted_disposals, 1)} predicted disposals</strong> v {f.opponent_name ?? 'opponent not recorded'} ({f.stage_label} {f.season}){f.interval_low !== null && f.interval_high !== null ? `; ${formatStat((f.interval_level ?? 0) * 100)}% interval ${formatStat(f.interval_low, 1)}–${formatStat(f.interval_high, 1)}` : '; no interval available'}.</p>
          <p className="muted">Selection: {f.selection_status === 'confirmed' ? 'confirmed' : 'not confirmed'} ({f.eligibility_basis}). Based on {f.history_games} prior games. Kickoff {f.scheduled_at ? <Instant iso={f.scheduled_at} /> : (f.scheduled_local ?? 'not recorded')}. Model {f.model_id}, origin {f.origin}.</p>
          {f.warnings.length ? <ul>{f.warnings.map((w) => <li key={w}>{w}</li>)}</ul> : null}
        </div>
      ) : (
        <p>No current forecast for this player in this release. See <a href={withBase(base, 'predictions/')}>Predictions</a> for the forecast status.</p>
      )}
    </section>
  );
}

function PlayerBody({ p }: { p: PlayerDetail }) {
  const seasons = [...p.seasons].sort((a, b) => b.season - a.season);
  const [season, setSeason] = useState(seasons[0]?.season ?? null);
  const line = seasons.find((s) => s.season === season) ?? null;
  const games = useResource('player_season_games', line ? line.games_resource : null);
  const base = siteBase();
  return (
    <div className="stack">
      <div className="cluster"><WatchButton id={p.id} name={p.name} /></div>
      <section aria-labelledby="bio-h">
        <h2 id="bio-h">Profile</h2>
        <dl className="grid">
          <div><dt className="muted">Born</dt><dd>{formatDateOnly(p.birth_date)} <span className="muted">({QUALITY[p.birth_date_quality]})</span></dd></div>
          <div><dt className="muted">Debut</dt><dd>{formatDateOnly(p.debut_date)}</dd></div>
          <div><dt className="muted">Height</dt><dd>{p.height_cm === null ? <span className="missing">not recorded</span> : `${formatStat(p.height_cm)} cm`}</dd></div>
          <div><dt className="muted">Weight</dt><dd>{p.weight_kg === null ? <span className="missing">not recorded</span> : `${formatStat(p.weight_kg)} kg`}</dd></div>
          <div><dt className="muted">Clubs</dt><dd>{p.clubs.map((c, i) => <span key={c.club_id}>{i ? ', ' : ''}<a href={withBase(base, `team/?id=${c.club_id}`)}>{c.name}</a></span>)}</dd></div>
          <div><dt className="muted">Career games (rows)</dt><dd className="tabular">{p.career_games}{p.career_counter_max !== null && p.career_counter_max !== p.career_games ? ` (source career counter: ${p.career_counter_max})` : ''}</dd></div>
          <div><dt className="muted">Identity</dt><dd>{p.identity_status}{p.aliases.length ? `; also known as ${p.aliases.join(', ')}` : ''}</dd></div>
        </dl>
        <p className="muted">Coverage: {p.coverage_note} Sources: {p.sources.map((s) => s.label + (s.note ? ` (${s.note})` : '')).join('; ')}. <a href={withBase(base, 'methodology/')}>Methodology</a>.</p>
      </section>
      <Forecast f={p.forecast} />
      <section aria-labelledby="career-h">
        <h2 id="career-h">Career statistics</h2>
        <StatTable caption="Career statistics" stats={p.career} />
      </section>
      <section aria-labelledby="seasons-h">
        <h2 id="seasons-h">Seasons</h2>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Season summary">
          <table>
            <caption>Season-by-season summary</caption>
            <thead><tr><th scope="col">Season</th><th scope="col">Clubs</th><th scope="col" className="num">Games</th>{(p.seasons[0]?.stats ?? []).map((s) => <th scope="col" className="num" key={s.stat}>{s.stat} per game</th>)}</tr></thead>
            <tbody>
              {seasons.map((s) => (
                <tr key={s.season}><th scope="row">{s.season}</th><td>{s.clubs.join(', ')}</td><td className="num">{s.games}</td>{s.stats.map((st) => <td className="num" key={st.stat}><Stat value={st.mean} digits={1} /></td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section aria-labelledby="log-h">
        <h2 id="log-h">Game log</h2>
        {seasons.length === 0 ? <p className="muted">No seasons recorded.</p> : (
          <>
            <div className="field">
              <label htmlFor="log-season">Season</label>
              <select id="log-season" value={season ?? ''} onChange={(e) => setSeason(Number(e.target.value))}>
                {seasons.map((s) => <option key={s.season} value={s.season}>{s.season}</option>)}
              </select>
            </div>
            <DataState state={games.state} retry={games.retry} what="game log">
              {(g) => <GameLog g={g} />}
            </DataState>
          </>
        )}
      </section>
    </div>
  );
}

function GameLog({ g }: { g: PlayerSeasonGames }) {
  const base = siteBase();
  const main = g.stat_columns.includes('disposals') ? 'disposals' : g.stat_columns[0];
  return (
    <div className="stack">
      {main ? (
        <LineChart
          id={`form-${g.season}`} title={`${g.season} form: ${main} by game`} description={`${main} in each ${g.season} game in order`}
          xLabel="Game" yLabel={main} series={[{ name: main, points: g.games.map((x, i) => ({ x: `G${i + 1}`, y: x.stats[g.stat_columns.indexOf(main)] ?? null })) }]}
        />
      ) : null}
      <div className="table-wrap" tabIndex={0} role="region" aria-label={`Scrollable table: ${`${g.season} game log`}`}>
        <table>
          <caption>{g.season} game log</caption>
          <thead><tr><th scope="col">Date</th><th scope="col">Stage</th><th scope="col">Opponent</th><th scope="col">Result</th>{g.stat_columns.map((c) => <th scope="col" className="num" key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {g.games.map((x) => (
              <tr key={x.match_id}>
                <th scope="row"><a href={withBase(base, `match/?id=${x.match_id.replaceAll(':', '__')}`)}>{formatDateOnly(x.match_date)}</a>{x.date_quality !== 'fixture_verified' && x.date_quality !== 'source' ? <span className="muted"> ({x.date_quality})</span> : null}</th>
                <td>{x.stage_label}</td>
                <td>{x.opponent_name ?? <span className="missing">not recorded</span>}</td>
                <td>{x.result ?? <span className="missing">not recorded</span>}</td>
                {g.stat_columns.map((c, i) => <td className="num" key={c}><Stat value={x.stats[i] ?? null} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
