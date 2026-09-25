import { useEffect, useState } from 'react';
import type { BoxScoreRow, MatchDetail } from '../lib/contracts';
import { encodeId, isSafeKey, withBase } from '../lib/ids';
import { formatDateOnly, formatStat } from '../lib/format';
import { isReplay, resultLine, statusLabel, teamScoreText } from '../lib/matches';
import { DataState } from './common/DataState';
import { siteBase, useHeading, useResource, useUrlSearch } from './common/runtime';
import { NoScriptNotice } from './common/NoScript';
import { Stat } from './common/Stat';

export default function MatchView({ downloadHref }: { downloadHref: string }) {
  const [search] = useUrlSearch();
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  const raw = new URLSearchParams(search).get('id') ?? '';
  const key = raw.includes(':') ? encodeId(raw) : raw;
  const valid = key && isSafeKey(key);
  const detail = useResource('match_detail', ready && valid ? `matches/detail/${key}.json` : null);
  useHeading(detail.state.status === 'success' ? `${detail.state.data.summary.home.name} v ${detail.state.data.summary.away.name}, ${detail.state.data.summary.stage_label} ${detail.state.data.summary.season}` : null);
  const base = siteBase();
  if (!ready) return <NoScriptNotice what="match view" href={downloadHref} linkText="browse downloads" />;
  if (!valid) return <div className="banner banner-info"><h2>Match not found</h2><p>That is not a valid match ID. <a href={withBase(base, 'matches/')}>Browse matches</a>.</p></div>;
  return (
    <DataState state={detail.state} retry={detail.retry} what="match"
      notFound={<div className="banner banner-info" data-state="notfound"><h2>Match not found</h2><p>No match <code>{raw}</code> in this release. <a href={withBase(base, 'matches/')}>Browse matches</a>.</p></div>}>
      {(d) => <MatchBody d={d} />}
    </DataState>
  );
}

function Box({ rows, cols, caption }: { rows: BoxScoreRow[]; cols: string[]; caption: string }) {
  const base = siteBase();
  if (!rows.length) return <p className="muted">{caption}: no player statistics recorded.</p>;
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={`Scrollable table: ${caption}`}>
      <table>
        <caption>{caption}</caption>
        <thead><tr><th scope="col">Player</th>{cols.map((c) => <th scope="col" className="num" key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.player_id}><th scope="row"><a href={withBase(base, `player/?id=${encodeId(r.player_id)}`)}>{r.name}</a></th>{cols.map((c) => <td className="num" key={c}><Stat value={r.stats[c]} /></td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MatchBody({ d }: { d: MatchDetail }) {
  const s = d.summary;
  const base = siteBase();
  const q = (g: number | null, b: number | null) => (g === null || b === null ? <span className="missing">not recorded</span> : `${g}.${b} (${g * 6 + b})`);
  return (
    <div className="stack">
      <div className="card">
        <p><strong>{s.home.name}</strong> {s.status === 'scheduled' ? '' : teamScoreText(s.home)} v <strong>{s.away.name}</strong> {s.status === 'scheduled' ? '' : teamScoreText(s.away)}</p>
        <p>{resultLine(s)} · {statusLabel(s.status)} · {s.stage_label} ({s.stage_type}){isReplay(s) ? <span className="badge badge-muted">Replay #{s.replay_occurrence}</span> : null}</p>
        <p className="muted">{s.local_start ? `${s.local_start} local time` : formatDateOnly(s.match_date)} · {s.venue ?? 'venue not recorded'} · attendance {d.attendance === null ? 'not recorded' : formatStat(d.attendance)}</p>
        {s.status === 'postponed' ? <p className="banner banner-stale">This match was postponed. A rescheduled fixture, if any, appears separately in the <a href={withBase(base, `matches/?season=${s.season}&stage=${s.stage_id}`)}>{s.stage_label} list</a>.</p> : null}
        {isReplay(s) || s.stage_type === 'final' ? <p><a href={withBase(base, `matches/?season=${s.season}&stage=${s.stage_id}`)}>All matches in this stage (including replays)</a></p> : null}
      </div>
      <section aria-labelledby="q-h">
        <h2 id="q-h">Scores by quarter</h2>
        {d.quarters.length ? (
          <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Quarter scores">
            <table>
              <caption>Cumulative score by quarter</caption>
              <thead><tr><th scope="col">Quarter</th><th scope="col" className="num">{s.home.name}</th><th scope="col" className="num">{s.away.name}</th></tr></thead>
              <tbody>{d.quarters.map((x) => <tr key={x.quarter}><th scope="row">{x.quarter === 'final' ? 'Final' : x.quarter.toUpperCase()}</th><td className="num">{q(x.home_goals, x.home_behinds)}</td><td className="num">{q(x.away_goals, x.away_behinds)}</td></tr>)}</tbody>
            </table>
          </div>
        ) : <p className="muted">No quarter scores recorded.</p>}
      </section>
      <section aria-labelledby="box-h">
        <h2 id="box-h">Box scores</h2>
        <Box rows={d.home_players} cols={d.stat_columns} caption={`${s.home.name} players`} />
        <Box rows={d.away_players} cols={d.stat_columns} caption={`${s.away.name} players`} />
      </section>
      {d.live_snapshots.length ? (
        <section aria-labelledby="live-h">
          <h2 id="live-h">Live snapshots</h2>
          <ul>{d.live_snapshots.map((k) => { const g = /^live\/([^/]+)\//.exec(k)?.[1]; return g ? <li key={k}><a href={withBase(base, `live/?match=${g}`)}>Snapshot {g}</a></li> : null; })}</ul>
        </section>
      ) : null}
      <p className="muted">Sources: {d.sources.map((x) => x.label + (x.note ? ` (${x.note})` : '')).join('; ')}</p>
    </div>
  );
}
