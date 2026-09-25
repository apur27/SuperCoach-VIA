import { useEffect, useMemo, useRef, useState } from 'react';
import type { PredictionRow, PredictionSet } from '../lib/contracts';
import { parseUrlState, serializeUrlState, sortRows, type SortDir, type StateSpec } from '../lib/filters';
import { encodeId, withBase } from '../lib/ids';
import { formatStat } from '../lib/format';
import { reasonText } from '../lib/matches';
import { normalizeText } from '../lib/search';
import { toCsv } from '../lib/csv';
import { DataState } from './common/DataState';
import { useResource, useUrlSearch, siteBase } from './common/runtime';
import { SortHeader } from './common/SortHeader';
import { Stat } from './common/Stat';
import { Instant } from './common/Instant';
import { WatchButton } from './common/watch';
import { NoScriptNotice } from './common/NoScript';
import { downloadText } from './common/download';

const SORTS = ['player', 'club', 'opponent', 'kickoff', 'predicted', 'interval', 'recent', 'history'] as const;
const spec = {
  stage: { kind: 'token' },
  team: { kind: 'token' },
  q: { kind: 'text', maxLength: 60 },
  sel: { kind: 'enum', values: ['all', 'confirmed', 'unconfirmed'], default: 'all' },
  sort: { kind: 'enum', values: SORTS },
  dir: { kind: 'enum', values: ['asc', 'desc'], default: 'asc' },
} as const satisfies StateSpec;

const sortKey: Record<(typeof SORTS)[number], (r: PredictionRow) => number | string | null> = {
  player: (r) => r.player_name,
  club: (r) => r.club_name,
  opponent: (r) => r.opponent_name,
  kickoff: (r) => r.scheduled_at,
  predicted: (r) => r.predicted_disposals,
  interval: (r) => (r.interval_low === null || r.interval_high === null ? null : r.interval_high - r.interval_low),
  recent: (r) => r.recent_mean_5,
  history: (r) => r.history_games,
};

export default function PredictionsExplorer({ downloadHref }: { downloadHref: string }) {
  const [search, setSearch] = useUrlSearch();
  const state = parseUrlState(search, spec);
  const index = useResource('prediction_index', 'predictions/index.json');
  const sets = index.state.status === 'success' ? index.state.data.sets : [];
  const validStage = sets.find((s) => s.stage_id === state.stage);
  const currentPath = index.state.status === 'success' ? index.state.data.current : null;
  const setPath = validStage?.resource ?? currentPath ?? sets[0]?.resource ?? null;
  const set = useResource('prediction_set', index.state.status === 'success' ? setPath : null);
  const [compare, setCompare] = useState<string[]>([]);
  const [qInput, setQInput] = useState(state.q ?? '');
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => setQInput(state.q ?? ''), [state.q]);

  const update = (patch: Partial<typeof state>, mode: 'push' | 'replace' = 'push') =>
    setSearch(serializeUrlState({ ...state, ...patch }, spec), mode);
  const onQuery = (v: string) => {
    setQInput(v);
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => update({ q: v.trim() || undefined }, 'replace'), 150);
  };

  return (
    <div>
      <NoScriptNotice what="predictions table" href={downloadHref} linkText="download all prediction rows (CSV)" />
      <DataState state={index.state} retry={index.retry} what="prediction index">
        {(idx) => {
          if (idx.status === 'unavailable' || idx.sets.length === 0 || !setPath) {
            return (
              <div className="banner banner-info" data-testid="set-status">
                <p><strong>No current forecast.</strong> {reasonText(idx.reason)}</p>
                <p>Past results remain available under <a href={withBase(siteBase(), 'matches/')}>Matches</a> and <a href={withBase(siteBase(), 'accuracy/')}>Accuracy</a>.</p>
              </div>
            );
          }
          return (
            <>
              <div className="controls" role="group" aria-label="Prediction filters">
                <div className="field">
                  <label htmlFor="pred-set">Forecast set</label>
                  <select id="pred-set" value={validStage?.stage_id ?? idx.sets.find((s) => s.resource === setPath)?.stage_id ?? ''} onChange={(e) => update({ stage: e.target.value, team: undefined })}>
                    {idx.sets.map((s) => <option key={s.resource} value={s.stage_id}>{s.season} {s.stage_label} ({s.status}, {s.rows} rows)</option>)}
                  </select>
                </div>
                <PredictionFilters set={set.state.status === 'success' ? set.state.data : null} state={state} update={update} qInput={qInput} onQuery={onQuery} />
              </div>
              <DataState state={set.state} retry={set.retry} what="predictions">
                {(ps) => <PredictionTable ps={ps} state={state} update={update} compare={compare} setCompare={setCompare} downloadHref={downloadHref} />}
              </DataState>
            </>
          );
        }}
      </DataState>
    </div>
  );
}

type St = ReturnType<typeof parseUrlState<typeof spec>>;
type Upd = (patch: Partial<St>, mode?: 'push' | 'replace') => void;

function PredictionFilters({ set, state, update, qInput, onQuery }: { set: PredictionSet | null; state: St; update: Upd; qInput: string; onQuery: (v: string) => void }) {
  const clubs = useMemo(() => {
    const m = new Map<string, string>();
    for (const r of set?.rows ?? []) m.set(r.club_id, r.club_name);
    return [...m].sort((a, b) => a[1].localeCompare(b[1]));
  }, [set]);
  return (
    <>
      <div className="field">
        <label htmlFor="pred-team">Team</label>
        <select id="pred-team" value={clubs.some(([id]) => id === state.team) ? state.team : ''} onChange={(e) => update({ team: e.target.value || undefined })}>
          <option value="">All teams</option>
          {clubs.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
        </select>
      </div>
      <div className="field">
        <label htmlFor="pred-q">Player name</label>
        <input id="pred-q" type="search" value={qInput} maxLength={60} autoComplete="off" onChange={(e) => onQuery(e.target.value)} />
      </div>
      <div className="field">
        <label htmlFor="pred-sel">Selection</label>
        <select id="pred-sel" value={state.sel ?? 'all'} onChange={(e) => update({ sel: e.target.value as St['sel'] })}>
          <option value="all">All</option>
          <option value="confirmed">Confirmed in team</option>
          <option value="unconfirmed">Not confirmed</option>
        </select>
      </div>
    </>
  );
}

function PredictionTable({ ps, state, update, compare, setCompare, downloadHref }: { ps: PredictionSet; state: St; update: Upd; compare: string[]; setCompare: (c: string[]) => void; downloadHref: string }) {
  const base = siteBase();
  const q = normalizeText(state.q ?? '');
  const teamValid = ps.rows.some((r) => r.club_id === state.team);
  const rows = ps.rows.filter(
    (r) => (!teamValid || r.club_id === state.team) && (!q || normalizeText(r.player_name).includes(q)) && (state.sel === 'all' || !state.sel || r.selection_status === state.sel),
  );
  const dir: SortDir = state.dir ?? 'asc';
  const sorted = state.sort ? sortRows(rows, sortKey[state.sort], dir) : rows;
  const showInterval = ps.interval.available;
  const onSort = (column: string, d: SortDir) => update({ sort: column as St['sort'], dir: d });
  const toggleCompare = (id: string, on: boolean) => setCompare(on ? [...compare, id].slice(0, 4) : compare.filter((x) => x !== id));
  const exportFiltered = () => {
    const header = ['player_id', 'player_name', 'club', 'opponent', 'stage', 'scheduled_at', 'predicted_disposals', 'interval_low', 'interval_high', 'recent_mean_5', 'history_games', 'selection_status'];
    downloadText(`predictions-${ps.season}-${ps.stage_id}-filtered.csv`, toCsv(header, sorted.map((r) => [r.player_id, r.player_name, r.club_name, r.opponent_name, r.stage_label, r.scheduled_at, r.predicted_disposals, r.interval_low, r.interval_high, r.recent_mean_5, r.history_games, r.selection_status])), 'text/csv');
  };
  return (
    <div className="stack">
      <div data-testid="set-status">
        {ps.status === 'available' ? (
          <p className="banner banner-ok"><strong>{ps.stage_label} {ps.season}: forecast available.</strong> Units: {ps.units} per player-game. Cutoff <Instant iso={ps.forecast_cutoff} />; generated <Instant iso={ps.generated_at} />.</p>
        ) : ps.status === 'expired' ? (
          <div className="banner banner-stale"><p><strong>This forecast has expired.</strong> {reasonText(ps.reason ?? 'expired')} Shown for reference only.</p></div>
        ) : (
          <div className="banner banner-info"><p><strong>Forecast unavailable.</strong> {reasonText(ps.reason)}</p></div>
        )}
        <p className="muted">
          Model: {ps.model ? `${ps.model.name} (${ps.model.kind})` : 'not recorded'}.{' '}
          {showInterval ? `Intervals: ${formatStat((ps.interval.level ?? 0) * 100)}% ${ps.interval.method ?? ''}${ps.interval.calibrated ? ', calibrated' : ', not calibrated'}.` : `No interval: ${ps.interval.reason ?? 'not available for this set'}.`}
        </p>
      </div>
      <p role="status" aria-live="polite" className="tabular">{sorted.length} of {ps.rows.length} rows shown</p>
      <div className="cluster">
        <a className="button secondary" href={downloadHref}>Download all rows (CSV, release as-of)</a>
        <button type="button" className="secondary" onClick={exportFiltered}>Download filtered rows (CSV, {sorted.length})</button>
        <button type="button" className="secondary" onClick={() => void navigator.clipboard?.writeText(window.location.href)}>Copy link to this view</button>
        {compare.length > 0 ? (
          <a className="button" href={withBase(base, `compare/?players=${compare.map(encodeId).join(',')}`)}>Compare selected ({compare.length})</a>
        ) : null}
      </div>
      {sorted.length === 0 ? <p data-state="empty">No predictions match these filters.</p> : (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Predictions table">
          <table>
            <caption>Predicted disposals — {ps.stage_label} {ps.season}</caption>
            <thead>
              <tr>
                <SortHeader label="Player" column="player" sort={state.sort} dir={dir} onSort={onSort} />
                <SortHeader label="Club" column="club" sort={state.sort} dir={dir} onSort={onSort} className="col-optional" />
                <SortHeader label="Opponent" column="opponent" sort={state.sort} dir={dir} onSort={onSort} className="col-optional" />
                <SortHeader label="Kickoff" column="kickoff" sort={state.sort} dir={dir} onSort={onSort} className="col-optional" />
                <SortHeader label="Predicted" column="predicted" sort={state.sort} dir={dir} onSort={onSort} numeric />
                {showInterval ? <SortHeader label="Interval (width)" column="interval" sort={state.sort} dir={dir} onSort={onSort} numeric /> : null}
                <SortHeader label="Recent mean (5)" column="recent" sort={state.sort} dir={dir} onSort={onSort} numeric className="col-optional" />
                <SortHeader label="History games" column="history" sort={state.sort} dir={dir} onSort={onSort} numeric className="col-optional" />
                <th scope="col">Selection</th>
                <th scope="col">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.prediction_id}>
                  <th scope="row"><a href={withBase(base, `player/?id=${encodeId(r.player_id)}`)}>{r.player_name}</a>{r.warnings.length ? <span className="muted"> ({r.warnings.join('; ')})</span> : null}</th>
                  <td data-col="club" className="col-optional">{r.club_name}</td>
                  <td className="col-optional">{r.opponent_name ?? <span className="missing">not recorded</span>}</td>
                  <td className="col-optional">{r.scheduled_at ? <Instant iso={r.scheduled_at} /> : (r.scheduled_local ?? <span className="missing">not recorded</span>)}</td>
                  <td className="num" data-col="predicted">{formatStat(r.predicted_disposals, 1)}</td>
                  {showInterval ? (
                    <td className="num">{r.interval_low === null || r.interval_high === null ? <span className="missing">not available</span> : `${formatStat(r.interval_low, 1)}–${formatStat(r.interval_high, 1)}`}</td>
                  ) : null}
                  <td className="num col-optional"><Stat value={r.recent_mean_5} digits={1} /></td>
                  <td className="num col-optional">{r.history_games}</td>
                  <td>{r.selection_status === 'confirmed' ? 'Confirmed' : 'Not confirmed'}</td>
                  <td>
                    <div className="cluster">
                      <WatchButton id={r.player_id} name={r.player_name} compact />
                      <label className="cluster">
                        <input type="checkbox" checked={compare.includes(r.player_id)} disabled={!compare.includes(r.player_id) && compare.length >= 4} onChange={(e) => toggleCompare(r.player_id, e.target.checked)} aria-label={`Compare ${r.player_name}`} />
                        <span aria-hidden="true">Compare</span>
                      </label>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {ps.omissions.length ? (
        <details>
          <summary>Players without a prediction ({ps.omissions.reduce((a, o) => a + o.count, 0)})</summary>
          <ul>{ps.omissions.map((o) => <li key={o.reason}>{o.reason.replaceAll('_', ' ')}: {o.count}{o.detail ? ` — ${o.detail}` : ''}</li>)}</ul>
        </details>
      ) : null}
    </div>
  );
}
