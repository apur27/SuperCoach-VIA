import type { HistoryIndex, HistoryRow } from '../lib/contracts';
import { parseUrlState, serializeUrlState, sortRows, type SortDir, type StateSpec } from '../lib/filters';
import { encodeId, withBase } from '../lib/ids';
import { formatPercent, formatStat } from '../lib/format';
import { DataState } from './common/DataState';
import { siteBase, useResource, useUrlSearch } from './common/runtime';
import { SortHeader } from './common/SortHeader';
import { BarChart } from './common/Charts';

const SORTS = ['rank', 'name', 'value', 'games', 'coverage'] as const;
const key: Record<(typeof SORTS)[number], (r: HistoryRow) => number | string | null> = {
  rank: (r) => r.rank, name: (r) => r.name, value: (r) => r.value, games: (r) => r.observed_games, coverage: (r) => r.coverage,
};
const SCOPE_LABEL = { career: 'Career', single_season: 'Single season', ranking: 'Ranking' } as const;

export default function HistoryExplorer({ index }: { index: HistoryIndex }) {
  const [search, setSearch] = useUrlSearch();
  const spec = {
    scope: { kind: 'enum', values: ['career', 'single_season', 'ranking'] },
    category: { kind: 'token' },
    era: { kind: 'token' },
    sort: { kind: 'enum', values: SORTS },
    dir: { kind: 'enum', values: ['asc', 'desc'], default: 'asc' },
  } as const satisfies StateSpec;
  const state = parseUrlState(search, spec);
  const scopes = [...new Set(index.tables.map((t) => t.scope))];
  const scope = state.scope && scopes.includes(state.scope) ? state.scope : scopes[0];
  const tables = index.tables.filter((t) => t.scope === scope);
  const table = tables.find((t) => t.category === state.category) ?? tables[0];
  const era = table && state.era && table.eras.includes(state.era) ? state.era : table?.eras[0];
  const path = table && era ? table.resources[era] ?? null : null;
  const data = useResource('history_table', path);
  const update = (patch: Partial<typeof state>) => setSearch(serializeUrlState({ ...state, ...patch }, spec));
  const base = siteBase();
  const dir: SortDir = state.dir ?? 'asc';
  if (!table) return <p data-state="empty">No history tables in this release.</p>;
  return (
    <div>
      <div className="controls" role="group" aria-label="History controls">
        <fieldset>
          <legend>Scope</legend>
          <div className="segmented">
            {scopes.map((s) => <button key={s} type="button" className={s === scope ? '' : 'secondary'} aria-pressed={s === scope} onClick={() => update({ scope: s, category: undefined, era: undefined })}>{SCOPE_LABEL[s]}</button>)}
          </div>
        </fieldset>
        <div className="field">
          <label htmlFor="h-cat">Category</label>
          <select id="h-cat" value={table.category} onChange={(e) => update({ category: e.target.value, era: undefined })}>{tables.map((t) => <option key={t.category} value={t.category}>{t.title}</option>)}</select>
        </div>
        <div className="field">
          <label htmlFor="h-era">Era</label>
          <select id="h-era" value={era} onChange={(e) => update({ era: e.target.value })}>{table.eras.map((e) => <option key={e} value={e}>{e}</option>)}</select>
        </div>
      </div>
      <DataState state={data.state} retry={data.retry} what="history table" isEmpty={(t) => t.rows.length === 0}>
        {(t) => {
          const rows = sortRows(t.rows, key[state.sort ?? 'rank'], state.sort ? dir : 'asc');
          const onSort = (column: string, d: SortDir) => update({ sort: column as (typeof SORTS)[number], dir: d });
          return (
            <div className="stack">
              <p className="muted">Method: {t.method} (version {t.method_version}). {t.coverage_note}</p>
              {t.warning ? <div className="banner banner-stale" data-testid="history-warning"><p><strong>Historical methodology warning.</strong> {t.warning}</p></div> : null}
              <BarChart id="hist-chart" title={`${t.title}: top ${Math.min(10, t.rows.length)} (${t.era})`} description={`${t.rows[0]?.value_label ?? 'value'} by player`} categoryLabel="Player" valueLabel={t.rows[0]?.value_label ?? 'Value'}
                data={t.rows.slice(0, 10).map((r) => ({ label: r.name, value: r.value }))} />
              <div className="table-wrap" tabIndex={0} role="region" aria-label={`Scrollable table: ${t.title}`}>
                <table>
                  <caption>{t.title} — {t.era}</caption>
                  <thead>
                    <tr>
                      <SortHeader label="Rank" column="rank" sort={state.sort} dir={dir} onSort={onSort} numeric />
                      <SortHeader label="Player" column="name" sort={state.sort} dir={dir} onSort={onSort} />
                      <th scope="col" className="col-optional">Clubs</th>
                      <th scope="col" className="col-optional">Seasons</th>
                      <SortHeader label={t.rows[0]?.value_label ?? 'Value'} column="value" sort={state.sort} dir={dir} onSort={onSort} numeric />
                      <SortHeader label="Games with data" column="games" sort={state.sort} dir={dir} onSort={onSort} numeric className="col-optional" />
                      <SortHeader label="Coverage" column="coverage" sort={state.sort} dir={dir} onSort={onSort} numeric />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={`${r.rank}-${r.name}`}>
                        <td className="num">{r.rank}</td>
                        <th scope="row">{r.player_id ? <a href={withBase(base, `player/?id=${encodeId(r.player_id)}`)}>{r.name}</a> : r.name}</th>
                        <td className="col-optional">{r.clubs.join(', ')}</td>
                        <td className="col-optional">{r.seasons ?? <span className="missing">not recorded</span>}</td>
                        <td className="num">{formatStat(r.value, r.value % 1 ? 1 : 0)}</td>
                        <td className="num col-optional">{r.observed_games === null ? <span className="missing">not recorded</span> : `${r.observed_games}${r.eligible_games !== null ? ` of ${r.eligible_games}` : ''}`}</td>
                        <td className="num">{formatPercent(r.coverage)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        }}
      </DataState>
    </div>
  );
}
