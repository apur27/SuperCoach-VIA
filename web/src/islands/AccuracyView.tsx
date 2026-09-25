import type { AccuracyIndex, AccuracyReport, MetricBlock } from '../lib/contracts';
import { parseUrlState, serializeUrlState, type StateSpec } from '../lib/filters';
import { isSafeResourcePath, withBase } from '../lib/ids';
import { formatPercent, formatStat } from '../lib/format';
import { DataState } from './common/DataState';
import { siteBase, useResource, useUrlSearch } from './common/runtime';
import { BarChart } from './common/Charts';

const ORIGINS = ['prospective', 'replay', 'legacy_unknown'] as const;
const ORIGIN_LABEL = { prospective: 'Prospective (made before the match)', replay: 'Replay (reconstructed afterwards)', legacy_unknown: 'Legacy (origin unknown)' } as const;
const METRICS: [keyof MetricBlock, string, 'num' | 'pct' | 'int'][] = [
  ['n', 'Scored player-games', 'int'], ['mae', 'Mean absolute error', 'num'], ['rmse', 'RMSE', 'num'], ['bias', 'Bias (predicted − actual)', 'num'],
  ['median_ae', 'Median absolute error', 'num'], ['within_5', 'Within 5 disposals', 'pct'], ['within_10', 'Within 10 disposals', 'pct'],
];
const fmt = (v: number | null, k: 'num' | 'pct' | 'int') => (v === null ? <span className="missing">not recorded</span> : k === 'pct' ? formatPercent(v) : k === 'int' ? formatStat(v) : formatStat(v, 2));

export default function AccuracyView({ index, releaseId }: { index: AccuracyIndex; releaseId: string }) {
  const [search, setSearch] = useUrlSearch();
  const spec = { origin: { kind: 'enum', values: ORIGINS }, report: { kind: 'token' }, dim: { kind: 'token' } } as const satisfies StateSpec;
  const state = parseUrlState(search, spec);
  const origins = ORIGINS.filter((o) => index.reports.some((r) => r.origin === o));
  const origin = state.origin && origins.includes(state.origin) ? state.origin : origins[0];
  const reports = index.reports.filter((r) => r.origin === origin);
  const entry = reports.find((r) => `${r.model_id}-${r.season ?? 'all'}` === state.report) ?? reports.find((r) => r.model_id === index.champion_model_id) ?? reports[0];
  const data = useResource('accuracy_report', entry?.resource ?? null);
  const update = (patch: Partial<typeof state>) => setSearch(serializeUrlState({ ...state, ...patch }, spec));
  const base = siteBase();
  if (!origins.length) return <p data-state="empty">No accuracy reports in this release.</p>;
  return (
    <div>
      <div className="controls" role="group" aria-label="Accuracy controls">
        <fieldset>
          <legend>Prediction origin (never combined)</legend>
          <div className="segmented">
            {origins.map((o) => <button key={o} type="button" aria-pressed={o === origin} className={o === origin ? '' : 'secondary'} onClick={() => update({ origin: o, report: undefined, dim: undefined })}>{ORIGIN_LABEL[o]}</button>)}
          </div>
        </fieldset>
        {reports.length > 1 ? (
          <div className="field">
            <label htmlFor="acc-report">Report</label>
            <select id="acc-report" value={entry ? `${entry.model_id}-${entry.season ?? 'all'}` : ''} onChange={(e) => update({ report: e.target.value })}>
              {reports.map((r) => <option key={r.resource} value={`${r.model_id}-${r.season ?? 'all'}`}>{r.label}</option>)}
            </select>
          </div>
        ) : null}
      </div>
      <DataState state={data.state} retry={data.retry} what="accuracy report">
        {(r) => <Report r={r} dim={state.dim} setDim={(d) => update({ dim: d })} releaseId={releaseId} base={base} />}
      </DataState>
    </div>
  );
}

function Report({ r, dim, setDim, releaseId, base }: { r: AccuracyReport; dim: string | undefined; setDim: (d: string | undefined) => void; releaseId: string; base: string }) {
  const dims = [...new Set(r.cohorts.map((c) => c.dimension))];
  const d = dim && dims.includes(dim) ? dim : dims[0];
  const cohorts = r.cohorts.filter((c) => c.dimension === d);
  const p = r.populations;
  return (
    <div className="stack">
      <h2>{r.label}</h2>
      <p className="muted">Model <code>{r.model_id}</code> vs baseline <code>{r.baseline_id ?? 'none'}</code> · origin <strong>{r.origin}</strong> · season {r.season ?? 'all'}. {r.promotion}</p>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Headline metrics">
        <table>
          <caption>Headline accuracy: champion vs baseline (pooled over all scored player-games)</caption>
          <thead><tr><th scope="col">Metric</th><th scope="col" className="num">Model</th><th scope="col" className="num">Baseline</th></tr></thead>
          <tbody>
            {METRICS.map(([k, label, kind]) => <tr key={k}><th scope="row">{label}</th><td className="num">{fmt(r.headline[k], kind)}</td><td className="num">{r.baseline_headline ? fmt(r.baseline_headline[k], kind) : <span className="missing">no baseline</span>}</td></tr>)}
            <tr><th scope="row">Mean of per-round MAE (unweighted, for reference)</th><td className="num">{fmt(r.mean_of_rounds_mae, 'num')}</td><td className="num">—</td></tr>
          </tbody>
        </table>
      </div>
      <section aria-labelledby="pop-h">
        <h3 id="pop-h">Denominators and exclusions</h3>
        <p className="tabular">Intended {p.intended} · predicted {p.predicted} · joined to results {p.joined} · played {p.played} · missing {p.missing} · excluded {p.excluded}.</p>
        {Object.keys(p.exclusion_reasons).length ? <ul>{Object.entries(p.exclusion_reasons).map(([k, v]) => <li key={k}>{k.replaceAll('_', ' ')}: {v}</li>)}</ul> : null}
      </section>
      <section aria-labelledby="int-h">
        <h3 id="int-h">Intervals</h3>
        <p>{r.interval.available ? `${formatStat((r.interval.level ?? 0) * 100)}% intervals (${r.interval.method ?? 'method not recorded'}${r.interval.calibrated ? ', calibrated' : ''}). Held-out coverage ${formatPercent(r.interval_coverage)}; median width ${r.interval_median_width === null ? 'not recorded' : formatStat(r.interval_median_width, 1)} disposals.` : `No intervals: ${r.interval.reason ?? 'not available'}.`}</p>
      </section>
      <section aria-labelledby="coh-h">
        <h3 id="coh-h">Cohorts</h3>
        {dims.length ? (
          <>
            <div className="field">
              <label htmlFor="acc-dim">Cohort dimension</label>
              <select id="acc-dim" value={d} onChange={(e) => setDim(e.target.value)}>{dims.map((x) => <option key={x} value={x}>{x}</option>)}</select>
            </div>
            <BarChart id="acc-chart" title={`MAE by ${d}`} description={`Model mean absolute error for each ${d} cohort`} categoryLabel={d ?? 'Cohort'} valueLabel="Model MAE" data={cohorts.map((c) => ({ label: c.cohort, value: c.model.mae }))} />
            <div className="table-wrap" tabIndex={0} role="region" aria-label={`Scrollable table: ${`Cohorts by ${d}`}`}>
              <table>
                <caption>Accuracy by {d}</caption>
                <thead><tr><th scope="col">Cohort</th><th scope="col" className="num">n</th><th scope="col" className="num">Model MAE</th><th scope="col" className="num">Baseline MAE</th><th scope="col" className="num">Bias</th><th scope="col">Sample</th></tr></thead>
                <tbody>{cohorts.map((c) => <tr key={c.cohort}><th scope="row">{c.cohort}</th><td className="num">{c.model.n}</td><td className="num">{fmt(c.model.mae, 'num')}</td><td className="num">{c.baseline ? fmt(c.baseline.mae, 'num') : <span className="missing">none</span>}</td><td className="num">{fmt(c.model.bias, 'num')}</td><td>{c.sufficient ? 'Sufficient' : 'Too small to judge'}</td></tr>)}</tbody>
              </table>
            </div>
          </>
        ) : <p className="muted">No cohort breakdown.</p>}
      </section>
      {r.notes.length ? <ul>{r.notes.map((n) => <li key={n}>{n}</li>)}</ul> : null}
      {r.rows_resource && isSafeResourcePath(r.rows_resource) ? <p><a className="button secondary" href={withBase(base, `data/${releaseId}/${r.rows_resource}`)}>Download the scored rows behind these numbers (CSV, all rows)</a></p> : null}
    </div>
  );
}
