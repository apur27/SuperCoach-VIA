import type { ListsIndex, ListsSeason } from '../lib/contracts';
import { parseUrlState, serializeUrlState, type StateSpec } from '../lib/filters';
import { encodeId, withBase } from '../lib/ids';
import { formatDateOnly } from '../lib/format';
import { DataState } from './common/DataState';
import { siteBase, useResource, useUrlSearch } from './common/runtime';

export default function ListsView({ index }: { index: ListsIndex }) {
  const [search, setSearch] = useUrlSearch();
  const spec = { season: { kind: 'int', min: 1850, max: 2100 }, club: { kind: 'text', maxLength: 80 }, kind: { kind: 'enum', values: ['drafts', 'contracts', 'schools'], default: 'drafts' } } as const satisfies StateSpec;
  const state = parseUrlState(search, spec);
  const seasons = [...index.seasons].sort((a, b) => b - a);
  const season = state.season !== undefined && seasons.includes(state.season) ? state.season : seasons[0];
  const path = season !== undefined ? index.resources[String(season)] ?? null : null;
  const data = useResource('lists_season', path);
  const update = (patch: Partial<typeof state>) => setSearch(serializeUrlState({ ...state, ...patch }, spec));
  if (!seasons.length) return <p data-state="empty">No list data in this release.</p>;
  return (
    <div>
      <div className="controls" role="group" aria-label="List filters">
        <div className="field"><label htmlFor="l-season">Year</label>
          <select id="l-season" value={season} onChange={(e) => update({ season: Number(e.target.value), club: undefined })}>{seasons.map((s) => <option key={s} value={s}>{s}</option>)}</select></div>
        <div className="field"><label htmlFor="l-kind">List</label>
          <select id="l-kind" value={state.kind ?? 'drafts'} onChange={(e) => update({ kind: e.target.value as typeof state.kind })}><option value="drafts">National and rookie drafts</option><option value="contracts">Contract observations</option><option value="schools">School classifications</option></select></div>
        {data.state.status === 'success' ? (
          <div className="field"><label htmlFor="l-club">Club</label>
            <select id="l-club" value={state.club ?? ''} onChange={(e) => update({ club: e.target.value || undefined })}>
              <option value="">All clubs</option>
              {[...new Set([...data.state.data.drafts.map((d) => d.club), ...data.state.data.contracts.map((c) => c.club)].filter((c): c is string => Boolean(c)))].sort().map((c) => <option key={c} value={c}>{c}</option>)}
            </select></div>
        ) : null}
      </div>
      <DataState state={data.state} retry={data.retry} what="lists">
        {(l) => <Lists l={l} kind={state.kind ?? 'drafts'} club={state.club} />}
      </DataState>
    </div>
  );
}

function Name({ name, id }: { name: string; id: string | null }) {
  return id ? <a href={withBase(siteBase(), `player/?id=${encodeId(id)}`)}>{name}</a> : <>{name}</>;
}

function Lists({ l, kind, club }: { l: ListsSeason; kind: string; club: string | undefined }) {
  const na = <span className="missing">not recorded</span>;
  return (
    <div className="stack">
      <div className="banner banner-stale"><p><strong>Source and freshness:</strong> {l.source_note}</p><p>Sources: {l.sources.map((s) => s.label).join('; ')}. Contract rows are dated observations, not guaranteed current contracts.</p></div>
      {kind === 'drafts' ? (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Drafts">
          <table><caption>{l.season} drafts</caption>
            <thead><tr><th scope="col">Player</th><th scope="col">Draft</th><th scope="col" className="num">Round</th><th scope="col" className="num">Pick</th><th scope="col">Club</th><th scope="col">Recruited from</th></tr></thead>
            <tbody>{l.drafts.filter((d) => !club || d.club === club).map((d, i) => <tr key={i}><th scope="row"><Name name={d.player_name} id={d.player_id} /></th><td>{d.event_type.replaceAll('_', ' ')}</td><td className="num">{d.draft_round ?? na}</td><td className="num">{d.pick ?? na}</td><td>{d.club ?? na}</td><td>{d.recruited_from ?? na}</td></tr>)}</tbody>
          </table></div>
      ) : kind === 'contracts' ? (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Contract observations">
          <table><caption>{l.season} contract observations</caption>
            <thead><tr><th scope="col">Player</th><th scope="col">Club</th><th scope="col" className="num">Contracted to</th><th scope="col">Free-agency category</th><th scope="col">Observed</th><th scope="col">Source type</th></tr></thead>
            <tbody>{l.contracts.filter((c) => !club || c.club === club).map((c, i) => <tr key={i}><th scope="row"><Name name={c.player_name} id={c.player_id} /></th><td>{c.club ?? na}</td><td className="num">{c.contract_end ?? na}</td><td>{c.fa_category ?? na}</td><td>{formatDateOnly(c.observed_at)}</td><td>{c.source_type}{c.notes ? ` — ${c.notes}` : ''}</td></tr>)}</tbody>
          </table></div>
      ) : (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: School classifications">
          <table><caption>{l.season} school classifications</caption>
            <thead><tr><th scope="col">Player</th><th scope="col" className="num">Draft year</th><th scope="col" className="num">Pick</th><th scope="col">School</th><th scope="col">Type</th><th scope="col">Confidence</th></tr></thead>
            <tbody>{l.schools.map((s, i) => <tr key={i}><th scope="row"><Name name={s.player_name} id={s.player_id} /></th><td className="num">{s.draft_year ?? na}</td><td className="num">{s.pick ?? na}</td><td>{s.school ?? na}</td><td>{s.school_type ?? na}</td><td>{s.confidence}</td></tr>)}</tbody>
          </table></div>
      )}
    </div>
  );
}
