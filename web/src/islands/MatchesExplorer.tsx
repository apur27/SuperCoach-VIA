import { useMemo } from 'react';
import type { MatchIndex } from '../lib/contracts';
import { parseUrlState, serializeUrlState, type StateSpec } from '../lib/filters';
import { DataState } from './common/DataState';
import { useResource, useUrlSearch } from './common/runtime';
import { MatchRowsTable } from './common/MatchRows';

const STATUSES = ['scheduled', 'in_progress', 'complete', 'postponed', 'cancelled', 'unknown'] as const;

export default function MatchesExplorer({ seasons }: { seasons: number[] }) {
  const [search, setSearch] = useUrlSearch();
  const spec = {
    season: { kind: 'int', min: 1850, max: 2100 },
    stage: { kind: 'token' },
    club: { kind: 'token' },
    status: { kind: 'enum', values: STATUSES },
  } as const satisfies StateSpec;
  const state = parseUrlState(search, spec);
  const season = state.season !== undefined && seasons.includes(state.season) ? state.season : seasons[0];
  const index = useResource('match_index', season !== undefined ? `matches/${season}/index.json` : null);
  const update = (patch: Partial<typeof state>) => setSearch(serializeUrlState({ ...state, ...patch }, spec));
  if (!seasons.length) return <p data-state="empty">No match seasons in this release.</p>;
  return (
    <div>
      <div className="controls" role="group" aria-label="Match filters">
        <div className="field">
          <label htmlFor="m-season">Season</label>
          <select id="m-season" value={season} onChange={(e) => update({ season: Number(e.target.value), stage: undefined })}>
            {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        {index.state.status === 'success' ? <Filters idx={index.state.data} state={state} update={update} /> : null}
      </div>
      <DataState state={index.state} retry={index.retry} what="matches" isEmpty={(d) => d.matches.length === 0} empty={<p data-state="empty">No matches recorded for {season}.</p>}>
        {(idx) => {
          const rows = idx.matches.filter((m) => (!state.stage || m.stage_id === state.stage) && (!state.club || m.home.club_id === state.club || m.away.club_id === state.club) && (!state.status || m.status === state.status))
            .sort((a, b) => a.stage_order - b.stage_order || (a.match_date ?? '').localeCompare(b.match_date ?? '') || a.replay_occurrence - b.replay_occurrence);
          return (
            <>
              <p role="status" aria-live="polite">{rows.length} of {idx.matches.length} matches shown</p>
              {rows.length ? <MatchRowsTable matches={rows} caption={`${idx.season} matches`} /> : <p data-state="empty">No matches match these filters.</p>}
              <p className="muted">A score of 0 is a recorded zero; “score not recorded” means the source has no score. Replays are listed after the original match.</p>
            </>
          );
        }}
      </DataState>
    </div>
  );
}

function Filters({ idx, state, update }: { idx: MatchIndex; state: { stage?: string; club?: string; status?: string }; update: (p: Record<string, string | undefined>) => void }) {
  const stages = useMemo(() => {
    const m = new Map<string, { label: string; order: number }>();
    for (const x of idx.matches) if (!m.has(x.stage_id)) m.set(x.stage_id, { label: x.stage_label.replace(/ \(replay\)$/, ''), order: x.stage_order });
    return [...m].sort((a, b) => a[1].order - b[1].order);
  }, [idx]);
  const clubs = useMemo(() => {
    const m = new Map<string, string>();
    for (const x of idx.matches) {
      m.set(x.home.club_id, x.home.name);
      m.set(x.away.club_id, x.away.name);
    }
    return [...m].sort((a, b) => a[1].localeCompare(b[1]));
  }, [idx]);
  return (
    <>
      <div className="field">
        <label htmlFor="m-stage">Stage</label>
        <select id="m-stage" value={stages.some(([id]) => id === state.stage) ? state.stage : ''} onChange={(e) => update({ stage: e.target.value || undefined })}>
          <option value="">All stages</option>
          {stages.map(([id, s]) => <option key={id} value={id}>{s.label}</option>)}
        </select>
      </div>
      <div className="field">
        <label htmlFor="m-club">Club</label>
        <select id="m-club" value={clubs.some(([id]) => id === state.club) ? state.club : ''} onChange={(e) => update({ club: e.target.value || undefined })}>
          <option value="">All clubs</option>
          {clubs.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
        </select>
      </div>
      <div className="field">
        <label htmlFor="m-status">Status</label>
        <select id="m-status" value={state.status ?? ''} onChange={(e) => update({ status: e.target.value || undefined })}>
          <option value="">Any status</option>
          {STATUSES.map((s) => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
        </select>
      </div>
    </>
  );
}
