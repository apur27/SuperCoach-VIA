import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { PlayerIndexEntry } from '../lib/contracts';
import { paginate, parseUrlState, serializeUrlState, type StateSpec } from '../lib/filters';
import { withBase } from '../lib/ids';
import { disambiguate, eraLabel, searchPlayers } from '../lib/search';
import { DataState } from './common/DataState';
import { siteBase, useResource, useUrlSearch } from './common/runtime';
import { Pagination } from './common/Pagination';
import { NoScriptNotice } from './common/NoScript';

const spec = {
  q: { kind: 'text', maxLength: 60 },
  status: { kind: 'enum', values: ['all', 'active', 'retired'], default: 'all' },
  club: { kind: 'text', maxLength: 80 },
  season: { kind: 'int', min: 1850, max: 2100 },
  page: { kind: 'int', min: 1, max: 100000, default: 1 },
  size: { kind: 'enum', values: ['25', '50', '100'], default: '25' },
} as const satisfies StateSpec;
type St = ReturnType<typeof parseUrlState<typeof spec>>;

export default function PlayerSearch({ downloadHref }: { downloadHref: string }) {
  const [search, setSearch] = useUrlSearch();
  const state = parseUrlState(search, spec);
  const wantsIndex = Boolean(state.q || state.club || state.season || (state.status && state.status !== 'all'));
  const [activated, setActivated] = useState(false);
  const index = useResource('player_index', activated || wantsIndex ? 'players/index.json' : null);
  const [qInput, setQInput] = useState(state.q ?? '');
  const [active, setActive] = useState(-1);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const listId = useId();
  useEffect(() => setQInput(state.q ?? ''), [state.q]);

  const update = (patch: Partial<St>, mode: 'push' | 'replace' = 'push') => {
    const filterChanged = Object.keys(patch).some((k) => k !== 'page');
    setSearch(serializeUrlState({ ...state, ...(filterChanged ? { page: 1 } : {}), ...patch }, spec), mode);
    setActive(-1);
  };
  const onInput = (v: string) => {
    setQInput(v);
    setActivated(true);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => update({ q: v.trim() || undefined }, 'replace'), 150);
  };

  const players = index.state.status === 'success' ? index.state.data.players : [];
  const clubs = useMemo(() => [...new Set(players.flatMap((p) => p.clubs))].sort(), [players]);
  const results = useMemo(() => {
    if (!players.length) return [];
    return searchPlayers(players, state.q ?? '', {
      ...(state.status === 'active' ? { active: true } : state.status === 'retired' ? { active: false } : {}),
      ...(state.club ? { club: state.club } : {}),
      ...(state.season ? { season: state.season } : {}),
    });
  }, [players, state.q, state.status, state.club, state.season]);
  const labels = useMemo(() => disambiguate(results), [results]);
  const size = Number(state.size ?? '25');
  const pageData = paginate(results, state.page ?? 1, size);
  const base = siteBase();
  const go = (e: PlayerIndexEntry) => window.location.assign(withBase(base, `player/?id=${e.key}`));

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, pageData.items.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === 'Enter') {
      const hit = pageData.items[active];
      if (hit) {
        e.preventDefault();
        go(hit);
      }
    } else if (e.key === 'Escape') setActive(-1);
  };
  const hasQuery = Boolean(state.q || state.club || state.season || (state.status && state.status !== 'all'));

  return (
    <div>
      <NoScriptNotice what="player search" href={downloadHref} linkText="download the full player directory (CSV)" />
      <form className="controls" role="search" onSubmit={(e) => e.preventDefault()}>
        <div className="field">
          <label htmlFor="player-q">Search all players</label>
          <input
            id="player-q" type="search" role="combobox" aria-autocomplete="list" aria-expanded={pageData.items.length > 0 && hasQuery}
            aria-controls={listId} aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined} aria-describedby={`${listId}-hint`}
            value={qInput} maxLength={60} autoComplete="off" onFocus={() => setActivated(true)} onChange={(e) => onInput(e.target.value)} onKeyDown={onKey}
          />
          <small id={`${listId}-hint`} className="muted">Accents and case are ignored. Use arrow keys and Enter to open a player.</small>
        </div>
        <div className="field">
          <label htmlFor="player-status">Status</label>
          <select id="player-status" value={state.status ?? 'all'} onChange={(e) => { setActivated(true); update({ status: e.target.value as St['status'] }); }}>
            <option value="all">All players</option><option value="active">Active</option><option value="retired">Not active</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="player-club">Club</label>
          <select id="player-club" value={state.club ?? ''} onFocus={() => setActivated(true)} onChange={(e) => update({ club: e.target.value || undefined })}>
            <option value="">All clubs</option>
            {clubs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="player-season">Played in season</label>
          <input id="player-season" type="text" inputMode="numeric" pattern="[0-9]{4}" size={6} defaultValue={state.season ?? ''} key={state.season ?? 'none'}
            onFocus={() => setActivated(true)} onBlur={(e) => { const v = Number(e.target.value); update({ season: /^\d{4}$/.test(e.target.value) ? v : undefined }); }} />
        </div>
      </form>
      <DataState state={index.state} retry={index.retry} what="player directory" idle={<p className="muted">Start typing to search every player in the release. The directory loads when you start a search.</p>}>
        {() => (
          <>
            <p role="status" aria-live="polite" className="tabular">
              {hasQuery ? `${results.length} ${results.length === 1 ? 'player' : 'players'} found` : 'Type a name or choose a filter to search.'}
            </p>
            {hasQuery && results.length > 0 ? (
              <>
                <ul className="combo-results" role="listbox" id={listId} aria-label="Matching players">
                  {pageData.items.map((p, i) => (
                    <li key={p.id} id={`${listId}-${i}`} role="option" aria-selected={i === active} onClick={() => go(p)}>
                      <div className="option-body">
                        <span className="name">{p.name}</span>
                        <span className="muted">{labels.get(p.id) ?? `${p.clubs.join(', ') || 'club unknown'}, ${eraLabel(p)}`}</span>
                        <span className="muted tabular">{p.games} games{p.active ? ' · active' : ''}</span>
                      </div>
                    </li>
                  ))}
                </ul>
                <Pagination label="Player results" page={pageData.page} pages={pageData.pages} total={pageData.total} size={state.size ?? '25'} onPage={(p) => update({ page: p })} onSize={(s) => update({ size: s as St['size'] })} />
              </>
            ) : hasQuery ? <p data-state="empty">No players match. Check spelling or clear a filter.</p> : null}
          </>
        )}
      </DataState>
    </div>
  );
}
