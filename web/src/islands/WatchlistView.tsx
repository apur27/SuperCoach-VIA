import { useEffect, useRef, useState } from 'react';
import type { PlayerDetail } from '../lib/contracts';
import { encodeId, withBase } from '../lib/ids';
import { formatStat } from '../lib/format';
import { parseWatchlistImport, MAX_IMPORT_BYTES } from '../lib/watchlist';
import { classifyError, getLoader, siteBase, type ErrorKind } from './common/runtime';
import { useWatchlist } from './common/watch';
import { NoScriptNotice } from './common/NoScript';
import { downloadText } from './common/download';
import { Stat } from './common/Stat';
import { expandStats } from '../lib/stats';

type Row = { state: 'loading' } | { state: 'ok'; p: PlayerDetail } | { state: 'error'; kind: ErrorKind };

export default function WatchlistView({ downloadHref }: { downloadHref: string }) {
  const { ids, persistent, store } = useWatchlist();
  const [ready, setReady] = useState(false);
  const [rows, setRows] = useState<Record<string, Row>>({});
  const [message, setMessage] = useState<{ kind: 'error' | 'ok'; text: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => setReady(true), []);

  useEffect(() => {
    let cancelled = false;
    const todo = ids.filter((id) => !rows[id]);
    if (!todo.length) return undefined;
    setRows((r) => ({ ...r, ...Object.fromEntries(todo.map((id) => [id, { state: 'loading' } as Row])) }));
    (async () => {
      const loader = await getLoader().catch(() => null);
      const queue = [...todo];
      const worker = async () => {
        for (let id = queue.shift(); id; id = queue.shift()) {
          try {
            if (!loader) throw new Error('no loader');
            const p = await loader.load('player_detail', `players/${encodeId(id)}.json`);
            if (!cancelled) setRows((r) => ({ ...r, [id]: { state: 'ok', p } }));
          } catch (e) {
            if (!cancelled) setRows((r) => ({ ...r, [id]: { state: 'error', kind: classifyError(e).kind } }));
          }
        }
      };
      await Promise.all(Array.from({ length: 6 }, worker));
    })();
    return () => {
      cancelled = true;
    };
  }, [ids.join(',')]);

  const base = siteBase();
  if (!ready) return <NoScriptNotice what="watchlist (stored in your browser)" href={downloadHref} linkText="download all player data (CSV)" />;

  const onImport = async (file: File | undefined) => {
    if (!file) return;
    if (file.size > MAX_IMPORT_BYTES) {
      setMessage({ kind: 'error', text: 'File is larger than 100 KiB.' });
      return;
    }
    const res = parseWatchlistImport(await file.text());
    if (!res.ok) setMessage({ kind: 'error', text: res.error });
    else {
      store?.replace(res.ids);
      setMessage({ kind: 'ok', text: `Imported ${res.ids.length} players.` });
    }
    if (fileRef.current) fileRef.current.value = '';
  };

  return (
    <div className="stack">
      {!persistent ? <p className="banner banner-stale">Browser storage is unavailable, so your watchlist is saved only for this page visit. Export it to keep a copy.</p> : <p className="muted">Saved only in this browser (no account, nothing sent to a server). Up to 100 players.</p>}
      {message ? <p className={`banner ${message.kind === 'error' ? 'banner-error' : 'banner-ok'}`} role={message.kind === 'error' ? 'alert' : 'status'}>{message.text}</p> : null}
      {ids.length === 0 ? (
        <p data-state="empty">Your watchlist is empty. Add players from a <a href={withBase(base, 'players/')}>player page</a> or <a href={withBase(base, 'predictions/')}>Predictions</a>.</p>
      ) : (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Watchlist">
          <table>
            <caption>Watched players ({ids.length})</caption>
            <thead><tr><th scope="col">Player</th><th scope="col" className="num">Latest season disposals per game</th><th scope="col">Current forecast</th><th scope="col">Remove</th></tr></thead>
            <tbody>
              {ids.map((id) => {
                const r = rows[id] ?? { state: 'loading' };
                const name = r.state === 'ok' ? r.p.name : id;
                const latest = r.state === 'ok' ? [...r.p.seasons].sort((a, b) => b.season - a.season)[0] : undefined;
                return (
                  <tr key={id}>
                    <th scope="row">{r.state === 'ok' ? <a href={withBase(base, `player/?id=${encodeId(id)}`)}>{r.p.name}</a> : r.state === 'loading' ? <span>{id} (loading)</span> : <span className="missing">{id}: {r.kind === 'notfound' ? 'not in this release' : 'could not load'}</span>}</th>
                    <td className="num">{latest ? <>{latest.season}: <Stat value={r.state === 'ok' ? expandStats(r.p.stat_names, latest.stats, latest.games).find((s) => s.stat === 'disposals')?.mean : null} digits={1} /></> : <span className="missing">not recorded</span>}</td>
                    <td>{r.state === 'ok' && r.p.forecast ? `${formatStat(r.p.forecast.predicted_disposals, 1)} disposals v ${r.p.forecast.opponent_name ?? '?'}` : <span className="muted">none</span>}</td>
                    <td><button type="button" className="secondary small" aria-label={`Remove ${name}`} onClick={() => store?.remove(id)}>Remove</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="cluster">
        <button type="button" className="secondary" disabled={!ids.length} onClick={() => downloadText('supercoach-via-watchlist.json', store?.exportJson() ?? '', 'application/json')}>Export watchlist (JSON)</button>
        <button type="button" className="secondary" disabled={!ids.length} onClick={() => { if (window.confirm('Remove every player from your watchlist?')) store?.clear(); }}>Clear watchlist</button>
      </div>
      <div className="field">
        <label htmlFor="wl-import">Import watchlist JSON</label>
        <input id="wl-import" ref={fileRef} type="file" accept="application/json,.json" onChange={(e) => void onImport(e.target.files?.[0])} />
        <small className="muted">A file exported from this page (max 100 KiB). Importing replaces the current list.</small>
      </div>
    </div>
  );
}
