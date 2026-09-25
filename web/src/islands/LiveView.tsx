import { useEffect, useRef, useState } from 'react';
import type { LiveIndex, LiveSnapshot } from '../lib/contracts';
import { isSafeKey, withBase } from '../lib/ids';
import { liveFreshness, POLL_MS } from '../lib/live';
import { teamScoreText } from '../lib/matches';
import { classifyError, getLoader, siteBase, useHeading, useUrlSearch, type ErrorKind } from './common/runtime';
import { announceReleaseUnavailable } from '../lib/prefs';
import { NoScriptNotice } from './common/NoScript';
import { Instant } from './common/Instant';
import { Stat } from './common/Stat';

export default function LiveView({ index, downloadHref }: { index: LiveIndex; downloadHref: string }) {
  const [search] = useUrlSearch();
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  const game = new URLSearchParams(search).get('match');
  const entry = game && isSafeKey(game) ? index.matches.find((m) => m.source_game_id === game) : undefined;
  const base = siteBase();
  useHeading(entry ? `Live: ${entry.label}` : null);
  if (!ready) return <NoScriptNotice what="live view" href={downloadHref} linkText="browse downloads" />;
  if (!game) {
    return index.matches.length ? (
      <ul className="list-plain">{index.matches.map((m) => <li key={m.source_game_id}><a href={withBase(base, `live/?match=${m.source_game_id}`)}>{m.label}</a> <span className="muted">{m.final ? 'final' : 'in progress at last snapshot'}</span></li>)}</ul>
    ) : <p data-state="empty">No live snapshots in this release.</p>;
  }
  if (!entry) return <div className="banner banner-info" data-state="notfound"><h2>Live match not found</h2><p>No live snapshot for that match in this release. <a href={withBase(base, 'live/')}>All live snapshots</a>.</p></div>;
  return <Poller resource={entry.resource} delivery={index.delivery} />;
}

function Poller({ resource, delivery }: { resource: string; delivery: string }) {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null);
  const [error, setError] = useState<ErrorKind | null>(null);
  const [checkedAt, setCheckedAt] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const snapRef = useRef<LiveSnapshot | null>(null);

  useEffect(() => {
    let stopped = false;
    const ac = new AbortController();
    const schedule = () => {
      if (timer.current) clearTimeout(timer.current);
      const s = snapRef.current;
      if (stopped || (s && s.final) || document.visibilityState !== 'visible') return;
      timer.current = setTimeout(tick, POLL_MS);
    };
    const tick = async () => {
      try {
        const loader = await getLoader();
        const s = await loader.load('live_snapshot', resource, { signal: ac.signal, fresh: true });
        if (stopped) return;
        // Never let an older snapshot replace a newer accepted one.
        const prev = snapRef.current;
        if (!prev || !prev.fetched_at || !s.fetched_at || Date.parse(s.fetched_at) >= Date.parse(prev.fetched_at)) {
          snapRef.current = s;
          setSnap(s);
        }
        setError(null);
      } catch (e) {
        if (stopped || (e as Error).name === 'AbortError') return;
        const c = classifyError(e);
        if (c.kind === 'release') announceReleaseUnavailable();
        setError(c.kind);
      }
      setCheckedAt(Date.now());
      schedule();
    };
    const onVis = () => {
      if (document.visibilityState === 'visible') void tick();
      else if (timer.current) clearTimeout(timer.current);
    };
    document.addEventListener('visibilitychange', onVis);
    void tick();
    return () => {
      stopped = true;
      ac.abort();
      if (timer.current) clearTimeout(timer.current);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [resource]);

  const fresh = snap ? liveFreshness(snap, checkedAt ?? Date.now()) : null;
  return (
    <div className="stack">
      <div data-testid="live-status" role="status" className={`banner ${fresh?.state === 'live' ? 'banner-ok' : fresh?.state === 'final' ? 'banner-info' : 'banner-stale'}`}>
        {snap ? (
          <>
            <p><strong>{fresh?.state === 'final' ? 'Final: match complete. Updates have stopped.' : fresh?.state === 'live' ? 'Live snapshot (updates every 90 seconds while this tab is visible).' : 'Delayed or stale: the last accepted snapshot is more than 5 minutes old.'}</strong></p>
            <p>Last accepted update: <Instant iso={snap.fetched_at} />. {checkedAt ? <>Last checked: <Instant iso={new Date(checkedAt).toISOString()} />.</> : null}</p>
          </>
        ) : error ? null : <p>Loading the last accepted snapshot…</p>}
        {error ? <p><strong>{error === 'release' ? 'A newer release is available.' : 'Disconnected: the latest snapshot could not be loaded.'}</strong> {snap ? 'Showing the last accepted snapshot.' : ''}</p> : null}
        <p className="muted">Delivery: {delivery}</p>
      </div>
      {snap ? <SnapshotBody s={snap} /> : null}
    </div>
  );
}

function SnapshotBody({ s }: { s: LiveSnapshot }) {
  const base = siteBase();
  const cols = s.reliable_fields;
  return (
    <>
      <div className="card">
        <p className="score">{s.home.name} {teamScoreText(s.home)} v {s.away.name} {teamScoreText(s.away)}</p>
        <p>Status: {s.status ?? 'not recorded'} · quarter {s.quarter ?? 'not recorded'}</p>
        {s.match_id ? <p><a href={withBase(base, `match/?id=${s.match_id.replaceAll(':', '__')}`)}>Match page</a></p> : null}
      </div>
      {s.unavailable_fields.length ? <p className="muted">Not shown because the feed is unreliable for them: {s.unavailable_fields.join(', ')}.</p> : null}
      {s.players.length ? (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Live player statistics">
          <table>
            <caption>Player statistics (reliable fields only)</caption>
            <thead><tr><th scope="col">Player</th>{cols.map((c) => <th scope="col" className="num" key={c}>{c}</th>)}</tr></thead>
            <tbody>{s.players.map((p) => <tr key={p.player_id}><th scope="row">{p.name}</th>{cols.map((c) => <td className="num" key={c}><Stat value={p.stats[c]} /></td>)}</tr>)}</tbody>
          </table>
        </div>
      ) : null}
      {s.timeline.length ? (
        <section aria-labelledby="tl-h"><h2 id="tl-h">Quarter timeline</h2>
          <ol>{s.timeline.map((t, i) => <li key={i}>{Object.entries(t).map(([k, v]) => `${k.replaceAll('_', ' ')}: ${v ?? 'not recorded'}`).join(' · ')}</li>)}</ol></section>
      ) : null}
      {s.reads.length ? <section aria-labelledby="reads-h"><h2 id="reads-h">Deterministic reads</h2><ul>{s.reads.map((r) => <li key={r}><strong>Rule-based read:</strong> {r}</li>)}</ul></section> : null}
      {s.anomalies.length ? <section aria-labelledby="an-h"><h2 id="an-h">Feed anomalies</h2><ul>{s.anomalies.map((a) => <li key={a}>{a}</li>)}</ul></section> : null}
    </>
  );
}
