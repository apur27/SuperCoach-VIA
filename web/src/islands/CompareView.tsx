import { useEffect, useState } from 'react';
import type { PlayerDetail } from '../lib/contracts';
import { encodeId, parsePlayerIdParam, withBase } from '../lib/ids';
import { formatPercent, formatStat } from '../lib/format';
import { toCsv } from '../lib/csv';
import { classifyError, getLoader, siteBase, useUrlSearch, type ErrorKind } from './common/runtime';
import { announceReleaseUnavailable } from '../lib/prefs';
import { NoScriptNotice } from './common/NoScript';
import { downloadText } from './common/download';
import { LineChart } from './common/Charts';

const MAX = 4;
type Entry = { key: string; id: string; state: 'loading' } | { key: string; id: string; state: 'ok'; p: PlayerDetail } | { key: string; id: string; state: 'error'; kind: ErrorKind };

export function parseCompareParam(raw: string | null): { keys: string[]; ignored: number } {
  const parts = (raw ?? '').split(',').map((s) => s.trim()).filter(Boolean);
  const keys: string[] = [];
  let ignored = 0;
  for (const part of parts) {
    const parsed = parsePlayerIdParam(part);
    if (!parsed || keys.includes(parsed.key) || keys.length >= MAX) ignored += 1;
    else keys.push(parsed.key);
  }
  return { keys, ignored };
}

export default function CompareView({ downloadHref }: { downloadHref: string }) {
  const [search, setSearch] = useUrlSearch();
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  const { keys, ignored } = parseCompareParam(new URLSearchParams(search).get('players'));
  const [entries, setEntries] = useState<Entry[]>([]);
  const keyStr = keys.join(',');
  useEffect(() => {
    if (!ready) return undefined;
    const ac = new AbortController();
    setEntries(keys.map((key) => ({ key, id: key.replaceAll('__', ':'), state: 'loading' })));
    keys.forEach((key, i) => {
      getLoader()
        .then((l) => l.load('player_detail', `players/${key}.json`, { signal: ac.signal }))
        .then((p) => !ac.signal.aborted && setEntries((prev) => prev.map((e, j) => (j === i ? { key, id: p.id, state: 'ok', p } : e))))
        .catch((err: unknown) => {
          if (ac.signal.aborted) return;
          const c = classifyError(err);
          if (c.kind === 'release') announceReleaseUnavailable();
          setEntries((prev) => prev.map((e, j) => (j === i ? { key, id: e.id, state: 'error', kind: c.kind } : e)));
        });
    });
    return () => ac.abort();
  }, [keyStr, ready]);

  const base = siteBase();
  if (!ready) return <NoScriptNotice what="comparison" href={downloadHref} linkText="download all player data (CSV)" />;
  const setKeys = (next: string[]) => setSearch(next.length ? `?players=${next.join(',')}` : '');
  const players = entries.filter((e): e is Extract<Entry, { state: 'ok' }> => e.state === 'ok').map((e) => e.p);
  const statNames = players.length ? players.map((p) => new Set(p.career.map((s) => s.stat))).reduce((a, b) => new Set([...a].filter((x) => b.has(x)))) : new Set<string>();
  const common = [...statNames];
  const warnings: string[] = [];
  if (ignored) warnings.push(`${ignored} player ID(s) ignored: invalid, duplicate, or more than ${MAX}.`);
  const firsts = players.map((p) => Math.min(...p.seasons.map((s) => s.season)));
  if (players.length > 1 && Math.max(...firsts) - Math.min(...firsts) >= 15) warnings.push('These players come from different eras; statistics recorded and game styles differ, so direct comparison is limited.');
  const small = players.filter((p) => p.career_games < 20);
  if (small.length) warnings.push(`Small samples (under 20 games): ${small.map((p) => p.name).join(', ')}.`);
  const lowCov = players.filter((p) => p.career.some((s) => s.coverage !== null && s.coverage < 0.9) || p.career.some((s) => s.coverage === null));
  if (lowCov.length) warnings.push(`Incomplete statistics coverage for: ${lowCov.map((p) => p.name).join(', ')}.`);
  const shareUrl = typeof window === 'undefined' ? '' : window.location.href;
  const exportCsv = () => {
    const header = ['statistic', ...players.flatMap((p) => [`${p.name} per game`, `${p.name} games with data`, `${p.name} coverage`])];
    const rows = common.map((stat) => [stat, ...players.flatMap((p) => {
      const s = p.career.find((x) => x.stat === stat)!;
      return [s.mean, s.observed_games, s.coverage];
    })]);
    downloadText('supercoach-via-comparison.csv', toCsv(header, rows), 'text/csv');
  };

  return (
    <div className="stack">
      {keys.length === 0 ? <p>Choose up to four players to compare. Use <a href={withBase(base, 'players/')}>Players</a> or the compare boxes on <a href={withBase(base, 'predictions/')}>Predictions</a>.</p> : null}
      {warnings.length ? <div className="banner banner-stale" data-testid="compare-warnings"><ul>{warnings.map((w) => <li key={w}>{w}</li>)}</ul></div> : null}
      <ul className="cluster list-plain" aria-label="Players being compared">
        {entries.map((e) => (
          <li key={e.key} className="card">
            {e.state === 'ok' ? <a href={withBase(base, `player/?id=${encodeId(e.p.id)}`)}>{e.p.name}</a> : e.state === 'loading' ? <span role="status">Loading {e.id}…</span> : <span className="missing">{e.kind === 'notfound' ? `${e.id}: not in this release` : `${e.id}: could not load`}</span>}{' '}
            <button type="button" className="secondary small" aria-label={`Remove ${e.state === 'ok' ? e.p.name : e.id}`} onClick={() => setKeys(keys.filter((k) => k !== e.key))}>Remove</button>
          </li>
        ))}
      </ul>
      {players.length > 0 ? (
        <>
          <div className="field">
            <label htmlFor="share-url">Shareable link</label>
            <input id="share-url" type="text" readOnly value={shareUrl} onFocus={(e) => e.target.select()} />
          </div>
          <div className="cluster">
            <button type="button" className="secondary" onClick={() => void navigator.clipboard?.writeText(shareUrl)}>Copy link</button>
            <button type="button" className="secondary" onClick={exportCsv}>Download comparison CSV (career, these {players.length} players)</button>
          </div>
          <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable table: Career comparison">
            <table>
              <caption>Career comparison (per game, with games that have data)</caption>
              <thead>
                <tr><th scope="col">Statistic</th>{players.map((p) => <th scope="col" className="num" key={p.id}>{p.name}</th>)}</tr>
              </thead>
              <tbody>
                <tr><th scope="row">Games</th>{players.map((p) => <td className="num" key={p.id}>{p.career_games}</td>)}</tr>
                <tr><th scope="row">Seasons</th>{players.map((p) => <td className="num" key={p.id}>{Math.min(...p.seasons.map((s) => s.season))}–{Math.max(...p.seasons.map((s) => s.season))}</td>)}</tr>
                {common.map((stat) => (
                  <tr key={stat}>
                    <th scope="row">{stat}</th>
                    {players.map((p) => {
                      const s = p.career.find((x) => x.stat === stat)!;
                      return <td className="num" key={p.id}>{s.mean === null ? <span className="missing">not recorded</span> : formatStat(s.mean, 1)} <span className="muted">({s.observed_games}/{s.eligible_games} games, {formatPercent(s.coverage)})</span></td>;
                    })}
                  </tr>
                ))}
                <tr><th scope="row">Current forecast</th>{players.map((p) => <td className="num" key={p.id}>{p.forecast ? `${formatStat(p.forecast.predicted_disposals, 1)} disposals` : <span className="missing">none</span>}</td>)}</tr>
              </tbody>
            </table>
          </div>
          {common.includes('disposals') && players.length <= 2 ? (
            <LineChart id="compare-chart" title="Disposals per game by season" description="Season mean disposals for each compared player" xLabel="Season" yLabel="Disposals per game"
              series={players.map((p) => ({ name: p.name, points: [...p.seasons].sort((a, b) => a.season - b.season).map((s) => ({ x: String(s.season), y: s.stats.find((x) => x.stat === 'disposals')?.mean ?? null })) }))} />
          ) : null}
        </>
      ) : null}
    </div>
  );
}
