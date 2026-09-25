import { describe, expect, it } from 'vitest';
import { createWatchlistStore, parseWatchlistImport, WATCHLIST_KEY, MAX_WATCHLIST, MAX_IMPORT_BYTES } from '../../src/lib/watchlist';

class MemStorage {
  map = new Map<string, string>();
  getItem(k: string) { return this.map.get(k) ?? null; }
  setItem(k: string, v: string) { this.map.set(k, v); }
  removeItem(k: string) { this.map.delete(k); }
}

describe('watchlist', () => {
  it('uses the versioned key and persists a validated list', () => {
    const s = new MemStorage();
    const w = createWatchlistStore(() => s);
    expect(WATCHLIST_KEY).toBe('supercoach-via:watchlist:v1');
    w.add('legacy:demo_player_a1');
    w.add('legacy:demo_player_a1');
    expect(w.list()).toEqual(['legacy:demo_player_a1']);
    expect(JSON.parse(s.getItem(WATCHLIST_KEY)!)).toEqual({ version: 1, ids: ['legacy:demo_player_a1'] });
    w.remove('legacy:demo_player_a1');
    expect(w.list()).toEqual([]);
  });
  it('ignores malformed stored data without crashing', () => {
    const s = new MemStorage();
    s.setItem(WATCHLIST_KEY, '{not json');
    expect(createWatchlistStore(() => s).list()).toEqual([]);
    s.setItem(WATCHLIST_KEY, JSON.stringify({ version: 2, ids: ['x'] }));
    expect(createWatchlistStore(() => s).list()).toEqual([]);
    s.setItem(WATCHLIST_KEY, JSON.stringify({ version: 1, ids: ['ok:id', '../bad', 5] }));
    expect(createWatchlistStore(() => s).list()).toEqual(['ok:id']);
  });
  it('falls back to memory when storage throws or is disabled', () => {
    const throwing = { getItem() { throw new Error('denied'); }, setItem() { throw new DOMException('quota', 'QuotaExceededError'); }, removeItem() { throw new Error('x'); } };
    const w = createWatchlistStore(() => throwing);
    w.add('legacy:a');
    expect(w.list()).toEqual(['legacy:a']);
    expect(w.persistent()).toBe(false);
    const w2 = createWatchlistStore(() => { throw new Error('SecurityError'); });
    w2.add('legacy:b');
    expect(w2.list()).toEqual(['legacy:b']);
  });
  it('caps the list at 100 ids', () => {
    const w = createWatchlistStore(() => new MemStorage());
    for (let i = 0; i < 120; i += 1) w.add(`legacy:p${i}`);
    expect(w.list()).toHaveLength(MAX_WATCHLIST);
    expect(w.add('legacy:extra')).toBe(false);
  });
  it('validates imports: size, JSON, structure', () => {
    expect(parseWatchlistImport('x'.repeat(MAX_IMPORT_BYTES + 1))).toEqual({ ok: false, error: 'File is larger than 100 KiB.' });
    expect(parseWatchlistImport('{bad')).toEqual({ ok: false, error: 'File is not valid JSON.' });
    expect(parseWatchlistImport('{"version":1,"ids":"x"}')).toEqual({ ok: false, error: 'Unrecognised watchlist format.' });
    expect(parseWatchlistImport('{"version":1,"ids":["legacy:a"],"extra":1}')).toEqual({ ok: false, error: 'Unrecognised watchlist format.' });
    expect(parseWatchlistImport('{"version":1,"ids":["legacy:a","../x"]}')).toEqual({ ok: false, error: 'Watchlist contains an invalid player ID.' });
    const many = JSON.stringify({ version: 1, ids: Array.from({ length: 101 }, (_, i) => `legacy:p${i}`) });
    expect(parseWatchlistImport(many)).toEqual({ ok: false, error: 'Watchlist has more than 100 players.' });
    expect(parseWatchlistImport('{"version":1,"ids":["legacy:a","legacy:a"]}')).toEqual({ ok: true, ids: ['legacy:a'] });
  });
  it('exports JSON in the import format', () => {
    const w = createWatchlistStore(() => new MemStorage());
    w.add('legacy:a');
    expect(parseWatchlistImport(w.exportJson())).toEqual({ ok: true, ids: ['legacy:a'] });
  });
  it('notifies subscribers', () => {
    const w = createWatchlistStore(() => new MemStorage());
    const seen: string[][] = [];
    w.subscribe((ids) => seen.push(ids));
    w.add('legacy:a');
    w.clear();
    expect(seen).toEqual([['legacy:a'], []]);
  });
});
