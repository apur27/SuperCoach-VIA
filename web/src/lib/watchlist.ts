/** Local-only watchlist (PLAN 9.3). Storage is optional; failures fall back to memory. */
import { isSafeId } from './ids';

export const WATCHLIST_KEY = 'supercoach-via:watchlist:v1';
export const MAX_WATCHLIST = 100;
export const MAX_IMPORT_BYTES = 100 * 1024;

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

type Parsed = { ok: true; ids: string[] } | { ok: false; error: string };

function validateShape(value: unknown): Parsed {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, error: 'Unrecognised watchlist format.' };
  const keys = Object.keys(value).sort();
  const v = value as { version?: unknown; ids?: unknown };
  if (keys.join(',') !== 'ids,version' || v.version !== 1 || !Array.isArray(v.ids)) {
    return { ok: false, error: 'Unrecognised watchlist format.' };
  }
  if (v.ids.some((id) => typeof id !== 'string' || !isSafeId(id))) return { ok: false, error: 'Watchlist contains an invalid player ID.' };
  const ids = [...new Set(v.ids as string[])];
  if (ids.length > MAX_WATCHLIST) return { ok: false, error: 'Watchlist has more than 100 players.' };
  return { ok: true, ids };
}

export function parseWatchlistImport(text: string): Parsed {
  if (new TextEncoder().encode(text).length > MAX_IMPORT_BYTES) return { ok: false, error: 'File is larger than 100 KiB.' };
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return { ok: false, error: 'File is not valid JSON.' };
  }
  return validateShape(value);
}

function readStored(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const value = JSON.parse(raw) as unknown;
    if (!value || typeof value !== 'object') return [];
    const v = value as { version?: unknown; ids?: unknown };
    if (v.version !== 1 || !Array.isArray(v.ids)) return [];
    // Salvage valid entries from partially corrupted local data.
    return [...new Set(v.ids.filter((id): id is string => typeof id === 'string' && isSafeId(id)))].slice(0, MAX_WATCHLIST);
  } catch {
    return [];
  }
}

export interface WatchlistStore {
  list(): string[];
  has(id: string): boolean;
  add(id: string): boolean;
  remove(id: string): void;
  clear(): void;
  replace(ids: string[]): void;
  exportJson(): string;
  persistent(): boolean;
  subscribe(fn: (ids: string[]) => void): () => void;
}

export function createWatchlistStore(getStorage: () => StorageLike | null | undefined): WatchlistStore {
  let storage: StorageLike | null = null;
  try {
    storage = getStorage() ?? null;
  } catch {
    storage = null;
  }
  let ids: string[] = [];
  let persistent = storage !== null;
  try {
    ids = readStored(storage?.getItem(WATCHLIST_KEY) ?? null);
  } catch {
    persistent = false;
  }
  const subs = new Set<(ids: string[]) => void>();

  function save() {
    if (storage && persistent) {
      try {
        storage.setItem(WATCHLIST_KEY, JSON.stringify({ version: 1, ids }));
      } catch {
        persistent = false; // quota or disabled storage: keep working in memory
      }
    }
    for (const fn of subs) fn([...ids]);
  }

  return {
    list: () => [...ids],
    has: (id) => ids.includes(id),
    add(id) {
      if (!isSafeId(id) || ids.includes(id)) return ids.includes(id);
      if (ids.length >= MAX_WATCHLIST) return false;
      ids = [...ids, id];
      save();
      return true;
    },
    remove(id) {
      ids = ids.filter((x) => x !== id);
      save();
    },
    clear() {
      ids = [];
      save();
    },
    replace(next) {
      ids = [...new Set(next.filter(isSafeId))].slice(0, MAX_WATCHLIST);
      save();
    },
    exportJson: () => JSON.stringify({ version: 1, ids }, null, 2),
    persistent: () => persistent,
    subscribe(fn) {
      subs.add(fn);
      return () => subs.delete(fn);
    },
  };
}

let shared: WatchlistStore | null = null;
/** Browser singleton (lazy so SSR never touches localStorage). */
export function browserWatchlist(): WatchlistStore {
  shared ??= createWatchlistStore(() => window.localStorage);
  return shared;
}
