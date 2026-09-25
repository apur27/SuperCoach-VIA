/** Player directory search: normalized substring/token matching (PLAN 9.3). */
import type { PlayerIndexEntry } from './contracts';

export function normalizeText(s: string): string {
  return s
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

export function matchesQuery(entry: PlayerIndexEntry, normalizedQuery: string): boolean {
  if (!normalizedQuery) return false;
  const hay = `${normalizeText(entry.name)} ${entry.search}`;
  return normalizedQuery.split(' ').every((tok) => hay.includes(tok));
}

export interface SearchFilters {
  active?: boolean;
  club?: string;
  season?: number;
}

export function passesFilters(e: PlayerIndexEntry, f: SearchFilters): boolean {
  if (f.active !== undefined && e.active !== f.active) return false;
  if (f.club && !e.clubs.includes(f.club)) return false;
  if (f.season !== undefined) {
    if (e.first_season === null || e.last_season === null) return false;
    if (f.season < e.first_season || f.season > e.last_season) return false;
  }
  return true;
}

export function searchPlayers(list: readonly PlayerIndexEntry[], query: string, filters: SearchFilters = {}): PlayerIndexEntry[] {
  const q = normalizeText(query);
  const hits = list.filter((e) => (q ? matchesQuery(e, q) : true) && passesFilters(e, filters));
  if (!q) return hits;
  const score = (e: PlayerIndexEntry) => (normalizeText(e.name).startsWith(q) ? 0 : 1);
  return hits.map((e, i) => ({ e, i, s: score(e) })).sort((a, b) => a.s - b.s || a.i - b.i).map((x) => x.e);
}

export function eraLabel(e: Pick<PlayerIndexEntry, 'first_season' | 'last_season'>): string {
  if (e.first_season === null && e.last_season === null) return 'seasons unknown';
  if (e.first_season === e.last_season) return String(e.first_season);
  return `${e.first_season ?? '?'}–${e.last_season ?? '?'}`;
}

/** Disambiguation label for players who share a display name. */
export function disambiguate(list: readonly PlayerIndexEntry[]): Map<string, string> {
  const byName = new Map<string, PlayerIndexEntry[]>();
  for (const e of list) {
    const k = normalizeText(e.name);
    byName.set(k, [...(byName.get(k) ?? []), e]);
  }
  const out = new Map<string, string>();
  for (const group of byName.values()) {
    if (group.length < 2) continue;
    for (const e of group) out.set(e.id, `${e.clubs.join(', ') || 'club unknown'}, ${eraLabel(e)}`);
  }
  return out;
}
