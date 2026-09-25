import { describe, expect, it } from 'vitest';
import { normalizeText, matchesQuery, searchPlayers, disambiguate } from '../../src/lib/search';
import type { PlayerIndexEntry } from '../../src/lib/contracts';

const entry = (o: Partial<PlayerIndexEntry>): PlayerIndexEntry => ({
  id: 'legacy:x', key: 'legacy__x', name: 'X', clubs: [], first_season: null, last_season: null, games: 0, active: false, search: '', ...o,
});

describe('search', () => {
  it('normalizes diacritics, case and whitespace', () => {
    expect(normalizeText('  Zoë   ÄRGER ')).toBe('zoe arger');
    expect(normalizeText('Ōkami-Jones')).toBe('okami-jones');
  });
  it('matches substrings across all tokens in any order', () => {
    const e = entry({ name: 'Demo Zoë Ärger', search: 'demo zoe arger demo club c' });
    expect(matchesQuery(e, normalizeText('zoe'))).toBe(true);
    expect(matchesQuery(e, normalizeText('Ärg zo'))).toBe(true);
    expect(matchesQuery(e, normalizeText('club c zoe'))).toBe(true);
    expect(matchesQuery(e, normalizeText('zoe xyz'))).toBe(false);
    expect(matchesQuery(e, '')).toBe(false);
  });
  it('ranks name-prefix matches first and filters', () => {
    const list = [entry({ id: 'a', name: 'Alpha Demo', search: 'alpha demo' }), entry({ id: 'b', name: 'Demo Beta', search: 'demo beta' })];
    expect(searchPlayers(list, 'demo').map((e) => e.id)).toEqual(['b', 'a']);
    expect(searchPlayers(list, 'demo', { active: true })).toEqual([]);
  });
  it('disambiguates same-name players by clubs and era', () => {
    const a = entry({ id: 'p1', name: 'Demo Same Name', clubs: ['Demo Club A'], first_season: 2020, last_season: 2026 });
    const b = entry({ id: 'p2', name: 'Demo Same Name', clubs: ['Demo Club Old'], first_season: 1990, last_season: 1994 });
    const c = entry({ id: 'p3', name: 'Unique', clubs: ['Demo Club B'], first_season: 2026, last_season: 2026 });
    const labels = disambiguate([a, b, c]);
    expect(labels.get('p1')).toBe('Demo Club A, 2020–2026');
    expect(labels.get('p2')).toBe('Demo Club Old, 1990–1994');
    expect(labels.get('p3')).toBeUndefined();
  });
});
