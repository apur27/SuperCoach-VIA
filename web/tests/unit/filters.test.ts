import { describe, expect, it } from 'vitest';
import { compareValues, sortRows, paginate, parseUrlState, serializeUrlState, type StateSpec } from '../../src/lib/filters';

const spec = {
  season: { kind: 'int', min: 1897, max: 2100 },
  team: { kind: 'token' },
  q: { kind: 'text', maxLength: 60 },
  sort: { kind: 'enum', values: ['name', 'predicted'] },
  dir: { kind: 'enum', values: ['asc', 'desc'], default: 'asc' },
  page: { kind: 'int', min: 1, max: 10000, default: 1 },
  size: { kind: 'enum', values: ['25', '50', '100'], default: '25' },
} as const satisfies StateSpec;

describe('url state', () => {
  it('parses documented, valid fields only', () => {
    const s = parseUrlState('?season=2026&team=demo_a&q=%20Zo%C3%AB%20&sort=predicted&dir=desc&page=2&size=50&evil=1', spec);
    expect(s).toEqual({ season: 2026, team: 'demo_a', q: 'Zoë', sort: 'predicted', dir: 'desc', page: 2, size: '50' });
  });
  it('drops invalid values and applies defaults', () => {
    const s = parseUrlState('?season=abc&team=../x&sort=hack&dir=sideways&page=-3&size=7&q=' + 'x'.repeat(100), spec);
    expect(s).toEqual({ dir: 'asc', page: 1, size: '25', q: 'x'.repeat(60) });
    expect(parseUrlState('?season=1800', spec).season).toBeUndefined();
  });
  it('serializes without defaults and in stable order', () => {
    expect(serializeUrlState({ season: 2026, dir: 'asc', page: 1, size: '25', q: 'a b' }, spec)).toBe('?season=2026&q=a+b');
    expect(serializeUrlState({ dir: 'asc', page: 1, size: '25' }, spec)).toBe('');
  });
});

describe('sorting', () => {
  it('sorts numbers numerically with nulls last in both directions', () => {
    const rows = [{ v: 10 }, { v: null }, { v: 9 }, { v: 100 }, { v: undefined }];
    expect(sortRows(rows, (r) => r.v, 'asc').map((r) => r.v)).toEqual([9, 10, 100, null, undefined]);
    expect(sortRows(rows, (r) => r.v, 'desc').map((r) => r.v)).toEqual([100, 10, 9, null, undefined]);
  });
  it('sorts strings with locale compare and is stable', () => {
    const rows = [{ n: 'b', i: 1 }, { n: 'a', i: 2 }, { n: 'b', i: 3 }];
    expect(sortRows(rows, (r) => r.n, 'asc').map((r) => r.i)).toEqual([2, 1, 3]);
    expect(compareValues('Zoë', 'zoe')).not.toBeNaN();
  });
});

describe('paginate', () => {
  it('clamps page and reports totals', () => {
    const items = Array.from({ length: 60 }, (_, i) => i);
    expect(paginate(items, 1, 25)).toMatchObject({ page: 1, pages: 3, items: items.slice(0, 25) });
    expect(paginate(items, 9, 25)).toMatchObject({ page: 3, pages: 3, items: items.slice(50) });
    expect(paginate([], 1, 25)).toMatchObject({ page: 1, pages: 1, items: [] });
  });
});
