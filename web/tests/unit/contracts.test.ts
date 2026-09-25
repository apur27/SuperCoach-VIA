// Cross-language contract tests: the TypeScript types and runtime validators must be
// generated from the Python-owned JSON Schemas, and the validators must catch
// nullability / enum / date drift the way the Pydantic models do.
import { describe, expect, it } from 'vitest';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { validate, kindForPath } from '../../src/lib/contracts';
import { validateAsync } from '../../src/lib/validate-client';
import { RESOURCE_KINDS } from '../../src/lib/contracts.generated';
import { generate } from '../../scripts/gen-types.mjs';

const WEB = resolve(__dirname, '../..');
const FIX = resolve(WEB, 'tests/fixtures/demo-release');

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

describe('generated contracts', () => {
  it('are up to date with ../schemas', async () => {
    const { ts, js, dts, split } = await generate();
    for (const [f, c] of Object.entries(split as unknown as Record<string, string>)) {
      expect(readFileSync(resolve(WEB, 'src/lib/validators', f), 'utf8'), f).toBe(c);
    }
    expect(readFileSync(resolve(WEB, 'src/lib/contracts.generated.ts'), 'utf8')).toBe(ts);
    expect(readFileSync(resolve(WEB, 'src/lib/validators.generated.js'), 'utf8')).toBe(js);
    expect(readFileSync(resolve(WEB, 'src/lib/validators.generated.d.ts'), 'utf8')).toBe(dts);
  });
  it('cover every schema file', () => {
    const files = readdirSync(resolve(WEB, '../schemas')).filter((f) => f.endsWith('.schema.json')).map((f) => f.replace('.schema.json', '')).sort();
    expect([...RESOURCE_KINDS].sort()).toEqual(files);
  });
});

function validateTree(root: string) {
  const files = walk(root).filter((f) => f.endsWith('.json'));
  expect(files.length).toBeGreaterThan(10);
  for (const f of files) {
    const rel = relative(root, f).split('\\').join('/');
    const kind = kindForPath(rel);
    expect(kind, `no resource kind for ${rel}`).not.toBeNull();
    const data = JSON.parse(readFileSync(f, 'utf8'));
    const res = validate(kind!, data);
    expect(res.ok, `${rel}: ${res.ok ? '' : res.error}`).toBe(true);
    // Positional stats must align with the parent's column list (not expressible in JSON Schema).
    const aligned = (cols: string[], rows: { stats: unknown[] }[]) => {
      for (const r of rows) expect(r.stats.length, `${rel}: stats length`).toBe(cols.length);
    };
    if (kind === 'match_detail') aligned(data.stat_columns, [...data.home_players, ...data.away_players]);
    if (kind === 'player_season_games') aligned(data.stat_columns, data.games);
    // Live rows are keyed; only reliable fields may appear (an absent key = not reported).
    if (kind === 'live_snapshot') {
      for (const r of data.players) for (const k of Object.keys(r.stats)) expect(data.reliable_fields, `${rel}: live stat ${k}`).toContain(k);
    }
  }
}

describe('fixture releases validate against the schemas', () => {
  it('web DEMO fixture', () => {
    validateTree(FIX);
    const rel = JSON.parse(readFileSync(join(FIX, 'release.json'), 'utf8'));
    expect(rel.demo).toBe(true);
    expect(rel.base_label).toBe('DEMO');
  });
  const pyDemo = resolve(WEB, '../dist/demo/public');
  it.runIf(existsSync(join(pyDemo, 'release.json')))('python demo release (dist/demo/public)', () => validateTree(pyDemo));
});

describe('per-kind browser validators agree with the full validator set', () => {
  it('on every fixture file and on drifted payloads', async () => {
    for (const f of walk(FIX).filter((x) => x.endsWith('.json'))) {
      const rel = relative(FIX, f).split('\\').join('/');
      const kind = kindForPath(rel)!;
      const data = JSON.parse(readFileSync(f, 'utf8'));
      expect((await validateAsync(kind, data)).ok, rel).toBe(true);
    }
    const o = JSON.parse(readFileSync(join(FIX, 'overview.json'), 'utf8'));
    o.freshness.generated_at = '2026-09-25T00:00:00';
    expect((await validateAsync('overview', o)).ok).toBe(false);
  });
});

describe('drift detection', () => {
  const overview = () => JSON.parse(readFileSync(join(FIX, 'overview.json'), 'utf8'));
  it('rejects null where the contract is non-nullable', () => {
    const o = overview();
    o.freshness.stale = null;
    expect(validate('overview', o).ok).toBe(false);
  });
  it('rejects a missing required nullable field (null must be explicit)', () => {
    const o = overview();
    delete o.freshness.published_at;
    expect(validate('overview', o).ok).toBe(false);
  });
  it('rejects unknown enum values', () => {
    const o = overview();
    o.freshness.dataset_status = 'final';
    expect(validate('overview', o).ok).toBe(false);
  });
  it('rejects timestamps without a zone and non-date dates', () => {
    const o = overview();
    o.freshness.generated_at = '2026-09-25T00:00:00';
    expect(validate('overview', o).ok).toBe(false);
    const o2 = overview();
    o2.freshness.latest_completed_match_date = '2026-02-30';
    expect(validate('overview', o2).ok).toBe(false);
    const o3 = overview();
    o3.freshness.latest_completed_match_date = '2026-09-12T00:00:00Z';
    expect(validate('overview', o3).ok).toBe(false);
  });
  it('rejects unknown properties', () => {
    const o = overview();
    o.extra = 1;
    expect(validate('overview', o).ok).toBe(false);
  });
  it('rejects NaN-like strings in numeric fields', () => {
    const o = overview();
    o.leaders[0].value = 'NaN';
    expect(validate('overview', o).ok).toBe(false);
  });
  it('maps resource paths to kinds', () => {
    expect(kindForPath('players/index.json')).toBe('player_index');
    expect(kindForPath('players/legacy__x.json')).toBe('player_detail');
    expect(kindForPath('player-games/legacy__x/2026.json')).toBe('player_season_games');
    expect(kindForPath('matches/detail/demo__1.json')).toBe('match_detail');
    expect(kindForPath('matches/2026/index.json')).toBe('match_index');
    expect(kindForPath('teams/demo_a/2026.json')).toBe('team_season');
    expect(kindForPath('history/cat/era.json')).toBe('history_table');
    expect(kindForPath('accuracy/m/2026.json')).toBe('accuracy_report');
    expect(kindForPath('lists/2026.json')).toBe('lists_season');
    expect(kindForPath('articles/slug.json')).toBe('article');
    expect(kindForPath('live/g/latest.json')).toBe('live_snapshot');
    expect(kindForPath('predictions/2026/r01.json')).toBe('prediction_set');
    expect(kindForPath('downloads/players.csv')).toBeNull();
  });
});
