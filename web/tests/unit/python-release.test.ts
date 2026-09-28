// Reads a release directory the Python builder wrote (SCVIA_PY_PUBLIC).
// Skipped unless that directory is supplied, so the unit tier stays hermetic
// when the builder has not just run.
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { applyMatchFacts, gameRows } from '../../src/lib/stats';
import { BROWNLOW_VIEW_COLUMNS, ERA_VIEW_COLUMNS, parseCsv, project } from '../../src/lib/tabular';

const dir = process.env.SCVIA_PY_PUBLIC;

describe.skipIf(!dir)('Python-built release consumed by the browser readers', () => {
  it('restores shared match facts and serves non-empty era and proxy tables', () => {
    const root = join(dir!, 'player-games');
    const key = readdirSync(root)[0];
    if (!key) throw new Error('release has no player games');
    const file = readdirSync(join(root, key)).find((n) => n.endsWith('.json'));
    if (!file) throw new Error('release game log is missing');
    const log = JSON.parse(readFileSync(join(root, key, file), 'utf8'));
    expect(log.match_facts).toMatch(/^matches\/\d+\/index\.json$/);
    expect(log.games.opponent_name).toEqual([]);
    const index = JSON.parse(readFileSync(join(dir!, log.match_facts), 'utf8'));
    const rows = applyMatchFacts(gameRows(log.games), index);
    const row = rows[0];
    if (!row) throw new Error('game log restored no rows');
    expect(row.opponent_name).toBeTruthy();
    expect(row.stage_label).toBeTruthy();

    const downloads = JSON.parse(readFileSync(join(dir!, 'downloads.json'), 'utf8'));
    const eraItem = downloads.items.find((i: { key: string }) => i.key === 'era_summary_csv');
    const proxyItem = downloads.items.find((i: { key: string }) => i.key === 'brownlow_proxy_csv');
    const era = project(parseCsv(readFileSync(join(dir!, eraItem.path), 'utf8')), ERA_VIEW_COLUMNS);
    expect(era.rows.length).toBeGreaterThan(0);
    const proxy = project(parseCsv(readFileSync(join(dir!, proxyItem.path), 'utf8')), BROWNLOW_VIEW_COLUMNS);
    expect(proxy.rows.length).toBeGreaterThan(0);
    const label = proxy.columns.indexOf('label');
    expect(proxy.rows.every((r) => /proxy/i.test(r[label] ?? ''))).toBe(true);
  });
});
