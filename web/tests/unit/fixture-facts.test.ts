// The browser fixture must carry the same facts the Python builder publishes:
// shared match columns, a non-empty era summary, and a labelled Brownlow proxy.
import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { applyMatchFacts, gameRows } from '../../src/lib/stats';
import { BROWNLOW_VIEW_COLUMNS, ERA_VIEW_COLUMNS, parseCsv, project } from '../../src/lib/tabular';

const FIX = resolve(__dirname, '../fixtures/demo-release');

function firstLog(): { player_id: string; season: number; match_facts: string; games: Parameters<typeof gameRows>[0] } {
  const root = join(FIX, 'player-games');
  const key = readdirSync(root)[0];
  if (!key) throw new Error('fixture has no player games');
  const season = readdirSync(join(root, key)).find((n) => n.endsWith('.json'));
  if (!season) throw new Error('fixture game log is missing');
  return JSON.parse(readFileSync(join(root, key, season), 'utf8'));
}

describe('browser fixture facts', () => {
  it('restores opponent and stage from the shared match index', () => {
    const log = firstLog();
    expect(log.match_facts).toMatch(/^matches\/\d+\/index\.json$/);
    expect(log.games.opponent_name).toEqual([]);
    expect(log.games.stage_label).toEqual([]);
    expect(log.games.match_id.length).toBeGreaterThan(0);
    const index = JSON.parse(readFileSync(join(FIX, log.match_facts), 'utf8'));
    const rows = applyMatchFacts(gameRows(log.games), index);
    const row = rows[0];
    if (!row) throw new Error('game log restored no rows');
    expect(row.stage_label.length).toBeGreaterThan(0);
    expect(row.opponent_name).toBeTruthy();
    expect(row.stats).toEqual(log.games.stats[0]);
  });

  it('publishes a non-empty era summary and a proxy-labelled Brownlow table', () => {
    const downloads = JSON.parse(readFileSync(join(FIX, 'downloads.json'), 'utf8'));
    const era = downloads.items.find((i: { key: string }) => i.key === 'era_summary_csv');
    const proxy = downloads.items.find((i: { key: string }) => i.key === 'brownlow_proxy_csv');
    expect(era.rows).toBeGreaterThan(0);
    expect(proxy.rows).toBeGreaterThan(0);
    const eraView = project(parseCsv(readFileSync(join(FIX, era.path), 'utf8')), ERA_VIEW_COLUMNS);
    expect(eraView.rows.length).toBeGreaterThan(0);
    expect(eraView.rows.some((r) => r[0] === '1990s')).toBe(true);
    const proxyView = project(parseCsv(readFileSync(join(FIX, proxy.path), 'utf8')), BROWNLOW_VIEW_COLUMNS);
    const proxyRow = proxyView.rows[0];
    if (!proxyRow) throw new Error('proxy table is empty');
    expect(proxyRow[proxyView.columns.indexOf('label')]).toMatch(/not Brownlow votes/);
  });
});
