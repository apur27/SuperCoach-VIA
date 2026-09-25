// P03: the budget script charges each route for the JSON kinds and JS it actually requests on
// first load. If a route starts fetching something new, this fails until the map is updated.
import { test, expect } from './helpers';
import { resolve } from 'node:path';
import { kindForPath } from '../../src/lib/contracts';
import { ROUTES } from '../../scripts/budget-routes.mjs';
import { measureSite } from '../../scripts/budget-lib.mjs';

const QUERY: Record<string, string> = {
  'player/': '?id=legacy__demo_player_a1',
  'compare/': '?players=legacy__demo_player_a1,legacy__demo_player_b1',
  'team/': '?id=demo_a&season=2026',
  'match/': '?id=demo__2026__r01__a-b',
  'live/': '?match=demo-live-final',
};

test.describe('budget route map', () => {
  test.skip(({ baseURL }) => !baseURL, 'needs a base URL');
  for (const spec of ROUTES) {
    test(`/${spec.route} initial requests are all charged by the budget`, async ({ page }, info) => {
      const dist = resolve(info.project.name === 'subpath' ? '.e2e-dist/sub' : '.e2e-dist/root');
      const measured = measureSite(dist, { routes: [spec] }).routes[0]!;
      const json = new Set<string>();
      const js = new Set<string>();
      const base = new URL(info.project.use.baseURL!).pathname;
      page.on('request', (r) => {
        const p = new URL(r.url()).pathname.slice(base.length);
        const m = /^data\/[^/]+\/(.+\.json)$/.exec(p);
        if (m) json.add(kindForPath(m[1]!) ?? `unknown:${m[1]}`);
        if (p.endsWith('.js')) js.add(p);
      });
      await page.goto(`${spec.route}${QUERY[spec.route] ?? ''}`);
      await page.waitForLoadState('networkidle');
      expect([...json].sort()).toEqual([...new Set(spec.json.map((j) => j.kind))].sort());
      for (const f of js) expect(measured.js_files, `${f} requested but not charged`).toContain(f);
    });
  }
});
