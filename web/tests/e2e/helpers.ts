import { test as base, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

export const RELEASE_ID = process.env.SCVIA_E2E_RELEASE_ID ?? '20260925T000000Z-demo';
export const DEMO = (process.env.SCVIA_E2E_DEMO ?? 'fixture') === 'fixture';

export const ROUTES: { path: string; h1: RegExp }[] = [
  { path: '', h1: /overview/i },
  { path: 'predictions/', h1: /predictions/i },
  { path: 'players/', h1: /players/i },
  { path: 'player/?id=legacy__demo_player_a1', h1: /player/i },
  { path: 'compare/?players=legacy__demo_player_a1,legacy__demo_player_b1', h1: /compare/i },
  { path: 'teams/', h1: /teams/i },
  { path: 'team/?id=demo_a&season=2026', h1: /Demo Club A 2026/ },
  { path: 'matches/', h1: /matches/i },
  { path: 'match/?id=demo__2026__r01__a-b', h1: /Demo Club A v Demo Club B/ },
  { path: 'history/', h1: /history/i },
  { path: 'accuracy/', h1: /accuracy/i },
  { path: 'lists/', h1: /lists/i },
  { path: 'articles/', h1: /articles/i },
  { path: 'articles/demo-article-one/', h1: /demo article one/i },
  { path: 'live/?match=demo-live-final', h1: /live/i },
  { path: 'watchlist/', h1: /watchlist/i },
  { path: 'downloads/', h1: /downloads/i },
  { path: 'data-status/', h1: /data status/i },
  { path: 'methodology/', h1: /methodology/i },
];

export interface PageProblems { console: string[]; csp: string[]; pageErrors: string[]; external: string[] }

/** Collects console errors, CSP violations, uncaught errors and any non-same-origin request. */
export const test = base.extend<{ problems: PageProblems }>({
  problems: async ({ page, baseURL }, use) => {
    const problems: PageProblems = { console: [], csp: [], pageErrors: [], external: [] };
    const origin = new URL(baseURL!).origin;
    page.on('console', (msg) => {
      const text = msg.text();
      if (/Content Security Policy|Refused to/i.test(text)) problems.csp.push(text);
      else if (msg.type() === 'error' && !/Failed to load resource/.test(text)) problems.console.push(text);
    });
    page.on('pageerror', (e) => problems.pageErrors.push(e.message));
    page.on('request', (r) => {
      const u = r.url();
      if (!u.startsWith(origin) && !u.startsWith('data:')) problems.external.push(u);
    });
    await page.addInitScript(() => {
      document.addEventListener('securitypolicyviolation', (e) => {
        console.error(`Content Security Policy violation: ${e.violatedDirective} ${e.blockedURI}`);
      });
    });
    await use(problems);
  },
});

export function expectClean(p: PageProblems) {
  expect(p.csp, 'CSP violations').toEqual([]);
  expect(p.pageErrors, 'uncaught page errors').toEqual([]);
  expect(p.console, 'console errors').toEqual([]);
  expect(p.external, 'external requests').toEqual([]);
}

export async function axe(page: Page) {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice']).analyze();
  return results.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.length, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) }));
}

export async function noHorizontalOverflow(page: Page) {
  return page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
}

export { expect };

/**
 * Serve a replacement body for a release resource AND a consistent release.json (updated
 * size/sha256), simulating a genuinely different release payload rather than tampering.
 */
export async function overrideResource(page: Page, relPath: string, body: string) {
  const { createHash } = await import('node:crypto');
  const bytes = Buffer.from(body);
  await page.route(`**/data/*/${relPath}`, (route) => route.fulfill({ contentType: 'application/json', body: bytes }));
  await page.route('**/data/*/release.json', async (route) => {
    const res = await route.fetch();
    const manifest = await res.json();
    for (const ref of Object.values(manifest.resources) as { path: string; sha256: string; bytes: number }[]) {
      if (ref.path === relPath) {
        ref.sha256 = createHash('sha256').update(bytes).digest('hex');
        ref.bytes = bytes.length;
      }
    }
    await route.fulfill({ response: res, body: JSON.stringify(manifest) });
  });
}
