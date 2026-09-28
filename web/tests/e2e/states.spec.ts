// W05 + W09: loading/empty/error-retry/stale states via interception; malformed payloads;
// old-release missing-resource recovery.
import { test, expect, overrideResource } from './helpers';

test.describe('fetch states', () => {
  test('server error shows retry, and retry recovers', async ({ page }) => {
    let fail = true;
    await page.route('**/teams/demo_a/2026.json', (route) => (fail ? route.fulfill({ status: 503, body: '' }) : route.continue()));
    await page.goto('team/?id=demo_a&season=2026');
    const alert = page.locator('[data-state="error"]');
    await expect(alert).toContainText(/Could not load/);
    fail = false;
    await alert.getByRole('button', { name: 'Try again' }).click();
    await expect(page.getByRole('table', { name: /2026 ladder/i })).toBeVisible();
  });

  test('malformed JSON is refused with a verification error', async ({ page }) => {
    await page.route('**/matches/detail/*.json', (route) => route.fulfill({ contentType: 'application/json', body: '{"summary": 1' }));
    await page.goto('match/?id=k.ZGVtbzoyMDI2OnIwMTphLWI');
    await expect(page.locator('[data-state="error"]')).toHaveAttribute('data-error-kind', 'invalid');
    await expect(page.locator('[data-state="error"]')).toContainText(/could not be verified/);
  });

  test('schema-drifted payload is refused (null where not allowed)', async ({ page }) => {
    await page.route('**/players/k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex.json', async (route) => {
      const res = await route.fetch();
      const body = await res.json();
      body.career_games = null;
      await route.fulfill({ response: res, body: JSON.stringify(body) });
    });
    await page.goto('player/?id=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex');
    await expect(page.locator('[data-state="error"]')).toHaveAttribute('data-error-kind', 'invalid');
  });

  test('checksum mismatch against release.json is refused', async ({ page }) => {
    await page.route('**/teams/index.json', async (route) => {
      const res = await route.fetch();
      const text = (await res.text()).replace('Demo Club A', 'Demo Club Z');
      await route.fulfill({ response: res, body: text });
    });
    await page.goto('team/?id=demo_a&season=2026');
    await expect(page.locator('[data-state="error"][data-error-kind="invalid"]').first()).toBeVisible();
  });

  test('empty data shows an explicit empty state', async ({ page }) => {
    await overrideResource(page, 'matches/2026/index.json', JSON.stringify({ season: 2026, matches: [] }));
    await page.goto('matches/?season=2026');
    await expect(page.locator('[data-state="empty"]')).toContainText(/No matches/);
  });

  test('old release: missing resource shows "A newer release is available" and reload', async ({ page }) => {
    await page.route('**/data/*/release.json', (route) => route.fulfill({ status: 404, body: 'gone' }));
    await page.goto('player/?id=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex');
    await expect(page.locator('[data-release-notice]')).toBeVisible();
    await expect(page.locator('[data-release-notice]')).toContainText('A newer release is available');
    await expect(page.getByRole('button', { name: /Reload to get the newest release/ })).toBeVisible();
    await expect(page.locator('[data-state="error"]')).toHaveAttribute('data-error-kind', 'release');
  });

  test('an archived in-progress snapshot is labelled old and is not polled', async ({ page }) => {
    await page.clock.install({ time: new Date('2026-09-25T02:00:00Z') });
    let hits = 0;
    await page.route('**/live/demo-live-progress/latest.json', (route) => {
      hits += 1;
      return route.continue();
    });
    await page.goto('live/?match=demo-live-progress');
    await expect(page.getByTestId('live-status')).toContainText(/already old/i);
    await expect(page.getByTestId('live-status')).toContainText(/Last accepted update/);
    const before = hits;
    await page.clock.runFor(300_000);
    expect(hits).toBe(before);
  });

  test('final live snapshot stops polling', async ({ page }) => {
    await page.clock.install({ time: new Date('2026-09-25T02:00:00Z') });
    let hits = 0;
    await page.route('**/live/demo-live-final/latest.json', (route) => { hits += 1; return route.continue(); });
    await page.goto('live/?match=demo-live-final');
    await expect(page.getByTestId('live-status')).toContainText(/final/i);
    const n = hits;
    await page.clock.runFor(300_000);
    expect(hits).toBe(n);
  });
});
