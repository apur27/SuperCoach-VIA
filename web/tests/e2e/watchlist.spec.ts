// W04: watchlist persistence, import errors, export, storage failure fallback.
import { test, expect } from './helpers';

test.describe('watchlist', () => {
  test('persists across reloads under the versioned key', async ({ page }) => {
    await page.goto('player/?id=legacy__demo_player_a1');
    await page.getByRole('button', { name: /Add to watchlist/ }).click();
    await page.goto('watchlist/');
    await expect(page.getByRole('link', { name: 'Demo Player A1' })).toBeVisible();
    const stored = await page.evaluate(() => localStorage.getItem('supercoach-via:watchlist:v1'));
    expect(JSON.parse(stored!)).toEqual({ version: 1, ids: ['legacy:demo_player_a1'] });
    await page.reload();
    await expect(page.getByRole('link', { name: 'Demo Player A1' })).toBeVisible();
    await page.getByRole('button', { name: 'Remove Demo Player A1' }).click();
    await expect(page.getByText(/watchlist is empty/i)).toBeVisible();
  });

  test('malformed stored data does not crash', async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('supercoach-via:watchlist:v1', '{broken'));
    await page.goto('watchlist/');
    await expect(page.getByText(/watchlist is empty/i)).toBeVisible();
  });

  test('import validates and reports errors; valid import works; export offered', async ({ page }) => {
    await page.goto('watchlist/');
    const input = page.getByLabel('Import watchlist JSON');
    await input.setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{nope') });
    await expect(page.getByRole('alert')).toContainText('not valid JSON');
    await input.setInputFiles({ name: 'big.json', mimeType: 'application/json', buffer: Buffer.alloc(101 * 1024, 32) });
    await expect(page.getByRole('alert')).toContainText('larger than 100 KiB');
    await input.setInputFiles({ name: 'shape.json', mimeType: 'application/json', buffer: Buffer.from('{"version":1,"ids":["../x"]}') });
    await expect(page.getByRole('alert')).toContainText('invalid player ID');
    await input.setInputFiles({ name: 'ok.json', mimeType: 'application/json', buffer: Buffer.from('{"version":1,"ids":["legacy:demo_player_b1"]}') });
    await expect(page.getByRole('link', { name: 'Demo Player B1' })).toBeVisible();
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: /Export watchlist/ }).click();
    expect((await download).suggestedFilename()).toBe('supercoach-via-watchlist.json');
  });

  test('disabled storage falls back to memory with a notice', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('denied', 'SecurityError'); } });
    });
    await page.goto('watchlist/');
    await expect(page.getByText(/saved only for this page visit/i)).toBeVisible();
  });
});
