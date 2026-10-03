// JS-disabled readability: summaries and articles readable; detail shells show context + download.
import { test, expect } from '@playwright/test';

test.use({ javaScriptEnabled: false });

test('overview is readable without JavaScript', async ({ page }) => {
  await page.goto('');
  await expect(page.locator('h1')).toHaveText('Explore AFL players, matches and history');
  await expect(page.getByRole('navigation', { name: 'Main', exact: true }).getByRole('link', { name: 'Teams', exact: true })).toBeVisible();
  const all = page.getByRole('table', { name: 'All season leaders', exact: true });
  await expect(all).toBeHidden();
  await page.getByText('View all season leaders in this snapshot', { exact: true }).click();
  await expect(all).toBeVisible();
  await expect(page.getByTestId('forecast-status')).toBeVisible();
  await expect(page.getByTestId('recent')).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Players' })).toBeVisible();
});

test('predictions summary is server-rendered', async ({ page }) => {
  await page.goto('predictions/');
  await expect(page.getByTestId('predictions-summary')).toContainText(/Demo Round 5/);
});

test('article is readable without JavaScript', async ({ page }) => {
  await page.goto('articles/demo-article-one/');
  await expect(page.locator('article')).toContainText('DEMO section');
});

for (const p of ['player/?id=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex', 'compare/?players=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex', 'team/?id=demo_a&season=2026', 'match/?id=k.ZGVtbzoyMDI2OnIwMTphLWI', 'live/?match=demo-live-final', 'watchlist/']) {
  test(`detail shell ${p} shows context and a download link, no spinner`, async ({ page }) => {
    await page.goto(p);
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.getByText(/needs JavaScript/i)).toBeVisible();
    await expect(page.getByRole('link', { name: /download/i }).first()).toBeVisible();
    await expect(page.getByText(/Loading/)).toHaveCount(0);
  });
}
