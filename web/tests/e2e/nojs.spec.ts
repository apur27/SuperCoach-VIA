// JS-disabled readability: summaries and articles readable; detail shells show context + download.
import { test, expect } from '@playwright/test';

test.use({ javaScriptEnabled: false });

test('overview is readable without JavaScript', async ({ page }) => {
  await page.goto('');
  await expect(page.locator('h1')).toContainText('overview');
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

for (const p of ['player/?id=legacy__demo_player_a1', 'compare/?players=legacy__demo_player_a1', 'team/?id=demo_a&season=2026', 'match/?id=demo__2026__r01__a-b', 'live/?match=demo-live-final', 'watchlist/']) {
  test(`detail shell ${p} shows context and a download link, no spinner`, async ({ page }) => {
    await page.goto(p);
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.getByText(/needs JavaScript/i)).toBeVisible();
    await expect(page.getByRole('link', { name: /download/i }).first()).toBeVisible();
    await expect(page.getByText(/Loading/)).toHaveCount(0);
  });
}
