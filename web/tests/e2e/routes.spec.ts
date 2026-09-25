import { test, expect, expectClean, axe, ROUTES } from './helpers';
import { mkdirSync, writeFileSync } from 'node:fs';

for (const r of ROUTES) {
  test.describe(`route /${r.path}`, () => {
    test('renders shell, one H1, landmarks, freshness; no CSP/console problems', async ({ page, problems }) => {
      await page.goto(r.path);
      await expect(page.locator('h1')).toHaveCount(1);
      await expect(page.locator('h1')).toHaveText(r.h1);
      await expect(page).toHaveTitle(/SuperCoach VIA/);
      expect(await page.locator('meta[name="description"]').getAttribute('content')).toBeTruthy();
      await expect(page.locator('a.skip-link')).toHaveAttribute('href', '#main');
      await expect(page.locator('header')).toHaveCount(1);
      await expect(page.locator('main#main')).toHaveCount(1);
      await expect(page.locator('footer')).toHaveCount(1);
      await expect(page.getByTestId('freshness')).toContainText('Coverage through');
      await expect(page.getByTestId('freshness').getByRole('link', { name: 'Methodology' })).toBeVisible();
      await expect(page.locator('html')).toHaveAttribute('data-release', /.+/);
      await page.waitForLoadState('networkidle');
      await expect(page.locator('[data-state="loading"]')).toHaveCount(0);
      expectClean(problems);
    });
    test('axe scan', async ({ page }, info) => {
      await page.goto(r.path);
      await page.waitForLoadState('networkidle');
      const violations = await axe(page);
      mkdirSync('test-results/axe', { recursive: true });
      writeFileSync(`test-results/axe/${info.project.name}-${r.path.replace(/[^a-z0-9]+/gi, '_') || 'home'}.json`, JSON.stringify(violations, null, 2));
      expect(violations).toEqual([]);
    });
    test('axe scan at 375px (mobile) and in the dark theme', async ({ page }, info) => {
      const name = r.path.replace(/[^a-z0-9]+/gi, '_') || 'home';
      mkdirSync('test-results/axe', { recursive: true });
      await page.setViewportSize({ width: 375, height: 812 });
      await page.goto(r.path);
      await page.waitForLoadState('networkidle');
      const mobile = await axe(page);
      writeFileSync(`test-results/axe/${info.project.name}-mobile-${name}.json`, JSON.stringify(mobile, null, 2));
      await page.setViewportSize({ width: 1440, height: 900 });
      await page.emulateMedia({ colorScheme: 'dark' });
      await page.reload();
      await page.waitForLoadState('networkidle');
      const dark = await axe(page);
      writeFileSync(`test-results/axe/${info.project.name}-dark-${name}.json`, JSON.stringify(dark, null, 2));
      expect(mobile, 'mobile').toEqual([]);
      expect(dark, 'dark').toEqual([]);
    });
  });
}

test('detail pages show breadcrumbs', async ({ page }) => {
  for (const p of ['player/?id=legacy__demo_player_a1', 'team/?id=demo_a&season=2026', 'match/?id=demo__2026__r01__a-b', 'articles/demo-article-one/']) {
    await page.goto(p);
    await expect(page.getByRole('navigation', { name: 'Breadcrumb' })).toBeVisible();
  }
});

test('404 page for unknown routes', async ({ page }) => {
  const res = await page.goto('no-such-page/');
  expect(res?.status()).toBe(404);
  await expect(page.locator('h1')).toHaveText(/not found/i);
});
