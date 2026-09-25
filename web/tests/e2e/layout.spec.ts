// W06/W07/W08: responsive layouts, 200% zoom, keyboard flows, charts, base path + deep links.
import { test, expect, noHorizontalOverflow, ROUTES } from './helpers';

const WIDTHS = [320, 375, 768, 1440];

for (const width of WIDTHS) {
  test(`no page-wide horizontal overflow at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    for (const r of ROUTES) {
      await page.goto(r.path);
      await page.waitForLoadState('networkidle');
      expect(await noHorizontalOverflow(page), `${r.path} at ${width}`).toBe(true);
    }
  });
}

test('200% zoom (1280px viewport at 2x text/zoom equivalent: 640 CSS px) stays usable', async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 640, height: 400 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  for (const p of ['', 'predictions/', 'player/?id=legacy__demo_player_a1', 'data-status/']) {
    await page.goto(p);
    await page.waitForLoadState('networkidle');
    expect(await noHorizontalOverflow(page), p).toBe(true);
    await expect(page.locator('h1')).toBeVisible();
  }
  await ctx.close();
});

test('keyboard: skip link, mobile nav disclosure, More menu', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await page.goto('');
  await page.keyboard.press('Tab');
  await expect(page.locator('a.skip-link')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#main$/);
  const toggle = page.getByRole('button', { name: 'Menu' });
  await expect(page.getByRole('navigation', { name: 'Main' })).toBeHidden();
  await toggle.focus();
  await page.keyboard.press('Enter');
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await expect(page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Players' })).toBeVisible();
  const more = page.getByRole('button', { name: /^More/ });
  await more.focus();
  await page.keyboard.press('Enter');
  await expect(more).toHaveAttribute('aria-expanded', 'true');
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Data status' }).focus();
  await page.keyboard.press('Escape');
  await expect(more).toHaveAttribute('aria-expanded', 'false');
  await expect(more).toBeFocused();
  await toggle.focus();
  await page.keyboard.press('Escape');
});

test('keyboard: desktop More menu reachable and focus visible', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('');
  const more = page.getByRole('button', { name: /^More/ });
  await more.focus();
  const outline = await more.evaluate((el) => getComputedStyle(el).outlineStyle);
  expect(outline).not.toBe('none');
  await page.keyboard.press('Enter');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Lists' }).first()).toBeFocused();
});

test('keyboard-only predictions flow', async ({ page }) => {
  await page.goto('predictions/');
  // the filters render with the loaded set; focusing before then races the island's re-render
  await expect(page.getByRole('table', { name: /predicted disposals/i })).toBeVisible();
  await page.getByLabel('Team').focus();
  await page.keyboard.press('ArrowDown');
  await expect(page).toHaveURL(/team=/);
  await page.getByRole('button', { name: /^Predicted/ }).focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/sort=predicted/);
});

test('accessible charts expose names and data tables', async ({ page }) => {
  await page.goto('player/?id=legacy__demo_player_a1');
  const chart = page.getByRole('img', { name: /form/i });
  await expect(chart).toBeVisible();
  await page.getByText(/Data table: .*form/i).click();
  await expect(page.getByRole('table', { name: /form/i })).toBeVisible();
  await page.goto('history/');
  await expect(page.getByRole('img').first()).toBeVisible();
});

test('theme choice persists and dark tokens apply', async ({ page }) => {
  await page.goto('');
  await page.getByLabel('Theme').selectOption('dark');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(bg).toBe('rgb(13, 22, 38)');
});

test('time zone option relabels instants', async ({ page }) => {
  await page.goto('data-status/');
  const t = page.locator('time[data-instant]').first();
  const mel = await t.textContent();
  await page.getByLabel('Times').selectOption('UTC');
  await expect(t).not.toHaveText(mel!);
  await expect(t).toContainText('UTC');
});

test('deep-link reload keeps state and all links respect the base path', async ({ page, baseURL }) => {
  await page.goto('team/?id=demo_b&season=2025');
  await expect(page.locator('h1')).toContainText('Demo Club B');
  await page.reload();
  await expect(page.locator('h1')).toContainText('Demo Club B');
  await expect(page.getByLabel('Season', { exact: true })).toHaveValue('2025');
  const basePath = new URL(baseURL!).pathname;
  const hrefs = await page.locator('a[href^="/"]').evaluateAll((els) => els.map((e) => e.getAttribute('href')));
  for (const h of hrefs) expect(h!.startsWith(basePath), h!).toBe(true);
  const assets = await page.locator('script[src], link[href]').evaluateAll((els) => els.map((e) => e.getAttribute('src') ?? e.getAttribute('href')));
  for (const a of assets) expect(new URL(a!, page.url()).pathname.startsWith(basePath), a!).toBe(true);
});
