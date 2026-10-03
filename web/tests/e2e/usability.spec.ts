import { readFileSync } from 'node:fs';
import { test, expect, noHorizontalOverflow, overrideResource } from './helpers';

const PLAYER = 'player/?id=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex';

for (const width of [320, 390, 768]) {
  test(`compact header and intentional mobile navigation at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('');
    const brand = await page.locator('.brand').boundingBox();
    const menu = page.getByRole('button', { name: 'Menu', exact: true });
    const button = await menu.boundingBox();
    expect(button).not.toBeNull();
    expect(Math.abs(brand!.y - button!.y)).toBeLessThan(16);
    if (width === 390) expect((await page.locator('.site-header').boundingBox())!.height).toBeLessThanOrEqual(112);
    await menu.focus();
    await page.keyboard.press('Enter');
    const nav = page.getByRole('navigation', { name: 'Main', exact: true });
    await expect(nav.getByRole('link', { name: 'Home', exact: true })).toBeVisible();
    await expect(nav.getByRole('link', { name: 'Rankings', exact: true })).toBeVisible();
    await nav.getByRole('link', { name: 'Rankings', exact: true }).focus();
    await page.keyboard.press('Escape');
    await expect(menu).toBeFocused();
    await expect(nav).toBeHidden();
  });
}

test('desktop navigation and content share a gutter', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('');
  const nav = await page.locator('.nav-primary').boundingBox();
  const main = await page.locator('main').boundingBox();
  expect(Math.abs(nav!.x - main!.x)).toBeLessThanOrEqual(1);
});

test('homepage explains the site, offers destinations and keeps highlights complete', async ({ page, baseURL }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto('');
  await expect(page.locator('h1')).toHaveText('Explore AFL players, matches and history');
  const search = page.getByRole('search', { name: 'Quick player search' });
  expect((await search.boundingBox())!.y).toBeLessThan(900);
  const destinations = page.getByRole('navigation', { name: 'Explore AFL data' });
  for (const [name, path] of [['Rankings', 'history/'], ['Matches', 'matches/'], ['Compare players', 'compare/'], ['Downloads', 'downloads/']] as const) {
    await expect(destinations.getByRole('link', { name, exact: true })).toHaveAttribute('href', `${new URL(baseURL!).pathname}${path}`);
  }
  await expect(page.getByTestId('warnings')).toBeVisible();
  await expect(page.getByTestId('forecast-status')).toBeVisible();
  await expect(page.getByTestId('upcoming')).toBeVisible();
  const preview = page.getByRole('table', { name: 'Season leaders preview', exact: true });
  expect(await preview.locator('tbody tr').count()).toBeLessThanOrEqual(5);
  const all = page.getByRole('table', { name: 'All season leaders', exact: true });
  await expect(all).toBeHidden();
  const disclosure = page.getByText('View all season leaders in this snapshot', { exact: true });
  await disclosure.focus();
  await page.keyboard.press('Enter');
  await expect(all).toBeVisible();
  const expected = await page.evaluate(async () => (await (await fetch(`${document.documentElement.dataset.base}data/${document.documentElement.dataset.release}/overview.json`)).json()).leaders.length);
  await expect(all.locator('tbody tr')).toHaveCount(expected);
  await expect(page.getByText(/DEMO excerpt/)).toHaveCount(0);
  await expect(page.locator('main').getByRole('link', { name: 'All articles', exact: true })).toBeVisible();
  await search.getByRole('searchbox').fill('Demo Player A1');
  await search.getByRole('button', { name: 'Search players' }).click();
  await expect(page).toHaveURL(/players\/\?q=Demo\+Player\+A1/);
  await page.getByRole('option', { name: /^Demo Player A1 / }).click();
  await expect(page.getByRole('heading', { name: 'Demo Player A1', exact: true })).toBeVisible();
});

for (const width of [320, 390]) {
  for (const theme of ['light', 'dark']) {
    test(`table identities and all columns survive scrolling at ${width}px in ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('');
      await page.getByLabel('Theme', { exact: true }).selectOption(theme);
      const checkRegion = async (region: ReturnType<typeof page.getByRole>, identity: string) => {
        await expect(region).toBeVisible();
        await expect(region.locator('xpath=preceding-sibling::p[1]')).toHaveText('Scroll sideways for more columns');
        const firstIdentity = region.locator('tbody tr').first().getByRole('rowheader');
        const before = await firstIdentity.textContent();
        await region.evaluate((el) => { el.scrollLeft = el.scrollWidth; });
        const bounds = await region.boundingBox();
        const pinned = await firstIdentity.boundingBox();
        expect(pinned!.x).toBeGreaterThanOrEqual(bounds!.x - 1);
        expect(pinned!.x + pinned!.width).toBeLessThan(bounds!.x + bounds!.width - 32);
        expect(await firstIdentity.textContent()).toBe(before);
        expect((await region.locator('thead th').filter({ hasText: identity }).first().boundingBox())!.x).toBeGreaterThanOrEqual(bounds!.x - 1);
        const last = await region.locator('tbody tr').first().locator('td').last().boundingBox();
        expect(last!.x + last!.width).toBeLessThanOrEqual(bounds!.x + bounds!.width + 1);
        await region.scrollIntoViewIfNeeded();
        await region.focus();
        await expect(region).toBeFocused();
        const farRight = await region.evaluate((el) => el.scrollLeft);
        await region.press('ArrowLeft', { delay: 50 });
        await expect.poll(() => region.evaluate((el) => el.scrollLeft)).toBeLessThan(farRight);
        expect(await noHorizontalOverflow(page)).toBe(true);
      };
      await checkRegion(page.getByTestId('recent'), 'Match');
      await page.goto('history/');
      const history = page.getByRole('region', { name: 'Scrollable table: DEMO career disposals', exact: true });
      await expect(history.getByRole('columnheader', { name: /^Games with data/ })).toBeVisible();
      await expect(history.getByRole('columnheader', { name: 'Clubs' })).toBeVisible();
      await expect(history.getByRole('columnheader', { name: 'Seasons' })).toBeVisible();
      await checkRegion(history, 'Player');
      await history.evaluate((el) => { el.scrollLeft = 0; });
      await history.getByRole('button', { name: /^Coverage/ }).click();
      await expect(page).toHaveURL(/sort=coverage/);
      await page.goto(PLAYER);
      await expect(page.getByRole('heading', { name: 'Demo Player A1', exact: true })).toBeVisible();
      await page.getByRole('navigation', { name: 'Player sections' }).getByRole('link', { name: 'Career statistics' }).click();
      await expect(page).toHaveURL(/#career-h$/);
      await checkRegion(page.getByRole('region', { name: 'Scrollable table: Career statistics', exact: true }), 'Statistic');
      await checkRegion(page.getByRole('region', { name: 'Scrollable table: Season summary', exact: true }), 'Season');
      await checkRegion(page.getByRole('region', { name: /Scrollable table: .* game log/ }), 'Date');
    });
  }
}

test('snapshot selection is explained in details and freshness remains compact in every zone', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 900 });
  await page.goto('');
  const strip = page.getByTestId('freshness');
  for (const zone of ['Australia/Melbourne', 'UTC', 'local']) {
    await page.getByLabel('Times').selectOption(zone);
    expect((await strip.boundingBox())!.height).toBeLessThanOrEqual(170);
  }
  await expect(strip).toContainText('Freshness at build');
  await expect(strip.getByText('Snapshot selection', { exact: true })).toBeHidden();
  await strip.getByText('Release details', { exact: true }).click();
  await expect(strip).toContainText('Snapshot selection records whether this build used the locally selected dataset. See Data status for source audit findings.');
  await expect(strip.getByText('Snapshot selection', { exact: true })).toBeVisible();
});

// The compact fixture uses short labels. Replace only its display keys in-memory to
// exercise a published-width metric without changing the release on disk or its values.
for (const width of [320, 390, 640]) {
  for (const theme of ['light', 'dark']) {
    test(`long player headers stay readable beside sticky identities at ${width}px in ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: width === 640 ? 400 : 900 });
      const path = 'players/k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex.json';
      const detail = JSON.parse(readFileSync(new URL(`../fixtures/demo-release/${path}`, import.meta.url), 'utf8'));
      detail.stat_names[detail.stat_names.length - 1] = 'uncontested_possessions';
      await overrideResource(page, path, JSON.stringify(detail));
      await page.goto(PLAYER);
      await page.getByLabel('Theme', { exact: true }).selectOption(theme);
      for (const [regionName, label] of [
        ['Scrollable table: Season summary', 'Uncontested possessions per game'],
        ['Scrollable table: 2026 game log', 'Uncontested possessions'],
      ] as const) {
        const region = page.getByRole('region', { name: regionName, exact: true });
        if (regionName.endsWith('game log')) {
          await expect(region).toBeVisible();
          await region.locator('thead th').last().evaluate((el) => { el.textContent = 'Uncontested possessions'; });
        }
        const heading = region.getByRole('columnheader', { name: label, exact: true });
        await expect(heading).toBeVisible();
        await region.evaluate((el) => { el.scrollLeft = el.scrollWidth; });
        const bounds = await region.boundingBox();
        const identity = await region.locator('thead .row-identity').boundingBox();
        const lines = await heading.evaluate((el) => {
          const text = document.createRange();
          text.selectNodeContents(el);
          return [...text.getClientRects()].map((r) => ({ left: r.left, right: r.right }));
        });
        expect(lines.length).toBeGreaterThan(0);
        for (const line of lines) {
          expect(line.left, label).toBeGreaterThanOrEqual(identity!.x + identity!.width);
          expect(line.right, label).toBeLessThanOrEqual(bounds!.x + bounds!.width - 1);
        }
        const values = await region.locator('tbody tr').first().locator('td').last().boundingBox();
        expect(values!.x + values!.width).toBeLessThanOrEqual(bounds!.x + bounds!.width + 1);
        await region.scrollIntoViewIfNeeded();
        await region.focus();
        await expect(region).toBeFocused();
        const farRight = await region.evaluate((el) => el.scrollLeft);
        await region.press('ArrowLeft', { delay: 50 });
        await expect.poll(() => region.evaluate((el) => el.scrollLeft)).toBeLessThan(farRight);
        expect(await noHorizontalOverflow(page)).toBe(true);
      }
    });
  }
}
