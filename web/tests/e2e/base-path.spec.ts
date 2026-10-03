import { test, expect, expectClean } from './helpers';

// Verify the deployed casing itself, because a case-insensitive assertion can hide
// links that return 404 on Pages even when the local directory name is unchanged.
test('canonical base preserves exact case for assets and direct player links', async ({ page, baseURL, problems }) => {
  const expectedBase = new URL(baseURL!).pathname;
  const failures: string[] = [];
  page.on('response', (response) => {
    if (response.status() >= 400) failures.push(`${response.status()} ${new URL(response.url()).pathname}`);
  });
  const home = await page.goto(baseURL!);
  expect(home?.status()).toBe(200);
  await expect(page.locator('html')).toHaveAttribute('data-base', expectedBase);
  expect(new URL(page.url()).pathname).toBe(expectedBase);
  const assets = await page.locator('script[src], link[rel="stylesheet"], link[rel="icon"]').evaluateAll((els) => els.map((el) => el.getAttribute('src') ?? el.getAttribute('href')));
  expect(assets.length).toBeGreaterThan(0);
  for (const asset of assets) expect(new URL(asset!, page.url()).pathname.startsWith(expectedBase), asset!).toBe(true);
  const playerPath = `${expectedBase}player/?id=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex`;
  const player = await page.goto(new URL(playerPath, baseURL!).href);
  expect(player?.status()).toBe(200);
  expect(new URL(page.url()).pathname).toBe(`${expectedBase}player/`);
  await expect(page.getByRole('heading', { name: 'Demo Player A1', exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Demo Player A1', exact: true })).toBeVisible();
  const match = page.getByRole('region', { name: 'Scrollable table: 2026 game log', exact: true }).getByRole('link').first();
  const matchHref = await match.getAttribute('href');
  expect(new URL(matchHref!, baseURL!).pathname).toBe(`${expectedBase}match/`);
  expect(new URL(matchHref!, baseURL!).search.startsWith('?id=')).toBe(true);
  await match.click();
  expect(new URL(page.url()).pathname).toBe(`${expectedBase}match/`);
  await expect(page.locator('h1')).toContainText('Demo Club');
  await page.waitForLoadState('networkidle');
  expect(failures).toEqual([]);
  expectClean(problems);
});
