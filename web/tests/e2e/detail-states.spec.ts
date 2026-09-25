// PLAN 9.2/9.3: helpful not-found for unknown IDs on every detail shell; URL state with
// back/forward on explorers; last good view kept (and labelled) while revalidating; partial views.
import { test, expect, expectClean } from './helpers';

test.describe('unknown IDs', () => {
  const CASES: [string, RegExp, RegExp][] = [
    ['match/?id=demo__2026__r99__nobody', /match not found/i, /browse matches/i],
    ['team/?id=demo_zz&season=2026', /team not found/i, /teams/i],
    ['live/?match=no-such-feed', /live match not found/i, /./],
  ];
  for (const [path, heading, link] of CASES) {
    test(`${path} explains the ID is not in this release and links onward`, async ({ page, problems }) => {
      await page.goto(path);
      await expect(page.getByRole('heading', { name: heading })).toBeVisible();
      await expect(page.locator('[data-state="notfound"]').getByRole('link', { name: link }).first()).toBeVisible();
      await expect(page.locator('[data-state="loading"]')).toHaveCount(0);
      expectClean(problems);
    });
  }
  test('a hostile match ID is refused before any fetch', async ({ page }) => {
    const fetched: string[] = [];
    page.on('request', (r) => { if (r.url().includes('/matches/detail/')) fetched.push(r.url()); });
    await page.goto('match/?id=..%2F..%2Frelease');
    await expect(page.getByRole('heading', { name: /match not found/i })).toBeVisible();
    await expect(page.getByText(/not a valid match ID/i)).toBeVisible();
    expect(fetched).toEqual([]);
  });
  test('an unknown article slug is a real 404 page', async ({ page }) => {
    const res = await page.goto('articles/no-such-article/');
    expect(res?.status()).toBe(404);
    await expect(page.locator('h1')).toHaveText(/not found/i);
  });
});

test.describe('explorer URL state', () => {
  test('matches: season/status filters restore on back and forward', async ({ page }) => {
    await page.goto('matches/');
    const season = page.getByLabel('Season', { exact: true });
    await expect(page.getByRole('status').filter({ hasText: /matches shown/ })).toBeVisible();
    await season.selectOption('2025');
    await expect(page).toHaveURL(/season=2025/);
    await expect(page.getByRole('table', { name: /2025 matches/ })).toBeVisible();
    await page.getByLabel('Status').selectOption('complete');
    await expect(page).toHaveURL(/status=complete/);
    await page.goBack();
    await expect(page).not.toHaveURL(/status=/);
    await expect(page.getByLabel('Status')).toHaveValue('');
    await expect(season).toHaveValue('2025');
    await page.goBack();
    await expect(page).not.toHaveURL(/season=2025/);
    await expect(season).toHaveValue('2026');
    await page.goForward();
    await expect(season).toHaveValue('2025');
    await expect(page.getByRole('table', { name: /2025 matches/ })).toBeVisible();
  });

  test('players: search text and page restore on back', async ({ page }) => {
    await page.goto('players/?q=demo');
    await expect(page.getByRole('status').filter({ hasText: /players found/ })).toBeVisible();
    await page.getByRole('button', { name: 'Next' }).click();
    await expect(page).toHaveURL(/page=2/);
    await page.goBack();
    await expect(page).not.toHaveURL(/page=2/);
    await expect(page.getByLabel('Search all players')).toHaveValue('demo');
    await expect(page.getByRole('status').filter({ hasText: /players found/ })).toBeVisible();
  });
});

test.describe('revalidation and partial views', () => {
  test('switching season keeps the last good view, labelled, until the new data arrives', async ({ page }) => {
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    await page.route('**/data/*/matches/2025/index.json', async (route) => {
      await gate;
      await route.continue();
    });
    await page.goto('matches/?season=2026');
    await expect(page.getByRole('table', { name: /2026 matches/ })).toBeVisible();
    await page.getByLabel('Season', { exact: true }).selectOption('2025');
    await expect(page.getByText(/Updating matches… showing the previous view/)).toBeVisible();
    await expect(page.getByRole('table', { name: /2026 matches/ })).toBeVisible();
    await expect(page.locator('[aria-busy="true"]')).toHaveCount(1);
    release();
    await expect(page.getByRole('table', { name: /2025 matches/ })).toBeVisible();
    await expect(page.getByText(/Updating matches/)).toHaveCount(0);
  });

  test('a failed refresh keeps the previous view and marks it possibly out of date', async ({ page }) => {
    await page.route('**/data/*/matches/2025/index.json', (route) => route.fulfill({ status: 503, body: 'down' }));
    await page.goto('matches/?season=2026');
    await expect(page.getByRole('table', { name: /2026 matches/ })).toBeVisible();
    await page.getByLabel('Season', { exact: true }).selectOption('2025');
    await expect(page.locator('[data-state="error"]')).toBeVisible();
    await expect(page.getByText(/Showing the last successfully loaded matches; it may be out of date/)).toBeVisible();
    await expect(page.getByRole('table', { name: /2026 matches/ })).toBeVisible();
  });

  test('compare: one unavailable player is reported while the others still render (partial)', async ({ page }) => {
    await page.route('**/data/*/players/legacy__demo_player_b1.json', (route) => route.fulfill({ status: 503, body: 'down' }));
    await page.goto('compare/?players=legacy__demo_player_a1,legacy__demo_player_b1,legacy__nobody_here');
    await expect(page.getByRole('table', { name: /career comparison/i })).toContainText('Demo Player A1');
    await expect(page.getByText(/demo_player_b1: could not load/)).toBeVisible();
    await expect(page.getByText(/nobody_here: not in this release/)).toBeVisible();
  });
});
