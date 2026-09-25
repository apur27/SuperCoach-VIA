// W02: player search, disambiguation, lazy index, keyboard selection, pagination.
import { test, expect, expectClean } from './helpers';

test.describe('player directory', () => {
  test('index is only fetched on interaction', async ({ page }) => {
    const requests: string[] = [];
    page.on('request', (r) => requests.push(r.url()));
    await page.goto('players/');
    await page.waitForLoadState('networkidle');
    expect(requests.some((u) => u.includes('/players/index.json'))).toBe(false);
    await page.getByLabel('Search all players').focus();
    await page.waitForResponse((r) => r.url().includes('/players/index.json'));
  });

  test('diacritic-insensitive search with live result count and keyboard selection', async ({ page, problems }) => {
    await page.goto('players/');
    const box = page.getByLabel('Search all players');
    await box.fill('zoe arger');
    await expect(page.getByRole('status').filter({ hasText: /1 player found/ })).toBeVisible();
    await expect(page).toHaveURL(/q=zoe/);
    await box.press('ArrowDown');
    await expect(page.getByRole('option', { name: /Demo Zoë Ärger/ })).toHaveAttribute('aria-selected', 'true');
    await box.press('Enter');
    await expect(page).toHaveURL(/player\/\?id=legacy__demo_zoe_arger/);
    await expect(page.locator('h1')).toContainText('Demo Zoë Ärger');
    expectClean(problems);
  });

  test('same-name players are disambiguated by clubs and era', async ({ page }) => {
    await page.goto('players/?q=same+name');
    const opts = page.getByRole('listbox').getByRole('option', { name: /Demo Same Name/ });
    await expect(opts).toHaveCount(2);
    await expect(opts.nth(0)).toContainText(/Demo Club A, Demo Club C, 2025–2026|Demo Club Old \(historical\), 1994/);
    const texts = await opts.allTextContents();
    expect(new Set(texts).size).toBe(2);
  });

  test('filters and pagination 25/50/100 with page reset on filter change', async ({ page }) => {
    await page.goto('players/?q=demo');
    await expect(page.getByRole('status').filter({ hasText: /66 players found/ })).toBeVisible();
    await expect(page.getByRole('listbox').getByRole('option')).toHaveCount(25);
    await page.getByRole('button', { name: 'Next' }).click();
    await expect(page).toHaveURL(/page=2/);
    await page.getByLabel('Rows per page').selectOption('50');
    await expect(page).not.toHaveURL(/page=2/);
    await expect(page.getByRole('listbox').getByRole('option')).toHaveCount(50);
    await page.getByLabel('Status').selectOption('retired');
    await expect(page.getByRole('status').filter({ hasText: /2 players found/ })).toBeVisible();
  });

  test('unknown player id shows a helpful not-found state', async ({ page }) => {
    await page.goto('player/?id=legacy__nobody_here');
    await expect(page.getByRole('heading', { name: /player not found/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /search all players/i })).toBeVisible();
    await page.goto('player/?id=..%2F..%2Fetc');
    await expect(page.getByText(/not a valid player ID/i)).toBeVisible();
  });

  test('player detail: null coverage is not zero; one season log fetched at a time', async ({ page }) => {
    const logs: string[] = [];
    page.on('request', (r) => { if (r.url().includes('/player-games/')) logs.push(r.url()); });
    await page.goto('player/?id=legacy__demo_player_sparse');
    await expect(page.locator('h1')).toContainText('Demo Player Sparse');
    await expect(page.getByRole('table', { name: /career/i })).toContainText('not recorded');
    await page.goto('player/?id=legacy__demo_player_a1');
    await expect(page.getByRole('table', { name: /game log/i })).toBeVisible();
    expect(logs.filter((u) => u.includes('demo_player_a1'))).toHaveLength(1);
    await page.getByLabel('Season', { exact: true }).selectOption('2025');
    await expect(page.getByRole('table', { name: /2025 game log/i })).toBeVisible();
    expect(logs.filter((u) => u.includes('demo_player_a1'))).toHaveLength(2);
    await expect(page.getByRole('img', { name: /form/i })).toBeVisible();
    await expect(page.getByTestId('player-forecast')).toContainText(/predicted/i);
  });
});
