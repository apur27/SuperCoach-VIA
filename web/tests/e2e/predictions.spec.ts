// W01: predictions filter / sort / back / share, expired and no-fixture states.
import { test, expect, expectClean, overrideResource } from './helpers';

test.describe('predictions explorer', () => {
  test('filters, sorts, restores on back/forward and shares via URL', async ({ page, problems, context }) => {
    await page.goto('predictions/');
    const table = page.getByRole('table', { name: /predicted disposals/i });
    await expect(table).toBeVisible();
    await expect(table.locator('tbody tr')).toHaveCount(14);
    await expect(page.getByRole('columnheader', { name: /interval/i })).toBeVisible();

    await page.getByLabel('Team').selectOption('demo_b');
    await expect(page).toHaveURL(/team=demo_b/);
    const clubCells = table.locator('tbody tr td[data-col="club"]');
    await expect(clubCells.first()).toHaveText('Demo Club B');
    for (const t of await clubCells.allTextContents()) expect(t).toBe('Demo Club B');
    const filteredCount = await table.locator('tbody tr').count();
    expect(filteredCount).toBeLessThan(14);

    await page.getByRole('button', { name: /^Predicted/ }).click();
    await expect(page).toHaveURL(/sort=predicted/);
    const header = page.getByRole('columnheader', { name: /Predicted/ });
    await expect(header).toHaveAttribute('aria-sort', 'descending');
    const values = (await table.locator('tbody td[data-col="predicted"]').allTextContents()).map(Number);
    expect([...values].sort((a, b) => b - a)).toEqual(values);

    await page.goBack();
    await expect(page).not.toHaveURL(/sort=/);
    await expect(page.getByLabel('Team')).toHaveValue('demo_b');
    await expect(header).toHaveAttribute('aria-sort', 'none');
    await page.goBack();
    await expect(page).not.toHaveURL(/team=/);
    await expect(page.getByLabel('Team')).toHaveValue('');
    await expect(table.locator('tbody tr')).toHaveCount(14);
    await page.goForward();
    await expect(page.getByLabel('Team')).toHaveValue('demo_b');

    await page.getByLabel('Player name').fill('Filler');
    await expect(page).toHaveURL(/q=Filler/);
    const shared = page.url();
    const other = await context.newPage();
    await other.goto(shared);
    await expect(other.getByLabel('Team')).toHaveValue('demo_b');
    await expect(other.getByLabel('Player name')).toHaveValue('Filler');
    await expect(other.getByRole('status').filter({ hasText: /rows? shown/ })).toBeVisible();
    expectClean(problems);
  });

  test('invalid URL values are ignored, not trusted', async ({ page }) => {
    await page.goto('predictions/?team=..%2F..%2Fx&sort=evil&dir=up&stage=%3Cscript%3E');
    await expect(page.getByRole('table', { name: /predicted disposals/i }).locator('tbody tr')).toHaveCount(14);
    await expect(page.getByLabel('Team')).toHaveValue('');
  });

  test('expired set is labelled and has no interval column', async ({ page }) => {
    await page.goto('predictions/');
    await page.getByLabel('Forecast set').selectOption('qf1');
    await expect(page.getByTestId('set-status')).toContainText(/expired/i);
    await expect(page.getByRole('columnheader', { name: /interval/i })).toHaveCount(0);
  });

  test('no valid future fixture is stated honestly', async ({ page }) => {
    await overrideResource(page, 'predictions/index.json', JSON.stringify({ current: null, status: 'unavailable', reason: 'no_valid_future_fixture', sets: [] }));
    await page.goto('predictions/');
    await expect(page.getByTestId('set-status')).toContainText('No valid future fixture');
    await expect(page.getByRole('table', { name: /predicted disposals/i })).toHaveCount(0);
  });

  test('watchlist and compare actions', async ({ page }) => {
    await page.goto('predictions/');
    const watch = page.getByRole('button', { name: 'Watch Demo Player A1' });
    await watch.click();
    await expect(watch).toHaveAttribute('aria-pressed', 'true');
    await page.getByRole('checkbox', { name: 'Compare Demo Player A1' }).check();
    await page.getByRole('checkbox', { name: 'Compare Demo Player B1' }).check();
    await page.getByRole('link', { name: /Compare selected \(2\)/ }).click();
    await expect(page).toHaveURL(/compare\/\?players=legacy__demo_player_a1(%2C|,)legacy__demo_player_b1/);
  });
});
