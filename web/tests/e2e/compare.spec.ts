// W03: compare up to four validated players, warnings, copyable URL, CSV export.
import { test, expect, expectClean } from './helpers';

test.describe('compare', () => {
  test('compares players with common stats and era warning', async ({ page, problems }) => {
    await page.goto('compare/?players=legacy__demo_player_a1,legacy__demo_same_name_1990');
    const table = page.getByRole('table', { name: /career comparison/i });
    await expect(table).toBeVisible();
    await expect(table.getByRole('columnheader', { name: 'Demo Player A1' })).toBeVisible();
    await expect(table.getByRole('columnheader', { name: /Demo Same Name/ })).toBeVisible();
    await expect(page.getByTestId('compare-warnings')).toContainText(/era/i);
    await expect(page.getByLabel('Shareable link')).toHaveValue(/compare\/\?players=legacy__demo_player_a1/);
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: /Download comparison CSV/ }).click();
    const d = await download;
    expect(d.suggestedFilename()).toMatch(/\.csv$/);
    expectClean(problems);
  });

  test('rejects invalid ids and caps at four', async ({ page }) => {
    await page.goto('compare/?players=legacy__demo_player_a1,..%2Fx,legacy__demo_player_b1,legacy__demo_filler_01,legacy__demo_filler_02,legacy__demo_filler_03');
    await expect(page.getByTestId('compare-warnings')).toContainText(/ignored/i);
    await expect(page.getByRole('table', { name: /career comparison/i }).locator('thead th[scope="col"]')).toHaveCount(5);
  });

  test('add and remove players updates the URL', async ({ page }) => {
    await page.goto('compare/?players=legacy__demo_player_a1');
    await page.getByRole('button', { name: 'Remove Demo Player A1' }).click();
    await expect(page).toHaveURL(/compare\/$/);
    await expect(page.getByText(/choose up to four players/i)).toBeVisible();
  });
});
