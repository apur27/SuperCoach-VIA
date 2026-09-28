// Era and Brownlow tables, and game-log opponents restored from the shared match index.
import { test, expect } from './helpers';

const PLAYER_A1 = 'k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex';

test('era summary table has published rows', async ({ page }) => {
  await page.goto('history/eras/');
  await expect(page.getByRole('heading', { name: 'Era summary' })).toBeVisible();
  const table = page.getByRole('table', { name: /era summary/i });
  await expect(table).toBeVisible();
  await expect(table.getByRole('cell', { name: '1990s' })).toBeVisible();
  await expect(table.getByRole('cell', { name: 'partial' })).toBeVisible();
  await expect(page.getByText(/not that the value was zero/i)).toBeVisible();
});

test('Brownlow proxy table is populated and stays labelled as a proxy', async ({ page }) => {
  await page.goto('history/brownlow/');
  await expect(page.getByText('Proxy, not votes.')).toBeVisible();
  const table = page.getByRole('table', { name: /not Brownlow votes/i });
  await expect(table).toBeVisible();
  await expect(table.getByRole('cell', { name: 'Demo Player A1' })).toBeVisible();
  await expect(table.getByRole('cell', { name: /not Brownlow votes/ }).first()).toBeVisible();
});

test('player game log shows the opponent from the shared match index', async ({ page }) => {
  await page.goto(`player/?id=${PLAYER_A1}`);
  await expect(page.getByRole('heading', { name: 'Demo Player A1' })).toBeVisible();
  const log = page.getByRole('table', { name: /game log/i });
  await expect(log).toBeVisible();
  await expect(log.getByRole('cell', { name: 'Demo Club B' }).first()).toBeVisible();
});
