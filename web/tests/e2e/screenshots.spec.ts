import { test } from '@playwright/test';

const PAGES = [
  ['overview', ''],
  ['predictions', 'predictions/'],
  ['player', 'player/?id=legacy__demo_player_a1'],
  ['comparison', 'compare/?players=legacy__demo_player_a1,legacy__demo_player_b1'],
  ['data-status', 'data-status/'],
] as const;
const SIZES = [['mobile', 375, 812], ['desktop', 1440, 900]] as const;

for (const [name, path] of PAGES) {
  for (const [size, width, height] of SIZES) {
    test(`screenshot ${name} ${size}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await page.goto(path);
      await page.waitForLoadState('networkidle');
      await page.screenshot({ path: `test-results/screenshots/${name}-${size}.png`, fullPage: true });
    });
  }
}
