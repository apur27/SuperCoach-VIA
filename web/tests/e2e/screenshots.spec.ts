import { test } from '@playwright/test';

const PAGES = [
  ['overview', ''],
  ['predictions', 'predictions/'],
  ['player', 'player/?id=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex'],
  ['comparison', 'compare/?players=k.bGVnYWN5OmRlbW9fcGxheWVyX2Ex,k.bGVnYWN5OmRlbW9fcGxheWVyX2Ix'],
  ['data-status', 'data-status/'],
] as const;
const SIZES = [
  ['w320', 320, 700],
  ['w375', 375, 812],
  ['w768', 768, 900],
  ['w1440', 1440, 900],
] as const;
const THEMES = ['light', 'dark'] as const;

for (const [name, path] of PAGES) {
  for (const [size, width, height] of SIZES) {
    for (const theme of THEMES) {
      test(`screenshot ${name} ${size} ${theme}`, async ({ page }) => {
        await page.setViewportSize({ width, height });
        await page.emulateMedia({ colorScheme: theme });
        await page.goto(path);
        await page.waitForLoadState('networkidle');
        await page.getByLabel('Theme').selectOption(theme);
        await page.screenshot({ path: `test-results/screenshots/${name}-${size}-${theme}.png`, fullPage: true });
      });
    }
  }
}
