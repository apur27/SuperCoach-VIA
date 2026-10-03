import { readFileSync } from 'node:fs';
import type { Locator } from '@playwright/test';
import { test, expect, noHorizontalOverflow } from './helpers';

const heading = 'Why this repo exists';
const readme = readFileSync(new URL('../../../README.md', import.meta.url), 'utf8');
const sourceSection = readme.slice(readme.indexOf(`## ${heading}`)).split(/\n## /)[0]!;
const expectedBlocks = sourceSection.trim().split(/\n\s*\n/).map((block) => block.replace(/^## /, '').replace(/^> /, ''));
const normalizeWhitespace = (text: string) => text.replace(/\s+/g, ' ').trim();

async function expectVerbatimStory(story: Locator) {
  await expect(story.locator('.prose > blockquote')).toHaveCount(1);
  await expect(story.locator('.prose > p')).toHaveCount(6);
  const rendered = await story.locator('h2, .prose > blockquote, .prose > p').allTextContents();
  expect(rendered.map(normalizeWhitespace)).toEqual(expectedBlocks.map(normalizeWhitespace));
}

for (const width of [320, 390, 768, 1440]) {
  for (const theme of ['light', 'dark']) {
    test(`homepage purpose and native story link remain readable at ${width}px in ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('');
      await page.getByLabel('Theme', { exact: true }).selectOption(theme);
      await expect(page.locator('.home-intro .lede')).toHaveText('SuperCoach VIA brings together player and match history, comparisons, rankings and downloads.');
      const story = page.getByRole('region', { name: heading, exact: true });
      await expect(story).toBeVisible();
      await expectVerbatimStory(story);
      const searchBox = await page.getByRole('search', { name: 'Quick player search' }).boundingBox();
      const storyBox = await story.boundingBox();
      const resultsBox = await page.getByRole('heading', { name: 'Recent results', exact: true }).boundingBox();
      expect(searchBox!.y).toBeLessThan(storyBox!.y);
      expect(storyBox!.y + storyBox!.height).toBeLessThan(resultsBox!.y);
      if (width === 390) expect(searchBox!.y).toBeLessThan(900);
      const prose = await story.locator('.prose').boundingBox();
      expect(prose!.width).toBeLessThanOrEqual(720);
      expect(await noHorizontalOverflow(page)).toBe(true);
      const link = page.getByRole('link', { name: heading, exact: true });
      await expect(link).toHaveAttribute('href', '#why-this-project');
      await link.focus();
      await page.keyboard.press('Enter');
      await expect(page).toHaveURL(/#why-this-project$/);
      const headingBox = await story.getByRole('heading', { name: heading, exact: true }).boundingBox();
      expect(headingBox!.y).toBeGreaterThanOrEqual(0);
      expect(headingBox!.y).toBeLessThan(900);
      await expect(page.getByTestId('warnings')).toBeVisible();
      await expect(page.getByTestId('forecast-status')).toBeVisible();
    });
  }
}

test.describe('no JavaScript', () => {
  test.use({ javaScriptEnabled: false });
  test('homepage purpose story stays visible and its ordinary anchor works', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 900 });
    await page.goto('');
    const story = page.getByRole('region', { name: heading, exact: true });
    await expectVerbatimStory(story);
    await expect(story).toBeVisible();
    await page.getByRole('link', { name: heading, exact: true }).click();
    await expect(page).toHaveURL(/#why-this-project$/);
    await expect(story.getByRole('heading', { name: heading, exact: true })).toBeVisible();
  });
});
