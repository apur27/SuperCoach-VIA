import { test, expect, noHorizontalOverflow } from './helpers';

const themes = [
  'It is noncommercial, has no affiliation with gambling services and is not intended to encourage betting.',
  'I have played SuperCoach with the same group for over a decade.',
  'friends and colleagues who introduced me to AFL and SuperCoach',
  'Cranbourne Junior Football Club, who welcomed my son',
  'volunteers giving their time on cold mornings',
  'AFL’s contribution to a multicultural Australia',
  'honours the players of the past, present and future',
];

for (const width of [320, 390, 768, 1440]) {
  for (const theme of ['light', 'dark']) {
    test(`homepage purpose and native story link remain readable at ${width}px in ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('');
      await page.getByLabel('Theme', { exact: true }).selectOption(theme);
      await expect(page.locator('.home-intro .lede')).toHaveText('A noncommercial AFL project, inspired by years of SuperCoach with friends and gratitude for the people who make the game welcoming.');
      const story = page.getByRole('region', { name: 'Why this project exists', exact: true });
      await expect(story).toBeVisible();
      await expect(story.getByText('From the project creator', { exact: true })).toBeVisible();
      for (const text of themes) await expect(story).toContainText(text);
      const searchBox = await page.getByRole('search', { name: 'Quick player search' }).boundingBox();
      const storyBox = await story.boundingBox();
      const resultsBox = await page.getByRole('heading', { name: 'Recent results', exact: true }).boundingBox();
      expect(searchBox!.y).toBeLessThan(storyBox!.y);
      expect(storyBox!.y + storyBox!.height).toBeLessThan(resultsBox!.y);
      if (width === 390) expect(searchBox!.y).toBeLessThan(900);
      const prose = await story.locator('.prose').boundingBox();
      expect(prose!.width).toBeLessThanOrEqual(720);
      expect(await noHorizontalOverflow(page)).toBe(true);
      const link = page.getByRole('link', { name: 'Why this project exists', exact: true });
      await expect(link).toHaveAttribute('href', '#why-this-project');
      await link.focus();
      await page.keyboard.press('Enter');
      await expect(page).toHaveURL(/#why-this-project$/);
      const heading = await story.getByRole('heading', { name: 'Why this project exists', exact: true }).boundingBox();
      expect(heading!.y).toBeGreaterThanOrEqual(0);
      expect(heading!.y).toBeLessThan(900);
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
    const story = page.getByRole('region', { name: 'Why this project exists', exact: true });
    for (const text of themes) await expect(story).toContainText(text);
    await expect(story).toBeVisible();
    await page.getByRole('link', { name: 'Why this project exists', exact: true }).click();
    await expect(page).toHaveURL(/#why-this-project$/);
    await expect(story.getByRole('heading', { name: 'Why this project exists', exact: true })).toBeVisible();
  });
});
