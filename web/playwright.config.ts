import { defineConfig, devices } from '@playwright/test';

// Tests run against the PRODUCTION build (scripts/build-e2e.mjs) served on 127.0.0.1,
// at both supported base paths.
export default defineConfig({
  testDir: 'tests/e2e',
  outputDir: 'test-results/artifacts',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: process.env.CI ? 2 : 4,
  reporter: [['list'], ['json', { outputFile: 'test-results/e2e-report.json' }]],
  use: {
    // The locally cached browser is full Chromium (new headless mode).
    channel: 'chromium',
    trace: 'retain-on-failure',
    timezoneId: 'Australia/Melbourne',
    locale: 'en-AU',
    // Optional override for machines whose preinstalled Chromium differs from this Playwright pin.
    ...(process.env.SCVIA_CHROMIUM_PATH ? { launchOptions: { executablePath: process.env.SCVIA_CHROMIUM_PATH } } : {}),
  },
  projects: [
    { name: 'root', use: { ...devices['Desktop Chrome'], channel: 'chromium', baseURL: 'http://127.0.0.1:4401/' } },
    { name: 'subpath', testIgnore: /screenshots\.spec\.ts/, use: { ...devices['Desktop Chrome'], channel: 'chromium', baseURL: 'http://127.0.0.1:4402/SuperCoach-VIA/' } },
  ],
  webServer: [
    { command: 'node scripts/serve.mjs --dir .e2e-dist/root --base / --port 4401', url: 'http://127.0.0.1:4401/', reuseExistingServer: false },
    { command: 'node scripts/serve.mjs --dir .e2e-dist/sub --base /SuperCoach-VIA/ --port 4402', url: 'http://127.0.0.1:4402/SuperCoach-VIA/', reuseExistingServer: false },
  ],
});
