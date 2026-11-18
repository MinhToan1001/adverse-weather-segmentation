import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  testDir: './tests', testMatch: '**/*.spec.mjs', workers: 1, timeout: 180_000,
  outputDir: '../outputs/web-studio/playwright',
  reporter: [['list'], ['junit', { outputFile: fileURLToPath(new URL('../outputs/web-studio/browser-tests.xml', import.meta.url)) }]],
  use: { baseURL: process.env.STUDIO_URL || 'http://127.0.0.1:3000',
    browserName: 'chromium', channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true, viewport: { width: 1600, height: 1060 }, trace: 'retain-on-failure' },
});
