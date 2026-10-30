import { defineConfig } from '@playwright/test';

const env = (globalThis as any).process?.env ?? {};

export default defineConfig({
  testDir: './tests',
  timeout: 60_000,
  use: {
    baseURL: env.MOTION_EDIT_E2E_BASE_URL || 'http://127.0.0.1:8094',
    headless: true,
    launchOptions: {
      executablePath: env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined,
      args: ['--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
    },
  },
});
