import { expect, test } from '@playwright/test';

const env = (globalThis as any).process?.env ?? {};
const motionId = env.MOTION_EDIT_E2E_MOTION_ID as string | undefined;
test.skip(!motionId, 'requires a real playback server and registered motion');

async function openPlayback(page: any) {
  const response = await page.request.post('/api/session/load', { data: { motion_id: motionId } });
  expect(response.ok()).toBeTruthy();
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/, { timeout: 45_000 });
}

test('display layers remain independent, frames are bounded, playback reuses geometry', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await openPlayback(page);
  await page.locator('[data-tab="display"]').click();
  await page.locator('#surfaceToggle').check();
  await page.locator('#terrainToggle').uncheck();
  await expect(page.locator('#scene')).toHaveAttribute('data-surfaces-visible', 'true');
  await expect(page.locator('#scene')).toHaveAttribute('data-terrain-visible', 'false');
  await page.locator('#terrainToggle').check();
  await page.locator('#surfaceToggle').uncheck();
  await expect(page.locator('#scene')).toHaveAttribute('data-surfaces-visible', 'false');
  await expect(page.locator('#scene')).toHaveAttribute('data-terrain-visible', 'true');
  await page.locator('[data-tab="contact"]').click();
  await page.waitForTimeout(150);
  const before = Number(await page.locator('#scene').getAttribute('data-geometry-count'));
  for (let frame = 0; frame < 80; frame++) {
    await page.locator('#frameInput').fill(String(frame));
    await page.locator('#frameInput').dispatchEvent('change');
  }
  await page.waitForTimeout(150);
  expect(Number(await page.locator('#scene').getAttribute('data-geometry-count'))).toBe(before);
  await page.locator('#frameInput').fill('999999');
  await page.locator('#frameInput').dispatchEvent('change');
  expect(await page.locator('#frameInput').inputValue()).toBe(await page.locator('#frameInput').getAttribute('max'));
  await page.locator('#fpsInput').fill('');
  await page.locator('#fpsInput').dispatchEvent('change');
  expect(Number(await page.locator('#fpsInput').inputValue())).toBeGreaterThan(0);
  await page.locator('#fpsInput').fill('999');
  await page.locator('#fpsInput').dispatchEvent('change');
  await expect(page.locator('#fpsInput')).toHaveValue('240');
  await page.locator('#playBtn').click();
  await page.waitForTimeout(200);
  await page.locator('#playBtn').click();
  expect(errors).toEqual([]);
});

test('registered motion picker searches the same motion identities', async ({ page }) => {
  await openPlayback(page);
  await page.locator('#openMotionBtn').click();
  await expect(page.locator('#motionPicker')).toBeVisible();
  await page.locator('#motionSearch').fill('nonexistent-motion-xyz');
  await expect(page.locator('#motionPickerList')).toHaveText('没有匹配的轨迹');
  await page.locator('#motionSearch').fill(motionId!);
  await expect(page.locator('.motion-picker-item').first()).toContainText(motionId!);
  await page.locator('.motion-picker-item').first().click();
  await expect(page.locator('#motionPicker')).not.toBeVisible();
  await expect(page.locator('#motionLabel')).toHaveText(motionId!);
});

test('session load errors persist and do not silently open a different motion', async ({ page }) => {
  let loads = 0;
  page.on('request', request => { if (request.url().endsWith('/api/session/load')) loads++; });
  await page.route('**/api/session', route => route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'session load failed for acceptance' }) }));
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toContainText('session load failed for acceptance');
  await page.waitForTimeout(3200);
  await expect(page.locator('#sceneStatus')).toBeVisible();
  await expect(page.locator('#retryLoad')).toBeVisible();
  expect(loads).toBe(0);
  await page.unroute('**/api/session');
  await page.locator('#retryLoad').click();
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/, { timeout: 45_000 });
});


test('registered playback exposes no editing or generation controls', async ({ page }) => {
  const mutations: string[] = [];
  page.on('request', request => {
    if (/\/api\/session\/(?:generate|save|validate|move|restore|undo|redo|settings)/.test(request.url())) mutations.push(request.url());
  });
  await openPlayback(page);
  await expect(page.locator('#saveBtn,#undoBtn,#redoBtn,#generateBtn,[data-tab="output"]')).toHaveCount(0);
  await expect(page.locator('#playBtn')).toBeEnabled();
  await page.locator('#nextBtn').click();
  await page.locator('[data-tab="display"]').click();
  await expect(page.locator('#contactSampleToggle')).toBeVisible();
  expect(mutations).toEqual([]);
});
