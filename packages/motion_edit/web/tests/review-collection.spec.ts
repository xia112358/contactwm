import { expect, test } from '@playwright/test';

const env = (globalThis as any).process?.env ?? {};
const cases = env.MOTION_EDIT_E2E_CASES as string | undefined;
test.skip(!cases, 'set MOTION_EDIT_E2E_CASES to a real playback cases.json');

async function loadCase(page: any, index = 1) {
  const response = await page.request.post('/api/review/load', { data: { path: cases, case_index: index } });
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json();
}

test('shared open dialog switches between registered playback and case review', async ({ page }) => {
  test.skip(!env.MOTION_EDIT_E2E_MOTION_ID, 'requires a registered motion');
  const response = await page.request.post('/api/session/load', { data: { motion_id: env.MOTION_EDIT_E2E_MOTION_ID } });
  expect(response.ok()).toBeTruthy();
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/, { timeout: 45_000 });
  await page.locator('#openMotionBtn').click();
  await page.locator('#reviewPath').fill(cases!);
  await page.locator('#reviewOpen').click();
  await expect(page.locator('#motionPicker')).not.toBeVisible();
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/, { timeout: 45_000 });
  await expect(page.locator('#reviewCase')).toBeVisible();
  await expect(page.locator('#saveBtn,[data-tab="output"]')).toHaveCount(0);
  await page.locator('#openMotionBtn').click();
  await page.locator('#motionSearch').fill(env.MOTION_EDIT_E2E_MOTION_ID);
  await page.locator('.motion-picker-item').first().click();
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/, { timeout: 45_000 });
  await expect(page.locator('#motionLabel')).toHaveText(env.MOTION_EDIT_E2E_MOTION_ID);
  await expect(page.locator('#saveBtn,[data-tab="output"]')).toHaveCount(0);
});

for (const width of [1440, 390]) {
  test(`case review, diagnostics and comparison at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 960 });
    const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
    const payload = await loadCase(page);
    await page.goto('/');
    await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/, { timeout: 45_000 });
    await expect(page.locator('#reviewCase')).toHaveValue('1');
    await expect(page.locator('#reviewReference')).toHaveValue('0');
    await expect(page.locator('[data-tab="output"]')).not.toBeVisible();
    await expect(page.locator('#saveBtn')).not.toBeVisible();
    await expect(page.locator('#episodeSelect')).not.toBeVisible();
    await expect(page.locator('#reviewPanel')).toContainText(payload.review.notice);
    await expect(page.locator('#playbackNotice')).not.toBeVisible();
    await expect(page.locator('#fpsInput')).toBeVisible();
    await expect(page.locator('#scene')).toHaveAttribute('data-reference-frame', '0');
    await page.locator('#frameInput').fill('20'); await page.locator('#frameInput').dispatchEvent('change');
    await expect(page.locator('#scene')).toHaveAttribute('data-reference-frame', '20');
    await page.locator('#reviewLayout').selectOption('overlay');
    await expect(page.locator('#comparisonClock')).toContainText('按时间对齐');
    await page.locator('#reviewCase').selectOption('4');
    await expect(page.locator('#reviewCase')).toHaveValue('4');
    await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/);
    await expect(page.locator('#reviewReference')).toHaveValue('3');
    await page.locator('#reviewWorst').click();
    const active = await page.request.get('/api/session');
    const depth = (await active.json()).review.depth_mm as number[];
    const peak = depth.reduce((best, value, i) => value > depth[best] ? i : best, 0);
    await expect(page.locator('#frameInput')).toHaveValue(String(peak));
    await expect(page.locator('#reviewMetrics')).toContainText(Math.max(...depth).toFixed(2));
    await page.locator('#reviewReference').selectOption('-1');
    await expect(page.locator('#reviewReference')).toHaveValue('-1');
    await expect(page.locator('#reviewLayout')).not.toBeVisible();
    await page.locator('#reviewReference').selectOption('3');
    await expect(page.locator('#reviewReference')).toHaveValue('3');
    await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/);
    await page.locator('[data-tab="display"]').click();
    await page.locator('#terrainOpacity').evaluate((input: HTMLInputElement) => { input.value = '0.4'; input.dispatchEvent(new Event('input')); });
    await page.locator('#focusRobot').click();
    await page.locator('#terrainToggle').uncheck();
    await expect(page.locator('#scene')).toHaveAttribute('data-terrain-visible', 'false');
    await page.locator('#terrainToggle').check();
    const before = Number(await page.locator('#scene').getAttribute('data-geometry-count'));
    for (let frame = 0; frame < 20; frame++) {
      await page.locator('#frameInput').fill(String(frame)); await page.locator('#frameInput').dispatchEvent('change');
    }
    await page.waitForTimeout(100);
    expect(Number(await page.locator('#scene').getAttribute('data-geometry-count'))).toBe(before);
    await page.locator('[data-tab="contact"]').click();
    const bounds = await page.locator('#reviewCase').boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0); expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
    expect(payload.contact_force.masks).toEqual([]);
    expect(errors).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath(`review-${width}.png`), fullPage: true });
  });
}

test('reference uses its own fps and is hidden after its duration', async ({ page }) => {
  const payload = await loadCase(page);
  payload.review.comparison.fps = 25;
  payload.review.comparison.qpos = payload.review.comparison.qpos.slice(0, 11);
  await page.route('**/api/session', route => route.fulfill({ json: payload }));
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/);
  await page.locator('#frameInput').fill('10'); await page.locator('#frameInput').dispatchEvent('change');
  await expect(page.locator('#scene')).toHaveAttribute('data-reference-frame', '5');
  await page.locator('#frameInput').fill('21'); await page.locator('#frameInput').dispatchEvent('change');
  await expect(page.locator('#scene')).toHaveAttribute('data-reference-frame', 'none');
  await expect(page.locator('#comparisonClock')).toContainText('已结束');
});

test('stationary and one-point support phases render without exceptions', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  const payload = await loadCase(page);
  const phase = (start: number, end: number) => ({ start, end, part: 'left_foot',
    aggregate: Array.from({length: end - start + 1}, () => [0, 0, 0]),
    rollouts: [Array.from({length: end - start + 1}, () => [0, 0, 0])], aggregate_net_cm: 0, rollout_median_net_cm: 0 });
  payload.review = { kind: 'support_motion', source: 'browser-fixture', notice: '固定材料点诊断', phases: [phase(0, 0), phase(1, 3)] };
  await page.route('**/api/session', route => route.fulfill({ json: payload }));
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/);
  await page.getByLabel('持续接触阶段').selectOption('1');
  await expect(page.locator('#frameInput')).toHaveValue('1');
  await expect(page.locator('#reviewPanel')).toContainText('0.00 cm');
  expect(errors).toEqual([]);
});

test('failed scene asset disables playback until retry succeeds', async ({ page }) => {
  await loadCase(page);
  await page.route('**/*.obj', route => route.abort());
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveAttribute('role', 'alert');
  await expect(page.locator('#playBtn')).toBeDisabled();
  await page.unroute('**/*.obj');
  await page.locator('#retryLoad').click();
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/, { timeout: 45_000 });
  await expect(page.locator('#playBtn')).toBeEnabled();
});

test('single-pose reports retain iteration selection and shared terrain controls', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  const payload = await loadCase(page);
  const record = (label: string, iteration: string, q: number[]) => ({ label, method: 'method', iteration, q,
    normal_starts: [], normals: [], diagnostics: {} });
  payload.review = { kind: 'report', source: 'pose-fixture', notice: '历史诊断', reference: 'original',
    records: [record('original', 'initial', payload.qpos[0]), record('method_final', 'final', payload.qpos[20])] };
  payload.qpos = [payload.qpos[0]];
  await page.route('**/api/session', route => route.fulfill({ json: payload }));
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/);
  await expect(page.locator('#reviewIteration')).toHaveValue('method_final');
  await expect(page.locator('#playBtn')).toBeDisabled();
  await page.locator('#reviewIteration').selectOption('original');
  await expect(page.locator('#reviewPanel')).toContainText('材料点平均 dx');
  await page.locator('[data-tab="display"]').click();
  await expect(page.locator('#terrainOpacity')).toBeVisible();
  await page.locator('#terrainToggle').uncheck();
  expect(errors).toEqual([]);
});
