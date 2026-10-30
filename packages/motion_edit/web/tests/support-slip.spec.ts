import { expect, test } from '@playwright/test';
const review=(globalThis as any).process?.env?.MOTION_EDIT_E2E_REVIEW as string | undefined;
test.skip(!review,'set MOTION_EDIT_E2E_REVIEW to a verified support_slip report');

test('shows actual contact displacement and unknown loaded slip in shared scene',async({page},testInfo)=>{
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  const load=await page.request.post('/api/review/load',{data:{path:review}});
  expect(load.ok()).toBeTruthy();
  const payload=await load.json();expect(payload.terrain_obj_url).toBeTruthy();
  const first=payload.review.slip_frames.findIndex((points:any[])=>points.length>0);
  expect(first).toBeGreaterThanOrEqual(0);
  expect(payload.review.slip_frames[first].every((p:any)=>Number.isInteger(p.surface))).toBeTruthy();
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/,{timeout:45_000});
  await page.locator('#frameInput').fill(String(payload.qpos.length>340?340:first));await page.locator('#frameInput').dispatchEvent('change');
  await expect(page.locator('#reviewPanel')).toContainText(payload.review.support_frames?'承重点相对切向速度':'承重滑移：未知');
  await expect(page.locator('#reviewPanel')).toContainText('中位');
  await expect(page.locator('#reviewPanel')).not.toContainText('NaN');
  const gain=page.locator('#reviewPanel input[type=number]');await gain.fill('10');await gain.dispatchEvent('change');
  await expect(page.locator('#reviewPanel')).toContainText('不改变统计值');
  expect(Number(await page.locator('#scene').getAttribute('data-robot-meshes'))).toBeGreaterThan(30);
  await page.getByRole('button',{name:'聚焦当前接触点'}).click();
  await page.screenshot({path:testInfo.outputPath('support_slip_browser.png')});
  expect(errors).toEqual([]);
});
