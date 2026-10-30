import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// Exercise the real viewer with a tiny deterministic motion, without requiring
// training outputs or modifying registered motions on a running backend.
test('read-only viewer loads motion, scrubs and shows contact forces without editor requests', async ({ page }) => {
  // Intercept built assets without occupying a visualization service port.
  const dist = resolve(process.cwd(), '../motion_edit/web_dist');
  await page.route('**/*', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/') {
      await route.fulfill({ contentType: 'text/html', body: readFileSync(resolve(dist, 'index.html')) });
    } else if (/^\/assets\/[a-zA-Z0-9_.-]+$/.test(pathname)) {
      const contentType = pathname.endsWith('.js') ? 'application/javascript' : 'text/css';
      await route.fulfill({ contentType, body: readFileSync(resolve(dist, pathname.slice(1))) });
    } else {
      await route.fallback();
    }
  });
  const errors: string[] = [];
  const writes: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (/\/api\/session\/(generate|move|restore|save|validate|undo|redo|settings)/.test(request.url())) writes.push(request.url());
  });
  const qpos = Array.from({ length: 20 }, (_, i) => [i * .01, 0, .5, 1, 0, 0, 0]);
  const payload = {
    motion_id: 'playback-fixture', read_only: true, provenance: 'source', source_motion_id: 'playback-fixture',
    capabilities: { playback: true, edit_contacts: false, generate: false },
    fps: 50, qpos, joint_names: [], robot_urdf_url: '/fixture/robot.urdf', terrain_obj_url: '/fixture/terrain.obj',
    graph: { anchors: [], transitions: [{ start_frame: 5, end_frame: 11, metadata: {} }] }, surfaces: [],
    contact_force: {
      mask_status: 'verified_newton_primary_faces', part_order: ['left_foot'],
      forces: qpos.map(() => [[0, 0, 100]]), masks: qpos.map(() => [true]),
      positions: qpos.map(() => [[0, 0, 0]]), position_valid: qpos.map(() => [true]),
      sample_points: [], sample_forces: [],
    },
  };
  await page.route('**/api/motions', route => route.fulfill({ json: { motions: [{ motion_id: payload.motion_id, label: payload.motion_id, ready: true, provenance: 'source' }] } }));
  await page.route('**/api/session', route => route.fulfill({ json: payload }));
  await page.route('**/api/recent-motions', route => route.fulfill({ json: { items: [], active_motion_id: payload.motion_id } }));
  await page.route('**/fixture/robot.urdf', route => route.fulfill({
    contentType: 'application/xml',
    body: '<robot name="fixture"><link name="pelvis"><visual><geometry><mesh filename="/fixture/mesh.obj"/></geometry></visual></link></robot>',
  }));
  await page.route('**/fixture/*.obj', route => route.fulfill({ contentType: 'text/plain', body: 'v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n' }));
  await page.goto('/');
  await expect(page.locator('#sceneStatus')).toHaveClass(/hidden/);
  await expect(page.locator('.brand strong')).toHaveText('Motion Review');
  await expect(page.locator('#saveBtn,#undoBtn,#redoBtn,#generateBtn,[data-tab="output"],.anchor-browser')).toHaveCount(0);
  await expect(page.locator('#scene')).toHaveAttribute('data-visible-contact-parts', '1');
  await page.locator('#nextBtn').click();
  await expect(page.locator('#frameInput')).toHaveValue('1');
  await page.locator('#nextCut').click();
  await expect(page.locator('#frameInput')).toHaveValue('5');
  await page.locator('#frameInput').fill('999');
  await page.locator('#frameInput').dispatchEvent('change');
  await expect(page.locator('#frameInput')).toHaveValue('19');
  await page.locator('[data-tab="display"]').click();
  await page.locator('#contactSampleToggle').uncheck();
  await page.locator('#terrainToggle').uncheck();
  await expect(page.locator('#scene')).toHaveAttribute('data-terrain-visible', 'false');
  await page.locator('#openMotionBtn').click();
  await expect(page.locator('.motion-picker-item')).toContainText('playback-fixture');
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});
