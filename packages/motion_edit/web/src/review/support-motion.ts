import * as THREE from 'three';
import { clearResources } from '../core/resources';
import type { Review, Vec3 } from '../core/types';

type Phase = NonNullable<Review['phases']>[number];

export class SupportMotionOverlay {
  private group = new THREE.Group();
  private phases: Phase[] = [];
  private selected = 0;
  private marker: THREE.Mesh | null = null;
  private details: HTMLElement | null = null;
  private rawGroup = new THREE.Group();
  private showRaw = true;

  constructor(private scene: THREE.Scene, private jump: (frame: number) => void,
              private focus: (track: Vec3[]) => void) {
    scene.add(this.group);
  }

  private clear() {
    clearResources(this.group);
    this.rawGroup.clear();
    this.marker = null;
  }

  setup(review: Review | undefined) {
    this.clear(); this.phases = review?.kind === 'support_motion' ? review.phases ?? [] : [];
    this.selected = 0;
    const panel = document.getElementById('reviewPanel')!;
    if (!this.phases.length) return;
    panel.hidden = false;
    const notice = document.createElement('p'); notice.textContent = review!.notice; panel.append(notice);
    const field = document.createElement('label'); field.className = 'stack-field';
    const title = document.createElement('span'); title.textContent = '持续接触阶段';
    const select = document.createElement('select'); select.setAttribute('aria-label', '持续接触阶段');
    this.phases.forEach((phase, index) => select.add(new Option(`${phase.start}–${phase.end} ${phase.part}`, String(index))));
    select.onchange = () => { this.selected = Number(select.value); this.draw(); this.jump(this.phases[this.selected].start); this.focusSelected(); };
    field.append(title, select); panel.append(field);
    const toggle = document.createElement('label'); toggle.className = 'toggle';
    const check = document.createElement('input'); check.type = 'checkbox'; check.checked = this.showRaw;
    check.onchange = () => { this.showRaw = check.checked; this.rawGroup.visible = this.showRaw; };
    toggle.append(check, document.createTextNode('显示原始 rollout 轨迹')); panel.append(toggle);
    this.details = document.createElement('p'); panel.append(this.details);
    this.draw();
  }

  private dot(position: Vec3, color: number, radius: number) {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(radius, 10, 8), new THREE.MeshBasicMaterial({ color, depthTest: false }));
    marker.position.set(...position); marker.renderOrder = 20; this.group.add(marker);
    return marker;
  }

  private draw() {
    this.clear();
    const phase = this.phases[this.selected]; if (!phase) return;
    const color = phase.part === 'left_foot' ? 0xf06449 : 0x31a7d8;
    const points = (track: Vec3[]) => track.map(value => new THREE.Vector3(...value));
    this.group.add(this.rawGroup);
    phase.rollouts.forEach(track => {
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points(track)),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: .12, depthTest: false, depthWrite: false }));
      line.renderOrder = 15; this.rawGroup.add(line);
    });
    this.rawGroup.visible = this.showRaw;
    const path = points(phase.aggregate);
    const curve = new THREE.CurvePath<THREE.Vector3>();
    for (let i = 1; i < path.length; i++) {
      if (path[i].distanceToSquared(path[i - 1]) > 1e-12)
        curve.add(new THREE.LineCurve3(path[i - 1], path[i]));
    }
    if (curve.curves.length) {
      const line = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, path.length * 2), .0015, 6, false),
        new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false }));
      line.renderOrder = 16; this.group.add(line);
    } else {
      this.dot(phase.aggregate[0], color, .003);
    }
    const start = path[0], end = path.at(-1)!;
    const horizontal = end.clone().sub(start); horizontal.z = 0;
    if (horizontal.length() > 1e-6) {
      const arrow = new THREE.ArrowHelper(horizontal.clone().normalize(), start, horizontal.length(), color,
        Math.min(.004, horizontal.length() / 6), .0015);
      arrow.traverse((object: any) => { if (object.material) object.material.depthTest = false; });
      arrow.renderOrder = 18; this.group.add(arrow);
    }
    this.marker = this.dot(phase.aggregate[0], 0xffffff, .0025);
    if (this.details) this.details.textContent =
      `聚合净位移 ${phase.aggregate_net_cm.toFixed(2)} cm；原始 rollout 中位数 ${phase.rollout_median_net_cm.toFixed(2)} cm。亮色宽线：聚合；淡色细线：原始执行；同一固定材料点。`;
  }

  update(frame: number) {
    const phase = this.phases[this.selected];
    if (!phase || !this.marker) return;
    this.marker.visible = frame >= phase.start && frame <= phase.end;
    if (this.marker.visible) this.marker.position.set(...phase.aggregate[frame - phase.start]);
  }

  focusSelected() {
    const phase = this.phases[this.selected];
    if (phase) this.focus(phase.aggregate);
  }
}
