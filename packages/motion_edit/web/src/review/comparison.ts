import * as THREE from 'three';
import { clearResources } from '../core/resources';
import type { Review, ReviewRecord } from '../core/types';

// A layer of the existing scene, not another renderer, server, or playback clock.
export class Comparison {
  private group = new THREE.Group();
  private annotations = new THREE.Group();
  private reference: any = null;
  private current: any = null;
  private review: Review | undefined;
  private selected: ReviewRecord | undefined;
  private spacing = 1;
  private link = 'left_ankle_roll_link';
  private normals = true;
  private displacement = true;
  private scale = 3;
  private panel: HTMLElement;
  private details: HTMLElement;
  constructor(private scene: THREE.Scene, private onPose: (q: number[]) => void) {
    this.panel = document.getElementById('reviewPanel')!;
    this.details = document.createElement('pre');
    this.details.style.whiteSpace = 'pre-wrap';
    this.scene.add(this.group, this.annotations);
  }
  private clearAnnotations() {
    clearResources(this.annotations);
  }
  setup(review: Review | undefined, robot: any, terrain: THREE.Object3D | null, names: string[]) {
    this.group.clear(); this.clearAnnotations(); this.panel.replaceChildren();
    this.review = review; this.current = robot; this.reference = null; this.selected = undefined;
    this.panel.hidden = !review;
    if (!review || review.kind === 'support_motion' || review.kind === 'support_slip') return;
    const notice = document.createElement('p'); notice.textContent = review.notice;
    this.panel.append(notice);
    if (review.kind !== 'report' || !review.records?.length) return;
    const records = review.records;
    const original = records.find(row => row.label === review.reference) || records[0];
    this.reference = robot.clone();
    this.spacing = terrain ? Math.max(1, new THREE.Box3().setFromObject(terrain).getSize(new THREE.Vector3()).y / 2 + .5) : 1;
    const q = original.q;
    this.reference.position.set(q[0], q[1] - this.spacing, q[2]);
    this.reference.quaternion.set(q[4], q[5], q[6], q[3]);
    names.forEach((name, i) => this.reference.setJointValue(name, q[7 + i]));
    this.group.add(this.reference);
    if (terrain) {
      const other = terrain.clone(); other.position.y -= this.spacing; this.group.add(other);
      terrain.position.y += this.spacing;
    }
    const heading = document.createElement('p'); heading.textContent = '左：原始姿态　右：选中结果'; this.panel.append(heading);
    const method = document.createElement('select'), iteration = document.createElement('select');
    method.setAttribute('aria-label', '实验方法'); method.id = 'reviewMethod';
    iteration.setAttribute('aria-label', '保存的迭代'); iteration.id = 'reviewIteration';
    [...new Set(records.map(row => row.method))].forEach(name => method.add(new Option(name, name)));
    const choose = () => {
      this.selected = records.find(row => row.label === iteration.value)!;
      this.onPose(this.selected.q);
    };
    const changeMethod = () => {
      iteration.replaceChildren();
      const rows = records.filter(row => row.method === method.value);
      const order = (r: ReviewRecord) => r.iteration === 'initial' ? -1 : r.iteration === 'final' ? Infinity : Number(r.iteration.replace('step', '')) || 0;
      rows.sort((a, b) => order(a) - order(b));
      rows.forEach(row => iteration.add(new Option(row.iteration, row.label)));
      iteration.value = (rows.find(row => row.iteration === 'final') || rows[0]).label;
      choose();
    };
    method.onchange = changeMethod; iteration.onchange = choose;
    const field = (title: string, control: HTMLElement) => { const label = document.createElement('label'); label.className = 'stack-field'; const span = document.createElement('span'); span.textContent = title; label.append(span, control); this.panel.append(label); };
    field('实验方法', method); field('保存的迭代', iteration);
    const link = document.createElement('select'); link.id = 'reviewLink'; link.setAttribute('aria-label', '位移部位');
    Object.keys(robot.links).forEach(name => link.add(new Option(name, name)));
    link.value = robot.links[this.link] ? this.link : Object.keys(robot.links)[0]; this.link = link.value;
    link.onchange = () => { this.link = link.value; this.onPose(this.selected!.q); };
    field('位移部位', link);
    const toggle = (label: string, initial: boolean, change: (value: boolean) => void) => {
      const container = document.createElement('label'); container.className = 'toggle';
      const input = document.createElement('input'); input.type = 'checkbox'; input.checked = initial;
      input.onchange = () => { change(input.checked); this.onPose(this.selected!.q); };
      container.append(input, document.createTextNode(label)); this.panel.append(container);
    };
    toggle('穿透法线（报告记录）', this.normals, value => this.normals = value);
    toggle('材料点实际位移 dx', this.displacement, value => this.displacement = value);
    const scale = document.createElement('input'); scale.type = 'range'; scale.min = '1'; scale.max = '20'; scale.value = String(this.scale);
    scale.setAttribute('aria-label', '位移显示倍率');
    scale.oninput = () => { this.scale = Number(scale.value); this.onPose(this.selected!.q); };
    field('位移显示倍率', scale);
    this.panel.append(this.details);
    method.value = records.find(row => row.label !== original.label)?.method || original.method;
    changeMethod();
  }
  update() {
    if (!this.reference || !this.current || !this.selected || !this.review?.records) return;
    this.current.position.y += this.spacing;
    this.current.updateMatrixWorld(true); this.reference.updateMatrixWorld(true);
    this.clearAnnotations();
    const original = this.review.records.find(r => r.label === this.review!.reference) || this.review.records[0];
    [original, this.selected].forEach((row, side) => {
      const offset = side === 0 ? -this.spacing : this.spacing;
      if (this.normals) row.normal_starts.forEach((p, i) => {
        const n = new THREE.Vector3(...row.normals[i]);
        if (n.length() > 0) this.annotations.add(new THREE.ArrowHelper(n.normalize(), new THREE.Vector3(p[0], p[1] + offset, p[2]), .08, 0xff5555, .015, .008));
      });
      // Historical markers are deliberately grey and never used as active contacts.
      const points = row.diagnostics.contact_position_w as number[][] | undefined;
      const mask = row.diagnostics.actual_contact as boolean[] | undefined;
      points?.forEach((p, i) => {
        if (!mask?.[i] || !p.every(Number.isFinite)) return;
        const dot = new THREE.Mesh(new THREE.SphereGeometry(.012, 8, 6), new THREE.MeshBasicMaterial({ color: 0xaaaaaa }));
        dot.position.set(p[0], p[1] + offset, p[2]); this.annotations.add(dot);
      });
    });
    const from = this.reference.links[this.link], to = this.current.links[this.link];
    const sourceMeshes: THREE.Mesh[] = [], targetMeshes: THREE.Mesh[] = [];
    from?.traverse((o: any) => { if (o.isMesh) sourceMeshes.push(o); });
    to?.traverse((o: any) => { if (o.isMesh) targetMeshes.push(o); });
    const mean = new THREE.Vector3(); let count = 0;
    sourceMeshes.forEach((mesh, j) => {
      const positions = mesh.geometry.getAttribute('position'); const target = targetMeshes[j];
      if (!positions || !target) return;
      for (let i = 0; i < positions.count; i += Math.max(1, Math.ceil(positions.count / 12))) {
        const a = new THREE.Vector3().fromBufferAttribute(positions, i).applyMatrix4(mesh.matrixWorld); a.y += 2 * this.spacing;
        const b = new THREE.Vector3().fromBufferAttribute(positions, i).applyMatrix4(target.matrixWorld);
        const delta = b.sub(a); mean.add(delta); count++;
        const length = delta.length();
        if (this.displacement && length > 1e-8) this.annotations.add(new THREE.ArrowHelper(delta.normalize(), a, length * this.scale, 0x40ef90, Math.min(.025, length * this.scale / 4), .01));
      }
    });
    if (count) mean.multiplyScalar(100 / count);
    const metrics = (row: ReviewRecord) => {
      const d = row.diagnostics;
      const format = (value: unknown) => typeof value === 'number' ? value.toFixed(3) : '未知';
      return `${row.label}\n全身负距离: ${format(d.full_depth_cm)} cm；双足: ${format(d.feet_depth_cm)} cm\n验收（报告）: ${d.static_recovery_pass ?? '未知'}；保留（报告）: ${d.all_required_kept ?? '未知'}`;
    };
    this.details.textContent = `${metrics(original)}\n\n${metrics(this.selected)}\n\n灰点：报告中的历史接触标记\n${this.link} 材料点平均 dx (cm): ${count ? mean.toArray().map(v => v.toFixed(3)).join(', ') : '无网格采样'}\n箭头显示倍率: ${this.scale}×`;
  }
  bounds(box: THREE.Box3) { if (this.reference) box.expandByObject(this.group); }
  terrainVisible(visible: boolean) { this.group.children.forEach(obj => { if (obj !== this.reference) obj.visible = visible; }); }
}
