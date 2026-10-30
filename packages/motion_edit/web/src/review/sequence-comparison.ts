import * as THREE from 'three';
import { disposeResources } from '../core/resources';
import type { Review } from '../core/types';

/** A second motion in the shared scene, driven by the selected motion's clock. */
export class SequenceComparison {
  private reference: any = null;
  private terrain: THREE.Object3D | null = null;
  private currentTerrain: THREE.Object3D | null = null;
  private data: Review['comparison'];
  private materials: THREE.Material[] = [];
  private spacing = 1;
  private mode: 'beside' | 'overlay' = 'beside';
  private terrainShown = true;
  private opacity = 1;
  private available = true;
  referenceFrame: number | null = null;

  constructor(private scene: THREE.Scene) {}

  clear() {
    if (this.reference) this.scene.remove(this.reference);
    // The cloned robot shares source geometries but owns its tinted materials.
    this.materials.forEach(material => material.dispose()); this.materials = [];
    if (this.terrain) { this.scene.remove(this.terrain); disposeResources(this.terrain); }
    if (this.currentTerrain) this.currentTerrain.position.y = 0;
    this.reference = null; this.terrain = null; this.currentTerrain = null;
    this.data = undefined; this.referenceFrame = null;
  }

  setup(data: Review['comparison'], robot: any, terrain: THREE.Object3D | null,
        referenceTerrain: THREE.Object3D | null) {
    this.clear(); this.data = data; this.currentTerrain = terrain;
    if (!data) return;
    this.reference = robot.clone(); this.terrain = referenceTerrain;
    this.reference.traverse((mesh: any) => {
      if (!mesh.isMesh) return;
      const tint = (material: THREE.Material) => {
        const copy = material.clone() as THREE.MeshStandardMaterial;
        copy.color?.set(0x58bce8); this.materials.push(copy); return copy;
      };
      mesh.material = Array.isArray(mesh.material) ? mesh.material.map(tint) : tint(mesh.material);
    });
    const size = (object: THREE.Object3D | null) => object
      ? new THREE.Box3().setFromObject(object).getSize(new THREE.Vector3()).y : 0;
    this.spacing = Math.max(1, (size(terrain) + size(referenceTerrain)) / 4 + .5);
    this.scene.add(this.reference);
    if (this.terrain) this.scene.add(this.terrain);
    this.setMode(this.mode);
  }

  setMode(mode: 'beside' | 'overlay') {
    this.mode = mode;
    this.materials.forEach(material => {
      material.transparent = mode === 'overlay'; material.opacity = mode === 'overlay' ? .4 : 1;
      material.depthWrite = mode !== 'overlay';
    });
    if (this.currentTerrain) this.currentTerrain.position.y = this.offset;
  }

  get layout() { return this.mode; }
  get offset() { return this.data && this.mode === 'beside' ? this.spacing : 0; }

  update(seconds: number, current: THREE.Object3D) {
    if (!this.data || !this.reference) return;
    current.position.y += this.offset;
    const end = (this.data.qpos.length - 1) / this.data.fps;
    this.available = seconds <= end + 1e-8;
    this.reference.visible = this.available;
    this.referenceFrame = this.available ? Math.min(this.data.qpos.length - 1, Math.round(seconds * this.data.fps)) : null;
    if (this.referenceFrame !== null) {
      const q = this.data.qpos[this.referenceFrame];
      this.reference.position.set(q[0], q[1] - this.offset, q[2]);
      this.reference.quaternion.set(q[4], q[5], q[6], q[3]);
      this.data.joint_names.forEach((name, index) => this.reference.setJointValue(name, q[7 + index]));
      this.reference.updateMatrixWorld(true);
    }
    if (this.terrain) this.terrain.position.y = -this.offset;
    this.display(this.terrainShown, this.opacity);
  }

  display(visible: boolean, opacity: number) {
    this.terrainShown = visible; this.opacity = opacity;
    if (!this.terrain) return;
    // Coincident terrain copies obscure the ghost in overlay mode.
    this.terrain.visible = visible && this.available && (this.mode === 'beside' || !this.currentTerrain);
    this.terrain.traverse((mesh: any) => {
      if (!mesh.isMesh) return;
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      materials.forEach((material: THREE.Material) => {
        material.transparent = opacity < 1; material.opacity = opacity; material.depthWrite = opacity === 1;
      });
    });
  }

  bounds(box: THREE.Box3, terrain = true) {
    if (this.reference?.visible) box.expandByObject(this.reference);
    if (terrain && this.terrain?.visible) box.expandByObject(this.terrain);
  }
}
