import * as THREE from 'three';

// ArrowHelper owns its materials, but its geometries are shared by Three.js.
export function disposeResources(root: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  root.traverse(object => {
    const mesh = object as THREE.Mesh;
    if (mesh.geometry && !(object.parent instanceof THREE.ArrowHelper)) geometries.add(mesh.geometry);
    const values = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    values.forEach(material => materials.add(material));
  });
  materials.forEach(material => {
    Object.values(material).forEach(value => { if (value instanceof THREE.Texture) textures.add(value); });
    material.dispose();
  });
  textures.forEach(texture => texture.dispose());
  geometries.forEach(geometry => geometry.dispose());
}

export function clearResources(group: THREE.Group): void {
  disposeResources(group);
  group.clear();
}
