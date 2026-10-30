"""Import verified SomaForge step/climb tasks without altering their source files."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {
    "motions-low-step": ("lowstep", "lowstep_boxes"),
    "motions-low-climb-76": ("lowclimb", "lowclimb_boxes"),
    "motions-high-climb-76": ("highclimb", "highclimb_boxes"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_terrain(source, destination, shape_count, boxes):
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    mesh = trimesh.load(source, force="mesh", process=True)
    parts = list(mesh.split(only_watertight=False))
    if not parts or any(not part.is_watertight or not part.is_convex for part in parts):
        raise ValueError(f"Expected separate closed convex box components: {source}")
    original_count = len(parts)
    floors = [part for part in parts if part.bounds[1, 2] < 0.01]
    if len(floors) != 1:
        raise ValueError("Expected one source floor")
    transform, extents = trimesh.bounds.oriented_bounds(floors[0])
    pose = np.linalg.inv(transform)
    from scipy.spatial.transform import Rotation
    quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
    colliders = [np.concatenate((pose[:3, 3], quaternion, extents)), *boxes]
    for index in range(original_count, shape_count):
        colliders.append(np.array([0, 0, -1000-index, 1, 0, 0, 0, .002, .002, .002]))
    stage = Usd.Stage.CreateNew(str(destination))
    root = UsdGeom.Xform.Define(stage, "/Terrain")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for index, box in enumerate(colliders):
        # Batched MuJoCo supports per-world geom pose/size, but mesh vertex
        # data is shared. Keep geometry out of per-task mesh vertices.
        prim = UsdGeom.Cube.Define(stage, f"/Terrain/box_{index}")
        prim.CreateSizeAttr(2.0)
        prim.AddTranslateOp().Set(Gf.Vec3d(*box[:3]))
        prim.AddOrientOp().Set(Gf.Quatf(float(box[3]), Gf.Vec3f(*box[4:7])))
        prim.AddScaleOp().Set(Gf.Vec3f(*(box[7:] * .5)))
        prim.CreateDisplayColorAttr([Gf.Vec3f(0.48, 0.55, 0.62)])
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    stage.GetRootLayer().Save()
    # A separate convex collider per original box prevents a hull bridging their gaps.
    return original_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--somaforge", type=Path, default=ROOT.parent / "somaforge")
    parser.add_argument("--component", choices=tuple(COMPONENTS), default="motions-low-step")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--contact_model", choices=("binary2", "force6"), default="binary2",
                        help="document the intended observation schema; runtime still selects its contact model explicitly")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    soma = args.somaforge.resolve()
    prefix, directory = COMPONENTS[args.component]
    output = (args.output or runtime_path('data') / directory).resolve()
    if (output / "task.json").exists():
        raise FileExistsError(f"Prepared manifest already exists; choose a new --output: {output}")
    if args.count is not None and args.count < 1:
        parser.error("--count must be positive")
    pass
    from somaforge_core.robot_assets import canonical_g1_asset_metadata, decode_robot_asset_json
    from somaforge_core.motion_schema import G1_29DOF_JOINT_ORDER, decode_kinematics_provenance
    from somaforge_core.published_data import resolve_published_path

    asset = canonical_g1_asset_metadata()
    source_manifest = soma / "runtime/current/motions/php_wbt_v1_20261016" / args.component / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    if manifest["robot_asset_sha256"] != asset["urdf_sha256"]:
        raise ValueError("Motion manifest uses a different robot")
    entries = manifest["motion_files"][:args.count]
    terrains = {entry["terrain_id"]: entry for entry in manifest["terrains"]}
    shape_count = 1 + max(len(np.load(resolve_published_path(terrains[entry["terrain_id"]]["source_file"]),
                                     allow_pickle=False)) for entry in entries)
    (output / "motions").mkdir(parents=True, exist_ok=True)
    (output / "terrains").mkdir(exist_ok=True)
    tasks = []
    for index, entry in enumerate(entries):
        source = source_manifest.parent / entry["motion_file"]
        terrain_entry = terrains[entry["terrain_id"]]
        terrain_source = source_manifest.parent / terrain_entry["terrain_file"]
        if digest(source) != entry["motion_sha256"] or digest(terrain_source) != terrain_entry["terrain_sha256"]:
            raise ValueError(f"Source fingerprint mismatch: {source}")
        raw_terrain = resolve_published_path(terrain_entry["source_file"])
        if digest(raw_terrain) != terrain_entry["source_sha256"]:
            raise ValueError(f"Raw terrain fingerprint mismatch: {raw_terrain}")
        boxes = np.load(raw_terrain, allow_pickle=False)[:, terrain_entry["php_variant"]]
        with np.load(source, allow_pickle=False) as data:
            decode_robot_asset_json(data["robot_asset_json"], context=str(source))
            decode_kinematics_provenance(data["kinematics_provenance_json"], context=str(source))
            fps = float(data["fps"])
            if fps != 50.0:
                raise ValueError(f"Expected 50 Hz canonical input, got {fps}")
            names = data["joint_names"].tolist()
            q = data["joint_pos"]
            joints = q[:, 7:][:, [names.index(name) for name in G1_29DOF_JOINT_ORDER]]
            # Source qpos quaternion is WXYZ; the upstream CSV loader consumes XYZW.
            csv = np.concatenate((q[:, :3], q[:, [4, 5, 6, 3]], joints), axis=1)
            if not np.isfinite(csv).all():
                raise ValueError(f"Non-finite reference: {source}")
        csv_path = output / "motions" / f"{prefix}_{index:04d}.csv"
        np.savetxt(csv_path, csv, delimiter=",", fmt="%.9g")
        terrain_usd = output / "terrains" / f"terrain_{index:04d}.usda"
        if terrain_usd.exists():
            terrain_usd.unlink()  # Only this experiment's generated, reproducible USD.
        count = write_terrain(terrain_source, terrain_usd, shape_count, boxes)
        if count != len(boxes) + 1:
            raise ValueError(f"Expected {len(boxes)} obstacles plus the original floor, got {count}")
        tasks.append({
            "name": csv_path.stem, "source_motion_id": entry["motion_id"],
            "source_motion": str(source), "source_motion_sha256": digest(source),
            "csv": str(csv_path), "csv_sha256": digest(csv_path),
            "source_terrain": str(terrain_source), "source_terrain_sha256": digest(terrain_source),
            "terrain_usd": str(terrain_usd), "terrain_usd_sha256": digest(terrain_usd),
            "boxes": boxes.tolist(), "frames": len(csv), "fps": fps,
            "layout_padding_colliders": shape_count - count,
        })
    # Auxiliary MJCF is for visualization only; physics uses the untouched canonical USD.
    import mujoco
    urdf = Path(asset["urdf_path"])
    text = re.sub(r"<!--.*?-->", "", urdf.read_text(), flags=re.S)
    text = re.sub(r"<mujoco>.*?</mujoco>",
                  f'<mujoco><compiler meshdir="{urdf.parent}" discardvisual="false" '
                  'fusestatic="false" strippath="false" balanceinertia="true"/></mujoco>', text, flags=re.S)
    temporary = output / "canonical_visual.urdf"
    temporary.write_text(text)
    model = mujoco.MjModel.from_xml_path(str(temporary))
    visual = output / "canonical_visual.xml"
    mujoco.mj_saveLastXML(str(visual), model)
    xml = visual.read_text()
    xml, changed = re.subn(r'(<body name="pelvis"[^>]*>)', r'\1\n<freejoint name="floating_base_joint"/>', xml, count=1)
    if changed != 1:
        raise ValueError("Auxiliary model lost the pelvis link")
    visual.write_text(xml)
    prepared = {
        "schema": "supermimic_somaforge_lowstep_v1", "somaforge_root": str(soma),
        "source_component": args.component,
        "somaforge_commit": subprocess.check_output(["git", "-C", str(soma), "rev-parse", "HEAD"], text=True).strip(),
        "source_manifest": str(source_manifest), "source_manifest_sha256": digest(source_manifest),
        "robot_asset": asset, "visualization_mjcf": str(visual), "visualization_mjcf_sha256": digest(visual),
        "motion_dir": str(output / "motions"),
        "source_fps": 50.0,
        "terrain_observation": ("local_heightmap_11x11_force6x3_heading_v1" if args.contact_model == "force6"
                                else "local_heightmap_11x11_contact2_v1"),
        "contact_labels": ("runtime measured six-part total contact forces (normal + friction), not offline motion labels"
                           if args.contact_model == "force6"
                           else "measured simulated foot normal forces; learned next-contact targets"),
        "terrain_layout": {"shapes_per_world": shape_count, "collider_representation": "primitive_box_pose_size_v1",
                           "bounding_volume_schema": "per_world_box_aabb_rbound_v1",
                           "padding": "2mm cubes below z=-1000m; structural layout only"},
        "tasks": tasks,
    }
    path = output / "task.json"
    path.write_text(json.dumps(prepared, indent=2) + "\n")
    print(f"Prepared {len(tasks)} paired tasks, {sum(t['frames'] for t in tasks)} frames: {path}")


if __name__ == "__main__":
    main()
