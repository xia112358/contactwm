"""Authoritative collision revisions of the same canonical G1 kinematic asset.

The surface revision removes helper sphere geometry and duplicate collision
meshes. Rigid-body frames, inertia, joints and source asset bytes are retained.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LEGACY = "legacy_v1"
SURFACE = "surface_v2"
SURFACE_USD_RELATIVE = Path("assets/g1_surface_v2_usd/g1_29dof_surface_v2.usda")
SURFACE_USD_SHA256 = "a288f117aa838134eea219bbb88ae422b574b948a180886b8f371073d58f3893"
HELPER_LINKS = tuple(
    name for side in ("left", "right") for name in (
        *(f"{side}_ankle_roll_sphere_{i}_link" for i in range(1, 6)),
        f"{side}_sphere_hand_tip_link", f"{side}_ankle_intermediate_1_link"))


def profile_metadata(profile):
    if profile not in (LEGACY, SURFACE):
        raise ValueError("Unknown G1 collision profile: " + str(profile))
    definition = dict(schema="somaforge_collision_profile_v1", profile=profile,
        disabled_helper_links=HELPER_LINKS if profile == SURFACE else (),
        unique_collision_meshes=profile == SURFACE,
        kinematics="canonical_unchanged", inertia="canonical_unchanged")
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    return dict(collision_profile=profile, collision_profile_sha256=digest)


def require_profile(metadata, expected, *, context):
    """Physical execution/support contracts must match, independently of FK."""
    wanted = profile_metadata(expected)
    actual = (metadata or {}).get("collision_profile", LEGACY)
    if actual != expected:
        raise ValueError(f"{context}: collision profile {actual!r} differs from {expected!r}; "
                         "old contact/force evidence requires new execution")
    present = (metadata or {}).get("collision_profile_sha256")
    if present is not None and present != wanted["collision_profile_sha256"]:
        raise ValueError(f"{context}: corrupt collision profile fingerprint")
    if expected != LEGACY and present is None:
        raise ValueError(f"{context}: missing collision profile fingerprint")


def contact_body_names(profile):
    from .contact_schema import CONTACT_BODY_NAMES_BY_PART
    profile_metadata(profile)
    if profile == LEGACY:
        return CONTACT_BODY_NAMES_BY_PART
    return {**CONTACT_BODY_NAMES_BY_PART,
        "left_foot": ("left_ankle_roll_link",), "right_foot": ("right_ankle_roll_link",),
        "left_hand": ("left_sphere_hand_link",), "right_hand": ("right_sphere_hand_link",),
        "left_heel": (), "left_toe": (), "right_heel": (), "right_toe": ()}


def execution_profile(env):
    """The configured physical revision, including historical serialized configs."""
    profile = getattr(env.robot_config.asset, "collision_profile", LEGACY)
    profile_metadata(profile)
    return profile


def audit_surface_mujoco(model):
    """Check the realized one-world solver model, rather than its filename."""
    import mujoco
    from .robot_assets import canonical_g1_urdf_path
    import xml.etree.ElementTree as ET
    links = {link.get("name") for link in ET.parse(canonical_g1_urdf_path()).getroot().findall("link")}
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    robot = {i for i, name in enumerate(names) if any(name == link or name.endswith("_" + link) or name.endswith("/" + link) for link in links)}
    counts = dict(mesh=0, sphere=0, cylinder=0, other=0)
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in robot or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        kind = int(model.geom_type[geom])
        key = {int(mujoco.mjtGeom.mjGEOM_MESH): "mesh", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
               int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder"}.get(kind, "other")
        counts[key] += 1
    if counts != dict(mesh=29, sphere=0, cylinder=4, other=0):
        raise ValueError("Solver model is not canonical surface-v2 collision geometry: " + repr(counts))
    return counts


def remove_duplicate_meshes(stage, robot_prim):
    """Identify duplicates by physical body, transformed vertices and topology."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    groups = {}
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        body = prim.GetParent()
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body.IsValid():
            raise ValueError("Collision mesh has no actual rigid body: " + str(prim.GetPath()))
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        world = (np.c_[points, np.ones(len(points))] @ np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
        data = (np.round(world, 6).tobytes() + np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes()
                + np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32).tobytes())
        groups.setdefault((str(body.GetPath()), hashlib.sha256(data).hexdigest()), []).append(str(prim.GetPath()))
    removed, retained = [], []
    for paths in groups.values():
        paths.sort(key=lambda p: (not p.rsplit("/", 1)[-1].endswith("_1"), p))
        retained.append(paths[0])
        for path in paths[1:]:
            prim = stage.GetPrimAtPath(path)
            while prim.IsInstanceProxy():
                parent = prim.GetParent()
                while parent.IsInstanceProxy():
                    parent = parent.GetParent()
                parent.SetInstanceable(False)
                prim = stage.GetPrimAtPath(path)
            prim.SetActive(False)
            removed.append(path)
    return dict(removed=removed, retained=retained)


def apply_surface_profile(stage, robot_prim):
    """Author only geometry overrides; retain every rigid-body/joint/inertia prim."""
    from pxr import Usd, UsdGeom, UsdPhysics
    duplicates = remove_duplicate_meshes(stage, robot_prim)
    spheres = []
    found = set()
    for prim in list(Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())):
        if not prim.IsA(UsdGeom.Sphere):
            continue
        owner = prim.GetParent().GetName()
        if owner not in HELPER_LINKS:
            raise ValueError("Unregistered sphere geometry in canonical G1: " + str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            spheres.append(str(prim.GetPath()));found.add(owner)
        prim.SetActive(False)
    if found != set(HELPER_LINKS):
        raise ValueError("Canonical helper collision topology differs: " + repr(sorted(set(HELPER_LINKS)-found)))
    return dict(**profile_metadata(SURFACE), removed_sphere_colliders=spheres,
                removed_duplicate_meshes=duplicates["removed"],retained_meshes=duplicates["retained"])


def audit_surface_stage(stage, robot_prim):
    from pxr import Usd, UsdGeom, UsdPhysics
    counts = dict(mesh=0, sphere=0, cylinder=0, other=0)
    bodies = set()
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.add(str(prim.GetPath()))
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        kind = "mesh" if prim.IsA(UsdGeom.Mesh) else "sphere" if prim.IsA(UsdGeom.Sphere) else "cylinder" if prim.IsA(UsdGeom.Cylinder) else "other"
        counts[kind] += 1
    if counts != dict(mesh=29, sphere=0, cylinder=4, other=0):
        raise ValueError("Unexpected surface-v2 physical colliders: " + repr(counts))
    for name in HELPER_LINKS:
        if not any(p.endswith("/"+name) for p in bodies):
            raise ValueError("Collision migration removed a body frame: " + name)
    return dict(**counts, rigid_bodies=len(bodies))


def generate_surface_usd(output):
    """Build a small deterministic overlay referencing the pinned base bundle."""
    from pxr import Usd
    from .robot_assets import canonical_g1_asset_metadata
    import os
    base = canonical_g1_asset_metadata()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite collision asset: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    layer = stage.GetRootLayer()
    layer.subLayerPaths = [os.path.relpath(base["usd_path"], output.parent)]
    layer.customLayerData = {**profile_metadata(SURFACE), "base_usd_bundle_sha256": base["usd_bundle_sha256"]}
    stage.SetDefaultPrim(stage.GetPrimAtPath("/g1_29dof"))
    report = apply_surface_profile(stage, stage.GetDefaultPrim())
    report["colliders"] = audit_surface_stage(stage, stage.GetDefaultPrim())
    layer.Save()
    return report
