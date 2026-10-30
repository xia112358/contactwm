"""Remove duplicate collision meshes on the runtime stage, preserving asset files."""
from __future__ import annotations

import hashlib
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

COLLISION_SCHEMA = "canonical_unique_collision_meshes_v1"


def remove_duplicate_collision_meshes(stage, robot_prim):
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
            raise RuntimeError(f"Collision mesh has no rigid body: {prim.GetPath()}")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        world = (np.c_[points, np.ones(len(points))] @ np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
        geometry = (np.round(world, 6).tobytes()
                    + np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes()
                    + np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32).tobytes())
        key = (str(body.GetPath()), hashlib.sha256(geometry).hexdigest())
        groups.setdefault(key, []).append(str(prim.GetPath()))

    removed, retained = [], []
    for paths in groups.values():
        # The explicit collision copy ends in *_link_1; the other copy is
        # the same mesh under the visual-name branch. Keep the former.
        paths.sort(key=lambda path: (not path.rsplit("/", 1)[-1].endswith("_1"), path))
        retained.append(paths[0])
        for path in paths[1:]:
            prim = stage.GetPrimAtPath(path)
            # USD instance proxies cannot receive overrides until their
            # enclosing instance is made editable on this runtime stage.
            while prim.IsInstanceProxy():
                parent = prim.GetParent()
                while parent.IsInstanceProxy():
                    parent = parent.GetParent()
                parent.SetInstanceable(False)
                prim = stage.GetPrimAtPath(path)
            prim.SetActive(False)
            removed.append(path)
    return {"schema": COLLISION_SCHEMA, "removed": removed, "retained": retained}
