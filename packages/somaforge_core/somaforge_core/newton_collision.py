"""Direct Newton collision queries without Isaac Lab, Kit, or COAL."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from somaforge_core import G1_29DOF_JOINT_ORDER, canonical_g1_urdf_path

from somaforge_core.newton_collision_filter import (
    apply_self_collision_filters_to_builder,
)

NEWTON_SHAPE_MARGIN_M = 0.01
NEWTON_MAX_TRIANGLE_PAIRS = 2_500_000
COLLISION_WITNESS_SCHEMA = 'newton_geometry_surface_witness_v3'
_BASE_G1_BUILDERS: dict[str, Any] = {}


def _leaf(value: object) -> str:
    return str(value).rsplit("/", 1)[-1]


def _split_transforms(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.asarray(value)
    if transforms.dtype.names:
        fields = list(transforms.dtype.names)
        return (
            np.asarray(transforms[fields[0]], dtype=np.float32),
            np.asarray(transforms[fields[1]], dtype=np.float32),
        )
    flat = np.asarray(transforms, dtype=np.float32).reshape(-1, 7)
    return flat[:, :3], flat[:, 3:7]


def _quat_rotate_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotation = np.asarray(quat, dtype=np.float64)
    q_xyz = rotation[..., :3]
    q_w = rotation[..., 3, None]
    value = np.asarray(vector, dtype=np.float64)
    uv = np.cross(q_xyz, value)
    uuv = np.cross(q_xyz, uv)
    return value + 2.0 * (q_w * uv + uuv)


def _effective_surface_witnesses(point0, point1, normal, thickness0, thickness1,
                                shape_margin0, shape_margin1):
    """Recover geometry witnesses without importing inflated solver margins.

    Newton stores support points and radius-plus-margin thickness. Subtracting
    all thickness produces separation of its inflated solver surfaces. Geometry
    separation restores the actual model margins; a sphere's center therefore
    shifts by its radius, while an unrounded mesh witness remains unchanged.
    All inputs come from the current model/contact arrays, with no guessed radii.
    """
    geometry_offset0 = np.asarray(thickness0) - np.asarray(shape_margin0)
    geometry_offset1 = np.asarray(thickness1) - np.asarray(shape_margin1)
    surface0 = np.asarray(point0) + np.asarray(normal) * geometry_offset0[:, None]
    surface1 = np.asarray(point1) - np.asarray(normal) * geometry_offset1[:, None]
    distance = np.einsum('ij,ij->i', normal, surface1 - surface0)
    return surface0, surface1, distance


@dataclass(frozen=True)
class NewtonCollisionContacts:
    shape0: np.ndarray
    shape1: np.ndarray
    body0: np.ndarray
    body1: np.ndarray
    point0_w: np.ndarray
    point1_w: np.ndarray
    normal_a_to_b_w: np.ndarray
    geometry_distance_m: np.ndarray
    constraint_distance_m: np.ndarray
    margin0_m: np.ndarray
    margin1_m: np.ndarray
    shape_labels: tuple[str, ...]
    robot_body_names: tuple[str, ...]
    robot_body_indices: np.ndarray
    robot_points_w: np.ndarray
    terrain_points_w: np.ndarray
    outward_normals_w: np.ndarray


class DirectNewtonCollisionScene:
    """Canonical G1 and one target terrain in Newton's collision pipeline."""

    def __init__(
        self,
        terrain_mesh: str | Path | None,
        *,
        device: str = "cpu",
        ground_height_m: float = 0.0,
    ) -> None:
        import newton
        import trimesh
        import warp as wp

        self.newton = newton
        self.wp = wp
        self.device = str(device)
        builder_key = str(canonical_g1_urdf_path().resolve())
        base_builder = _BASE_G1_BUILDERS.get(builder_key)
        if base_builder is None:
            base_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            base_builder.default_shape_cfg.margin = NEWTON_SHAPE_MARGIN_M
            base_builder.default_shape_cfg.gap = NEWTON_SHAPE_MARGIN_M
            base_builder.add_urdf(
                builder_key,
                floating=True,
                enable_self_collisions=True,
                joint_ordering=None,
                bodies_follow_joint_ordering=False,
                collapse_fixed_joints=False,
                hide_visuals=True,
            )
            base_builder.approximate_meshes("convex_hull")
            apply_self_collision_filters_to_builder(
                base_builder,
                exclude_kinematic_distance=3,
            )
            _BASE_G1_BUILDERS[builder_key] = base_builder
        builder = copy.deepcopy(base_builder)

        if terrain_mesh is not None:
            terrain_path = Path(terrain_mesh).expanduser().resolve()
            if not terrain_path.is_file():
                raise FileNotFoundError(terrain_path)
            loaded = trimesh.load(str(terrain_path), process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = loaded.dump(concatenate=True)
            if not isinstance(loaded, trimesh.Trimesh):
                raise ValueError(f"target terrain is not a triangle mesh: {terrain_path}")
            mesh = newton.Mesh(
                np.asarray(loaded.vertices, dtype=np.float32),
                np.asarray(loaded.faces, dtype=np.int32).reshape(-1),
                compute_inertia=False,
                is_solid=True,
            )
            builder.add_shape_mesh(
                -1,
                mesh=mesh,
                scale=(1.0, 1.0, 1.0),
                label="terrain_obstacle",
            )
            builder.add_ground_plane(height=float(ground_height_m), label="terrain_ground")

        self.model = builder.finalize(device=self.device, requires_grad=True)
        self.state = self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            max_triangle_pairs=NEWTON_MAX_TRIANGLE_PAIRS,
        )
        self.contacts = self.pipeline.contacts()
        self._collision_graph = None
        self._collision_graph_binding = None
        self._qpos_query_graph = None
        self._qpos_query_graph_binding = None
        self.body_names = tuple(_leaf(item) for item in self.model.body_label)
        self.shape_labels = tuple(_leaf(item) for item in self.model.shape_label)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Newton body labels are not uniquely resolvable")
        self.body_index = {name: index for index, name in enumerate(self.body_names)}

        joint_labels = tuple(_leaf(item) for item in self.model.joint_label)
        joint_q_start = np.asarray(self.model.joint_q_start.numpy(), dtype=np.int64)
        missing = [name for name in G1_29DOF_JOINT_ORDER if name not in joint_labels]
        if missing:
            raise ValueError(f"Newton collision model is missing joints: {missing}")
        self.motion_joint_q_indices = np.asarray(
            [joint_q_start[joint_labels.index(name)] for name in G1_29DOF_JOINT_ORDER],
            dtype=np.int64,
        )

    def set_body_poses(
        self,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        body_names: Sequence[str],
    ) -> None:
        positions = np.asarray(body_pos_w, dtype=np.float32)
        quaternions = np.asarray(body_quat_w, dtype=np.float32)
        names = tuple(str(name) for name in body_names)
        if positions.shape != (len(names), 3):
            raise ValueError("body_pos_w does not match body_names")
        if quaternions.shape != (len(names), 4):
            raise ValueError("body_quat_w does not match body_names")
        missing = [name for name in names if name not in self.body_index]
        if missing:
            raise ValueError(f"Newton collision model is missing bodies: {missing}")

        model_pos, model_quat_xyzw = _split_transforms(self.state.body_q.numpy())
        for source_index, name in enumerate(names):
            model_index = self.body_index[name]
            model_pos[model_index] = positions[source_index]
            model_quat_xyzw[model_index] = quaternions[source_index, [1, 2, 3, 0]]
        transforms = np.concatenate((model_pos, model_quat_xyzw), axis=1)
        self.state.body_q.assign(transforms)

    def set_qpos(self, qpos: np.ndarray) -> None:
        self._assign_qpos(qpos)
        self.newton.eval_fk(
            self.model,
            self.state.joint_q,
            self.state.joint_qd,
            self.state,
        )

    def _assign_qpos(self, qpos: np.ndarray) -> None:
        value = np.asarray(qpos, dtype=np.float32)
        if value.shape != (36,):
            raise ValueError(f"Newton collision qpos must be (36,), got {value.shape}")
        joint_q = np.zeros(int(self.model.joint_coord_count), dtype=np.float32)
        joint_q[:3] = value[:3]
        joint_q[3:7] = value[[4, 5, 6, 3]]
        joint_q[self.motion_joint_q_indices] = value[7:]
        self.state.joint_q.assign(joint_q)
        self.state.joint_qd.zero_()

    def body_poses(
        self,
        body_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return current Newton FK poses in world frame and WXYZ order."""
        positions, quaternion_xyzw = _split_transforms(self.state.body_q.numpy())
        names = self.body_names if body_names is None else tuple(str(name) for name in body_names)
        missing = [name for name in names if name not in self.body_index]
        if missing:
            raise ValueError(f"Newton collision model is missing bodies: {missing}")
        indices = np.asarray([self.body_index[name] for name in names], dtype=np.int64)
        quaternion_wxyz = quaternion_xyzw[indices][:, [3, 0, 1, 2]]
        return positions[indices].copy(), quaternion_wxyz.copy()

    def prepare_collision_graph(self, *, include_fk: bool = False) -> None:
        """Capture the same collision kernels, without caching any pose/pair.

        Only opt-in private optimizer scenes use this path. Current state and
        model buffers are read on every replay, including the contact counters
        reset and broad phase. Replacing a bound buffer requires recapturing.
        """
        self._qpos_query_graph = None
        self._qpos_query_graph_binding = None
        with self.wp.ScopedCapture(device=self.device) as capture:
            self.pipeline.collide(self.state, self.contacts)
        self._collision_graph = capture.graph
        self._collision_graph_binding = (self.pipeline, self.state, self.contacts)
        if include_fk:
            # Capture the identical FK kernel together with collision kernels
            # for joint-pose queries. Body-pose queries retain collision-only
            # replay so externally supplied body transforms are never replaced.
            with self.wp.ScopedCapture(device=self.device) as capture:
                self.newton.eval_fk(self.model, self.state.joint_q,
                                    self.state.joint_qd, self.state)
                self.pipeline.collide(self.state, self.contacts)
            self._qpos_query_graph = capture.graph
            self._qpos_query_graph_binding = self._query_buffer_binding()

    def _query_buffer_binding(self):
        return (self.model, self.pipeline, self.state, self.contacts,
                self.state.joint_q, self.state.joint_qd, self.state.body_q)

    def collide(self) -> NewtonCollisionContacts:
        if self._collision_graph is None:
            self.pipeline.collide(self.state, self.contacts)
        else:
            if any(current is not captured for current, captured in zip(
                    (self.pipeline, self.state, self.contacts), self._collision_graph_binding)):
                raise RuntimeError('Newton collision buffers changed; recapture the query graph')
            self.wp.capture_launch(self._collision_graph)
        return self._read_contacts()

    def _read_contacts(self) -> NewtonCollisionContacts:
        self.wp.synchronize_device(self.device)
        count = int(np.asarray(self.contacts.rigid_contact_count.numpy()).reshape(-1)[0])
        count = max(0, min(count, int(self.contacts.rigid_contact_max)))
        shape0 = np.asarray(
            self.contacts.rigid_contact_shape0.numpy()[:count],
            dtype=np.int32,
        )
        shape1 = np.asarray(
            self.contacts.rigid_contact_shape1.numpy()[:count],
            dtype=np.int32,
        )
        shape_body = np.asarray(self.model.shape_body.numpy(), dtype=np.int32)
        body0 = np.where(shape0 >= 0, shape_body[np.maximum(shape0, 0)], -1)
        body1 = np.where(shape1 >= 0, shape_body[np.maximum(shape1, 0)], -1)
        body_position, body_quat_xyzw = _split_transforms(self.state.body_q.numpy())
        local0 = np.asarray(
            self.contacts.rigid_contact_point0.numpy()[:count],
            dtype=np.float64,
        )
        local1 = np.asarray(
            self.contacts.rigid_contact_point1.numpy()[:count],
            dtype=np.float64,
        )
        point0 = local0.copy()
        point1 = local1.copy()
        for bodies, local, points in ((body0, local0, point0), (body1, local1, point1)):
            valid = bodies >= 0
            selected = bodies[valid]
            points[valid] = body_position[selected] + _quat_rotate_xyzw(
                body_quat_xyzw[selected], local[valid])
        normal = np.asarray(
            self.contacts.rigid_contact_normal.numpy()[:count],
            dtype=np.float64,
        )
        margin0 = np.asarray(
            self.contacts.rigid_contact_margin0.numpy()[:count],
            dtype=np.float64,
        )
        margin1 = np.asarray(
            self.contacts.rigid_contact_margin1.numpy()[:count],
            dtype=np.float64,
        )
        configured_margin = np.asarray(self.model.shape_margin.numpy(), dtype=np.float64)
        shape_margin0 = configured_margin[shape0]
        shape_margin1 = configured_margin[shape1]
        constraint_distance = np.einsum('ij,ij->i', normal, point1-point0) - margin0-margin1
        point0, point1, geometry_distance = _effective_surface_witnesses(
            point0, point1, normal, margin0, margin1, shape_margin0, shape_margin1)

        robot_indices = np.flatnonzero((body0 >= 0) != (body1 >= 0))
        first_robot = body0[robot_indices] >= 0
        robot_bodies = np.where(first_robot, body0[robot_indices], body1[robot_indices])
        robot_names = tuple(self.body_names[body] for body in robot_bodies)
        robot_points = np.where(first_robot[:, None], point0[robot_indices], point1[robot_indices])
        terrain_points = np.where(first_robot[:, None], point1[robot_indices], point0[robot_indices])
        outward_normals = normal[robot_indices]*np.where(first_robot, -1., 1.)[:, None]
        outward_normals /= np.maximum(np.linalg.norm(outward_normals, axis=-1, keepdims=True), 1.e-12)
        return NewtonCollisionContacts(
            shape0=shape0,
            shape1=shape1,
            body0=body0,
            body1=body1,
            point0_w=point0,
            point1_w=point1,
            normal_a_to_b_w=normal,
            geometry_distance_m=geometry_distance,
            constraint_distance_m=constraint_distance,
            margin0_m=margin0,
            margin1_m=margin1,
            shape_labels=self.shape_labels,
            robot_body_names=tuple(robot_names),
            robot_body_indices=np.asarray(robot_indices, dtype=np.int32),
            robot_points_w=np.asarray(robot_points, dtype=np.float64).reshape(-1, 3),
            terrain_points_w=np.asarray(terrain_points, dtype=np.float64).reshape(-1, 3),
            outward_normals_w=np.asarray(outward_normals, dtype=np.float64).reshape(-1, 3),
        )

    def query_qpos(self, qpos: np.ndarray) -> NewtonCollisionContacts:
        if self._qpos_query_graph is not None:
            if any(current is not captured for current, captured in zip(
                    self._query_buffer_binding(), self._qpos_query_graph_binding)):
                raise RuntimeError('Newton FK/collision buffers changed; recapture the query graph')
            self._assign_qpos(qpos)
            self.wp.capture_launch(self._qpos_query_graph)
            return self._read_contacts()
        self.set_qpos(qpos)
        return self.collide()

    def query_body_poses(
        self,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        body_names: Sequence[str],
    ) -> NewtonCollisionContacts:
        self.set_body_poses(body_pos_w, body_quat_w, body_names)
        return self.collide()


__all__ = [
    "NEWTON_SHAPE_MARGIN_M",
    "DirectNewtonCollisionScene",
    "NewtonCollisionContacts",
]
