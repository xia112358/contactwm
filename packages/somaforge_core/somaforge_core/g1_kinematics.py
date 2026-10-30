from __future__ import annotations
import math
import xml.etree.ElementTree as ET
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.robot_assets import canonical_g1_urdf_path
from somaforge_core.motion_contracts import BODY_NAMES


def _rpy_matrix(rpy: tuple[float, float, float]) -> Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=torch.float32,
    )


def _quaternion_matrix_wxyz(quaternion: Tensor) -> Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _axis_angle_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    axis = F.normalize(axis, dim=-1)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(axis.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=angle.dtype, device=angle.device)
    identity = identity.expand(angle.shape + (3, 3))
    outer = axis[..., :, None] * axis[..., None, :]
    return (
        torch.cos(angle)[..., None, None] * identity
        + (1.0 - torch.cos(angle))[..., None, None] * outer
        + torch.sin(angle)[..., None, None] * skew
    )


def _rotation6d(matrix: Tensor) -> Tensor:
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def _matrix_from_rotation6d(rotation6d: Tensor) -> Tensor:
    columns = rotation6d.reshape(rotation6d.shape[:-1] + (3, 2))
    first = F.normalize(columns[..., :, 0], dim=-1)
    second_raw = columns[..., :, 1]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _rotation_vector_matrix(rotation_vector: Tensor) -> Tensor:
    """Map an unconstrained rotation vector to SO(3) with a stable exponential map."""

    x, y, z = rotation_vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(rotation_vector.shape[:-1] + (3, 3))
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
    first_order = torch.sinc(angle / math.pi)
    second_order = 0.5 * torch.sinc(angle / (2.0 * math.pi)).square()
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    identity = identity.expand(rotation_vector.shape[:-1] + (3, 3))
    return identity + first_order[..., None, None] * skew + second_order[..., None, None] * (skew @ skew)


def _rotation_chordal_error(first: Tensor, second: Tensor) -> Tensor:
    """Squared SO(3) chordal error, retaining all leading dimensions."""

    return (first - second).square().mean(dim=(-2, -1))


def _rotation_angle_degrees(first: Tensor, second: Tensor) -> Tensor:
    """Geodesic SO(3) angle in degrees for reporting only."""

    relative = first.transpose(-1, -2) @ second
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _rotation_log_vector(matrix: Tensor) -> Tensor:
    """Return the axis-angle logarithm of an SO(3) matrix."""

    skew = 0.5 * torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = (matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    angle = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
    scale = torch.where(sine > 1.0e-6, angle / sine.clamp_min(1.0e-6), torch.ones_like(sine))
    return scale[..., None] * skew


def _quaternion_multiply_wxyz(first: Tensor, second: Tensor) -> Tensor:
    aw, ax, ay, az = first.unbind(dim=-1)
    bw, bx, by, bz = second.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _canonical_boundary(position: Tensor, rotation6d: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rotation = _matrix_from_rotation6d(rotation6d)
    torso_position = position[:, 0]
    torso_rotation = rotation[:, 0]
    yaw = torch.atan2(torso_rotation[:, 1, 0], torso_rotation[:, 0, 0])
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    yaw_rotation = torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1).reshape(-1, 3, 3)
    world_to_canonical = yaw_rotation.transpose(-1, -2)
    canonical_position = torch.einsum("bij,bnj->bni", world_to_canonical, position - torso_position[:, None])
    canonical_rotation = torch.einsum("bij,bnjk->bnik", world_to_canonical, rotation)
    return canonical_position, _rotation6d(canonical_rotation), torso_position, yaw


class CanonicalG1ForwardKinematics(nn.Module):
    """Differentiable FK parsed only from SomaForge's canonical G1 asset."""

    def __init__(self) -> None:
        super().__init__()
        root = ET.parse(canonical_g1_urdf_path()).getroot()  # noqa: S314
        links = {element.get("name", "") for element in root.findall("link")}
        raw_joints = []
        child_links = set()
        q_index = {name: index for index, name in enumerate(G1_29DOF_JOINT_ORDER)}
        lower = torch.empty(len(q_index), dtype=torch.float32)
        upper = torch.empty(len(q_index), dtype=torch.float32)
        for joint in root.findall("joint"):
            name = joint.get("name", "")
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_name = parent.get("link", "")
            child_name = child.get("link", "")
            child_links.add(child_name)
            origin = joint.find("origin")
            xyz_text = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy_text = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            xyz = tuple(float(value) for value in xyz_text.split())
            rpy = tuple(float(value) for value in rpy_text.split())
            axis_node = joint.find("axis")
            axis_text = axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0"
            axis = tuple(float(value) for value in axis_text.split())
            index = q_index.get(name, -1)
            if index >= 0:
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"canonical G1 joint has no limit: {name}")
                lower[index] = float(limit.get("lower", "nan"))
                upper[index] = float(limit.get("upper", "nan"))
            raw_joints.append((name, parent_name, child_name, joint.get("type", "fixed"), xyz, rpy, axis, index))

        root_links = links - child_links
        if len(root_links) != 1:
            raise ValueError(f"canonical G1 URDF must have one root link, got {sorted(root_links)}")
        root_link = next(iter(root_links))
        children: dict[str, list[tuple]] = {}
        for record in raw_joints:
            children.setdefault(record[1], []).append(record)
        ordered = []
        stack = [root_link]
        while stack:
            parent_name = stack.pop()
            for record in children.get(parent_name, ()):
                ordered.append(record)
                stack.append(record[2])
        if len(ordered) != len(raw_joints):
            raise ValueError("canonical G1 URDF joint graph is disconnected")
        missing = set(BODY_NAMES) - links
        if missing:
            raise ValueError(f"canonical G1 URDF is missing keypoint links: {sorted(missing)}")

        self.root_link = root_link
        self.joint_records = tuple((record[1], record[2], record[3], record[7]) for record in ordered)
        record_by_child = {record[2]: record for record in ordered}
        body_joint_ancestor = torch.zeros((len(BODY_NAMES), len(q_index)), dtype=torch.bool)
        for body_index, body_name in enumerate(BODY_NAMES):
            link_name = body_name
            while link_name != root_link:
                record = record_by_child.get(link_name)
                if record is None:
                    raise ValueError(f"canonical G1 cannot trace {body_name!r} to root")
                if record[7] >= 0:
                    body_joint_ancestor[body_index, record[7]] = True
                link_name = record[1]
        self.register_buffer("body_joint_ancestor", body_joint_ancestor, persistent=False)
        self.register_buffer("origin_xyz", torch.tensor([record[4] for record in ordered], dtype=torch.float32))
        self.register_buffer("origin_rotation", torch.stack([_rpy_matrix(record[5]) for record in ordered]))
        self.register_buffer("joint_axis", torch.tensor([record[6] for record in ordered], dtype=torch.float32))
        self.register_buffer("joint_lower", lower)
        self.register_buffer("joint_upper", upper)

    def bound_joints(self, raw_joint: Tensor) -> Tensor:
        midpoint = 0.5 * (self.joint_lower + self.joint_upper)
        half_range = 0.5 * (self.joint_upper - self.joint_lower)
        return midpoint + half_range * torch.tanh(raw_joint)

    def link_poses(self, qpos: Tensor, link_names: tuple[str, ...]) -> tuple[Tensor, Tensor]:
        if qpos.shape[-1] != 36:
            raise ValueError(f"qpos must end in 36 values, got {tuple(qpos.shape)}")
        cached = getattr(self, "_link_pose_cache", None)
        if cached is not None and cached[0] is qpos and cached[1] is not None:
            world_position, world_rotation = cached[1], cached[2]
        else:
            world_position, world_rotation = self._world_poses(qpos)
            # A cache scope belongs to one exact prediction tensor. Diagnostic
            # FK of a target pose must not evict that differentiable graph.
            if cached is not None and cached[0] is qpos:
                self._link_pose_cache = (qpos, world_position, world_rotation)
        missing = set(link_names) - world_position.keys()
        if missing:
            raise ValueError(f"unknown canonical G1 links: {sorted(missing)}")
        position = torch.stack([world_position[name] for name in link_names], dim=-2)
        rotation = torch.stack([world_rotation[name] for name in link_names], dim=-3)
        return position, rotation

    def _world_poses(self, qpos: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Pure tensor traversal, separate from the eager prediction cache."""
        world_rotation = {self.root_link: _quaternion_matrix_wxyz(qpos[..., 3:7])}
        world_position = {self.root_link: qpos[..., :3]}
        for joint_index, (parent, child, joint_type, q_index) in enumerate(self.joint_records):
            parent_rotation = world_rotation[parent]
            parent_position = world_position[parent]
            origin_rotation = self.origin_rotation[joint_index].to(qpos)
            origin_position = self.origin_xyz[joint_index].to(qpos)
            child_position = parent_position + torch.einsum("...ij,j->...i", parent_rotation, origin_position)
            child_rotation = parent_rotation @ origin_rotation
            if q_index >= 0:
                if joint_type not in {"revolute", "continuous"}:
                    raise ValueError(f"unsupported actuated canonical G1 joint type: {joint_type}")
                axis = self.joint_axis[joint_index].to(qpos).expand(qpos.shape[:-1] + (3,))
                child_rotation = child_rotation @ _axis_angle_matrix(axis, qpos[..., 7 + q_index])
            world_position[child] = child_position
            world_rotation[child] = child_rotation
        return world_position, world_rotation

    def begin_link_pose_cache(self, qpos: Tensor) -> None:
        if getattr(self, "_link_pose_cache", None) is not None:
            raise RuntimeError("Nested forward-kinematics cache")
        self._link_pose_cache = (qpos, None, None)

    def end_link_pose_cache(self) -> None:
        self._link_pose_cache = None

    def forward(self, qpos: Tensor) -> tuple[Tensor, Tensor]:
        position, rotation = self.link_poses(qpos, BODY_NAMES)
        return position, _rotation6d(rotation)

    def point_geometric_jacobian(
        self,
        qpos: Tensor,
        point_position: Tensor,
        body_indices: tuple[int, ...],
    ) -> Tensor:
        """World-frame point Jacobian over the 35D floating-base tangent space."""

        if qpos.ndim != 2 or qpos.shape[-1] != 36:
            raise ValueError("qpos must have shape [B,36]")
        if point_position.shape != (len(qpos), len(body_indices), 3):
            raise ValueError("point_position must have shape [B,N,3]")

        world_rotation = {self.root_link: _quaternion_matrix_wxyz(qpos[:, 3:7])}
        world_position = {self.root_link: qpos[:, :3]}
        joint_origins: list[Tensor | None] = [None] * len(self.joint_lower)
        joint_axes: list[Tensor | None] = [None] * len(self.joint_lower)
        for joint_index, (parent, child, joint_type, q_index) in enumerate(self.joint_records):
            parent_rotation = world_rotation[parent]
            parent_position = world_position[parent]
            origin_rotation = self.origin_rotation[joint_index].to(qpos)
            origin_position = self.origin_xyz[joint_index].to(qpos)
            child_position = parent_position + torch.einsum("bij,j->bi", parent_rotation, origin_position)
            child_rotation = parent_rotation @ origin_rotation
            if q_index >= 0:
                if joint_type not in {"revolute", "continuous"}:
                    raise ValueError(f"unsupported actuated canonical G1 joint type: {joint_type}")
                axis = self.joint_axis[joint_index].to(qpos).expand((len(qpos), 3))
                joint_origins[q_index] = child_position
                joint_axes[q_index] = torch.einsum("bij,bj->bi", child_rotation, axis)
                child_rotation = child_rotation @ _axis_angle_matrix(axis, qpos[:, 7 + q_index])
            world_position[child] = child_position
            world_rotation[child] = child_rotation

        if any(value is None for value in joint_origins) or any(value is None for value in joint_axes):
            raise RuntimeError("canonical G1 Jacobian is missing an actuated joint")
        origins = torch.stack([value for value in joint_origins if value is not None], dim=1)
        axes = torch.stack([value for value in joint_axes if value is not None], dim=1)
        batch, points = point_position.shape[:2]
        identity = torch.eye(3, dtype=qpos.dtype, device=qpos.device).expand(batch, points, 3, 3)
        root_lever = point_position - qpos[:, None, :3]
        root_rotation = torch.linalg.cross(
            identity,
            root_lever[:, :, None].expand(-1, -1, 3, -1),
            dim=-1,
        ).transpose(-1, -2)
        joint_lever = point_position[:, :, None] - origins[:, None]
        joint_columns = torch.linalg.cross(axes[:, None], joint_lever, dim=-1).permute(0, 1, 3, 2)
        ancestor = self.body_joint_ancestor[list(body_indices)].to(qpos)
        joint_columns = joint_columns * ancestor[None, :, None]
        return torch.cat((identity, root_rotation, joint_columns), dim=-1)

