from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from somaforge_core import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.robot_assets import canonical_g1_urdf_path
from somaforge_core.motion_contracts import CONTACT_PARTS
from somaforge_core.g1_kinematics import _rpy_matrix, CanonicalG1ForwardKinematics


def _mesh_vertices(path: Path, scale: tuple[float, float, float], maximum_points: int) -> Tensor:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append(tuple(float(value) for value in values[1:4]))
    if not vertices:
        raise ValueError(f"canonical G1 collision mesh has no vertices: {path}")
    values = np.asarray(vertices, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
    if len(values) > maximum_points:
        indices = np.linspace(0, len(values) - 1, maximum_points, dtype=np.int64)
        values = values[indices]
    return torch.from_numpy(values)


def _sphere_points(radius: float) -> Tensor:
    directions = torch.tensor(
        (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
            (-1, -1, 1),
            (-1, 1, -1),
            (1, -1, -1),
            (-1, -1, -1),
        ),
        dtype=torch.float32,
    )
    return radius * F.normalize(directions, dim=-1)


def _cylinder_points(radius: float, length: float, count: int = 16) -> Tensor:
    angle = torch.arange(count, dtype=torch.float32) * (2.0 * math.pi / count)
    ring = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
    lower = torch.cat((ring, torch.full((count, 1), -0.5 * length)), dim=-1)
    upper = torch.cat((ring, torch.full((count, 1), 0.5 * length)), dim=-1)
    return torch.cat((lower, upper), dim=0)


class CanonicalG1CollisionPoints(nn.Module):
    """Differentiable samples of every canonical G1 collision geometry."""

    def __init__(self, maximum_mesh_points: int = 64) -> None:
        super().__init__()
        urdf_path = canonical_g1_urdf_path()
        root = ET.parse(urdf_path).getroot()  # noqa: S314
        contact_links = (
            CONTACT_BODY_NAMES_BY_PART["left_foot"],
            CONTACT_BODY_NAMES_BY_PART["right_foot"],
            CONTACT_BODY_NAMES_BY_PART["left_hand"],
            CONTACT_BODY_NAMES_BY_PART["right_hand"],
            CONTACT_BODY_NAMES_BY_PART["left_knee"],
            CONTACT_BODY_NAMES_BY_PART["right_knee"],
        )
        contact_part_by_link = {
            link_name: part_index for part_index, link_names in enumerate(contact_links) for link_name in link_names
        }
        link_names: list[str] = []
        local_points: list[Tensor] = []
        contact_parts: list[Tensor] = []
        for link in root.findall("link"):
            link_name = link.get("name", "")
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None or len(geometry) != 1:
                    continue
                shape = geometry[0]
                if shape.tag == "mesh":
                    scale = tuple(float(value) for value in shape.get("scale", "1 1 1").split())
                    points = _mesh_vertices(urdf_path.parent / shape.get("filename", ""), scale, maximum_mesh_points)
                elif shape.tag == "sphere":
                    points = _sphere_points(float(shape.get("radius", "nan")))
                elif shape.tag == "cylinder":
                    points = _cylinder_points(float(shape.get("radius", "nan")), float(shape.get("length", "nan")))
                else:
                    raise ValueError(f"unsupported canonical G1 collision geometry: {shape.tag}")
                origin = collision.find("origin")
                xyz_text = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
                rpy_text = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
                xyz = torch.tensor(tuple(float(value) for value in xyz_text.split()), dtype=torch.float32)
                rpy = tuple(float(value) for value in rpy_text.split())
                points = torch.einsum("ij,pj->pi", _rpy_matrix(rpy), points) + xyz
                link_names.append(link_name)
                local_points.append(points)
                contact_parts.append(torch.full((len(points),), contact_part_by_link.get(link_name, -1)))
        if not local_points:
            raise ValueError("canonical G1 URDF has no collision geometry")
        self.link_names = tuple(link_names)
        self.point_counts = tuple(len(points) for points in local_points)
        self.register_buffer("local_points", torch.cat(local_points, dim=0))
        self.register_buffer("contact_part", torch.cat(contact_parts, dim=0).to(torch.int64))

    def forward(self, fk: CanonicalG1ForwardKinematics, qpos: Tensor) -> tuple[Tensor, Tensor]:
        link_position, link_rotation = fk.link_poses(qpos, self.link_names)
        output = []
        start = 0
        for link_index, count in enumerate(self.point_counts):
            points = self.local_points[start : start + count].to(qpos)
            world = torch.einsum("...ij,pj->...pi", link_rotation[..., link_index, :, :], points)
            output.append(world + link_position[..., link_index, None, :])
            start += count
        return torch.cat(output, dim=-2), self.contact_part


def full_geometry_box_ground_penetration(
    points: Tensor,
    point_contact_part: Tensor,
    contact: Tensor,
    *,
    box_center: Tensor,
    box_rotation: Tensor,
    box_half_extents: Tensor,
    ground_height: Tensor,
    margin_m: float = 0.002,
    planned_contact_tolerance_m: float = 0.005,
) -> tuple[Tensor, Tensor]:
    """Return box and ground penetration depths for every geometry sample."""

    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B,T,P,3], got {tuple(points.shape)}")
    batch, frames, _, _ = points.shape
    if contact.shape != (batch, frames, len(CONTACT_PARTS)):
        raise ValueError(f"contact must be [B,T,{len(CONTACT_PARTS)}], got {tuple(contact.shape)}")
    center = box_center[:, None, None]
    rotation = box_rotation[:, None, None]
    local = torch.einsum("btpji,btpj->btpi", rotation, points - center)
    distance_to_faces = local.abs() - box_half_extents[:, None, None]
    outside = torch.linalg.vector_norm(F.relu(distance_to_faces), dim=-1)
    inside = torch.clamp(distance_to_faces.amax(dim=-1), max=0.0)
    box_signed_distance = outside + inside
    ground_signed_distance = points[..., 2] - ground_height[:, None, None]

    valid_part = point_contact_part >= 0
    part_index = point_contact_part.clamp_min(0)
    active = contact[..., part_index] > 0.5
    active = active & valid_part[None, None]
    tolerance = torch.where(
        active,
        points.new_tensor(planned_contact_tolerance_m),
        points.new_zeros(()),
    )
    box_penetration = F.relu(-box_signed_distance + margin_m - tolerance)
    ground_penetration = F.relu(-ground_signed_distance + margin_m - tolerance)
    return box_penetration, ground_penetration


def full_geometry_box_penetration(
    points: Tensor,
    point_contact_part: Tensor,
    contact: Tensor,
    *,
    box_center: Tensor,
    box_rotation: Tensor,
    box_half_extents: Tensor,
    ground_height: Tensor,
    margin_m: float = 0.002,
    planned_contact_tolerance_m: float = 0.005,
) -> Tensor:
    """Return the worst box-or-ground penetration for every geometry sample."""

    box_penetration, ground_penetration = full_geometry_box_ground_penetration(
        points,
        point_contact_part,
        contact,
        box_center=box_center,
        box_rotation=box_rotation,
        box_half_extents=box_half_extents,
        ground_height=ground_height,
        margin_m=margin_m,
        planned_contact_tolerance_m=planned_contact_tolerance_m,
    )
    return torch.maximum(box_penetration, ground_penetration)


def full_geometry_contact_surface_penalty(
    points: Tensor,
    target_points: Tensor,
    point_contact_part: Tensor,
    contact: Tensor,
    *,
    box_center: Tensor,
    box_rotation: Tensor,
    box_half_extents: Tensor,
    ground_height: Tensor,
) -> Tensor:
    """Match active contact geometry to its demonstrated support surface.

    Collision is a one-sided constraint and therefore cannot distinguish a
    safe contact from a hovering one.  The target geometry selects ground or
    box independently for every canonical contact sample; the generated
    sample is then matched to that same surface's signed distance.  This
    treats penetration and separation symmetrically without locking a link to
    a fixed world pose.
    """

    if points.shape != target_points.shape or points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points and target_points must have identical [B,T,P,3] shapes")

    def signed_distances(value: Tensor) -> tuple[Tensor, Tensor]:
        center = box_center[:, None, None]
        rotation = box_rotation[:, None, None]
        local = torch.einsum("btpji,btpj->btpi", rotation, value - center)
        distance_to_faces = local.abs() - box_half_extents[:, None, None]
        outside = torch.linalg.vector_norm(F.relu(distance_to_faces), dim=-1)
        inside = torch.clamp(distance_to_faces.amax(dim=-1), max=0.0)
        return outside + inside, value[..., 2] - ground_height[:, None, None]

    output_box, output_ground = signed_distances(points)
    target_box, target_ground = signed_distances(target_points)
    target_uses_box = target_box.abs() <= target_ground.abs()
    output_distance = torch.where(target_uses_box, output_box, output_ground)
    target_distance = torch.where(target_uses_box, target_box, target_ground)

    valid_part = point_contact_part >= 0
    part_index = point_contact_part.clamp_min(0)
    active = (contact[..., part_index] > 0.5) & valid_part[None, None]
    error = ((output_distance - target_distance) / 0.01).square()
    return (error * active).sum() / active.sum().clamp_min(1)


def full_geometry_box_collision_penalty(
    points: Tensor,
    point_contact_part: Tensor,
    contact: Tensor,
    *,
    box_center: Tensor,
    box_rotation: Tensor,
    box_half_extents: Tensor,
    ground_height: Tensor,
    margin_m: float = 0.002,
    planned_contact_tolerance_m: float = 0.005,
) -> Tensor:
    """Penalize canonical collision-surface samples inside ground or an OBB."""

    penetration = full_geometry_box_penetration(
        points,
        point_contact_part,
        contact,
        box_center=box_center,
        box_rotation=box_rotation,
        box_half_extents=box_half_extents,
        ground_height=ground_height,
        margin_m=margin_m,
        planned_contact_tolerance_m=planned_contact_tolerance_m,
    )
    return (penetration / 0.01).square().mean() + (penetration.amax(dim=(1, 2)) / 0.01).square().mean()

