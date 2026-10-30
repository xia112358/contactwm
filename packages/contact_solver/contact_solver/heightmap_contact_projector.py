"""Contact projection using only a height map and current contact observation.

The height map is the sole terrain geometry available to the optimizer.
Newton is queried once after optimization as an authoritative audit; its
surface catalog and hypothetical-pose witnesses never enter the solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import Tensor

from somaforge_core.contact_face_selection import select_contact_pairs

from contact_solver.contact_constrained_projector import CanonicalContactCollisionGeometry, ContactProjectionResult, _maximum_penetration, apply_tangent_delta
from somaforge_core.motion_contracts import CONTACT_PARTS
from contact_solver.collision_geometry import CanonicalG1CollisionPoints
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from somaforge_core.heightmap import HEIGHTMAP_COLS, HEIGHTMAP_FORWARD_MAX_M, HEIGHTMAP_FORWARD_MIN_M, HEIGHTMAP_LATERAL_MAX_M, HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_RESOLUTION_M, HEIGHTMAP_ROWS
from somaforge_core.heightmap import _root_yaw_basis


NewtonAudit = Callable[[np.ndarray, list[np.ndarray] | None], dict]


def observed_surface_indices(
    heightmap: np.ndarray,
    current_q: np.ndarray,
    current_contact: np.ndarray,
    current_points_w: np.ndarray,
) -> np.ndarray:
    """Classify current contacts using only observed height levels."""

    heightmap = np.asarray(heightmap, dtype=np.float32)
    current_q = np.asarray(current_q, dtype=np.float32)
    current_contact = np.asarray(current_contact, dtype=bool)
    current_points_w = np.asarray(current_points_w, dtype=np.float32)
    if heightmap.ndim == 2:
        heightmap = heightmap[None]
        current_q = current_q[None]
        current_contact = current_contact[None]
        current_points_w = current_points_w[None]
        squeeze = True
    else:
        squeeze = False
    if heightmap.shape[1:] != (HEIGHTMAP_ROWS, HEIGHTMAP_COLS):
        raise ValueError("heightmap batch has an invalid shape")
    if current_q.shape != (len(heightmap), 36):
        raise ValueError("current_q must be [B,36]")
    if current_contact.shape != (len(heightmap), len(CONTACT_PARTS)):
        raise ValueError("current_contact must be [B,6]")
    if current_points_w.shape != (len(heightmap), len(CONTACT_PARTS), 3):
        raise ValueError("current_points_w must be [B,6,3]")
    flat = heightmap.reshape(len(heightmap), -1)
    levels = np.stack((flat.min(-1), flat.max(-1)), -1)
    relative_z = current_points_w[..., 2] - current_q[:, None, 2]
    result = np.abs(relative_z[..., None] - levels[:, None]).argmin(-1).astype(np.int64)
    result[~current_contact] = -1
    return result[0] if squeeze else result


@dataclass(frozen=True)
class ObservedHeightmapSurfaces:
    """Two temporary horizontal regions extracted from one height observation.

    Surface indices are ordered by observed height.  They have no persistent
    semantic meaning such as ``ground`` or ``box_top``.
    """

    heightmap: Tensor
    collision_heightmap: Tensor
    observation_root: Tensor
    world_from_local: Tensor
    level_height_local: Tensor
    region_mask: Tensor
    outside_distance: Tensor

    @classmethod
    def from_observation(cls, heightmap: Tensor, current_q: Tensor) -> "ObservedHeightmapSurfaces":
        if heightmap.shape != (HEIGHTMAP_ROWS, HEIGHTMAP_COLS):
            raise ValueError(
                f"heightmap must be [{HEIGHTMAP_ROWS},{HEIGHTMAP_COLS}], got {tuple(heightmap.shape)}"
            )
        if current_q.shape != (36,):
            raise ValueError("current_q must have shape [36]")
        values = heightmap.detach().cpu().numpy().astype(np.float32)
        flat = values.reshape(-1)
        centers = np.asarray((float(flat.min()), float(flat.max())), dtype=np.float32)
        if centers[1] - centers[0] > np.finfo(np.float32).eps:
            for _ in range(16):
                assignment = np.abs(flat[:, None] - centers[None]).argmin(-1)
                updated = centers.copy()
                for index in range(2):
                    selected = flat[assignment == index]
                    if len(selected):
                        updated[index] = float(np.median(selected))
                updated.sort()
                if np.array_equal(updated, centers):
                    break
                centers = updated
        distance = np.abs(values[None] - centers[:, None, None])
        assignment = distance.argmin(0)
        masks = np.stack((assignment == 0, assignment == 1))
        if centers[1] == centers[0]:
            masks[:] = True
        outside = np.stack([
            distance_transform_edt(~mask).astype(np.float32) * HEIGHTMAP_RESOLUTION_M
            for mask in masks
        ])
        basis, _ = _root_yaw_basis(current_q[None])
        # A scan sample represents its 2 cm cell, not only its center.  The
        # one-cell upper envelope is conservative at discontinuous ledges and
        # is derived entirely from the observation (no box boundary input).
        collision_heightmap = F.max_pool2d(
            heightmap.reshape(1, 1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS),
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0]
        return cls(
            heightmap=heightmap.detach().clone(),
            collision_heightmap=collision_heightmap.detach().clone(),
            observation_root=current_q[:3].detach().clone(),
            world_from_local=basis[0].detach().clone(),
            level_height_local=torch.as_tensor(centers, device=heightmap.device, dtype=heightmap.dtype),
            region_mask=torch.as_tensor(masks, device=heightmap.device),
            outside_distance=torch.as_tensor(outside, device=heightmap.device, dtype=heightmap.dtype),
        )

    def _local_xy(self, points_w: Tensor) -> Tensor:
        relative = points_w - self.observation_root.to(points_w)
        local = torch.einsum("ij,...j->...i", self.world_from_local.to(points_w).T, relative)
        return local[..., :2]

    @staticmethod
    def _grid(local_xy: Tensor) -> Tensor:
        forward, lateral = local_xy.unbind(-1)
        grid_x = 2.0 * (lateral - HEIGHTMAP_LATERAL_MIN_M) / (
            HEIGHTMAP_LATERAL_MAX_M - HEIGHTMAP_LATERAL_MIN_M
        ) - 1.0
        grid_y = 2.0 * (forward - HEIGHTMAP_FORWARD_MIN_M) / (
            HEIGHTMAP_FORWARD_MAX_M - HEIGHTMAP_FORWARD_MIN_M
        ) - 1.0
        return torch.stack((grid_x, grid_y), -1)

    def sample(self, field: Tensor, points_w: Tensor) -> Tensor:
        shape = points_w.shape[:-1]
        grid = self._grid(self._local_xy(points_w)).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            field.to(points_w).reshape(1, 1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(shape)

    def terrain_height_world(self, points_w: Tensor) -> Tensor:
        return self.observation_root.to(points_w)[2] + self.sample(
            self.collision_heightmap, points_w
        )

    def surface_height_world(self, surface: int, like: Tensor) -> Tensor:
        if surface not in (0, 1):
            raise ValueError(f"observed surface index must be 0 or 1, got {surface}")
        return self.observation_root.to(like)[2] + self.level_height_local.to(like)[surface]

    def outside(self, surface: int, points_w: Tensor) -> Tensor:
        if surface not in (0, 1):
            raise ValueError(f"observed surface index must be 0 or 1, got {surface}")
        return self.sample(self.outside_distance[surface], points_w)

    def classify_contact_points(self, mask: np.ndarray, points_w: np.ndarray) -> np.ndarray:
        """Assign current Newton contact points to observation-derived regions."""

        mask = np.asarray(mask, dtype=bool)
        points = torch.as_tensor(points_w, device=self.heightmap.device, dtype=self.heightmap.dtype)
        result = np.full(len(CONTACT_PARTS), -1, dtype=np.int64)
        if not mask.any():
            return result
        with torch.no_grad():
            for part in np.flatnonzero(mask):
                point = points[part : part + 1]
                costs = []
                for surface in range(2):
                    vertical = (point[0, 2] - self.surface_height_world(surface, point)).abs()
                    costs.append(vertical + self.outside(surface, point)[0])
                result[part] = int(torch.stack(costs).argmin())
        return result


class HeightmapContactProjector:
    """Project a keyframe using observed height geometry, then audit once."""

    def __init__(
        self,
        audit_query: NewtonAudit,
        *,
        iterations: int = 8,
        damping: float = 2.0e-3,
        maximum_step: float = 0.20,
        residual_scale_m: float = 0.02,
        line_search_steps: int = 4,
        enforce_target_surface: bool = True,
    ) -> None:
        self.audit_query = audit_query
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.maximum_step = float(maximum_step)
        self.residual_scale_m = float(residual_scale_m)
        self.line_search_steps = int(line_search_steps)
        self.enforce_target_surface = bool(enforce_target_surface)
        self.fk = CanonicalG1ForwardKinematics()
        self.contact_geometry = CanonicalContactCollisionGeometry()
        # Robot geometry is proprioceptive model knowledge, not environment
        # privilege.  Keep every collision-mesh vertex so strict zero-
        # penetration auditing cannot expose a low vertex hidden by sampling.
        self.full_geometry = CanonicalG1CollisionPoints(2**31 - 1)

    def to(self, device: torch.device | str) -> "HeightmapContactProjector":
        self.fk.to(device)
        self.contact_geometry.to(device)
        self.full_geometry.to(device)
        return self

    def _part_residual(
        self,
        qpos: Tensor,
        part: int,
        surface: int,
        observation: ObservedHeightmapSurfaces,
    ) -> Tensor:
        candidates: list[Tensor] = []
        merits: list[Tensor] = []
        upward = qpos.new_tensor((0.0, 0.0, 1.0))
        numerical_clearance = qpos.new_tensor(64.0 * torch.finfo(qpos.dtype).eps) * (
            1.0 + observation.observation_root.to(qpos).abs().max()
        )
        target_z = observation.surface_height_world(surface, qpos) + numerical_clearance
        for shape in self.contact_geometry.shapes:
            if shape.part != part:
                continue
            link_position, link_rotation = self.fk.link_poses(qpos, (shape.link_name,))
            position, rotation = link_position[:, 0], link_rotation[:, 0]
            if shape.kind == "sphere":
                center = position + torch.einsum("bij,j->bi", rotation, shape.local_center.to(qpos))
                points = center - qpos.new_tensor(float(shape.radius)) * upward
            else:
                points = torch.einsum("bij,pj->bpi", rotation, shape.local_points.to(qpos))
                points = points + position[:, None]
                vertical = points[..., 2] - target_z
                outside = observation.outside(surface, points)
                nearest = (vertical.detach().square() + outside.detach().square()).argmin(-1)
                points = points[torch.arange(len(qpos), device=qpos.device), nearest]
            vertical = points[:, 2] - target_z
            outside = observation.outside(surface, points)
            residual = torch.stack((vertical, outside), -1) / self.residual_scale_m
            candidates.append(residual)
            merits.append(residual.detach().square().sum(-1))
        if not candidates:
            raise ValueError(f"canonical collision asset has no shapes for contact part {part}")
        choice = torch.stack(merits, -1).argmin(-1)
        stacked = torch.stack(candidates, 1)
        return stacked[torch.arange(len(qpos), device=qpos.device), choice]

    def _residual(
        self,
        qpos: Tensor,
        intended_contact: Tensor,
        intended_surface: Tensor,
        observation: ObservedHeightmapSurfaces,
    ) -> Tensor:
        residuals = []
        if self.enforce_target_surface:
            residuals.extend(
                self._part_residual(qpos, part, int(intended_surface[part]), observation).reshape(-1)
                for part in range(len(CONTACT_PARTS))
                if bool(intended_contact[part])
            )
        points, _ = self.full_geometry(self.fk, qpos)
        terrain_z = observation.terrain_height_world(points)
        numerical_clearance = qpos.new_tensor(64.0 * torch.finfo(qpos.dtype).eps) * (
            1.0 + observation.observation_root.to(qpos).abs().max()
        )
        signed_penetration = terrain_z + numerical_clearance - points[..., 2]
        # One exact worst vertex per collision shape preserves the strict
        # envelope without constructing an enormous dense Jacobian.
        shape_worst = []
        start = 0
        for count in self.full_geometry.point_counts:
            shape_worst.append(signed_penetration[:, start : start + count].max(-1).values)
            start += count
        penetration = torch.stack(shape_worst, -1).relu() / self.residual_scale_m
        residuals.append(penetration.reshape(-1))
        return torch.cat(residuals) if residuals else qpos.new_zeros((0,))

    @staticmethod
    def _audit_plan(observed: dict, surfaces: ObservedHeightmapSurfaces) -> tuple[np.ndarray, np.ndarray]:
        selected = select_contact_pairs(observed["pairs"], observed["surface_catalog"])
        mask = np.asarray(selected["contact_part_mask"][0], dtype=bool)
        points = np.asarray(selected["contact_position_w"][0], dtype=np.float32)
        return mask, surfaces.classify_contact_points(mask, points)

    def project(
        self,
        nominal_qpos: Tensor,
        intended_contact: Tensor,
        intended_surface: Tensor,
        observation: ObservedHeightmapSurfaces,
        *,
        safe_qpos: Tensor | None = None,
    ) -> ContactProjectionResult:
        intended_contact = intended_contact.to(device=nominal_qpos.device, dtype=torch.bool)
        intended_surface = intended_surface.to(device=nominal_qpos.device, dtype=torch.long)
        current = nominal_qpos.detach().clone()[None]
        nominal = current[0].clone()
        history: list[dict[str, float | int | bool]] = []
        for iteration in range(1, self.iterations + 1):
            delta = current.new_zeros((1, 35), requires_grad=True)
            candidate = apply_tangent_delta(current, delta, self.fk)
            residual = self._residual(candidate, intended_contact, intended_surface, observation)
            merit = float(torch.linalg.vector_norm(residual).detach())
            history.append({
                "iteration": iteration,
                "residual_norm": merit,
                "geometry_source_heightmap_only": True,
            })
            if merit < 1.0e-4 or residual.numel() == 0:
                current = candidate.detach()
                break
            eye = torch.eye(len(residual), dtype=residual.dtype, device=residual.device)
            jacobian = torch.autograd.grad(
                residual, delta, grad_outputs=eye, is_grads_batched=True
            )[0][:, 0]
            jacobian64, residual64 = jacobian.double(), residual.detach().double()
            system = jacobian64.T @ jacobian64 + self.damping**2 * torch.eye(
                35, dtype=torch.float64, device=residual.device
            )
            step = -torch.linalg.solve(system, jacobian64.T @ residual64).to(residual.dtype)
            norm = torch.linalg.vector_norm(step)
            step = step * (self.maximum_step / norm.clamp_min(self.maximum_step)).clamp_max(1.0)
            accepted = False
            for line_search in range(self.line_search_steps):
                scale = 0.5**line_search
                trial = apply_tangent_delta(current, (scale * step)[None], self.fk).detach()
                with torch.no_grad():
                    trial_merit = float(torch.linalg.vector_norm(
                        self._residual(trial, intended_contact, intended_surface, observation)
                    ))
                if trial_merit < merit:
                    current = trial
                    history[-1]["accepted_step_scale"] = scale
                    accepted = True
                    break
            if not accepted:
                history[-1]["accepted_step_scale"] = 0.0
                break

        # Newton is an output audit only.  Nothing from this query is fed back
        # into the optimizer or predictor.
        final_observed = self.audit_query(current.detach().cpu().numpy(), None)
        actual_mask, actual_surface = self._audit_plan(final_observed, observation)
        intended_np = intended_contact.detach().cpu().numpy()
        intended_surface_np = intended_surface.detach().cpu().numpy()
        shared = intended_np & actual_mask
        unrealized = int(np.count_nonzero(intended_np & ~actual_mask))
        extra = int(np.count_nonzero(~intended_np & actual_mask))
        surface_match = bool(np.array_equal(actual_surface[shared], intended_surface_np[shared]))
        attempted_penetration = _maximum_penetration(final_observed)
        attempted_separation = final_observed["full_robot_separation"][0]
        attempted_terrain_penetration = max(
            0.0,
            -float(attempted_separation["worst_terrain"]["dist"])
            if attempted_separation.get("worst_terrain") is not None else 0.0,
        )
        attempted_self_penetration = max(
            0.0,
            -float(attempted_separation["worst_self"]["dist"])
            if attempted_separation.get("worst_self") is not None else 0.0,
        )
        penetration = attempted_penetration
        fallback = False
        if penetration > 0.0 and safe_qpos is not None:
            current = safe_qpos.detach().to(current)[None]
            final_observed = self.audit_query(current.detach().cpu().numpy(), None)
            actual_mask, actual_surface = self._audit_plan(final_observed, observation)
            shared = intended_np & actual_mask
            unrealized = int(np.count_nonzero(intended_np & ~actual_mask))
            extra = int(np.count_nonzero(~intended_np & actual_mask))
            surface_match = bool(np.array_equal(actual_surface[shared], intended_surface_np[shared]))
            penetration = _maximum_penetration(final_observed)
            fallback = True
        converged = unrealized == 0 and extra == 0 and surface_match and penetration == 0.0 and not fallback
        history.append({
            "output_newton_audit": True,
            "unrealized_contacts": unrealized,
            "extra_contacts": extra,
            "surface_match": surface_match,
            "maximum_penetration_m": penetration,
            "attempted_terrain_penetration_m": attempted_terrain_penetration,
            "attempted_self_penetration_m": attempted_self_penetration,
            "attempted_worst_terrain": attempted_separation.get("worst_terrain"),
            "attempted_worst_self": attempted_separation.get("worst_self"),
        })
        return ContactProjectionResult(
            qpos=current[0],
            converged=converged,
            iterations=max(len(history) - 1, 0),
            intended_contact=intended_contact,
            intended_surface=intended_surface,
            actual_contact=torch.as_tensor(actual_mask, device=current.device),
            actual_surface=torch.as_tensor(actual_surface, device=current.device),
            unrealized_contacts=unrealized,
            extra_contacts=extra,
            maximum_penetration_m=penetration,
            root_displacement_m=float(torch.linalg.vector_norm(current[0, :3] - nominal[:3])),
            joint_displacement_rad=float((current[0, 7:] - nominal[7:]).abs().max()),
            safety_fallback_used=fallback,
            attempted_maximum_penetration_m=attempted_penetration,
            history=tuple(history),
            observed=final_observed,
        )


def verify_observation_only_prediction(
    audit_query: NewtonAudit,
    nominal_qpos: Tensor,
    intended_contact: Tensor,
    intended_surface: Tensor,
    observation: ObservedHeightmapSurfaces,
    *,
    safe_qpos: Tensor | None = None,
) -> ContactProjectionResult:
    """Audit a one-pass prediction without feeding geometry back to it."""

    intended_contact = intended_contact.to(device=nominal_qpos.device, dtype=torch.bool)
    intended_surface = intended_surface.to(device=nominal_qpos.device, dtype=torch.long)
    attempted = nominal_qpos.detach().clone()
    attempted_observed = audit_query(attempted[None].cpu().numpy(), None)
    attempted_mask, attempted_surface = HeightmapContactProjector._audit_plan(
        attempted_observed, observation
    )
    intended_np = intended_contact.cpu().numpy()
    intended_surface_np = intended_surface.cpu().numpy()
    shared = intended_np & attempted_mask
    unrealized = int(np.count_nonzero(intended_np & ~attempted_mask))
    extra = int(np.count_nonzero(~intended_np & attempted_mask))
    surface_match = bool(
        np.array_equal(attempted_surface[shared], intended_surface_np[shared])
    )
    attempted_penetration = _maximum_penetration(attempted_observed)
    accepted = (
        unrealized == 0
        and extra == 0
        and surface_match
        and attempted_penetration == 0.0
    )
    final_qpos = attempted
    final_observed = attempted_observed
    actual_mask = attempted_mask
    actual_surface = attempted_surface
    fallback = False
    if not accepted and safe_qpos is not None:
        final_qpos = safe_qpos.detach().to(attempted)
        final_observed = audit_query(final_qpos[None].cpu().numpy(), None)
        actual_mask, actual_surface = HeightmapContactProjector._audit_plan(
            final_observed, observation
        )
        fallback = True
    history = ({
        "output_newton_audit": True,
        "geometry_feedback_to_prediction": False,
        "topology_exact": unrealized == 0 and extra == 0 and surface_match,
        "unrealized_contacts": unrealized,
        "extra_contacts": extra,
        "surface_match": surface_match,
        "maximum_penetration_m": attempted_penetration,
        "attempted_contact": attempted_mask.tolist(),
        "attempted_surface": attempted_surface.tolist(),
    },)
    return ContactProjectionResult(
        qpos=final_qpos,
        converged=accepted,
        iterations=0,
        intended_contact=intended_contact,
        intended_surface=intended_surface,
        actual_contact=torch.as_tensor(actual_mask, device=nominal_qpos.device),
        actual_surface=torch.as_tensor(actual_surface, device=nominal_qpos.device),
        unrealized_contacts=unrealized,
        extra_contacts=extra,
        maximum_penetration_m=(0.0 if fallback else attempted_penetration),
        root_displacement_m=float(torch.linalg.vector_norm(final_qpos[:3] - attempted[:3])),
        joint_displacement_rad=float((final_qpos[7:] - attempted[7:]).abs().max()),
        safety_fallback_used=fallback,
        attempted_maximum_penetration_m=attempted_penetration,
        history=history,
        observed=final_observed,
    )


__all__ = [
    "HeightmapContactProjector",
    "ObservedHeightmapSurfaces",
    "verify_observation_only_prediction",
    "observed_surface_indices",
]
