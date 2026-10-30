"""Sequential Newton-witness projection for next-contact keyframes.

The predictor supplies a nominal pose and a contact intention.  This module
does not reinterpret contact from a geometric threshold: every iteration
refreshes Newton/MJWarp witnesses, and the final success decision uses the
shared solver-activation and primary-face selection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from somaforge_core import CONTACT_BODY_NAMES_BY_PART
from somaforge_core.contact_face_selection import select_contact_pairs, upward_face_mask
from somaforge_core.robot_assets import canonical_g1_urdf_path

from somaforge_core.motion_contracts import BODY_NAMES, CONTACT_PARTS
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _quaternion_multiply_wxyz, _rpy_matrix
from contact_solver.collision_geometry import _cylinder_points, _mesh_vertices
from contact_solver.newton_witness_loss import full_body_violation, query_local_distances


NewtonQuery = Callable[[np.ndarray, list[np.ndarray] | None], dict]


@dataclass(frozen=True)
class ContactProjectionResult:
    qpos: Tensor
    converged: bool
    iterations: int
    intended_contact: Tensor
    intended_surface: Tensor
    actual_contact: Tensor
    actual_surface: Tensor
    unrealized_contacts: int
    extra_contacts: int
    maximum_penetration_m: float
    root_displacement_m: float
    joint_displacement_rad: float
    safety_fallback_used: bool
    attempted_maximum_penetration_m: float
    history: tuple[dict[str, float | int | bool], ...]
    observed: dict


@dataclass
class _ContactCollisionShape:
    part: int
    link_name: str
    kind: str
    local_center: Tensor
    radius: float | None = None
    local_points: Tensor | None = None


class CanonicalContactCollisionGeometry:
    """Exact authoritative contact shapes used only to establish new pairs.

    Newton remains the source of contact truth.  These shapes provide an
    approach direction while a requested part/surface pair is outside the
    narrow-phase candidate set.  Spheres are evaluated analytically; convex
    meshes retain every source vertex instead of the lossy 64-point cloud used
    by training-time collision regularizers.
    """

    def __init__(self) -> None:
        urdf = canonical_g1_urdf_path()
        root = ET.parse(urdf).getroot()  # noqa: S314
        contact_links = (
            CONTACT_BODY_NAMES_BY_PART["left_foot"],
            CONTACT_BODY_NAMES_BY_PART["right_foot"],
            CONTACT_BODY_NAMES_BY_PART["left_hand"],
            CONTACT_BODY_NAMES_BY_PART["right_hand"],
            CONTACT_BODY_NAMES_BY_PART["left_knee"],
            CONTACT_BODY_NAMES_BY_PART["right_knee"],
        )
        part_by_link = {
            link_name: part
            for part, link_names in enumerate(contact_links)
            for link_name in link_names
        }
        shapes: list[_ContactCollisionShape] = []
        for link in root.findall("link"):
            link_name = link.get("name", "")
            if link_name not in part_by_link:
                continue
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None or len(geometry) != 1:
                    continue
                origin = collision.find("origin")
                xyz_text = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
                rpy_text = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
                center = torch.tensor(tuple(float(value) for value in xyz_text.split()))
                rotation = _rpy_matrix(tuple(float(value) for value in rpy_text.split()))
                shape = geometry[0]
                if shape.tag == "sphere":
                    shapes.append(_ContactCollisionShape(
                        part_by_link[link_name], link_name, "sphere", center,
                        radius=float(shape.get("radius", "nan")),
                    ))
                    continue
                if shape.tag == "mesh":
                    scale = tuple(float(value) for value in shape.get("scale", "1 1 1").split())
                    points = _mesh_vertices(
                        urdf.parent / shape.get("filename", ""), scale, 2**31 - 1
                    )
                elif shape.tag == "cylinder":
                    points = _cylinder_points(
                        float(shape.get("radius", "nan")),
                        float(shape.get("length", "nan")),
                        count=128,
                    )
                else:
                    raise ValueError(f"unsupported canonical G1 collision geometry: {shape.tag}")
                points = torch.einsum("ij,pj->pi", rotation, points) + center
                shapes.append(_ContactCollisionShape(
                    part_by_link[link_name], link_name, shape.tag, center,
                    local_points=points,
                ))
        if not shapes:
            raise ValueError("canonical G1 URDF has no contact collision geometry")
        self.shapes = shapes

    def to(self, device: torch.device | str) -> "CanonicalContactCollisionGeometry":
        for shape in self.shapes:
            shape.local_center = shape.local_center.to(device)
            if shape.local_points is not None:
                shape.local_points = shape.local_points.to(device)
        return self

    @staticmethod
    def _finite_face_residual(
        point: Tensor,
        normal: Tensor,
        offset: Tensor,
        region: Tensor | None,
        inset: Tensor,
    ) -> Tensor:
        residuals = [((point * normal).sum(-1) - offset)[:, None]]
        if region is not None:
            if region.ndim == 1:
                center, tangent_u, tangent_v = region[:3], region[3:6], region[6:9]
                half = (region[9:11] - inset).clamp_min(0.0)
            elif region.ndim == 2 and region.shape == (len(point), 11):
                center, tangent_u, tangent_v = region[:, :3], region[:, 3:6], region[:, 6:9]
                half = (region[:, 9:11] - inset.reshape(-1, 1)).clamp_min(0.0)
            else:
                raise ValueError("finite face region must be [11] or [B,11]")
            relative = point - center
            coordinate_u = (relative * tangent_u).sum(-1)
            coordinate_v = (relative * tangent_v).sum(-1)
            half_u = half[0] if half.ndim == 1 else half[:, 0]
            half_v = half[1] if half.ndim == 1 else half[:, 1]
            residuals.extend((
                (coordinate_u.abs() - half_u).relu()[:, None],
                (coordinate_v.abs() - half_v).relu()[:, None],
            ))
        return torch.cat(residuals, dim=-1)

    def approach_residual(
        self,
        fk: CanonicalG1ForwardKinematics,
        qpos: Tensor,
        part: int,
        normal: Tensor,
        offset: Tensor,
        region: Tensor | None,
        anchor: Tensor,
        inset: Tensor,
        *,
        patch: bool,
        link_pose_override: dict[str, tuple[Tensor, Tensor]] | None = None,
    ) -> Tensor:
        candidates: list[Tensor] = []
        merits: list[Tensor] = []
        shapes = [shape for shape in self.shapes if shape.part == part]
        if not shapes:
            raise ValueError(f"canonical collision asset has no shapes for contact part {part}")
        if link_pose_override is None:
            names = tuple(dict.fromkeys(shape.link_name for shape in shapes))
            positions, rotations = fk.link_poses(qpos, names)
            link_pose_override = {name: (positions[:, i], rotations[:, i]) for i, name in enumerate(names)}
        for shape in shapes:
            position, rotation = link_pose_override[shape.link_name]
            if shape.kind == "sphere":
                center = position + torch.einsum("bij,j->bi", rotation, shape.local_center.to(qpos))
                radius = qpos.new_tensor(float(shape.radius))
                if patch:
                    direction = anchor - center
                    witness = center + radius * direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
                    residual = witness - anchor
                else:
                    witness = center - radius * normal
                    residual = self._finite_face_residual(witness, normal, offset, region, inset)
            else:
                points = torch.einsum(
                    "bij,pj->bpi", rotation, shape.local_points.to(qpos)
                ) + position[:, None]
                if patch:
                    cost = (points.detach() - anchor[:, None]).square().sum(-1)
                else:
                    expanded_normal = normal if normal.ndim == 1 else normal[:, None]
                    signed = (points.detach() * expanded_normal).sum(-1) - offset.reshape(-1, 1)
                    cost = signed.square()
                    if region is not None:
                        if region.ndim == 1:
                            center, tangent_u, tangent_v = region[:3], region[3:6], region[6:9]
                            half = (region[9:11] - inset).clamp_min(0.0)
                        else:
                            center = region[:, None, :3]
                            tangent_u = region[:, None, 3:6]
                            tangent_v = region[:, None, 6:9]
                            half = (region[:, 9:11] - inset.reshape(-1, 1)).clamp_min(0.0)
                        relative = points.detach() - center
                        u = (relative * tangent_u).sum(-1)
                        v = (relative * tangent_v).sum(-1)
                        half_u = half[0] if half.ndim == 1 else half[:, 0:1]
                        half_v = half[1] if half.ndim == 1 else half[:, 1:2]
                        cost = cost + (u.abs() - half_u).relu().square()
                        cost = cost + (v.abs() - half_v).relu().square()
                nearest = cost.argmin(-1)
                witness = points[torch.arange(len(points), device=points.device), nearest]
                residual = (
                    witness - anchor
                    if patch else self._finite_face_residual(witness, normal, offset, region, inset)
                )
            candidates.append(residual)
            merits.append(residual.detach().square().sum(-1))
        if not candidates:
            raise ValueError(f"canonical collision asset has no shapes for contact part {part}")
        choice = torch.stack(merits, dim=-1).argmin(-1)
        stacked = torch.stack(candidates, dim=1)
        return stacked[torch.arange(len(qpos), device=qpos.device), choice]


def apply_tangent_delta(
    qpos: Tensor,
    delta: Tensor,
    fk: CanonicalG1ForwardKinematics,
) -> Tensor:
    """Apply a 35D floating-base tangent increment and enforce URDF limits."""

    if qpos.ndim != 2 or qpos.shape[-1] != 36:
        raise ValueError("qpos must have shape [B,36]")
    if delta.shape != (len(qpos), 35):
        raise ValueError("delta must have shape [B,35]")
    angle = torch.linalg.vector_norm(delta[:, 3:6], dim=-1, keepdim=True)
    vector_scale = 0.5 * torch.sinc(angle / (2.0 * torch.pi))
    rotation = torch.cat((torch.cos(0.5 * angle), vector_scale * delta[:, 3:6]), dim=-1)
    quaternion = F.normalize(_quaternion_multiply_wxyz(rotation, qpos[:, 3:7]), dim=-1)
    joints = torch.maximum(
        torch.minimum(qpos[:, 7:] + delta[:, 6:], fk.joint_upper),
        fk.joint_lower,
    )
    return torch.cat((qpos[:, :3] + delta[:, :3], quaternion, joints), dim=-1)


def _selected_actual(observed: dict) -> tuple[np.ndarray, np.ndarray]:
    if "pairs" not in observed or "surface_catalog" not in observed:
        raise ValueError("Newton result lacks pairs/surface_catalog; no geometric fallback is allowed")
    selected = select_contact_pairs(observed["pairs"], observed["surface_catalog"])
    return (
        np.asarray(selected["contact_part_mask"][0], dtype=bool),
        np.asarray(selected["contact_surface"][0], dtype=np.int64),
    )


def _maximum_penetration(observed: dict) -> float:
    rows = observed.get("full_robot_separation")
    if rows is None or len(rows) != 1:
        raise ValueError("Newton result lacks one-sample full_robot_separation")
    depth = 0.0
    for key in ("worst_terrain", "worst_self"):
        witness = rows[0].get(key)
        if witness is not None:
            depth = max(depth, -float(witness["dist"]))
    return max(depth, 0.0)


class NewtonContactProjector:
    """Project one nominal G1 keyframe onto a requested Newton contact topology.

    Newton supplies the active-set distances, actual ``includemargin`` values,
    material witnesses, and final truth.  Differentiable canonical-G1 geometry
    is used only as an approach residual when the target pair is too far away
    for Newton to emit a candidate.
    """

    def __init__(
        self,
        query: NewtonQuery,
        *,
        iterations: int = 8,
        damping: float = 2.0e-3,
        maximum_step: float = 0.20,
        residual_scale_m: float = 0.02,
        activation_clearance_fraction: float = 0.05,
        fullbody_clearance_fraction: float = 0.01,
        line_search_steps: int = 4,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if damping <= 0.0 or maximum_step <= 0.0 or residual_scale_m <= 0.0:
            raise ValueError("damping, maximum_step, and residual_scale_m must be positive")
        if not 0.0 < activation_clearance_fraction < 1.0:
            raise ValueError("activation_clearance_fraction must be in (0,1)")
        if not 0.0 < fullbody_clearance_fraction < activation_clearance_fraction:
            raise ValueError("fullbody_clearance_fraction must lie in (0, activation_clearance_fraction)")
        if line_search_steps < 1:
            raise ValueError("line_search_steps must be positive")
        self.query = query
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.maximum_step = float(maximum_step)
        self.residual_scale_m = float(residual_scale_m)
        self.activation_clearance_fraction = float(activation_clearance_fraction)
        self.fullbody_clearance_fraction = float(fullbody_clearance_fraction)
        self.line_search_steps = int(line_search_steps)
        self.fk = CanonicalG1ForwardKinematics()
        self.approach_geometry = CanonicalContactCollisionGeometry()

    def to(self, device: torch.device | str) -> "NewtonContactProjector":
        self.fk.to(device)
        self.approach_geometry.to(device)
        return self

    def _query_rows(self, qpos: Tensor) -> tuple[list, dict]:
        return query_local_distances(self.fk, qpos, query=self.query)

    def _approach_residual(
        self,
        qpos: Tensor,
        part: int,
        surface: int,
        anchor: Tensor,
        catalog: list[dict],
        surface_regions: dict[int, Tensor] | None = None,
        surface_inset: Tensor | None = None,
        *,
        patch: bool,
    ) -> Tensor:
        face = next((face for face in catalog if int(face["surface"]) == surface), None)
        if face is None:
            raise ValueError(f"Newton surface catalog lacks intended surface {surface}")
        normal = qpos.new_tensor(face["normal_w"])
        offset = qpos.new_tensor(float(face["plane_offset"]))
        region = None if surface_regions is None else surface_regions.get(surface)
        if region is not None:
            region = region.to(qpos)
        if surface_inset is None:
            surface_inset = qpos.new_zeros(())
        return self.approach_geometry.approach_residual(
            self.fk,
            qpos,
            part,
            normal,
            offset,
            region,
            anchor,
            surface_inset,
            patch=patch,
        ) / self.residual_scale_m

    def _residual(
        self,
        qpos: Tensor,
        rows: list,
        observed: dict,
        intended_contact: Tensor,
        intended_surface: Tensor,
        anchor: Tensor,
        surface_regions: dict[int, Tensor] | None = None,
        sticking_contact: Tensor | None = None,
        sticking_local_point: Tensor | None = None,
        *,
        patch: bool,
    ) -> tuple[Tensor, int]:
        if len(qpos) != 1 or len(rows) != 1:
            raise ValueError("NewtonContactProjector intentionally projects one keyframe at a time")
        catalog = observed["surface_catalog"]
        upward = upward_face_mask([face["normal_w"] for face in catalog])
        eligible = {
            int(face["surface"])
            for face, keep in zip(catalog, upward, strict=True)
            if bool(keep)
        }
        residuals: list[Tensor] = []
        missing = 0
        margins = [
            float(pair["includemargin"])
            for pair, _ in rows[0]
            if float(pair["includemargin"]) > 0.0
        ]
        if not margins:
            margins = [
                float(value)
                for value in observed.get("configured_terrain_includemargins", ())
                if float(value) > 0.0
            ]
        if not margins:
            raise ValueError("Newton emitted no positive includemargin for projection constraints")
        surface_inset = qpos.new_tensor(max(margins))
        for part in range(len(CONTACT_PARTS)):
            if bool(intended_contact[part]):
                surface = int(intended_surface[part])
                matches = [
                    (pair, distance)
                    for pair, distance in rows[0]
                    if int(pair["part"]) == part and int(pair["surface"]) == surface
                ]
                if matches:
                    # One manifold witness realizes a part/face contact.  The
                    # closest physical surface separation is driven to zero;
                    # other manifold points are not pinned independently.
                    chosen = min(matches, key=lambda item: abs(float(item[0]["dist"])))
                    # Stay strictly on the non-penetrating side while still
                    # lying well inside this pair's real Newton activation
                    # interval.  This is an optimization interior point, not
                    # a replacement contact threshold.
                    target_gap = (
                        self.activation_clearance_fraction
                        * float(chosen[0]["includemargin"])
                    )
                    residuals.append(
                        (chosen[1] - qpos.new_tensor(target_gap)).reshape(1)
                        / self.residual_scale_m
                    )
                else:
                    missing += 1
                    residuals.append(
                        self._approach_residual(
                            qpos,
                            part,
                            surface,
                            anchor[part : part + 1],
                            catalog,
                            surface_regions,
                            surface_inset,
                            patch=patch,
                        ).reshape(-1)
                    )
            else:
                # Inactive means no actual Newton constraint on any eligible
                # primary surface.  The threshold is each pair's own solver
                # includemargin, never a project-local distance constant.
                for pair, distance in rows[0]:
                    if int(pair["part"]) != part or int(pair["surface"]) not in eligible:
                        continue
                    margin = float(pair["includemargin"])
                    if float(pair["dist"]) < margin:
                        residuals.append((qpos.new_tensor(margin) - distance).reshape(1) / self.residual_scale_m)

        if sticking_contact is not None:
            if sticking_local_point is None:
                raise ValueError("sticking_contact requires sticking_local_point")
            link_position, link_rotation = self.fk.link_poses(qpos, BODY_NAMES[1:7])
            material_point = link_position + torch.einsum(
                "bpij,pj->bpi", link_rotation, sticking_local_point
            )
            for part in sticking_contact.nonzero(as_tuple=False).flatten().tolist():
                residuals.append(
                    (material_point[0, part] - anchor[part]) / self.residual_scale_m
                )

        signed_violation, invalid = full_body_violation(
            self.fk,
            qpos,
            observed["full_robot_separation"],
            signed=True,
            invalid_policy="detach",
            return_validity=True,
        )
        if int(invalid[0]) != 0:
            raise ValueError("Newton returned an invalid full-body witness during projection")
        clearance = self.fullbody_clearance_fraction * min(margins)
        clearance_violation = (signed_violation + qpos.new_tensor(clearance)).relu()
        if float(clearance_violation.detach()[0]) > 0.0:
            residuals.append(clearance_violation / self.residual_scale_m)
        if not residuals:
            return qpos.new_zeros((0,)), missing
        return torch.cat(residuals), missing

    def project(
        self,
        nominal_qpos: Tensor,
        intended_contact: Tensor,
        intended_surface: Tensor,
        anchor_w: Tensor,
        *,
        patch: bool,
        surface_regions: dict[int, Tensor] | None = None,
        safe_qpos: Tensor | None = None,
        sticking_contact: Tensor | None = None,
        sticking_local_point: Tensor | None = None,
        sticking_tolerance_m: float = 2.0e-3,
    ) -> ContactProjectionResult:
        if nominal_qpos.shape != (36,):
            raise ValueError("nominal_qpos must have shape [36]")
        if intended_contact.shape != (len(CONTACT_PARTS),):
            raise ValueError("intended_contact must have shape [6]")
        if intended_surface.shape != (len(CONTACT_PARTS),):
            raise ValueError("intended_surface must have shape [6]")
        if anchor_w.shape != (len(CONTACT_PARTS), 3):
            raise ValueError("anchor_w must have shape [6,3]")
        intended_contact = intended_contact.to(device=nominal_qpos.device, dtype=torch.bool)
        intended_surface = intended_surface.to(device=nominal_qpos.device, dtype=torch.long)
        anchor_w = anchor_w.to(nominal_qpos)
        if sticking_tolerance_m <= 0.0:
            raise ValueError("sticking_tolerance_m must be positive")
        if sticking_contact is None:
            sticking_contact = torch.zeros_like(intended_contact)
        else:
            if sticking_contact.shape != (len(CONTACT_PARTS),):
                raise ValueError("sticking_contact must have shape [6]")
            sticking_contact = sticking_contact.to(
                device=nominal_qpos.device, dtype=torch.bool
            )
        if bool((sticking_contact & ~intended_contact).any()):
            raise ValueError("sticking contacts must also be intended contacts")
        if sticking_local_point is None:
            if bool(sticking_contact.any()):
                raise ValueError("sticking contacts require local material points")
            sticking_local_point = nominal_qpos.new_zeros((len(CONTACT_PARTS), 3))
        elif sticking_local_point.shape != (len(CONTACT_PARTS), 3):
            raise ValueError("sticking_local_point must have shape [6,3]")
        else:
            sticking_local_point = sticking_local_point.to(nominal_qpos)
        if bool((intended_contact & (intended_surface < 0)).any()):
            raise ValueError("active intended contacts require a Newton surface id")

        def maximum_sticking_error(qpos: Tensor) -> float:
            if not bool(sticking_contact.any()):
                return 0.0
            position, rotation = self.fk.link_poses(qpos, BODY_NAMES[1:7])
            point = position + torch.einsum(
                "bpij,pj->bpi", rotation, sticking_local_point
            )
            error = torch.linalg.vector_norm(point[0] - anchor_w, dim=-1)
            return float(error[sticking_contact].max().detach())

        nominal = nominal_qpos.detach().clone()
        current = nominal[None]
        history: list[dict[str, float | int | bool]] = []
        converged = False
        final_observed: dict | None = None
        actual_mask = np.zeros(len(CONTACT_PARTS), dtype=bool)
        actual_surface = np.full(len(CONTACT_PARTS), -1, dtype=np.int64)
        for iteration in range(1, self.iterations + 1):
            delta = current.new_zeros((1, 35), requires_grad=True)
            candidate = apply_tangent_delta(current, delta, self.fk)
            rows, observed = self._query_rows(candidate)
            residual, missing = self._residual(
                candidate,
                rows,
                observed,
                intended_contact,
                intended_surface,
                anchor_w,
                surface_regions,
                sticking_contact,
                sticking_local_point,
                patch=patch,
            )
            actual_mask, actual_surface = _selected_actual(observed)
            unrealized = int(np.count_nonzero(intended_contact.cpu().numpy() & ~actual_mask))
            extra = int(np.count_nonzero(~intended_contact.cpu().numpy() & actual_mask))
            surface_match = bool(
                np.array_equal(
                    actual_surface[intended_contact.cpu().numpy() & actual_mask],
                    intended_surface.detach().cpu().numpy()[intended_contact.cpu().numpy() & actual_mask],
                )
            )
            penetration = _maximum_penetration(observed)
            sticking_error = maximum_sticking_error(candidate)
            converged = (
                unrealized == 0
                and extra == 0
                and surface_match
                and penetration == 0.0
                and sticking_error <= sticking_tolerance_m
            )
            history.append(
                {
                    "iteration": iteration,
                    "residual_norm": float(torch.linalg.vector_norm(residual).detach()),
                    "missing_target_pairs": missing,
                    "unrealized_contacts": unrealized,
                    "extra_contacts": extra,
                    "maximum_penetration_m": penetration,
                    "topology_exact": unrealized == 0 and extra == 0 and surface_match,
                    "maximum_sticking_error_m": sticking_error,
                }
            )
            final_observed = observed
            if converged:
                current = candidate.detach()
                break
            if residual.numel() == 0:
                current = candidate.detach()
                break
            eye = torch.eye(len(residual), dtype=residual.dtype, device=residual.device)
            jacobian = torch.autograd.grad(
                residual,
                delta,
                grad_outputs=eye,
                is_grads_batched=True,
            )[0][:, 0]
            # Solve in the fixed 35D tangent space.  Contact manifolds often
            # emit redundant rows, so the residual-space J J^T system can be
            # singular even though the damped tangent-space normal equation
            # is well posed.  Float64 keeps small damping effective beside
            # large rotational Jacobian entries.
            jacobian64 = jacobian.to(torch.float64)
            residual64 = residual.detach().to(torch.float64)
            system = jacobian64.T @ jacobian64
            system = system + self.damping**2 * torch.eye(
                35, dtype=torch.float64, device=residual.device
            )
            step = -torch.linalg.solve(system, jacobian64.T @ residual64).to(residual.dtype)
            norm = torch.linalg.vector_norm(step)
            step = step * (self.maximum_step / norm.clamp_min(self.maximum_step)).clamp_max(1.0)
            current_merit = float(torch.linalg.vector_norm(residual).detach())
            accepted = False
            for line_search in range(self.line_search_steps):
                scale = 0.5**line_search
                trial = apply_tangent_delta(current, (scale * step)[None], self.fk).detach()
                with torch.no_grad():
                    trial_rows, trial_observed = self._query_rows(trial)
                    trial_residual, _ = self._residual(
                        trial,
                        trial_rows,
                        trial_observed,
                        intended_contact,
                        intended_surface,
                        anchor_w,
                        surface_regions,
                        sticking_contact,
                        sticking_local_point,
                        patch=patch,
                    )
                    trial_merit = float(torch.linalg.vector_norm(trial_residual))
                if trial_merit < current_merit:
                    current = trial
                    history[-1]["accepted_step_scale"] = scale
                    history[-1]["accepted_merit"] = trial_merit
                    accepted = True
                    break
            if not accepted:
                history[-1]["accepted_step_scale"] = 0.0
                history[-1]["accepted_merit"] = current_merit
                break

        # Always refresh after the last update; an iteration's linearized
        # witnesses may not describe the final pose after its accepted step.
        final_rows, final_observed = self._query_rows(current)
        del final_rows
        actual_mask, actual_surface = _selected_actual(final_observed)
        intended_np = intended_contact.detach().cpu().numpy()
        intended_surface_np = intended_surface.detach().cpu().numpy()
        unrealized = int(np.count_nonzero(intended_np & ~actual_mask))
        extra = int(np.count_nonzero(~intended_np & actual_mask))
        shared = intended_np & actual_mask
        surface_match = bool(np.array_equal(actual_surface[shared], intended_surface_np[shared]))
        penetration = _maximum_penetration(final_observed)
        attempted_penetration = penetration
        safety_fallback_used = False
        if penetration > 0.0:
            if safe_qpos is None:
                raise RuntimeError(
                    "contact projection produced penetration and no verified safe fallback was supplied"
                )
            if safe_qpos.shape != (36,):
                raise ValueError("safe_qpos must have shape [36]")
            fallback = safe_qpos.detach().to(device=current.device, dtype=current.dtype)[None]
            _, fallback_observed = self._query_rows(fallback)
            fallback_penetration = _maximum_penetration(fallback_observed)
            if fallback_penetration > 0.0:
                raise RuntimeError(
                    "supplied safety fallback is itself penetrating: "
                    f"{fallback_penetration:.9f} m"
                )
            current = fallback
            final_observed = fallback_observed
            actual_mask, actual_surface = _selected_actual(final_observed)
            intended_np = intended_contact.detach().cpu().numpy()
            intended_surface_np = intended_surface.detach().cpu().numpy()
            unrealized = int(np.count_nonzero(intended_np & ~actual_mask))
            extra = int(np.count_nonzero(~intended_np & actual_mask))
            shared = intended_np & actual_mask
            surface_match = bool(
                np.array_equal(actual_surface[shared], intended_surface_np[shared])
            )
            penetration = 0.0
            safety_fallback_used = True
        converged = (
            unrealized == 0
            and extra == 0
            and surface_match
            and penetration == 0.0
            and maximum_sticking_error(current) <= sticking_tolerance_m
        )
        if safety_fallback_used:
            converged = False
        return ContactProjectionResult(
            qpos=current[0],
            converged=converged,
            iterations=len(history),
            intended_contact=intended_contact,
            intended_surface=intended_surface,
            actual_contact=torch.as_tensor(actual_mask, device=current.device),
            actual_surface=torch.as_tensor(actual_surface, device=current.device),
            unrealized_contacts=unrealized,
            extra_contacts=extra,
            maximum_penetration_m=penetration,
            root_displacement_m=float(torch.linalg.vector_norm(current[0, :3] - nominal[:3])),
            joint_displacement_rad=float((current[0, 7:] - nominal[7:]).abs().max()),
            safety_fallback_used=safety_fallback_used,
            attempted_maximum_penetration_m=attempted_penetration,
            history=tuple(history),
            observed=final_observed,
        )
