from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from somaforge_core.motion_contracts import CONTACT_PARTS, CONTACT_BODY_INDEX
from somaforge_core.prediction_contracts import TrajectoryPrediction, UnifiedInteractionPrediction
from somaforge_core.g1_kinematics import _matrix_from_rotation6d


@dataclass(frozen=True)
class InteractionProjectionResult:
    """A q-space constrained trajectory and its before/after audit."""

    output: TrajectoryPrediction
    diagnostics: dict[str, float]


def _endpoint_root_shift_penalty(qpos: Tensor, initial_q: Tensor, maximum_shift_m: float) -> Tensor:
    endpoint_shift = torch.linalg.vector_norm(qpos[:, -1, :3] - initial_q[:, -1, :3], dim=-1)
    return (torch.nn.functional.relu(endpoint_shift - maximum_shift_m) / 0.05).square().mean()


def _limit_endpoint_root_shift(q_tail: Tensor, initial_endpoint: Tensor, maximum_shift_m: float) -> Tensor:
    root = q_tail[..., :3]
    delta = root[:, -1] - initial_endpoint
    scale = (maximum_shift_m / torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-8)).clamp_max(1.0)
    limited_endpoint = initial_endpoint + scale[:, None] * delta
    limited_root = torch.cat((root[:, :-1], limited_endpoint[:, None]), dim=1)
    return torch.cat((limited_root, q_tail[..., 3:]), dim=-1)


def _collision_depth_cost(excess: Tensor, mode: str) -> Tensor:
    if mode == "quadratic":
        return excess.square()
    if mode != "exponential":
        raise ValueError(f"unknown collision cost mode: {mode}")
    # Float32 exponential up to 10 cm excess; a C1 linear continuation beyond
    # that keeps extreme bootstrap states finite without zeroing their gradient.
    excess = excess.float()
    capped = excess.clamp_max(10.0)
    return torch.expm1(capped) + torch.exp(capped) * (excess - capped)


def rollout_dense_collision_loss(
    penetration: Tensor, tolerance_m: float = 0.002, *, mode: str = "quadratic"
) -> Tensor:
    """Penalize collision depth, duration, and the worst dense-trajectory frame."""

    excess = torch.nn.functional.relu(penetration - tolerance_m) / 0.01
    frame_max = excess.amax(dim=-1)
    soft_occupied = 1.0 - torch.exp(-frame_max)
    cost = _collision_depth_cost(excess, mode)
    frame_cost = cost.amax(dim=-1)
    return (
        cost.mean()
        + 0.25 * frame_cost.mean()
        + 0.25 * frame_cost.max()
        + 0.25 * soft_occupied.mean()
    )


def rollout_ground_collision_loss(
    penetration: Tensor, tolerance_m: float = 0.002, *, mode: str = "quadratic"
) -> Tensor:
    """Penalize ground penetration without diluting it over geometry samples."""

    excess = torch.nn.functional.relu(penetration - tolerance_m) / 0.01
    frame_max = excess.amax(dim=-1)
    soft_occupied = 1.0 - torch.exp(-frame_max)
    frame_cost = _collision_depth_cost(frame_max, mode)
    return frame_cost.mean() + frame_cost.max() + 0.25 * soft_occupied.mean()


def contact_rotation_losses(
    predicted_rotation6d: Tensor,
    target_rotation6d: Tensor,
    contact_mask: Tensor,
    *,
    orientation_scale: float = 0.2,
    angular_step_scale: float = 0.05,
    balance_contact_parts: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return contact-link orientation and angular-step losses from FK rotations."""

    if predicted_rotation6d.shape != target_rotation6d.shape:
        raise ValueError("predicted and target contact rotations must have the same shape")
    if predicted_rotation6d.shape[-2:] != (len(CONTACT_BODY_INDEX), 6):
        raise ValueError("contact rotations must end in [contact_part, rotation6d]")
    expected_mask = predicted_rotation6d.shape[:-1]
    if contact_mask.shape != expected_mask:
        raise ValueError(f"contact mask must have shape {expected_mask}, got {tuple(contact_mask.shape)}")
    predicted = _matrix_from_rotation6d(predicted_rotation6d)
    target = _matrix_from_rotation6d(target_rotation6d)
    mask = contact_mask.to(predicted.dtype)
    chordal = ((predicted - target) / orientation_scale).square().mean(dim=(-2, -1))

    def masked_mean(value: Tensor, value_mask: Tensor) -> Tensor:
        if not balance_contact_parts:
            return (value * value_mask).sum() / value_mask.sum().clamp_min(1.0)
        reduce_dims = tuple(range(value_mask.ndim - 1))
        part_count = value_mask.sum(dim=reduce_dims)
        part_mean = (value * value_mask).sum(dim=reduce_dims) / part_count.clamp_min(1.0)
        valid_part = part_count > 0.0
        return (part_mean * valid_part).sum() / valid_part.sum().clamp_min(1)

    orientation = masked_mean(chordal, mask)
    if predicted.ndim < 5 or predicted.shape[1] < 2:
        return orientation, predicted.sum() * 0.0
    predicted_step = predicted[:, :-1].transpose(-1, -2) @ predicted[:, 1:]
    target_step = target[:, :-1].transpose(-1, -2) @ target[:, 1:]
    pair_mask = mask[:, :-1] * mask[:, 1:]
    angular_step = ((predicted_step - target_step) / angular_step_scale).square().mean(dim=(-2, -1))
    angular_velocity = masked_mean(angular_step, pair_mask)
    return orientation, angular_velocity


def _trajectory_contact_points(
    output: TrajectoryPrediction,
    start_local_offset: Tensor,
    end_local_offset: Tensor,
    phase: Tensor,
) -> Tensor:
    rotation = _matrix_from_rotation6d(output.keypoint_rotation6d)
    u = phase[..., None, None]
    local_offset = (1.0 - u) * start_local_offset[:, None] + u * end_local_offset[:, None]
    points = []
    for part, body in enumerate(CONTACT_BODY_INDEX):
        rotated = torch.einsum("btij,btj->bti", rotation[:, :, body], local_offset[:, :, part])
        points.append(output.keypoint_position[:, :, body] - rotated)
    return torch.stack(points, dim=2)


def project_interaction_q_trajectory(
    output: TrajectoryPrediction,
    *,
    fk: nn.Module,
    collision_geometry: nn.Module,
    phase: Tensor,
    start_contact: Tensor,
    start_contact_position: Tensor,
    end: UnifiedInteractionPrediction,
    box_center: Tensor,
    box_rotation: Tensor,
    box_half_extents: Tensor,
    ground_height: Tensor,
    steps: int = 120,
    learning_rate: float = 0.01,
    maximum_endpoint_root_shift_m: float = 0.15,
    endpoint_root_shift_weight: float = 5.0,
    collect_diagnostics: bool = True,
) -> InteractionProjectionResult:
    """Project a generated interaction in q-space onto support and collision constraints.

    The current boundary (frame zero) is immutable.  The predicted next q is
    allowed to move because otherwise an inaccurate persistent endpoint makes
    the constraints inconsistent.  Touchdown points remain fixed targets and
    persistent points remain fixed at their current world anchors.
    """

    from contact_solver.collision_geometry import full_geometry_box_penetration

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if maximum_endpoint_root_shift_m < 0.0:
        raise ValueError("maximum endpoint root shift must be non-negative")
    if endpoint_root_shift_weight < 0.0:
        raise ValueError("endpoint root shift weight must be non-negative")
    initial_q = output.auxiliary_qpos.detach()
    batch, frames, _ = initial_q.shape
    if phase.shape != (batch, frames):
        raise ValueError("phase must match the q trajectory batch and frame dimensions")

    start_position, start_rotation6d = fk(initial_q[:, 0])
    start_rotation = _matrix_from_rotation6d(start_rotation6d)
    start_offsets = []
    for part, body in enumerate(CONTACT_BODY_INDEX):
        start_offsets.append(
            torch.einsum(
                "bij,bj->bi",
                start_rotation[:, body].transpose(-1, -2),
                start_position[:, body] - start_contact_position[:, part],
            )
        )
    start_offsets = torch.stack(start_offsets, dim=1).detach()

    start_active = start_contact > 0.5
    persistent = end.persistent_support & start_active & end.active_contact
    touchdown = end.active_contact & ~persistent
    liftoff = start_active & ~persistent
    phase3 = phase[..., None]
    anchor_mask = persistent[:, None].expand(-1, frames, -1).clone()
    anchor_mask |= liftoff[:, None] & (phase3 <= 0.2)
    anchor_mask |= touchdown[:, None] & (phase3 >= 0.8)
    end_point = end.contact_position.detach()
    box_local = torch.einsum(
        "bij,bpj->bpi", box_rotation.transpose(-1, -2), end_point - box_center[:, None]
    )
    box_local_xy = torch.maximum(
        torch.minimum(box_local[..., :2], box_half_extents[:, None, :2]),
        -box_half_extents[:, None, :2],
    )
    box_local_top = torch.cat(
        (box_local_xy, box_half_extents[:, None, 2:3].expand(-1, len(CONTACT_PARTS), -1)), dim=-1
    )
    box_surface_point = box_center[:, None] + torch.einsum("bij,bpj->bpi", box_rotation, box_local_top)
    ground_surface_point = end_point.clone()
    ground_surface_point[..., 2] = ground_height[:, None]
    end_surface_point = torch.where(
        (end.contact_surface == 1)[..., None], box_surface_point, ground_surface_point
    ).detach()
    touchdown_window = touchdown[:, None] & (phase3 >= 0.8)
    anchor_target = torch.where(
        touchdown_window[..., None],
        end_surface_point[:, None],
        start_contact_position[:, None],
    ).detach()
    contact_schedule = anchor_mask.to(initial_q.dtype)

    def evaluate(qpos: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        position, rotation6d = fk(qpos)
        candidate = TrajectoryPrediction(position, rotation6d, output.contact_logits, qpos)
        points = _trajectory_contact_points(candidate, start_offsets, end.contact_local_offset.detach(), phase)
        anchor_error = torch.linalg.vector_norm(points - anchor_target, dim=-1)
        collision_points, point_part = collision_geometry(fk, qpos)
        penetration = full_geometry_box_penetration(
            collision_points,
            point_part,
            contact_schedule,
            box_center=box_center,
            box_rotation=box_rotation,
            box_half_extents=box_half_extents,
            ground_height=ground_height,
        )
        return position, rotation6d, anchor_error, penetration

    def audit(qpos: Tensor, prefix: str) -> dict[str, float]:
        with torch.no_grad():
            _, _, anchor_error, penetration = evaluate(qpos)
            selected = anchor_error[anchor_mask]
            return {
                f"{prefix}_anchor_mean_cm": float(selected.mean().item() * 100.0) if selected.numel() else 0.0,
                f"{prefix}_anchor_max_cm": float(selected.max().item() * 100.0) if selected.numel() else 0.0,
                f"{prefix}_penetration_max_cm": float(penetration.max().item() * 100.0),
                f"{prefix}_collision_free_fraction": float((penetration.amax(dim=-1) <= 0.002).float().mean().item()),
            }

    diagnostics = audit(initial_q, "before") if collect_diagnostics else {}
    if collect_diagnostics:
        touchdown_snap = torch.linalg.vector_norm(end_surface_point - end_point, dim=-1)[touchdown]
        diagnostics["touchdown_surface_snap_mean_cm"] = (
            float(touchdown_snap.mean().item() * 100.0) if touchdown_snap.numel() else 0.0
        )
        diagnostics["touchdown_surface_snap_max_cm"] = (
            float(touchdown_snap.max().item() * 100.0) if touchdown_snap.numel() else 0.0
        )
    if steps == 0:
        if collect_diagnostics:
            diagnostics.update(audit(initial_q, "after"))
        return InteractionProjectionResult(output, diagnostics)

    variable = initial_q[:, 1:].clone().requires_grad_(True)
    optimizer = torch.optim.Adam((variable,), lr=learning_rate)
    initial_velocity = initial_q[:, 1:] - initial_q[:, :-1]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        q_tail = _limit_endpoint_root_shift(variable, initial_q[:, -1, :3], maximum_endpoint_root_shift_m)
        quaternion = F.normalize(q_tail[..., 3:7], dim=-1)
        joints = torch.maximum(torch.minimum(q_tail[..., 7:], fk.joint_upper), fk.joint_lower)
        qpos = torch.cat((initial_q[:, :1], torch.cat((q_tail[..., :3], quaternion, joints), dim=-1)), dim=1)
        _, _, anchor_error, penetration = evaluate(qpos)
        selected = anchor_error[anchor_mask]
        anchor_loss = qpos.sum() * 0.0
        if selected.numel():
            normalized = selected / 0.005
            anchor_loss = normalized.square().mean() + 0.25 * normalized.square().max()
        frame_penetration = penetration.amax(dim=-1) / 0.005
        collision_loss = frame_penetration.square().mean() + 0.25 * frame_penetration.square().max()
        root_deviation = ((qpos[..., :3] - initial_q[..., :3]) / 0.05).square().mean()
        quaternion_deviation = 1.0 - (qpos[..., 3:7] * initial_q[..., 3:7]).sum(dim=-1).abs()
        joint_deviation = ((qpos[..., 7:] - initial_q[..., 7:]) / 0.2).square().mean()
        velocity = qpos[:, 1:] - qpos[:, :-1]
        velocity_deviation = ((velocity - initial_velocity) / 0.1).square().mean()
        excessive_endpoint_shift = _endpoint_root_shift_penalty(
            qpos, initial_q, maximum_endpoint_root_shift_m
        )
        loss = 10.0 * anchor_loss + 10.0 * collision_loss
        loss = loss + 0.05 * root_deviation + 0.05 * quaternion_deviation.mean()
        loss = loss + 0.02 * joint_deviation + 0.002 * velocity_deviation
        loss = loss + endpoint_root_shift_weight * excessive_endpoint_shift
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        q_tail = _limit_endpoint_root_shift(variable, initial_q[:, -1, :3], maximum_endpoint_root_shift_m)
        final_q = torch.cat(
            (
                initial_q[:, :1],
                torch.cat(
                    (
                        q_tail[..., :3],
                        F.normalize(q_tail[..., 3:7], dim=-1),
                        torch.maximum(torch.minimum(q_tail[..., 7:], fk.joint_upper), fk.joint_lower),
                    ),
                    dim=-1,
                ),
            ),
            dim=1,
        )
        position, rotation6d = fk(final_q)
        projected = TrajectoryPrediction(position, rotation6d, output.contact_logits, final_q)
        if collect_diagnostics:
            diagnostics.update(audit(final_q, "after"))
            diagnostics["endpoint_root_shift_cm"] = float(
                torch.linalg.vector_norm(final_q[:, -1, :3] - initial_q[:, -1, :3], dim=-1).mean().item() * 100.0
            )
            diagnostics["q_rms_joint_shift_deg"] = float(
                torch.rad2deg(torch.sqrt(torch.mean((final_q[..., 7:] - initial_q[..., 7:]).square()))).item()
            )
    return InteractionProjectionResult(projected, diagnostics)
