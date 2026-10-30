"""No-binding whole-body endpoint predictor with an explicit contact plan.

The pose decoder consumes exactly the same ``(contact, surface)`` plan that a
downstream projector receives.  There is deliberately no topology head,
binding head, predicted-contact feature path, or implicit fallback plan in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from somaforge_core.heightmap import _root_yaw_basis
from contact_solver.surface_geometry import joint_feasibility_loss
from generator.next_interaction import structural_pose_loss
from generator.endpoint_objective import (POSE_POSITION_SCALE_M, POSE_ROTATION_MATRIX_SCALE,
    POSE_JOINT_SCALE_RAD, STRUCTURE_WEIGHT)
from generator.next_interaction_heightmap_v3 import DenseHeightmapEncoder, DenseHeightmapInteractionPredictor, TerrainCrossBlock
from contact_solver.collision_geometry import CanonicalG1CollisionPoints
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _matrix_from_rotation6d, _quaternion_matrix_wxyz, _rotation6d
from contact_solver.contact_constrained_projector import CanonicalContactCollisionGeometry


@dataclass(frozen=True)
class ConditionedPosePrediction:
    qpos: Tensor


class ConditionedHeightmapPosePredictor(nn.Module):
    """Predict a nominal endpoint from observation and an explicit plan."""

    def __init__(self, width: int = 192, layers: int = 3) -> None:
        super().__init__()
        self.fk = CanonicalG1ForwardKinematics()
        self.geometry = CanonicalG1CollisionPoints(64)
        self.height = DenseHeightmapEncoder(width)
        self.global_encoder = nn.Linear(38, width)
        self.part_encoder = nn.Linear(13, width)
        self.part_identity = nn.Embedding(6, width)
        self.surface_embedding = nn.Embedding(3, width)
        self.terrain_reasoning = nn.ModuleList((TerrainCrossBlock(width), TerrainCrossBlock(width)))
        state_layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.shared = nn.TransformerEncoder(
            state_layer, layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)

        # 0=ground, 1=top, 2=inactive.  This is the only future-contact input.
        self.plan_surface_embedding = nn.Embedding(3, width)
        self.plan_fusion = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.pose_terrain = TerrainCrossBlock(width)
        pose_layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.interaction_decoder = nn.TransformerEncoder(
            pose_layer, 2, enable_nested_tensor=False
        )
        self.pose_head = nn.Linear(width, 36)
        nn.init.normal_(self.pose_head.weight, std=0.001)
        nn.init.zeros_(self.pose_head.bias)
        with torch.no_grad():
            self.pose_head.bias[3] = 1.0

    @staticmethod
    def _validate_plan(planned_contact: Tensor, planned_surface: Tensor, batch: int) -> Tensor:
        if planned_contact.shape != (batch, 6) or planned_contact.dtype != torch.bool:
            raise ValueError("planned_contact must be bool [B,6]")
        if planned_surface.shape != (batch, 6):
            raise ValueError("planned_surface must be [B,6]")
        active_surface = planned_surface[planned_contact]
        if bool(((active_surface < 0) | (active_surface > 1)).any()):
            raise ValueError("active planned contacts require ground/top surface IDs")
        return torch.where(planned_contact, planned_surface.clamp(0, 1), 2)

    def load_observation_encoder(self, state: dict[str, Tensor]) -> tuple[list[str], list[str]]:
        """Load only modules whose meaning is unchanged from dense V3."""

        prefixes = (
            "height.", "global_encoder.", "part_encoder.", "part_identity.",
            "surface_embedding.", "terrain_reasoning.", "shared.", "norm.",
        )
        selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
        incompatible = self.load_state_dict(selected, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(f"unexpected observation encoder keys: {unexpected}")
        loaded = sorted(selected)
        return loaded, list(incompatible.missing_keys)

    def forward(
        self,
        current_q: Tensor,
        current_contact: Tensor,
        current_anchor: Tensor,
        current_surface: Tensor,
        heightmap: Tensor,
        planned_contact: Tensor,
        planned_surface: Tensor,
    ) -> ConditionedPosePrediction:
        terrain, basis, yaw_quaternion, body, part = self._encode_observation(
            current_q, current_contact, current_anchor, current_surface, heightmap
        )
        qpos = self._decode_with_plan(
            current_q, terrain, basis, yaw_quaternion, body, part,
            planned_contact, planned_surface,
        )
        return ConditionedPosePrediction(qpos=qpos)

    def _encode_observation(
        self,
        current_q: Tensor,
        current_contact: Tensor,
        current_anchor: Tensor,
        current_surface: Tensor,
        heightmap: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = len(current_q)
        terrain = self.height(heightmap)
        basis, yaw_quaternion = _root_yaw_basis(current_q)
        positions, rotations6d = self.fk(current_q[:, None])
        positions = positions[:, 0, 1:7]
        rotations = _matrix_from_rotation6d(rotations6d[:, 0, 1:7])
        local_position = torch.einsum(
            "bij,bpj->bpi", basis.transpose(1, 2), positions - current_q[:, None, :3]
        )
        local_rotation = basis[:, None].transpose(-1, -2) @ rotations
        local_anchor = torch.einsum(
            "bij,bpj->bpi", basis.transpose(1, 2), current_anchor - current_q[:, None, :3]
        )
        current_surface_index = torch.where(
            current_contact, current_surface.clamp(0, 1), 2
        )
        part = self.part_encoder(torch.cat((
            local_position,
            _rotation6d(local_rotation),
            local_anchor * current_contact[..., None],
            current_contact[..., None].float(),
        ), -1))
        part = (
            part + self.part_identity.weight[None]
            + self.surface_embedding(current_surface_index)
        )
        state = self.global_encoder(self._local_state(current_q, basis))[:, None]
        queries = torch.cat((state, part), dim=1)
        for block in self.terrain_reasoning:
            queries = block(queries, terrain)
        encoded = self.shared(queries)
        body = self.norm(encoded[:, :1])
        part = self.norm(encoded[:, 1:])
        return terrain, basis, yaw_quaternion, body, part

    def _decode_with_plan(
        self,
        current_q: Tensor,
        terrain: Tensor,
        basis: Tensor,
        yaw_quaternion: Tensor,
        body: Tensor,
        part: Tensor,
        planned_contact: Tensor,
        planned_surface: Tensor,
    ) -> Tensor:
        plan_index = self._validate_plan(planned_contact, planned_surface, len(current_q))
        plan_weight = torch.nn.functional.one_hot(plan_index, 3).to(part)
        return self._decode_with_plan_weights(
            current_q, terrain, basis, yaw_quaternion, body, part, plan_weight
        )

    def _decode_with_plan_weights(
        self,
        current_q: Tensor,
        terrain: Tensor,
        basis: Tensor,
        yaw_quaternion: Tensor,
        body: Tensor,
        part: Tensor,
        plan_weight: Tensor,
    ) -> Tensor:
        """Decode one endpoint from ground/top/inactive weights per limb.

        Hard one-hot weights retain the exact inference contract.  A caller
        may use straight-through weights so pose-realization gradients also
        train the contact heads without changing the forward contact plan.
        """
        if plan_weight.shape != (len(current_q), 6, 3):
            raise ValueError("plan_weight must be [B,6,3]")
        planned_part = part + plan_weight.to(part) @ self.plan_surface_embedding.weight
        planned_part = planned_part + self.plan_fusion(planned_part)
        interaction = self.pose_terrain(torch.cat((body, planned_part), 1), terrain)
        interaction = self.interaction_decoder(interaction)
        pooled = self.norm(interaction[:, 0] + interaction[:, 1:].mean(1))
        return self._decode_pose(self.pose_head(pooled), current_q, basis, yaw_quaternion)

    # Keep the exact root-yaw pose representation used by V3/V4.
    _local_state = staticmethod(DenseHeightmapInteractionPredictor._local_state)
    _decode_pose = staticmethod(DenseHeightmapInteractionPredictor._decode_pose)
    encode_pose = DenseHeightmapInteractionPredictor.encode_pose
    decode_pose = DenseHeightmapInteractionPredictor.decode_pose


def interaction_root_progress_loss(
    current_q: Tensor,
    predicted_q: Tensor,
    target_q: Tensor,
    *,
    dead_zone_m: float = 0.05,
) -> Tensor:
    """Weak local task-progress supervision without absolute-root imitation.

    Displacements are expressed in the current root-yaw frame.  A 5 cm region
    around the demonstrated displacement is free, while motion opposite the
    demonstrated horizontal progress direction receives an extra penalty.
    """

    if current_q.shape != predicted_q.shape or current_q.shape != target_q.shape:
        raise ValueError("current_q, predicted_q, and target_q must have equal shape")
    if current_q.ndim != 2 or current_q.shape[1] != 36:
        raise ValueError("root progress inputs must be [B,36]")
    if dead_zone_m < 0.0:
        raise ValueError("dead_zone_m must be non-negative")
    basis, _ = _root_yaw_basis(current_q)
    predicted_delta = torch.einsum(
        "bij,bj->bi",
        basis.transpose(1, 2),
        predicted_q[:, :3] - current_q[:, :3],
    )
    target_delta = torch.einsum(
        "bij,bj->bi",
        basis.transpose(1, 2),
        target_q[:, :3] - current_q[:, :3],
    )
    outside_region = (
        (predicted_delta - target_delta).abs() - dead_zone_m
    ).clamp_min(0.0)
    coarse_position = (outside_region / 0.15).square().mean(-1)

    horizontal_target = target_delta[:, :2]
    target_distance = horizontal_target.norm(dim=-1)
    direction = horizontal_target / target_distance[:, None].clamp_min(1.0e-6)
    signed_progress = (predicted_delta[:, :2] * direction).sum(-1)
    moving = target_distance > dead_zone_m
    reverse = (torch.relu(-signed_progress) / 0.10).square() * moving
    return coarse_position + reverse


def interaction_root_heading_loss(
    predicted_q: Tensor,
    target_q: Tensor,
    *,
    dead_zone_rad: float = math.radians(2.0),
    scale_rad: float = math.radians(15.0),
) -> Tensor:
    """Weakly preserve task-relevant horizontal heading with periodic error.

    The root-yaw height map changes discontinuously when a box edge crosses a
    grid cell.  Contact feasibility alone does not constrain heading, so a
    sequence can keep every support while gradually rotating into an unseen
    observation.  This loss supervises only the yaw degree of freedom; it is
    loss-side information and does not add a model input or regress joints.
    """

    if predicted_q.shape != target_q.shape:
        raise ValueError("predicted_q and target_q must have equal shape")
    if predicted_q.ndim != 2 or predicted_q.shape[1] != 36:
        raise ValueError("root heading inputs must be [B,36]")
    if dead_zone_rad < 0.0 or scale_rad <= 0.0:
        raise ValueError("heading dead zone must be non-negative and scale positive")
    predicted_basis, _ = _root_yaw_basis(predicted_q)
    target_basis, _ = _root_yaw_basis(target_q)
    predicted_forward = predicted_basis[:, :, 0]
    target_forward = target_basis[:, :, 0]
    cosine = (target_forward * predicted_forward).sum(-1)
    sine = (
        target_forward[:, 0] * predicted_forward[:, 1]
        - target_forward[:, 1] * predicted_forward[:, 0]
    )
    error = torch.atan2(sine, cosine)
    outside = (error.abs() - dead_zone_rad).clamp_min(0.0)
    return (outside / scale_rad).square()


def interaction_root_contact_frame_loss(
    predicted_q: Tensor,
    reference_q: Tensor,
    contact_anchor: Tensor,
    planned_contact: Tensor,
    *,
    dead_zone_m: float = 0.03,
    scale_m: float = 0.15,
    angle_dead_zone_rad: float = math.radians(2.0),
    angle_scale_rad: float = math.radians(15.0),
) -> Tensor:
    """Supervise root placement relative to the planned interaction geometry.

    Planned anchors are expressed in both the predicted and verified expert
    root-yaw frames.  Thus the target varies with the actual interaction
    location instead of teaching an event a fixed root displacement.  Anchors
    are loss-side supervision only, and the dead zone prevents them from
    becoming exact binding-point constraints.
    """

    if predicted_q.shape != reference_q.shape:
        raise ValueError("predicted_q and reference_q must have equal shape")
    if predicted_q.ndim != 2 or predicted_q.shape[1] != 36:
        raise ValueError("root/contact-frame inputs must be [B,36]")
    if contact_anchor.shape != (len(predicted_q), 6, 3):
        raise ValueError("contact_anchor must be [B,6,3]")
    if planned_contact.shape != (len(predicted_q), 6) or planned_contact.dtype != torch.bool:
        raise ValueError("planned_contact must be bool [B,6]")
    if dead_zone_m < 0.0 or scale_m <= 0.0:
        raise ValueError("root/contact-frame dead zone must be non-negative and scale positive")
    if angle_dead_zone_rad < 0.0 or angle_scale_rad <= 0.0:
        raise ValueError("root/contact-frame angle scales are invalid")

    predicted_basis, _ = _root_yaw_basis(predicted_q)
    reference_basis, _ = _root_yaw_basis(reference_q)
    predicted_offset = torch.einsum(
        "bij,bpj->bpi",
        predicted_basis.transpose(1, 2),
        contact_anchor - predicted_q[:, None, :3],
    )
    reference_offset = torch.einsum(
        "bij,bpj->bpi",
        reference_basis.transpose(1, 2),
        contact_anchor - reference_q[:, None, :3],
    )
    predicted_radius = predicted_offset[..., :2].norm(dim=-1)
    reference_radius = reference_offset[..., :2].norm(dim=-1)
    radial = (
        (predicted_radius - reference_radius).abs() - dead_zone_m
    ).clamp_min(0.0).div(scale_m).square()
    vertical = (
        (predicted_offset[..., 2] - reference_offset[..., 2]).abs() - dead_zone_m
    ).clamp_min(0.0).div(scale_m).square()
    dot = (predicted_offset[..., :2] * reference_offset[..., :2]).sum(-1)
    cross = (
        reference_offset[..., 0] * predicted_offset[..., 1]
        - reference_offset[..., 1] * predicted_offset[..., 0]
    )
    angle = torch.atan2(cross, dot).abs()
    angular = (
        (angle - angle_dead_zone_rad).clamp_min(0.0) / angle_scale_rad
    ).square()
    # A contact almost vertically below the root does not define a reliable
    # horizontal bearing.  Its radial/vertical relationship remains valid.
    bearing_valid = (predicted_radius > 0.05) & (reference_radius > 0.05)
    normalized = radial + vertical + angular * bearing_valid
    active = planned_contact.to(normalized)
    count = active.sum(-1).clamp_min(1.0)
    terms = normalized * active
    return terms.sum(-1) / count + 0.25 * terms.amax(-1)


def balanced_squared_loss_weights(losses: Tensor, floor: float = 0.10) -> Tensor:
    """Balance quadratic-error gradients without suppressing hard examples.

    For a squared residual, gradient magnitude grows approximately with the
    square root of its loss.  Detached inverse-square-root weights therefore
    equalize gradient scale.  The former inverse-loss rule over-corrected and
    made the first large DAgger frontier receive *less* useful gradient than
    already mastered states.
    """

    if losses.ndim != 1:
        raise ValueError("balanced losses must be a vector")
    if floor <= 0.0:
        raise ValueError("balanced loss floor must be positive")
    weights = losses.detach().clamp_min(floor).rsqrt()
    return weights / weights.mean().clamp_min(1.0e-12)


def _interaction_pose_prior(
    model: nn.Module,
    qpos: Tensor,
    positions: Tensor,
    rotations: Tensor,
    target: dict[str, Tensor],
) -> Tensor:
    """Weakly preserve demonstrated body organization, never demonstrated q.

    The prior compares the six interaction bodies in each endpoint's root
    frame.  Global placement is left to contact realization/root progress and
    no individual joint angle is supervised.
    """

    with torch.no_grad():
        _, target_rotation6d = model.fk(target["q"][:, None])
        target_rotation = _matrix_from_rotation6d(target_rotation6d[:, 0])
        target_root_rotation = _quaternion_matrix_wxyz(target["q"][:, 3:7])
        target_local_position = torch.einsum(
            "bij,bpj->bpi",
            target_root_rotation.transpose(1, 2),
            target["body_position"][:, 1:7] - target["q"][:, None, :3],
        )
        target_local_rotation = (
            target_root_rotation[:, None].transpose(-1, -2)
            @ target_rotation[:, 1:7]
        )
    root_rotation = _quaternion_matrix_wxyz(qpos[:, 3:7])
    local_position = torch.einsum(
        "bij,bpj->bpi",
        root_rotation.transpose(1, 2),
        positions[:, 1:7] - qpos[:, None, :3],
    )
    local_rotation = root_rotation[:, None].transpose(-1, -2) @ rotations[:, 1:7]
    position = ((local_position - target_local_position) / 0.15).square().sum(-1)
    orientation = ((local_rotation - target_local_rotation) / 0.50).square().mean((-1, -2))
    return position.mean(-1) + 0.25 * position.amax(-1) + orientation.mean(-1)


def _nominal_taskspace_teacher_loss(
    qpos: Tensor,
    positions: Tensor,
    rotations: Tensor,
    target: dict[str, Tensor],
) -> Tensor:
    """Distill a successful conditioned nominal without regressing its q.

    The teacher may consume privileged loss-side contact conditions, but the
    student never does.  The pelvis plus six interaction-body poses are
    compared in the current observed root-yaw frame so the target describes
    one whole-body action from the current state.  Keeping the pelvis is
    essential: end-effectors alone permit a displaced root to be hidden by
    compensating joint angles.
    """

    required = ("nominal_teacher_body_position", "nominal_teacher_body_rotation")
    if any(key not in target for key in required):
        return qpos.sum(-1) * 0.0
    if "current_q" not in target:
        raise ValueError("nominal task-space teacher requires current_q")
    basis, _ = _root_yaw_basis(target["current_q"])
    origin = target["current_q"][:, :3]
    teacher_position = target["nominal_teacher_body_position"][:, :7]
    teacher_rotation = target["nominal_teacher_body_rotation"][:, :7]
    predicted_local = torch.einsum(
        "bij,bpj->bpi", basis.transpose(1, 2), positions[:, :7] - origin[:, None]
    )
    teacher_local = torch.einsum(
        "bij,bpj->bpi", basis.transpose(1, 2), teacher_position - origin[:, None]
    )
    predicted_rotation = basis[:, None].transpose(-1, -2) @ rotations[:, :7]
    teacher_rotation = basis[:, None].transpose(-1, -2) @ teacher_rotation
    position = ((predicted_local - teacher_local) / 0.15).square().sum(-1)
    orientation = ((predicted_rotation - teacher_rotation) / 0.50).square().mean((-1, -2))
    return position.mean(-1) + 0.25 * position.amax(-1) + orientation.mean(-1)


def _missing_intent_surface_distance(
    model: ConditionedHeightmapPosePredictor,
    qpos: Tensor,
    intended_surface: Tensor,
    scene: dict[str, Tensor],
    configured_margin: Tensor,
) -> Tensor:
    """Loss-only approach residual for a requested pair absent from Newton.

    This reuses the mechanism that made the verified projector converge: the
    authoritative URDF collision shapes approach a finite primary face until
    Newton can emit the requested pair.  The target is an interior point of
    the *actual configured solver margin*, not an invented zero-gap contact.
    Newton active+allocated remains the only contact truth.  None of these
    privileged loss tensors enter ``model.forward``.
    """
    if configured_margin.shape != (len(qpos),) or bool((configured_margin <= 0).any()):
        raise ValueError("missing-contact approach requires positive Newton margins")
    geometry = getattr(model, "_establishment_geometry", None)
    if geometry is None:
        geometry = CanonicalContactCollisionGeometry().to(qpos.device)
        model._establishment_geometry = geometry
    center = scene["box_center"]
    rotation = scene["box_rotation"]
    half = scene["box_half_extents"]
    ground = scene["ground_height"]
    top_normal = rotation[:, :, 2]
    top_center = center + top_normal * half[:, 2:3]
    top_offset = (top_center * top_normal).sum(-1)
    # Loss-only approach to the upper gap bound; not a replacement for
    # Newton activation. Full-body safety handles the lower bound separately.
    target_gap = configured_margin
    output = qpos.new_zeros((len(qpos), 6))
    names = tuple(dict.fromkeys(shape.link_name for shape in geometry.shapes))
    positions, rotations = model.fk.link_poses(qpos, names)
    link_index = {name: index for index, name in enumerate(names)}
    for part_index in range(6):
        part_names = tuple(dict.fromkeys(shape.link_name for shape in geometry.shapes if shape.part == part_index))
        for surface_index in (0, 1):
            selected = intended_surface[:, part_index] == surface_index
            if not bool(selected.any()):
                continue
            indices = selected.nonzero(as_tuple=False)[:, 0]
            subset = qpos[indices]
            margin = configured_margin[indices]
            if surface_index == 0:
                normal = subset.new_tensor((0.0, 0.0, 1.0)).expand(len(indices), -1)
                offset = ground[indices] + target_gap[indices]
                region = None
                inset = subset.new_zeros(len(indices))
            else:
                normal = top_normal[indices]
                offset = top_offset[indices] + target_gap[indices]
                region = torch.cat((
                    top_center[indices],
                    rotation[indices, :, 0],
                    rotation[indices, :, 1],
                    half[indices, :2],
                ), dim=-1)
                inset = margin
            residual = geometry.approach_residual(
                model.fk,
                subset,
                part_index,
                normal,
                offset,
                region,
                subset.new_zeros((len(indices), 3)),
                inset,
                patch=False,
                link_pose_override={name: (positions[indices, link_index[name]], rotations[indices, link_index[name]])
                                    for name in part_names},
            )
            # Keep finite-face tangential guidance, but do not pull an interior
            # point back out to the upper boundary of the contact interval.
            normal_component = (residual * normal).sum(-1, keepdim=True)
            residual = residual - normal_component.clamp_max(0) * normal
            output[indices, part_index] = residual.norm(dim=-1)
    return output


def conditioned_pose_objective(
    model: ConditionedHeightmapPosePredictor,
    prediction: ConditionedPosePrediction,
    target: dict[str, Tensor],
    scene: dict[str, Tensor],
    *,
    realization_contact: Tensor | None = None,
    realization_surface: Tensor | None = None,
    missing_pair_approach_weight: float = 1.0,
    structure_joint_max_weight: float = 0.0,
    query_override=None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Pose imitation plus unconditional Newton realization of the plan."""

    qpos = prediction.qpos
    positions, rotations6d = model.fk(qpos)
    rotations = _matrix_from_rotation6d(rotations6d)
    active = target["planned_contact"]
    count = active.sum(-1).clamp_min(1)
    material = positions[:, 1:7] + torch.einsum(
        "bpij,bpj->bpi", rotations[:, 1:7], target["material_local"]
    )
    contact_delta = material - target["anchor"]
    contact_error = contact_delta.norm(dim=-1)
    # The demonstration anchor only selects a weak tangential neighborhood.
    # Normal establishment is exclusively handled by the Newton-margin loss;
    # the former isotropic 4 cm ball could silently accept a hovering limb.
    top_normal = scene["box_rotation"][:, :, 2]
    normal = torch.where(
        target["planned_surface"][..., None] == 1,
        top_normal[:, None],
        contact_delta.new_tensor((0.0, 0.0, 1.0)),
    )
    tangent_delta = contact_delta - (contact_delta * normal).sum(-1, keepdim=True) * normal
    region_error = (tangent_delta.norm(dim=-1) - 0.04).clamp_min(0.0)
    normalized_contact = (region_error / 0.04).square() * active
    contact = normalized_contact.sum(-1) / count + 0.25 * normalized_contact.amax(-1)

    root_translation = ((qpos[:, :3] - target["q"][:, :3]) / POSE_POSITION_SCALE_M).square().mean(-1)
    root_rotation = (
        (_quaternion_matrix_wxyz(qpos[:, 3:7])
         - _quaternion_matrix_wxyz(target["q"][:, 3:7])).square().mean((-1, -2))
        / POSE_ROTATION_MATRIX_SCALE**2
    )
    joint = ((qpos[:, 7:] - target["q"][:, 7:]) / POSE_JOINT_SCALE_RAD).square().mean(-1)
    pose = root_translation + root_rotation + joint
    structure, structure_metrics = structural_pose_loss(
        model, qpos, positions, rotations, target, joint_max_weight=structure_joint_max_weight
    )
    interaction_pose_prior = _interaction_pose_prior(
        model, qpos, positions, rotations, target
    )
    nominal_taskspace_teacher = _nominal_taskspace_teacher_loss(
        qpos, positions, rotations, target
    )

    if realization_contact is None:
        realization_contact = active
    if realization_surface is None:
        realization_surface = target["planned_surface"]
    if realization_contact.shape != active.shape or realization_contact.dtype != torch.bool:
        raise ValueError("realization_contact must be bool [B,6]")
    if realization_surface.shape != active.shape:
        raise ValueError("realization_surface must be [B,6]")
    if bool(((realization_surface < 0) & realization_contact).any()):
        raise ValueError("active realization contacts require a support surface")
    required = (
        "newton_world_origin", "newton_world_basis", "newton_model_fingerprint"
    )
    if any(key not in target for key in required):
        raise ValueError(
            "conditioned pose training requires fresh Newton query metadata; "
            "a geometric distance fallback is forbidden"
        )
    from contact_solver.newton_plan_realization import newton_plan_realization

    realization, realization_metrics = newton_plan_realization(
        model,
        qpos,
        realization_contact,
        realization_surface,
        scene,
        world_frame=(target["newton_world_origin"], target["newton_world_basis"]),
        fingerprints=target["newton_model_fingerprint"],
        invalid_witness_policy="detach",
        query_override=query_override,
    )
    missing_intended = (
        realization_metrics["newton_missing_intended_mask"].bool()
        & realization_contact
    )
    missing_count = missing_intended.sum(-1).clamp_min(1)
    missing_distance = _missing_intent_surface_distance(
        model,
        qpos,
        realization_surface,
        scene,
        realization_metrics["newton_configured_includemargin"],
    )
    # Normalize by each sample's real solver margin and keep a non-vanishing
    # gradient until the requested pair appears.  This is privileged loss-only
    # geometry; no distance, face, normal, or margin enters model.forward.
    missing_terms = (
        missing_distance
        / realization_metrics["newton_configured_includemargin"][:, None]
    ) * missing_intended
    missing_approach = (
        missing_terms.sum(-1) / missing_count
        + 0.25 * missing_terms.amax(-1)
    )
    joint_limit, violation = joint_feasibility_loss(
        qpos[:, 7:], model.fk.joint_lower, model.fk.joint_upper
    )
    # Exact demonstration witnesses are diagnostics, not hidden binding
    # targets.  The decoder must realize the explicit limb+surface plan using
    # its collision geometry; Newton verifies the resulting hard contact.
    imitation = pose + STRUCTURE_WEIGHT * structure
    safety = joint_limit
    # A newly requested contact commonly has no Newton narrow-phase pair yet.
    # In that case ``realization`` has no term for the missing limb, so this
    # finite-surface approach residual is the only establishment gradient.  It
    # is deliberately a first-class loss (default weight 1), not the former
    # 0.1 diagnostic nudge.  Newton remains the sole authority for deciding
    # whether contact was actually activated and allocated.
    establishment = missing_pair_approach_weight * missing_approach
    loss = imitation + realization + establishment + safety
    metrics = {
        "loss": loss,
        "contact_cm": 100.0 * (contact_error * active).sum(-1) / count,
        "contact_max_cm": 100.0 * (contact_error * active).amax(-1),
        "interaction_location_loss": contact,
        "root_cm": 100.0 * (qpos[:, :3] - target["q"][:, :3]).norm(dim=-1),
        "joint_rmse_rad": (qpos[:, 7:] - target["q"][:, 7:]).square().mean(-1).sqrt(),
        "body_cm": 100.0 * (positions - target["body_position"]).norm(dim=-1).mean(-1),
        "penetration_cm": realization_metrics["newton_penetration_cm"],
        "joint_violation_rad": violation.amax(-1),
        "pose_loss": pose,
        "structure_loss": structure,
        "imitation_loss": imitation,
        "interaction_pose_prior_loss": interaction_pose_prior,
        "nominal_taskspace_teacher_loss": nominal_taskspace_teacher,
        "own_plan_realization_loss": realization,
        "missing_pair_approach_loss": missing_approach,
        "contact_establishment_loss": establishment,
        "missing_pair_surface_distance_cm": (
            100.0 * (missing_distance * missing_intended).sum(-1) / missing_count
        ),
        "safety_loss": safety,
        "collision_loss": realization_metrics["newton_collision_loss"],
        "joint_limit_loss": joint_limit,
    }
    metrics.update(realization_metrics)
    metrics.update(structure_metrics)
    return loss, metrics


def topology_gated_pose_loss(
    metrics: dict[str, Tensor],
    compatible_plan: Tensor,
) -> Tensor:
    """Keep pose learning on the demonstrated contact topology.

    An incompatible predicted plan is a planning-head error.  Letting its
    realization residual update the pose decoder teaches the decoder to carry
    out an incorrect and potentially infeasible interaction.  Safety remains
    unconditional for every generated pose.
    """

    if compatible_plan.ndim != 1 or compatible_plan.dtype != torch.bool:
        raise ValueError("compatible_plan must be bool [B]")
    required = (
        "imitation_loss",
        "own_plan_realization_loss",
        "contact_establishment_loss",
        "safety_loss",
    )
    if any(key not in metrics for key in required):
        raise ValueError("pose metrics lack topology-gated loss terms")
    if any(metrics[key].shape != compatible_plan.shape for key in required):
        raise ValueError("pose metrics and compatible_plan batch shapes differ")
    task = (
        metrics["imitation_loss"]
        + metrics["own_plan_realization_loss"]
        + metrics["contact_establishment_loss"]
    )
    return compatible_plan.to(task) * task + metrics["safety_loss"]


def canonical_contact_first_pose_loss(
    metrics: dict[str, Tensor],
    compatible_plan: Tensor,
    *,
    imitation_weight: float = 0.1,
    interaction_pose_prior_weight: float = 0.0,
    nominal_taskspace_teacher_weight: float = 0.0,
) -> Tensor:
    """Contact-first endpoint training with unconditional geometric safety.

    Newton contact realization, missing-contact establishment, and removal of
    unwanted contacts define the interaction task.  Demonstration pose fitting
    only selects a reasonable whole-body solution inside that task.  An
    incompatible autonomous plan must not teach the decoder to imitate the GT
    transition under a different contact condition.
    """

    if compatible_plan.ndim != 1 or compatible_plan.dtype != torch.bool:
        raise ValueError("compatible_plan must be bool [B]")
    if not 0.0 <= imitation_weight <= 1.0:
        raise ValueError("imitation_weight must be in [0,1]")
    if not 0.0 <= interaction_pose_prior_weight <= 1.0:
        raise ValueError("interaction_pose_prior_weight must be in [0,1]")
    if not 0.0 <= nominal_taskspace_teacher_weight <= 64.0:
        raise ValueError("nominal_taskspace_teacher_weight must be in [0,64]")
    required = (
        "imitation_loss",
        "interaction_pose_prior_loss",
        "nominal_taskspace_teacher_loss",
        "interaction_location_loss",
        "newton_plan_realization_loss",
        "newton_unwanted_contact_loss",
        "contact_establishment_loss",
        "newton_collision_loss",
        "joint_limit_loss",
    )
    if any(key not in metrics for key in required):
        raise ValueError("pose metrics lack canonical contact-first loss terms")
    if any(metrics[key].shape != compatible_plan.shape for key in required):
        raise ValueError("pose metrics and compatible_plan batch shapes differ")
    contact_task = (
        metrics["interaction_location_loss"]
        + metrics["newton_plan_realization_loss"]
        + metrics["newton_unwanted_contact_loss"]
        + metrics["contact_establishment_loss"]
        + interaction_pose_prior_weight * metrics["interaction_pose_prior_loss"]
        + nominal_taskspace_teacher_weight * metrics["nominal_taskspace_teacher_loss"]
        + imitation_weight * metrics["imitation_loss"]
    )
    safety = metrics["newton_collision_loss"] + metrics["joint_limit_loss"]
    return compatible_plan.to(contact_task) * contact_task + safety


__all__ = [
    "ConditionedHeightmapPosePredictor",
    "ConditionedPosePrediction",
    "conditioned_pose_objective",
    "interaction_root_progress_loss",
    "interaction_root_heading_loss",
    "interaction_root_contact_frame_loss",
    "balanced_squared_loss_weights",
    "canonical_contact_first_pose_loss",
    "topology_gated_pose_loss",
]
