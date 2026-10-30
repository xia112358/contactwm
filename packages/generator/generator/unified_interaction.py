"""Joint next-interaction endpoint prediction.

Unlike the legacy contact-first selector, this module has no action vocabulary
and no pose-realization stage.  One shared decoder predicts the endpoint q,
active contacts, surfaces, and duration together.  Contact geometry is derived
from that same q through canonical-G1 FK, so pose and contact cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from generator.contact_q_generator import ContactTransitionPlan, ResidualBlock, TerrainScanEncoder
from somaforge_core.motion_contracts import CONTACT_BODY_INDEX, CONTACT_PARTS
from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _quaternion_multiply_wxyz


from somaforge_core.prediction_contracts import UnifiedInteractionPrediction


@dataclass(frozen=True)
class UnifiedInteractionLoss:
    total: Tensor
    q_pose: Tensor
    contact_state: Tensor
    touchdown: Tensor
    persistent: Tensor
    surface: Tensor
    active_geometry: Tensor
    persistent_anchor: Tensor
    duration: Tensor
    topology_choice: Tensor
    surface_feasibility: Tensor
    inactive_clearance: Tensor
    contact_orientation: Tensor
    q_velocity: Tensor
    contact_velocity: Tensor




from contact_solver.trajectory_projection import InteractionProjectionResult


from contact_solver.trajectory_projection import _endpoint_root_shift_penalty


from contact_solver.trajectory_projection import _limit_endpoint_root_shift


from contact_solver.trajectory_projection import _collision_depth_cost


from contact_solver.trajectory_projection import rollout_dense_collision_loss


from contact_solver.trajectory_projection import rollout_ground_collision_loss


from contact_solver.trajectory_projection import contact_rotation_losses


class UnifiedInteractionPredictor(nn.Module):
    """Predict the next pose-contact interaction as one coupled random variable."""

    def __init__(
        self,
        endpoint_offsets: Tensor,
        q_mean: Tensor,
        q_std: Tensor,
        scan_mean: Tensor,
        scan_std: Tensor,
        duration_log_mean: Tensor,
        duration_log_std: Tensor,
        *,
        surface_count: int = 2,
        width: int = 256,
        blocks: int = 4,
        maximum_root_translation_m: float = 0.5,
        maximum_root_quaternion_delta: float = 0.7,
        maximum_joint_delta_rad: float = 1.0,
        maximum_contact_offset_delta_m: float = 0.08,
        minimum_duration_s: float = 0.12,
        maximum_duration_s: float = 8.66,
        topology_transitions: Tensor | None = None,
        phase_topologies: Tensor | None = None,
        phase_q_prototypes: Tensor | None = None,
        phase_q_velocity_prototypes: Tensor | None = None,
        predict_boundary_velocity: bool = True,
        use_phase_condition: bool = True,
        fixed_material_contacts: bool = False,
        remember_material_contacts: bool = False,
        persistent_constraint_iterations: int = 0,
        persistent_constraint_damping: float = 1.0e-3,
        persistent_constraint_maximum_step: float = 0.25,
    ) -> None:
        super().__init__()
        if endpoint_offsets.shape != (len(CONTACT_PARTS), surface_count, 3):
            raise ValueError("endpoint_offsets must have shape [6, surface_count, 3]")
        from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics

        self.fk = CanonicalG1ForwardKinematics()
        self.surface_count = int(surface_count)
        self.maximum_root_translation_m = float(maximum_root_translation_m)
        self.maximum_root_quaternion_delta = float(maximum_root_quaternion_delta)
        self.maximum_joint_delta_rad = float(maximum_joint_delta_rad)
        self.maximum_contact_offset_delta_m = float(maximum_contact_offset_delta_m)
        self.predict_boundary_velocity = bool(predict_boundary_velocity)
        self.use_phase_condition = bool(use_phase_condition)
        self.fixed_material_contacts = bool(fixed_material_contacts)
        self.remember_material_contacts = bool(remember_material_contacts)
        if self.fixed_material_contacts and self.remember_material_contacts:
            raise ValueError('Choose canonical fixed points or initialized material-point memory, not both')
        if not self.use_phase_condition:
            phase_topologies = None
            phase_q_prototypes = None
            phase_q_velocity_prototypes = None
        self.persistent_constraint_iterations = int(persistent_constraint_iterations)
        self.persistent_constraint_damping = float(persistent_constraint_damping)
        self.persistent_constraint_maximum_step = float(persistent_constraint_maximum_step)
        if self.persistent_constraint_iterations < 0:
            raise ValueError("persistent constraint iterations must be non-negative")
        if self.persistent_constraint_damping <= 0.0:
            raise ValueError("persistent constraint damping must be positive")
        if self.persistent_constraint_maximum_step <= 0.0:
            raise ValueError("persistent constraint maximum step must be positive")
        self.minimum_duration_s = float(minimum_duration_s)
        self.maximum_duration_s = float(maximum_duration_s)
        if not 0.0 < self.minimum_duration_s <= self.maximum_duration_s:
            raise ValueError("duration limits must be positive and ordered")
        if topology_transitions is None:
            topology_transitions = torch.empty((0, 4, len(CONTACT_PARTS)), dtype=torch.bool)
        topology_transitions = topology_transitions.to(torch.bool)
        if topology_transitions.ndim != 3 or topology_transitions.shape[1:] != (4, len(CONTACT_PARTS)):
            raise ValueError("topology_transitions must have shape [K,4,6]")
        if len(topology_transitions):
            start, persistent, touchdown, finish = topology_transitions.unbind(dim=1)
            if bool((persistent & ~start).any() or (persistent & ~finish).any()):
                raise ValueError("persistent contacts must be active at both boundaries")
            if bool((touchdown & ~finish).any() or (persistent & touchdown).any()):
                raise ValueError("touchdown must be an exclusive subset of end contacts")
        self.register_buffer("topology_transitions", topology_transitions, persistent=False)
        if phase_topologies is None:
            phase_topologies = torch.empty((0, 4, len(CONTACT_PARTS)), dtype=torch.bool)
        phase_topologies = phase_topologies.to(torch.bool)
        if phase_topologies.ndim != 3 or phase_topologies.shape[1:] != (4, len(CONTACT_PARTS)):
            raise ValueError("phase_topologies must have shape [E,4,6]")
        self.register_buffer("phase_topologies", phase_topologies, persistent=False)
        if phase_q_prototypes is None:
            phase_q_prototypes = torch.empty((0, 36), dtype=torch.float32)
        phase_q_prototypes = phase_q_prototypes.float()
        if phase_q_prototypes.ndim != 2 or phase_q_prototypes.shape[1] != 36:
            raise ValueError("phase_q_prototypes must have shape [E,36]")
        if len(phase_q_prototypes) and len(phase_q_prototypes) != len(phase_topologies):
            raise ValueError("phase q prototypes and topology schedule must have equal length")
        self.register_buffer("phase_q_prototypes", phase_q_prototypes, persistent=False)
        if phase_q_velocity_prototypes is None:
            phase_q_velocity_prototypes = torch.zeros((len(phase_topologies), 36), dtype=torch.float32)
        phase_q_velocity_prototypes = phase_q_velocity_prototypes.float()
        if phase_q_velocity_prototypes.shape != (len(phase_topologies), 36):
            raise ValueError("phase q velocity prototypes must have shape [E,36]")
        self.register_buffer(
            "phase_q_velocity_prototypes", phase_q_velocity_prototypes, persistent=False
        )
        self.phase_q_residual_adapter = nn.Parameter(torch.zeros((len(phase_topologies), 36), dtype=torch.float32))
        self.phase_q_feature_adapter = nn.Parameter(
            torch.zeros((len(phase_topologies), 36, width), dtype=torch.float32)
        )
        self.register_buffer("endpoint_offsets", endpoint_offsets.float())
        self.register_buffer("q_mean", q_mean.float())
        self.register_buffer("q_std", q_std.float())
        self.register_buffer("scan_mean", scan_mean.float())
        self.register_buffer("scan_std", scan_std.float())
        self.register_buffer("duration_log_mean", duration_log_mean.float().reshape(()))
        self.register_buffer("duration_log_std", duration_log_std.float().reshape(()))

        state_dim = 36 + len(CONTACT_PARTS) * (1 + 3 + 3 + surface_count) + 2 * len(CONTACT_PARTS) + int(self.use_phase_condition)
        if self.predict_boundary_velocity:
            state_dim += 36
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, width), nn.SiLU(), ResidualBlock(width))
        self.scan_encoder = TerrainScanEncoder(width)
        self.trunk = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            *(ResidualBlock(width) for _ in range(blocks)),
            nn.LayerNorm(width),
        )
        self.topology_encoder = nn.Linear(3 * len(CONTACT_PARTS), width)
        nn.init.zeros_(self.topology_encoder.weight)
        nn.init.zeros_(self.topology_encoder.bias)
        # A single head is intentional: none of these quantities is upstream
        # of another.  Geometry is then made exact by deriving it from FK.
        # Velocity is appended so every older head row retains its exact
        # meaning when a checkpoint is expanded into the interaction-state
        # model.  New velocity rows start at zero (a stationary boundary).
        output_dim = 36 + 3 * len(CONTACT_PARTS) + len(CONTACT_PARTS) * (surface_count + 3) + 1
        if self.predict_boundary_velocity:
            output_dim += 36
        self.interaction_head = nn.Linear(width, output_dim)
        nn.init.zeros_(self.interaction_head.weight)
        nn.init.zeros_(self.interaction_head.bias)

    def load_compatible_state_dict(
        self, state_dict: dict[str, Tensor], *, reset_q_head_for_prototypes: bool = False
    ) -> None:
        """Load pre-history checkpoints while preserving their exact initial behavior."""

        state_dict = dict(state_dict)
        key = "state_encoder.0.weight"
        incoming = state_dict[key]
        expected = self.state_dict()[key]
        if incoming.shape != expected.shape:
            if incoming.shape[0] != expected.shape[0] or incoming.shape[1] >= expected.shape[1]:
                raise ValueError(f"incompatible {key}: {tuple(incoming.shape)} versus {tuple(expected.shape)}")
            expanded = expected.new_zeros(expected.shape)
            expanded[:, : incoming.shape[1]] = incoming
            state_dict[key] = expanded
        for key in ("interaction_head.weight", "interaction_head.bias"):
            incoming = state_dict[key]
            expected = self.state_dict()[key]
            if incoming.shape != expected.shape:
                if incoming.shape[0] >= expected.shape[0] or incoming.shape[1:] != expected.shape[1:]:
                    raise ValueError(f"incompatible {key}: {tuple(incoming.shape)} versus {tuple(expected.shape)}")
                expanded = expected.new_zeros(expected.shape)
                expanded[: incoming.shape[0]] = incoming
                state_dict[key] = expanded
        for missing_key in (
            "topology_encoder.weight",
            "topology_encoder.bias",
            "phase_q_residual_adapter",
            "phase_q_feature_adapter",
        ):
            if missing_key not in state_dict:
                state_dict[missing_key] = self.state_dict()[missing_key]
        self.load_state_dict(state_dict)
        if reset_q_head_for_prototypes and len(self.phase_q_prototypes):
            with torch.no_grad():
                self.interaction_head.weight[:36].zero_()
                self.interaction_head.bias[:36].zero_()

    def phase_topology_condition(self, current_contact: Tensor, interaction_phase: Tensor) -> Tensor:
        if not len(self.phase_topologies):
            return current_contact.new_zeros((len(current_contact), 3 * len(CONTACT_PARTS)))
        phase_index = torch.round(
            interaction_phase.clamp(0.0, 1.0) * (len(self.phase_topologies) - 1)
        ).to(torch.long)
        scheduled = self.phase_topologies.to(current_contact.device)[phase_index]
        compatible = (scheduled[:, 0] == (current_contact > 0.5)).all(dim=-1)
        condition = torch.cat((scheduled[:, 1], scheduled[:, 2], scheduled[:, 3]), dim=-1).to(current_contact.dtype)
        return condition * compatible[:, None]

    def _apply_q_delta(
        self,
        current_qpos: Tensor,
        residual: Tensor,
        interaction_phase: Tensor | None = None,
        current_contact: Tensor | None = None,
    ) -> Tensor:
        base_qpos = current_qpos
        phase_index = None
        if len(self.phase_q_residual_adapter) and interaction_phase is not None:
            phase_index = torch.round(
                interaction_phase.clamp(0.0, 1.0) * (len(self.phase_q_residual_adapter) - 1)
            ).to(torch.long)
            residual = residual + self.phase_q_residual_adapter[phase_index]
        if len(self.phase_q_prototypes) and interaction_phase is not None:
            if phase_index is None:
                phase_index = torch.round(
                    interaction_phase.clamp(0.0, 1.0) * (len(self.phase_q_prototypes) - 1)
                ).to(torch.long)
            prototype = self.phase_q_prototypes.to(current_qpos.device)[phase_index]
            if len(self.phase_topologies) and current_contact is not None:
                scheduled = self.phase_topologies.to(current_qpos.device)[phase_index]
                compatible = (scheduled[:, 0] == (current_contact > 0.5)).all(dim=-1)
                base_qpos = torch.where(compatible[:, None], prototype, current_qpos)
            else:
                base_qpos = prototype
        root_position = base_qpos[:, :3] + self.maximum_root_translation_m * torch.tanh(residual[:, :3])
        root_quaternion = F.normalize(
            base_qpos[:, 3:7] + self.maximum_root_quaternion_delta * torch.tanh(residual[:, 3:7]), dim=-1
        )
        joints = base_qpos[:, 7:] + self.maximum_joint_delta_rad * torch.tanh(residual[:, 7:])
        joints = torch.maximum(torch.minimum(joints, self.fk.joint_upper), self.fk.joint_lower)
        return torch.cat((root_position, root_quaternion, joints), dim=-1)

    def contact_points(
        self,
        qpos: Tensor,
        contact_surface: Tensor,
        contact_local_offset: Tensor | None = None,
    ) -> Tensor:
        position, rotation6d = self.fk(qpos)
        rotation = _matrix_from_rotation6d(rotation6d)
        points = []
        for part, body in enumerate(CONTACT_BODY_INDEX):
            surface = contact_surface[:, part].clamp(0, self.surface_count - 1)
            offset = (
                self.endpoint_offsets[part, surface]
                if contact_local_offset is None
                else contact_local_offset[:, part]
            )
            points.append(position[:, body] - torch.einsum("bij,bj->bi", rotation[:, body], offset))
        return torch.stack(points, dim=1)

    @staticmethod
    def tangent_q_velocity(qpos: Tensor, q_velocity: Tensor) -> Tensor:
        """Keep quaternion coordinate velocity in the unit-sphere tangent."""

        quaternion_velocity = q_velocity[:, 3:7] - qpos[:, 3:7] * (
            qpos[:, 3:7] * q_velocity[:, 3:7]
        ).sum(dim=-1, keepdim=True)
        return torch.cat((q_velocity[:, :3], quaternion_velocity, q_velocity[:, 7:]), dim=-1)

    def contact_point_velocity(
        self,
        qpos: Tensor,
        q_velocity: Tensor,
        contact_surface: Tensor,
        contact_local_offset: Tensor,
        *,
        epsilon_s: float = 1.0e-3,
    ) -> Tensor:
        """Differentiate the same FK material points used by the keyframe."""

        q_velocity = self.tangent_q_velocity(qpos, q_velocity)
        before = qpos - epsilon_s * q_velocity
        after = qpos + epsilon_s * q_velocity
        before = torch.cat((before[:, :3], F.normalize(before[:, 3:7], dim=-1), before[:, 7:]), dim=-1)
        after = torch.cat((after[:, :3], F.normalize(after[:, 3:7], dim=-1), after[:, 7:]), dim=-1)
        return (
            self.contact_points(after, contact_surface, contact_local_offset)
            - self.contact_points(before, contact_surface, contact_local_offset)
        ) / (2.0 * epsilon_s)

    @staticmethod
    def _coordinate_to_generalized_velocity(qpos: Tensor, q_velocity: Tensor) -> Tensor:
        conjugate = torch.cat((qpos[:, 3:4], -qpos[:, 4:7]), dim=-1)
        angular_quaternion = 2.0 * _quaternion_multiply_wxyz(q_velocity[:, 3:7], conjugate)
        return torch.cat((q_velocity[:, :3], angular_quaternion[:, 1:4], q_velocity[:, 7:]), dim=-1)

    @staticmethod
    def _generalized_to_coordinate_velocity(qpos: Tensor, velocity: Tensor) -> Tensor:
        zero = velocity[:, :1] * 0.0
        angular_quaternion = torch.cat((zero, velocity[:, 3:6]), dim=-1)
        quaternion_velocity = 0.5 * _quaternion_multiply_wxyz(
            angular_quaternion, qpos[:, 3:7]
        )
        return torch.cat((velocity[:, :3], quaternion_velocity, velocity[:, 6:]), dim=-1)

    def decode_contact_consistent_velocity(
        self,
        qpos: Tensor,
        q_velocity: Tensor,
        contact_surface: Tensor,
        contact_local_offset: Tensor,
        stationary_contact: Tensor,
    ) -> Tensor:
        """Construct qdot in the feasible velocity space of material contacts."""

        q_velocity = self.tangent_q_velocity(qpos, q_velocity)
        generalized = self._coordinate_to_generalized_velocity(qpos, q_velocity)
        points = self.contact_points(qpos, contact_surface, contact_local_offset)
        jacobian = self.fk.point_geometric_jacobian(qpos, points, CONTACT_BODY_INDEX)
        row_mask = stationary_contact[..., None].expand(-1, -1, 3).reshape(len(qpos), -1).to(qpos)
        jacobian = jacobian.reshape(len(qpos), 3 * len(CONTACT_PARTS), 35) * row_mask[..., None]
        # The velocity coordinates are decoded in this instantaneous feasible
        # basis. Detaching the basis avoids unnecessary second-order FK terms;
        # the returned velocity remains differentiable with respect to the
        # predicted free coordinates.
        jacobian = jacobian.detach()
        identity = torch.eye(3 * len(CONTACT_PARTS), dtype=qpos.dtype, device=qpos.device)
        system = jacobian @ jacobian.transpose(-1, -2) + 1.0e-8 * identity[None]
        violation = jacobian @ generalized[..., None]
        correction = jacobian.transpose(-1, -2) @ torch.linalg.solve(system, violation)
        feasible = generalized - correction[..., 0]
        return self._generalized_to_coordinate_velocity(qpos, feasible)

    def boundary_stationary_contacts(
        self, persistent_support: Tensor, interaction_phase: Tensor
    ) -> Tensor:
        stationary = persistent_support
        if len(self.phase_topologies):
            phase_index = torch.round(
                interaction_phase.clamp(0.0, 1.0) * (len(self.phase_topologies) - 1)
            ).to(torch.long)
            next_index = (phase_index + 1).clamp_max(len(self.phase_topologies) - 1)
            next_persistent = self.phase_topologies.to(persistent_support.device)[next_index, 1]
            has_next = phase_index < len(self.phase_topologies) - 1
            stationary = stationary | (next_persistent & has_next[:, None])
        return stationary

    def _apply_tangent_delta(self, qpos: Tensor, delta: Tensor) -> Tensor:
        root_position = qpos[:, :3] + delta[:, :3]
        rotation_vector = delta[:, 3:6]
        angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
        vector_scale = 0.5 * torch.sinc(angle / (2.0 * torch.pi))
        delta_quaternion = torch.cat((torch.cos(0.5 * angle), vector_scale * rotation_vector), dim=-1)
        root_quaternion = F.normalize(
            _quaternion_multiply_wxyz(delta_quaternion, qpos[:, 3:7]), dim=-1
        )
        joints = qpos[:, 7:] + delta[:, 6:]
        joints = torch.maximum(torch.minimum(joints, self.fk.joint_upper), self.fk.joint_lower)
        return torch.cat((root_position, root_quaternion, joints), dim=-1)

    def constrain_persistent_contacts(
        self,
        qpos: Tensor,
        contact_surface: Tensor,
        contact_local_offset: Tensor,
        persistent: Tensor,
        anchor: Tensor,
        *,
        iterations: int | None = None,
    ) -> Tensor:
        """Impose persistent material-point constraints with differentiable DLS IK."""

        constraint_iterations = self.persistent_constraint_iterations if iterations is None else int(iterations)
        if constraint_iterations < 0:
            raise ValueError("iterations must be non-negative")
        if constraint_iterations == 0:
            return qpos
        row_mask = persistent[..., None].expand(-1, -1, 3).reshape(len(qpos), -1).to(qpos)
        identity = torch.eye(3 * len(CONTACT_PARTS), dtype=qpos.dtype, device=qpos.device)
        identity = identity.expand(len(qpos), -1, -1)
        constrained = qpos
        for _ in range(constraint_iterations):
            points = self.contact_points(constrained, contact_surface, contact_local_offset)
            jacobian = self.fk.point_geometric_jacobian(constrained, points, CONTACT_BODY_INDEX)
            jacobian = jacobian.reshape(len(qpos), 3 * len(CONTACT_PARTS), 35) * row_mask[..., None]
            residual = (anchor - points).reshape(len(qpos), -1) * row_mask
            system = jacobian @ jacobian.transpose(-1, -2)
            system = system + self.persistent_constraint_damping**2 * identity
            delta = jacobian.transpose(-1, -2) @ torch.linalg.solve(system, residual[..., None])
            delta = delta[..., 0]
            root_step = torch.linalg.vector_norm(delta[:, :3], dim=-1)
            angular_step = torch.linalg.vector_norm(delta[:, 3:], dim=-1)
            scale = torch.minimum(
                (self.persistent_constraint_maximum_step / root_step.clamp_min(1.0e-8)).clamp_max(1.0),
                (self.persistent_constraint_maximum_step / angular_step.clamp_min(1.0e-8)).clamp_max(1.0),
            )
            constrained = self._apply_tangent_delta(constrained, scale[:, None] * delta)
        return constrained

    def synchronize_contact_targets(
        self,
        prediction: UnifiedInteractionPrediction,
        contact_target: Tensor,
    ) -> UnifiedInteractionPrediction:
        """Decode one q/contact keyframe against a shared set of contact targets.

        ``contact_target`` may be discretely selected from the terrain scan, but
        it never replaces FK geometry on its own.  The constrained q is the
        authoritative keyframe and all body/contact geometry is recomputed from
        that same q, keeping pose and contact inseparable.
        """

        if contact_target.shape != prediction.contact_position.shape:
            raise ValueError("contact_target must have shape [B,6,3]")
        constrained_contact = prediction.persistent_support | prediction.touchdown
        qpos = self.constrain_persistent_contacts(
            prediction.qpos,
            prediction.contact_surface,
            prediction.contact_local_offset,
            constrained_contact,
            contact_target,
        )
        position, rotation6d = self.fk(qpos)
        contact_position = self.contact_points(
            qpos, prediction.contact_surface, prediction.contact_local_offset
        )
        q_velocity = self.decode_contact_consistent_velocity(
            qpos,
            prediction.q_velocity,
            prediction.contact_surface,
            prediction.contact_local_offset,
            prediction.stationary_contact,
        )
        return UnifiedInteractionPrediction(
            qpos=qpos,
            q_velocity=q_velocity,
            stationary_contact=prediction.stationary_contact,
            keypoint_position=position,
            keypoint_rotation6d=rotation6d,
            contact_logits=prediction.contact_logits,
            active_contact=prediction.active_contact,
            touchdown_logits=prediction.touchdown_logits,
            touchdown=prediction.touchdown,
            persistent_logits=prediction.persistent_logits,
            persistent_support=prediction.persistent_support,
            surface_logits=prediction.surface_logits,
            contact_surface=prediction.contact_surface,
            contact_local_offset=prediction.contact_local_offset,
            contact_position=contact_position,
            contact_velocity=self.contact_point_velocity(
                qpos,
                q_velocity,
                prediction.contact_surface,
                prediction.contact_local_offset,
            ),
            duration=prediction.duration,
        )

    def decode_topology(
        self,
        current_contact: Tensor,
        contact_logits: Tensor,
        persistent_logits: Tensor,
        touchdown_logits: Tensor,
        interaction_phase: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Jointly select the most likely transition supported by training data."""

        if not len(self.topology_transitions):
            active = contact_logits >= 0.0
            persistent = (persistent_logits >= 0.0) & (current_contact > 0.5) & active
            return active, persistent, active & ~persistent
        transition = self.topology_transitions.to(contact_logits.device)
        score = self.topology_scores(current_contact, contact_logits, persistent_logits, touchdown_logits)
        selected = transition[score.argmax(dim=-1)]
        if len(self.phase_topologies) and interaction_phase is not None:
            phase_index = torch.round(
                interaction_phase.clamp(0.0, 1.0) * (len(self.phase_topologies) - 1)
            ).to(torch.long)
            scheduled = self.phase_topologies.to(contact_logits.device)[phase_index]
            compatible = (scheduled[:, 0] == (current_contact > 0.5)).all(dim=-1)
            selected = torch.where(compatible[:, None, None], scheduled, selected)
        return selected[:, 3], selected[:, 1], selected[:, 2]

    def topology_scores(
        self,
        current_contact: Tensor,
        contact_logits: Tensor,
        persistent_logits: Tensor,
        touchdown_logits: Tensor,
    ) -> Tensor:
        if not len(self.topology_transitions):
            raise ValueError("topology scores require a non-empty transition table")
        transition = self.topology_transitions.to(contact_logits.device)
        candidate = torch.stack((transition[:, 3], transition[:, 1], transition[:, 2]), dim=1)
        logits = torch.stack((contact_logits, persistent_logits, touchdown_logits), dim=1)
        log_likelihood = (
            candidate[None] * F.logsigmoid(logits[:, None])
            + ~candidate[None] * F.logsigmoid(-logits[:, None])
        ).sum(dim=(2, 3))
        current = current_contact > 0.5
        start_mismatch = (transition[None, :, 0] != current[:, None]).sum(dim=-1)
        return log_likelihood - 100.0 * start_mismatch

    def forward(
        self,
        current_qpos: Tensor,
        current_contact: Tensor,
        current_contact_position: Tensor,
        current_contact_surface: Tensor,
        scan: Tensor,
        previous_persistent: Tensor | None = None,
        previous_touchdown: Tensor | None = None,
        interaction_phase: Tensor | None = None,
        current_q_velocity: Tensor | None = None,
        *,
        apply_persistent_constraint: bool = True,
        current_contact_local_offset: Tensor | None = None,
    ) -> UnifiedInteractionPrediction:
        if current_contact_position.shape[-2:] != (len(CONTACT_PARTS), 3):
            raise ValueError("current_contact_position must have shape [B,6,3]")
        if previous_persistent is None:
            previous_persistent = torch.zeros_like(current_contact)
        if previous_touchdown is None:
            previous_touchdown = torch.zeros_like(current_contact)
        if interaction_phase is None:
            interaction_phase = current_qpos.new_zeros((len(current_qpos),))
        if current_q_velocity is None:
            current_q_velocity = torch.zeros_like(current_qpos)
        if current_q_velocity.shape != current_qpos.shape:
            raise ValueError("current_q_velocity and current_qpos must share [B,36]")
        if self.predict_boundary_velocity:
            current_q_velocity = self.tangent_q_velocity(current_qpos, current_q_velocity)
        if previous_persistent.shape != current_contact.shape or previous_touchdown.shape != current_contact.shape:
            raise ValueError("previous interaction masks must match current_contact")
        if interaction_phase.shape != (len(current_qpos),):
            raise ValueError("interaction_phase must have shape [B]")
        q_feature = ((current_qpos - self.q_mean) / self.q_std).clamp(-10.0, 10.0)
        anchors = current_contact_position * current_contact[..., None]
        if (((current_contact_surface < 0) | (current_contact_surface >= self.surface_count)) &
                current_contact.bool()).any():
            raise ValueError('Observed Newton surface is outside predictor vocabulary; cannot clamp it into ground/top')
        current_surface = F.one_hot(
            current_contact_surface.clamp(0, self.surface_count - 1), self.surface_count
        ).to(current_qpos.dtype) * current_contact[..., None]
        current_position, current_rotation6d = self.fk(current_qpos)
        current_rotation = _matrix_from_rotation6d(current_rotation6d)
        current_local_offsets = []
        for part, body in enumerate(CONTACT_BODY_INDEX):
            surface = current_contact_surface[:, part].clamp(0, self.surface_count - 1)
            measured = torch.einsum(
                "bij,bj->bi",
                current_rotation[:, body].transpose(-1, -2),
                current_position[:, body] - current_contact_position[:, part],
            )
            canonical = self.endpoint_offsets[part, surface]
            inferred = torch.where(current_contact[:, part : part + 1] > 0.5, measured, canonical)
            if self.remember_material_contacts and current_contact_local_offset is not None:
                if current_contact_local_offset.shape != current_contact_position.shape:
                    raise ValueError('Material point memory must have shape [B,6,3]')
                inferred = current_contact_local_offset[:, part]
            current_local_offsets.append(canonical if self.fixed_material_contacts else inferred)
        current_local_offset = torch.stack(current_local_offsets, dim=1)
        state_values = (
            q_feature,
            current_contact,
            anchors.flatten(1),
            current_local_offset.flatten(1),
            current_surface.flatten(1),
            previous_persistent,
            previous_touchdown,
        )
        if self.use_phase_condition:
            state_values += (interaction_phase[:, None],)
        if self.predict_boundary_velocity:
            state_values += (current_q_velocity.clamp(-10.0, 10.0),)
        state = self.state_encoder(torch.cat(state_values, dim=-1))
        scan_feature = self.scan_encoder((scan - self.scan_mean) / self.scan_std)
        hidden = self.trunk(torch.cat((state, scan_feature), dim=-1))
        if self.use_phase_condition:
            hidden = hidden + self.topology_encoder(self.phase_topology_condition(current_contact, interaction_phase))
        raw = self.interaction_head(hidden)
        cursor = 0
        q_residual = raw[:, cursor : cursor + 36]
        cursor += 36
        if len(self.phase_q_feature_adapter):
            phase_index = torch.round(
                interaction_phase.clamp(0.0, 1.0) * (len(self.phase_q_feature_adapter) - 1)
            ).to(torch.long)
            q_residual = q_residual + torch.einsum(
                "bi,boi->bo", hidden, self.phase_q_feature_adapter[phase_index]
            )
        contact_logits = raw[:, cursor : cursor + len(CONTACT_PARTS)] + 2.0 * (2.0 * current_contact - 1.0)
        cursor += len(CONTACT_PARTS)
        touchdown_logits = raw[:, cursor : cursor + len(CONTACT_PARTS)] - 2.0
        cursor += len(CONTACT_PARTS)
        persistent_logits = raw[:, cursor : cursor + len(CONTACT_PARTS)] + 2.0 * (2.0 * current_contact - 1.0)
        cursor += len(CONTACT_PARTS)
        surface_logits = raw[:, cursor : cursor + len(CONTACT_PARTS) * self.surface_count].reshape(
            -1, len(CONTACT_PARTS), self.surface_count
        ) + 2.0 * current_surface
        cursor += len(CONTACT_PARTS) * self.surface_count
        contact_offset_residual = raw[:, cursor : cursor + len(CONTACT_PARTS) * 3].reshape(
            -1, len(CONTACT_PARTS), 3
        )
        cursor += len(CONTACT_PARTS) * 3
        duration = torch.exp(raw[:, cursor] * self.duration_log_std + self.duration_log_mean).clamp(
            self.minimum_duration_s, self.maximum_duration_s
        )
        cursor += 1
        q_velocity = torch.zeros_like(current_qpos)
        if self.predict_boundary_velocity:
            q_velocity_scale = current_qpos.new_tensor(
                (0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0) + (3.0,) * 29
            )
            q_velocity = q_velocity_scale * torch.tanh(raw[:, cursor : cursor + 36])
        if self.predict_boundary_velocity and len(self.phase_q_velocity_prototypes):
            velocity_phase_index = torch.round(
                interaction_phase.clamp(0.0, 1.0) * (len(self.phase_q_velocity_prototypes) - 1)
            ).to(torch.long)
            q_velocity = q_velocity + self.phase_q_velocity_prototypes.to(q_velocity.device)[
                velocity_phase_index
            ]
        active_contact, persistent_support, touchdown = self.decode_topology(
            current_contact, contact_logits, persistent_logits, touchdown_logits, interaction_phase
        )
        persistent = persistent_support & (current_contact > 0.5)
        qpos = self._apply_q_delta(current_qpos, q_residual, interaction_phase, current_contact)
        contact_surface = surface_logits.argmax(dim=-1)
        contact_surface = torch.where(persistent, current_contact_surface, contact_surface)
        predicted_canonical_offset = self.endpoint_offsets[
            torch.arange(len(CONTACT_PARTS), device=qpos.device)[None], contact_surface
        ]
        offset_base = torch.where(
            current_contact[..., None] > 0.5, current_local_offset, predicted_canonical_offset
        )
        if self.remember_material_contacts:
            # A new touchdown starts a new material-point choice. Adding to the
            # previous point on every touchdown makes an unbounded random walk.
            # Persistent contacts still preserve memory exactly below.
            offset_base = predicted_canonical_offset
        contact_local_offset = offset_base + self.maximum_contact_offset_delta_m * torch.tanh(contact_offset_residual)
        contact_local_offset = torch.where(persistent[..., None], current_local_offset, contact_local_offset)
        if self.fixed_material_contacts:
            contact_local_offset = predicted_canonical_offset
        if apply_persistent_constraint:
            qpos = self.constrain_persistent_contacts(
                qpos, contact_surface, contact_local_offset, persistent, current_contact_position
            )
        position, rotation6d = self.fk(qpos)
        contact_position = self.contact_points(qpos, contact_surface, contact_local_offset)
        stationary_contact = self.boundary_stationary_contacts(
            persistent_support, interaction_phase
        )
        contact_velocity = torch.zeros_like(contact_position)
        if self.predict_boundary_velocity:
            q_velocity = self.tangent_q_velocity(qpos, q_velocity)
            q_velocity = self.decode_contact_consistent_velocity(
                qpos, q_velocity, contact_surface, contact_local_offset, stationary_contact
            )
            contact_velocity = self.contact_point_velocity(
                qpos, q_velocity, contact_surface, contact_local_offset
            )
        return UnifiedInteractionPrediction(
            qpos=qpos,
            q_velocity=q_velocity,
            stationary_contact=stationary_contact,
            keypoint_position=position,
            keypoint_rotation6d=rotation6d,
            contact_logits=contact_logits,
            active_contact=active_contact,
            touchdown_logits=touchdown_logits,
            touchdown=touchdown,
            persistent_logits=persistent_logits,
            persistent_support=persistent_support,
            surface_logits=surface_logits,
            contact_surface=contact_surface,
            contact_local_offset=contact_local_offset,
            contact_position=contact_position,
            contact_velocity=contact_velocity,
            duration=duration,
        )


def unified_interaction_loss(
    prediction: UnifiedInteractionPrediction,
    *,
    target_qpos: Tensor,
    target_contact: Tensor,
    target_touchdown: Tensor,
    target_persistent: Tensor,
    target_surface: Tensor,
    target_contact_position: Tensor,
    target_duration: Tensor,
    current_contact: Tensor,
    current_contact_position: Tensor,
    target_contact_rotation6d: Tensor | None = None,
    contact_orientation_weight: float = 0.0,
    balance_contact_parts: bool = False,
    active_geometry_weight: float = 2.0,
    persistent_anchor_weight: float = 5.0,
    topology_choice_penalty: Tensor | None = None,
    surface_feasibility_penalty: Tensor | None = None,
    inactive_clearance_penalty: Tensor | None = None,
    target_q_velocity: Tensor | None = None,
    q_velocity_weight: float = 1.0,
    contact_velocity_weight: float = 5.0,
    target_stationary_contact: Tensor | None = None,
) -> UnifiedInteractionLoss:
    """Supervise pose and contact together, including support-anchor invariance."""

    q_pose = (
        ((prediction.qpos[:, :3] - target_qpos[:, :3]) / 0.02).square().mean()
        + (1.0 - (prediction.qpos[:, 3:7] * target_qpos[:, 3:7]).sum(dim=-1).abs()).mean()
        + ((prediction.qpos[:, 7:] - target_qpos[:, 7:]) / 0.1).square().mean()
    )
    contact_state = F.binary_cross_entropy_with_logits(prediction.contact_logits, target_contact)
    touchdown = F.binary_cross_entropy_with_logits(prediction.touchdown_logits, target_touchdown)
    persistent_state = F.binary_cross_entropy_with_logits(prediction.persistent_logits, target_persistent)
    active = target_contact > 0.5
    valid_geometry = active & (target_surface >= 0)
    def contact_part_mean(value: Tensor, mask: Tensor) -> Tensor:
        if not balance_contact_parts:
            return value[mask].mean()
        numeric_mask = mask.to(value.dtype)
        count = numeric_mask.sum(dim=0)
        mean = (value * numeric_mask).sum(dim=0) / count.clamp_min(1.0)
        valid_part = count > 0.0
        return (mean * valid_part).sum() / valid_part.sum().clamp_min(1)

    if bool(valid_geometry.any()):
        surface = F.cross_entropy(prediction.surface_logits[valid_geometry], target_surface[valid_geometry])
        geometry_error = (
            (prediction.contact_position - target_contact_position) / 0.01
        ).square().mean(dim=-1)
        active_geometry = contact_part_mean(geometry_error, valid_geometry)
    else:
        surface = prediction.surface_logits.sum() * 0.0
        active_geometry = prediction.contact_position.sum() * 0.0
    persistent = valid_geometry & (target_persistent > 0.5)
    if bool(persistent.any()):
        persistent_error = (
            (prediction.contact_position - current_contact_position) / 0.005
        ).square().mean(dim=-1)
        persistent_anchor = contact_part_mean(persistent_error, persistent)
    else:
        persistent_anchor = prediction.contact_position.sum() * 0.0
    duration = ((torch.log(prediction.duration) - torch.log(target_duration)) / 0.1).square().mean()
    topology_choice = prediction.qpos.sum() * 0.0 if topology_choice_penalty is None else topology_choice_penalty
    surface_feasibility = (
        prediction.qpos.sum() * 0.0 if surface_feasibility_penalty is None else surface_feasibility_penalty
    )
    inactive_clearance = (
        prediction.qpos.sum() * 0.0 if inactive_clearance_penalty is None else inactive_clearance_penalty
    )
    if topology_choice.ndim != 0 or surface_feasibility.ndim != 0 or inactive_clearance.ndim != 0:
        raise ValueError("geometry penalties must be scalar")
    if min(contact_orientation_weight, active_geometry_weight, persistent_anchor_weight) < 0.0:
        raise ValueError("contact loss weights must be non-negative")
    if target_contact_rotation6d is None:
        contact_orientation = prediction.qpos.sum() * 0.0
    else:
        contact_orientation, _ = contact_rotation_losses(
            prediction.keypoint_rotation6d[:, CONTACT_BODY_INDEX],
            target_contact_rotation6d,
            valid_geometry,
            balance_contact_parts=balance_contact_parts,
        )
    if target_q_velocity is None:
        q_velocity = prediction.q_velocity.sum() * 0.0
    else:
        velocity_scale = prediction.q_velocity.new_tensor(
            (0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5) + (1.0,) * 29
        )
        q_velocity = ((prediction.q_velocity - target_q_velocity) / velocity_scale).square().mean()
    stationary = target_persistent > 0.5 if target_stationary_contact is None else target_stationary_contact > 0.5
    if bool(stationary.any()):
        contact_speed = prediction.contact_velocity.square().sum(dim=-1)
        contact_velocity = contact_part_mean(contact_speed / 0.05**2, stationary)
    else:
        contact_velocity = prediction.contact_velocity.sum() * 0.0
    if min(q_velocity_weight, contact_velocity_weight) < 0.0:
        raise ValueError("velocity loss weights must be non-negative")
    total = q_pose + contact_state + touchdown + persistent_state + surface
    total = total + active_geometry_weight * active_geometry
    total = total + persistent_anchor_weight * persistent_anchor
    total = total + contact_orientation_weight * contact_orientation
    total = total + q_velocity_weight * q_velocity + contact_velocity_weight * contact_velocity
    total = total + duration + 2.0 * topology_choice + 5.0 * surface_feasibility + 10.0 * inactive_clearance
    return UnifiedInteractionLoss(
        total,
        q_pose,
        contact_state,
        touchdown,
        persistent_state,
        surface,
        active_geometry,
        persistent_anchor,
        duration,
        topology_choice,
        surface_feasibility,
        inactive_clearance,
        contact_orientation,
        q_velocity,
        contact_velocity,
    )




from contact_solver.trajectory_projection import _trajectory_contact_points


from contact_solver.trajectory_projection import project_interaction_q_trajectory
