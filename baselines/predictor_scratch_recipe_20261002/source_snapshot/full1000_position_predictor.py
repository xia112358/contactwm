"""Full1000 backbone with observed contact positions instead of surface IDs."""
from dataclasses import dataclass
import copy

import torch
from torch import nn

from generator.conditioned_pose_predictor import ConditionedHeightmapPosePredictor
from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _rotation6d
from somaforge_core.heightmap import heightmap_grid, HEIGHTMAP_ROWS, HEIGHTMAP_COLS, HEIGHTMAP_FORWARD_MIN_M, HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_RESOLUTION_M
from somaforge_core.heightmap import _root_yaw_basis
from generator.planned_contact_predictor import PlannedContactPrediction


@dataclass(frozen=True)
class Full1000PositionPrediction(PlannedContactPrediction):
    teacher_conditioned: torch.Tensor
    region_logits: torch.Tensor | None = None
    planned_regions: torch.Tensor | None = None
    conditioned_role: torch.Tensor | None = None


class Full1000PositionPredictor(ConditionedHeightmapPosePredictor):
    """Preserve the original state/terrain/pose architecture and explicit plan.

    Only current Newton contact points and the heightmap enter forward. The
    hard planned point is an observed grid sample; no semantic face category
    exists in either the model parameters or the forward interface.
    """
    EXECUTION_PREFIXES = ('execution_', 'plan_position_encoder.', 'region_plan_encoder.',
                          'plan_fusion.', 'pose_terrain.', 'interaction_decoder.', 'pose_head.')
    PLAN_HEAD_PREFIXES = ('role_head.', 'location_query.', 'location_key.', 'region_head.', 'region_prior')

    def __init__(self, width=192, layers=3, location_width=32, *, body_geometry=False, part_geometry=False,
                 region_plan=False, unified_contact=False, execution_plan_gradients=False, event_roles=False,
                 execution_observation_gradients=False):
        super().__init__(width, layers)
        if execution_plan_gradients:
            raise ValueError('Execution gradients into planning are disabled by the isolated contract')
        if body_geometry and part_geometry:
            raise ValueError('Choose one geometry observation ablation')
        self.surface_embedding = None
        self.plan_surface_embedding = None
        self.role_head = nn.Linear(width, 4)
        self.location_query = nn.Linear(width, location_width)
        self.location_key = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, location_width))
        self.plan_position_encoder = nn.Linear(4, width)
        self.event_roles = event_roles
        if event_roles:
            self.execution_role_encoder = nn.Embedding(4, width)
            nn.init.zeros_(self.execution_role_encoder.weight)
        self.body_geometry = body_geometry
        self.part_geometry = part_geometry
        self.region_plan = region_plan
        self.execution_plan_gradients = execution_plan_gradients
        self.execution_observation_gradients = execution_observation_gradients
        if region_plan or unified_contact:
            if body_geometry or part_geometry:
                raise ValueError('Regional experiment uses its own compact geometry input')
            from contact_solver.contact_regions import ContactRegions
            self.region_geometry = ContactRegions(self.fk)
        if region_plan:
            self.region_geometry_encoder = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, width))
            self.region_head = nn.Linear(width, 15)
            self.region_prior = nn.Parameter(torch.zeros(6, 15))
            self.region_plan_encoder = nn.Linear(4, width, bias=False)
            nn.init.zeros_(self.region_geometry_encoder[-1].weight)
            nn.init.zeros_(self.region_geometry_encoder[-1].bias)
            nn.init.zeros_(self.region_plan_encoder.weight)
            nn.init.zeros_(self.region_head.weight)
            nn.init.zeros_(self.region_head.bias)
        if part_geometry:
            from contact_solver.part_collision_geometry import PartCollisionGeometry
            self.part_geometry_observation = PartCollisionGeometry(self.fk)
            self.part_geometry_encoder = nn.Sequential(
                nn.Linear(PartCollisionGeometry.feature_count, width), nn.GELU(), nn.Linear(width, width))
            nn.init.zeros_(self.part_geometry_encoder[-1].weight)
            nn.init.zeros_(self.part_geometry_encoder[-1].bias)
        if body_geometry:
            names = tuple(sorted(set(self.geometry.link_names)))
            self.body_geometry_names = names
            lookup = {name: i for i, name in enumerate(names)}
            ids = torch.cat([torch.full((count,), lookup[name], dtype=torch.long)
                for name, count in zip(self.geometry.link_names, self.geometry.point_counts)])
            self.register_buffer('body_geometry_point_link', ids, persistent=False)
            self.body_geometry_encoder = nn.Sequential(nn.Linear(13, width), nn.GELU(), nn.Linear(width, width))
            self.body_geometry_identity = nn.Embedding(len(names), width)
            self.body_geometry_norm = nn.LayerNorm(width)
            self.body_geometry_attention = nn.MultiheadAttention(width, 6, batch_first=True)
            self.body_geometry_output = nn.Linear(width, width)
            nn.init.zeros_(self.body_geometry_output.weight)
            nn.init.zeros_(self.body_geometry_output.bias)

        # Observation gradients are configurable independently of hard plans.
        # Zero residuals preserve migrated predictions.
        self.execution_adapter = nn.Linear(width, width)
        nn.init.zeros_(self.execution_adapter.weight)
        nn.init.zeros_(self.execution_adapter.bias)
        self.execution_norm = copy.deepcopy(self.norm)
        geometry_width = 32 if region_plan else (PartCollisionGeometry.feature_count if part_geometry else (13 if body_geometry else 0))
        self.execution_geometry_encoder = nn.Linear(geometry_width, width) if geometry_width else None
        if self.execution_geometry_encoder is not None:
            nn.init.zeros_(self.execution_geometry_encoder.weight)
            nn.init.zeros_(self.execution_geometry_encoder.bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Legacy checkpoints have one shared norm. Copy its actual weights
        # and initialize only new execution residuals to zero, even on reload.
        if not any(name.startswith('execution_') for name in state_dict):
            state_dict = copy.copy(state_dict)
            for name, value in self.state_dict().items():
                if name.startswith('execution_norm.'):
                    state_dict[name] = state_dict[name.replace('execution_norm.', 'norm.', 1)].clone()
                elif name.startswith('execution_'):
                    state_dict[name] = torch.zeros_like(value)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def training_parameter_groups(self):
        groups = {'planner': [], 'executor': []}
        if self.execution_observation_gradients:
            groups['shared'] = []
        for name, parameter in self.named_parameters():
            if name.startswith(self.EXECUTION_PREFIXES):
                group = 'executor'
            elif self.execution_observation_gradients and not name.startswith(self.PLAN_HEAD_PREFIXES):
                group = 'shared'
            else:
                group = 'planner'
            groups[group].append(parameter)
        return groups

    def clip_training_gradients(self, max_norm=10.):
        # Clip heads and shared representation separately: an execution-driven
        # shared gradient must not rescale the planning-head gradient.
        # torch returns a CPU zero when a group has no gradients (e.g. the
        # frozen planner during execution-only adaptation).
        return {name: nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True).to(parameters[0].device)
                for name, parameters in self.training_parameter_groups().items()}

    def geometry_features(self, current_q, heightmap, basis):
        """Per-link collision extent and observed terrain clearance features.

        All points come from the canonical collision asset. Height differences
        are observation features only, never a contact predicate. Out-of-view
        points are explicitly masked rather than assigned a border height.
        """
        points, _ = self.geometry(self.fk, current_q)
        local = torch.einsum('bij,bpj->bpi', basis.transpose(1, 2), points-current_q[:, None, :3])
        b, _, _ = local.shape
        n = len(self.body_geometry_names)
        idx = self.body_geometry_point_link[None, :, None].expand(b, -1, 3)
        low = local.new_full((b, n, 3), torch.inf).scatter_reduce(1, idx, local, reduce='amin', include_self=True)
        high = local.new_full((b, n, 3), -torch.inf).scatter_reduce(1, idx, local, reduce='amax', include_self=True)
        total = local.new_zeros(b, n, 3).scatter_add(1, idx, local)
        ids = self.body_geometry_point_link[None].expand(b, -1)
        count = local.new_zeros(b, n).scatter_add(1, ids, torch.ones_like(local[..., 0]))
        mean = total/count[..., None]
        x = (local[..., 0]-HEIGHTMAP_FORWARD_MIN_M)/HEIGHTMAP_RESOLUTION_M
        y = (local[..., 1]-HEIGHTMAP_LATERAL_MIN_M)/HEIGHTMAP_RESOLUTION_M
        visible = (x >= 0) & (x <= HEIGHTMAP_ROWS-1) & (y >= 0) & (y <= HEIGHTMAP_COLS-1)
        cell = x.round().long().clamp(0, HEIGHTMAP_ROWS-1)*HEIGHTMAP_COLS+y.round().long().clamp(0, HEIGHTMAP_COLS-1)
        gap = local[..., 2]-heightmap.flatten(1).gather(1, cell)
        observed = local.new_zeros(b, n).scatter_add(1, ids, visible.to(local))
        gap_mean = local.new_zeros(b, n).scatter_add(1, ids, torch.where(visible, gap, 0))/observed.clamp_min(1)
        gap_min = local.new_full((b, n), torch.inf).scatter_reduce(1, ids,
            torch.where(visible, gap, torch.inf), reduce='amin', include_self=True)
        gap_max = local.new_full((b, n), -torch.inf).scatter_reduce(1, ids,
            torch.where(visible, gap, -torch.inf), reduce='amax', include_self=True)
        gap_min = torch.where(observed > 0, gap_min, 0)
        gap_max = torch.where(observed > 0, gap_max, 0)
        return torch.cat((mean, low, high, gap_mean[..., None], gap_min[..., None],
                          gap_max[..., None], (observed/count)[..., None]), -1)

    def _encode_points(self, current_q, current_contact, current_anchor, heightmap):
        geometry_features = None
        terrain = self.height(heightmap)
        basis, yaw = _root_yaw_basis(current_q)
        positions, rotations6d = self.fk(current_q[:, None])
        positions = positions[:, 0, 1:7]
        rotations = _matrix_from_rotation6d(rotations6d[:, 0, 1:7])
        local_position = torch.einsum('bij,bpj->bpi', basis.transpose(1, 2), positions - current_q[:, None, :3])
        local_rotation = basis[:, None].transpose(-1, -2) @ rotations
        local_anchor = torch.einsum('bij,bpj->bpi', basis.transpose(1, 2), current_anchor - current_q[:, None, :3])
        part = self.part_encoder(torch.cat((local_position, _rotation6d(local_rotation),
            local_anchor * current_contact[..., None], current_contact[..., None].float()), -1))
        part = part + self.part_identity.weight[None]
        if self.part_geometry:
            features = self.part_geometry_observation(local_position, local_rotation, heightmap)
            geometry_features = features
            part = part+self.part_geometry_encoder(features)
        if self.region_plan:
            features = self.region_geometry(local_position, local_rotation, heightmap)
            geometry_features = features
            part = part+self.region_geometry_encoder(features)
        state = self.global_encoder(self._local_state(current_q, basis))[:, None]
        queries = torch.cat((state, part), 1)
        if self.body_geometry:
            features = self.geometry_features(current_q, heightmap, basis)
            geometry_features = features
            memory = self.body_geometry_norm(self.body_geometry_encoder(features)
                                             + self.body_geometry_identity.weight[None])
            extra = self.body_geometry_attention(self.body_geometry_norm(queries), memory, memory,
                                                need_weights=False)[0]
            queries = queries+self.body_geometry_output(extra)
        for block in self.terrain_reasoning:
            queries = block(queries, terrain)
        encoded = self.shared(queries)
        return terrain, basis, yaw, self.norm(encoded[:, :1]), self.norm(encoded[:, 1:]), geometry_features

    def forward(self, current_q, current_contact, current_anchor, heightmap, *,
                teacher_contact=None, teacher_cell=None, teacher_mask=None, teacher_role=None,
                repair_mask=None, repair_role=None, repair_points_world=None, repair_regions=None):
        terrain, basis, yaw, body, part, geometry_features = self._encode_points(current_q, current_contact, current_anchor, heightmap)
        grid = torch.as_tensor(heightmap_grid(), device=heightmap.device, dtype=heightmap.dtype).reshape(-1, 2)
        geometry = torch.cat((grid[None].expand(len(current_q), -1, -1), heightmap.flatten(1)[..., None]), -1)
        # Hard plan heads remain supervised only by planning losses.
        role = self.role_head(part)
        keys = self.location_key(geometry)
        logits = torch.einsum('bpd,bnd->bpn', self.location_query(part), keys) / keys.shape[-1] ** .5
        contact, cell = role.argmax(-1) != 0, logits.argmax(-1)
        conditioned_role = role.argmax(-1)
        predicted_points = geometry.gather(1, cell[..., None].expand(-1, -1, 3))
        if teacher_mask is None:
            teacher_mask = torch.zeros(len(current_q), dtype=torch.bool, device=current_q.device)
        if teacher_mask.shape != (len(current_q),) or teacher_mask.dtype != torch.bool:
            raise ValueError('teacher mask must be boolean [B]')
        if bool(teacher_mask.any()):
            if self.event_roles:
                if teacher_role is None or teacher_role.shape != contact.shape:
                    raise ValueError('Event-role conditioning requires explicit teacher roles')
                if teacher_role.dtype != torch.long or bool(((teacher_role < 0) | (teacher_role > 3)).any()):
                    raise ValueError('Teacher roles must be long values in [0,3]')
                if teacher_contact is None or not torch.equal((teacher_role != 0)[teacher_mask], teacher_contact[teacher_mask]):
                    raise ValueError('Teacher role/contact mismatch')
                conditioned_role = torch.where(teacher_mask[:, None], teacher_role, conditioned_role)
            if teacher_contact is None or teacher_cell is None:
                raise ValueError('teacher samples require observed contact and cell labels')
            if teacher_contact.dtype != torch.bool or teacher_contact.shape != contact.shape or teacher_cell.shape != cell.shape:
                raise ValueError('teacher contact/cell shape mismatch')
            contact = torch.where(teacher_mask[:, None], teacher_contact, contact)
            cell = torch.where(teacher_mask[:, None], teacher_cell, cell)
        points = geometry.gather(1, cell[..., None].expand(-1, -1, 3))
        region_logits, planned_regions = None, None
        if self.region_plan:
            region_logits = (self.region_head(part)+self.region_prior).masked_fill(~self.region_geometry.valid_subsets, -1e4)
            subsets = self.region_geometry.subsets.to(part)
            hard = subsets[region_logits.argmax(-1)]
            planned_regions = hard.bool() & contact[..., None]

        if repair_mask is not None:
            if not self.event_roles or repair_mask.dtype != torch.bool or repair_mask.shape != (len(current_q),):
                raise ValueError('Plan repair requires event roles and a boolean batch mask')
            if bool(repair_mask.any()):
                if repair_role is None or repair_points_world is None:
                    raise ValueError('Repair requires the previously issued role and world points')
                if repair_role.shape != contact.shape or repair_role.dtype != torch.long or repair_points_world.shape != points.shape:
                    raise ValueError('Invalid stored plan shapes/dtypes')
                if bool(((repair_role < 0) | (repair_role > 3)).any()) or not bool(torch.isfinite(repair_points_world).all()):
                    raise ValueError('Invalid stored plan values')
                local = torch.einsum('bji,bpj->bpi', basis,
                    repair_points_world.detach()-current_q[:, None, :3])
                conditioned_role = torch.where(repair_mask[:, None], repair_role, conditioned_role)
                contact = conditioned_role != 0
                points = torch.where(repair_mask[:, None, None], local, points)
                cell = torch.where(repair_mask[:, None], -1, cell)
                if self.region_plan:
                    if repair_regions is None or repair_regions.shape != planned_regions.shape or repair_regions.dtype != torch.bool:
                        raise ValueError('Regional repair requires stored regional intent')
                    planned_regions = torch.where(repair_mask[:, None, None], repair_regions, planned_regions)

        # Share observation learning when enabled, while keeping explicit
        # contact/position/role/region intent detached below.
        if not self.execution_observation_gradients:
            terrain, body, part = terrain.detach(), body.detach(), part.detach()
        body = body+self.execution_adapter(body)
        part = part+self.execution_adapter(part)
        if geometry_features is not None:
            geometry_delta = self.execution_geometry_encoder(geometry_features.detach())
            if self.body_geometry:
                body = body+geometry_delta.mean(1, keepdim=True)
            else:
                part = part+geometry_delta
        plan = torch.cat((contact[..., None].to(part), points*contact[..., None]), -1).detach()
        planned_part = part+self.plan_position_encoder(plan)
        if self.event_roles:
            planned_part = planned_part+self.execution_role_encoder(conditioned_role.detach())
        if self.region_plan:
            planned_part = planned_part+self.region_plan_encoder(planned_regions.to(part).detach())
        planned_part = planned_part + self.plan_fusion(planned_part)
        interaction = self.pose_terrain(torch.cat((body, planned_part), 1), terrain)
        interaction = self.interaction_decoder(interaction)
        pooled = self.execution_norm(interaction[:, 0] + interaction[:, 1:].mean(1))
        qpos = self._decode_pose(self.pose_head(pooled), current_q, basis, yaw)
        return Full1000PositionPrediction(qpos, role, logits, predicted_points,
                                         contact, cell, points, teacher_mask, region_logits, planned_regions, conditioned_role)
