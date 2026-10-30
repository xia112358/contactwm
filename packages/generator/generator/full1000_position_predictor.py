"""V1 control architecture for explicit scratch training and checkpoint reading."""
import copy

import torch
from torch import nn

from generator.conditioned_pose_predictor import ConditionedHeightmapPosePredictor
from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _rotation6d
from somaforge_core.heightmap import _root_yaw_basis
from generator.position_plan import Full1000PositionPrediction, select_position_plan


class Full1000PositionPredictor(ConditionedHeightmapPosePredictor):
    """Preserve the original state/terrain/pose architecture and explicit plan.

    Depth mode encodes the ordered optical-depth image and actual Newton
    contact points. The hard planned point is a raw pixel's world XYZ vertex;
    historical checkpoints retain their recorded heightmap encoder. Surface identity is used
    only by the deterministic validity constraint, never by learned encoders.
    """
    schema = 'full1000_position_predictor_v1'
    EXECUTION_PREFIXES = ('execution_', 'plan_position_encoder.', 'region_plan_encoder.',
                          'plan_fusion.', 'pose_terrain.', 'interaction_decoder.', 'pose_head.')
    PLAN_HEAD_PREFIXES = ('role_head.', 'location_query.', 'location_key.', 'location_scan_key.', 'region_head.', 'region_prior')

    def __init__(self, width=192, layers=3, location_width=32, *, body_geometry=False, part_geometry=False,
                 region_plan=False, unified_contact=False, execution_plan_gradients=False, event_roles=False,
                 execution_observation_gradients=False, role_validity=False, world_surface_scan=False,
                 depth_observation=False, limb_contact_modes=False):
        super().__init__(width, layers)
        self.width, self.layers, self.location_width = width, layers, location_width
        self.unified_contact = unified_contact
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
        self.limb_contact_modes = limb_contact_modes
        if limb_contact_modes and not (event_roles and role_validity):
            raise ValueError("Limb contact modes require event roles and actual-contact validity")
        self.event_roles = event_roles
        self.role_validity = role_validity
        self.world_surface_scan = world_surface_scan
        self.depth_observation = depth_observation
        if depth_observation:
            if not world_surface_scan or body_geometry or part_geometry:
                raise ValueError('Depth observation requires scanned candidates and depth-compatible region geometry')
            from generator.depth_observation import DepthScanEncoder
            self.height = None
            self.depth = DepthScanEncoder(width)
            self.location_scan_key = nn.Linear(width, location_width, bias=False)
        if world_surface_scan and not role_validity:
            raise ValueError('World mesh selection requires actual-contact role validity')
        if role_validity and not event_roles:
            raise ValueError('Role validity requires explicit event roles')
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

    def architecture_contract(self):
        contract = dict(schema=self.schema, width=self.width, layers=self.layers,
            location_width=self.location_width, body_geometry=self.body_geometry,
            part_geometry=self.part_geometry, region_plan=self.region_plan,
            unified_contact=self.unified_contact, event_roles=self.event_roles,
            independent_observation_encoders=False,
            execution_plan_gradients=self.execution_plan_gradients,
            execution_observation_gradients=self.execution_observation_gradients,
            readout='legacy root+part mean')
        if self.limb_contact_modes:
            from generator.limb_contact import SCHEMA
            contract["limb_contact_modes"] = SCHEMA
        if self.role_validity:
            from generator.role_validity import ROLE_VALIDITY_SCHEMA
            contract['role_validity'] = ROLE_VALIDITY_SCHEMA
        if self.world_surface_scan:
            from somaforge_core.world_mesh_scan import SCAN_SCHEMA
            contract['interaction_candidates'] = SCAN_SCHEMA
            contract['interaction_coordinates'] = 'world mesh XYZ; no heightmap lifting'
        if self.depth_observation:
            from generator.depth_observation import depth_observation_contract
            contract['environment_observation'] = depth_observation_contract()
            contract['keep_target'] = 'actual_start_contact_point_no_location_selection_v1'
            from somaforge_core.world_mesh_scan import LEGACY_DEPTH_PROFILE, depth_camera_profile
            profile = getattr(self, 'depth_scan_profile', LEGACY_DEPTH_PROFILE)
            if profile != LEGACY_DEPTH_PROFILE:
                contract['depth_camera'] = dict(profile=profile, **depth_camera_profile(profile))
        return contract

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
        from generator.observation_geometry import body_geometry_features
        return body_geometry_features(self, current_q, heightmap, basis)

    def _encode_points(self, current_q, current_contact, current_anchor, heightmap=None, *, depth=None):
        geometry_features = None
        image_features = None
        if self.depth_observation:
            if heightmap is not None:
                raise ValueError('Depth Predictor does not accept a heightmap observation')
            if depth is None:
                raise ValueError('Depth Predictor is missing the 2D scan')
            terrain, image_features = self.depth(depth['depth_image'], depth['depth_valid'], depth['depth_intrinsics'])
        else:
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
            if self.depth_observation:
                from generator.depth_observation import depth_region_features
                features = depth_region_features(self.region_geometry, local_position, local_rotation,
                                                 current_q, basis, depth)
            else:
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
        return terrain, basis, yaw, self.norm(encoded[:, :1]), self.norm(encoded[:, 1:]), geometry_features, image_features

    def forward(self, current_q, current_contact, current_anchor, heightmap=None, *,
                current_contact_relations=None, contact_surface_ids=None, heightmap_surface=None,
                contact_relation_points=None,
                execution_observation_supervised=None,
                contact_patch_points=None, contact_patch_part=None, contact_patch_surface=None, contact_patch_valid=None,
                scan_points_world=None, scan_surface=None, scan_valid=None, scan_point_id=None, scan_spacing_m=None,
                depth_image=None, depth_valid=None, depth_intrinsics=None, depth_camera_rotation_world=None,
                depth_camera_position_world=None, scan_pixel_index=None,
                teacher_contact=None, teacher_cell=None, teacher_mask=None, teacher_role=None, teacher_regions=None,
                repair_mask=None, repair_role=None, repair_points_world=None, repair_regions=None,
                repair_surface=None, repair_point_id=None):
        depth = dict(depth_image=depth_image, depth_valid=depth_valid, depth_intrinsics=depth_intrinsics,
                     depth_camera_rotation_world=depth_camera_rotation_world,
                     depth_camera_position_world=depth_camera_position_world)
        terrain, basis, yaw, body, part, geometry_features, image_features = self._encode_points(
            current_q, current_contact, current_anchor, heightmap, depth=depth)
        selected = select_position_plan(self, current_q, heightmap, basis, part,
            current_contact=current_contact, current_contact_relations=current_contact_relations,
            contact_surface_ids=contact_surface_ids, heightmap_surface=heightmap_surface,
            contact_relation_points=contact_relation_points,
            scan_points_world=scan_points_world, scan_surface=scan_surface, scan_valid=scan_valid,
            scan_point_id=scan_point_id, scan_spacing_m=scan_spacing_m,
            depth_image=depth_image, scan_pixel_index=scan_pixel_index, image_features=image_features,
            teacher_contact=teacher_contact, teacher_cell=teacher_cell, teacher_mask=teacher_mask,
            teacher_role=teacher_role, teacher_regions=teacher_regions, repair_mask=repair_mask,
            repair_role=repair_role, repair_points_world=repair_points_world, repair_regions=repair_regions,
            repair_surface=repair_surface, repair_point_id=repair_point_id)
        contact, points = selected.conditioned_contact, selected.conditioned_points_local
        conditioned_role, planned_regions = selected.conditioned_role, selected.planned_regions

        # Share observation learning when enabled, while keeping explicit
        # contact/position/role/region intent detached below.
        if not self.execution_observation_gradients:
            terrain, body, part = terrain.detach(), body.detach(), part.detach()
        elif execution_observation_supervised is not None:
            mask = execution_observation_supervised
            if mask.dtype != torch.bool or mask.shape != (len(current_q),):
                raise ValueError('Execution observation supervision must be boolean [B]')
            # Only the execution read of shared features is gated. Planner
            # reads above retain their original gradients on every sample.
            terrain, body, part = [torch.where(mask[:, None, None], x, x.detach())
                for x in (terrain, body, part)]
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
        return selected.prediction(qpos)
