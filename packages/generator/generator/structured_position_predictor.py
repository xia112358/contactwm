"""Typed observations and contact intent with independent planning/execution.

Every root/joint output reads its own query token. No across-part latent average
or observation/intent addition is used. Canonical FK and collision buffers are
shared physical definitions, not trainable representations.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.g1_kinematics import (
    BODY_NAMES, CanonicalG1ForwardKinematics, _matrix_from_rotation6d,
    _quaternion_matrix_wxyz, _rotation6d,
)
from somaforge_core.heightmap import _root_yaw_basis
from contact_solver.collision_geometry import CanonicalG1CollisionPoints
from generator.next_interaction_heightmap_v3 import (
    DenseHeightmapEncoder, DenseHeightmapInteractionPredictor, TerrainCrossBlock,
)
from generator.position_plan import select_position_plan
from generator.observation_geometry import body_geometry_features

STRUCTURED_SCHEMA = 'full1000_position_predictor_v2'


def _transformer(width, layers):
    layer = nn.TransformerEncoderLayer(width, 6, 2*width, dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ObservationEncoder(nn.Module):
    """Ordered root, body-pose, actual-contact, joint and geometry tokens."""
    def __init__(self, width, layers, geometry_width=0, geometry_count=0, *, depth_observation=False):
        super().__init__()
        self.depth_observation = depth_observation
        if depth_observation:
            from generator.depth_observation import DepthScanEncoder
            self.depth = DepthScanEncoder(width)
            self.height = None
        else:
            self.height = DenseHeightmapEncoder(width)
        self.root_encoder = nn.Linear(6, width)
        self.part_encoder = nn.Linear(9, width)
        self.contact_encoder = nn.Linear(4, width)
        self.joint_encoder = nn.Linear(1, width)
        self.part_identity = nn.Embedding(6, width)
        self.joint_identity = nn.Embedding(29, width)
        self.type_identity = nn.Embedding(5, width)
        self.geometry_encoder = nn.Linear(geometry_width, width) if geometry_width else None
        self.geometry_identity = nn.Embedding(geometry_count, width) if geometry_width else None
        self.terrain_reasoning = nn.ModuleList([TerrainCrossBlock(width), TerrainCrossBlock(width)])
        self.interaction = _transformer(width, layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, root_rotation, part_pose, current_contact, local_anchor, joint_angles,
                heightmap, geometry_features=None, *, depth=None):
        image_features = None
        if self.depth_observation:
            if heightmap is not None:
                raise ValueError('Depth Predictor does not accept a heightmap observation')
            if depth is None:
                raise ValueError('Depth Predictor is missing the 2D scan')
            terrain, image_features = self.depth(depth['depth_image'], depth['depth_valid'], depth['depth_intrinsics'])
        else:
            terrain = self.height(heightmap)
        root = self.root_encoder(root_rotation)[:, None] + self.type_identity.weight[0]
        parts = self.part_encoder(part_pose) + self.part_identity.weight[None] + self.type_identity.weight[1]
        # Terrain cross-attention retains the spatial memory; it does not pool it.
        spatial = torch.cat((root, parts), 1)
        for block in self.terrain_reasoning:
            spatial = block(spatial, terrain)
        anchors = torch.where(current_contact[..., None], local_anchor, 0.)
        contacts = self.contact_encoder(torch.cat((anchors, current_contact[..., None].to(anchors)), -1))
        contacts = contacts + self.part_identity.weight[None] + self.type_identity.weight[2]
        joints = self.joint_encoder(joint_angles[..., None]/math.pi)
        joints = joints + self.joint_identity.weight[None] + self.type_identity.weight[3]
        tokens = [spatial, contacts, joints]
        if self.geometry_encoder is not None:
            if geometry_features is None:
                raise ValueError('Configured geometry observations are missing')
            tokens.append(self.geometry_encoder(geometry_features) + self.geometry_identity.weight[None]
                          + self.type_identity.weight[4])
        return terrain, self.norm(self.interaction(torch.cat(tokens, 1))), image_features


class JointPoseHead(nn.Module):
    """Each joint reads only its corresponding decoded feature, with its own row."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(29, width))
        self.bias = nn.Parameter(torch.zeros(29))
        nn.init.normal_(self.weight, std=.001)

    def forward(self, tokens):
        if tokens.shape[1:] != self.weight.shape:
            raise ValueError('Joint readout requires ordered [B,29,width] queries')
        return torch.einsum('bjd,jd->bj', tokens, self.weight) + self.bias


class StructuredPositionPredictor(nn.Module):
    """Version 2 scratch model; v1 weights require the explicit legacy reader."""
    schema = STRUCTURED_SCHEMA

    def __init__(self, width=192, layers=3, location_width=32, *, body_geometry=False,
                 part_geometry=False, region_plan=False, unified_contact=False, event_roles=False,
                 execution_plan_gradients=False, execution_observation_gradients=False, role_validity=False,
                 world_surface_scan=False, depth_observation=False):
        super().__init__()
        if width <= 0 or width % 6 or layers < 1 or location_width < 1:
            raise ValueError('Width must be positive/divisible by 6; layers/location width must be positive')
        if execution_plan_gradients or execution_observation_gradients:
            raise ValueError('Structured planner and executor cannot share trainable gradients')
        if body_geometry and part_geometry:
            raise ValueError('Choose one geometry observation ablation')
        if (region_plan or unified_contact) and (body_geometry or part_geometry):
            raise ValueError('Regional experiment uses its own compact geometry input')
        self.width, self.layers, self.location_width = width, layers, location_width
        self.body_geometry, self.part_geometry = body_geometry, part_geometry
        self.region_plan, self.unified_contact, self.event_roles = region_plan, unified_contact, event_roles
        self.role_validity = role_validity
        self.world_surface_scan = world_surface_scan
        self.depth_observation = depth_observation
        if depth_observation and (not world_surface_scan or body_geometry or part_geometry):
            raise ValueError('Depth observation requires scanned candidates and depth-compatible region geometry')
        if world_surface_scan and not role_validity:
            raise ValueError('World mesh selection requires actual-contact role validity')
        if role_validity and not event_roles:
            raise ValueError('Role validity requires explicit event roles')
        self.execution_plan_gradients = self.execution_observation_gradients = False
        self.fk = CanonicalG1ForwardKinematics()
        self.geometry = CanonicalG1CollisionPoints(64)
        geometry_width, geometry_count = 0, 0
        if region_plan or unified_contact:
            from contact_solver.contact_regions import ContactRegions
            self.region_geometry = ContactRegions(self.fk)
        if region_plan:
            geometry_width, geometry_count = 32, 6
        elif part_geometry:
            from contact_solver.part_collision_geometry import PartCollisionGeometry
            self.part_geometry_observation = PartCollisionGeometry(self.fk)
            geometry_width, geometry_count = PartCollisionGeometry.feature_count, 6
        elif body_geometry:
            self.body_geometry_names = tuple(sorted(set(self.geometry.link_names)))
            lookup = {name: i for i, name in enumerate(self.body_geometry_names)}
            ids = torch.cat([torch.full((count,), lookup[name], dtype=torch.long)
                for name, count in zip(self.geometry.link_names, self.geometry.point_counts)])
            self.register_buffer('body_geometry_point_link', ids, persistent=False)
            geometry_width, geometry_count = 13, len(self.body_geometry_names)

        # Separate construction, not aliases/copies of the same Parameter objects.
        self.planner_observation = ObservationEncoder(width, layers, geometry_width, geometry_count,
                                                      depth_observation=depth_observation)
        self.executor_observation = ObservationEncoder(width, layers, geometry_width, geometry_count,
                                                       depth_observation=depth_observation)
        self.role_head = nn.Linear(width, 4)
        self.location_query = nn.Linear(width, location_width)
        self.location_key = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, location_width))
        if depth_observation:
            self.location_scan_key = nn.Linear(width, location_width, bias=False)
        if region_plan:
            self.region_head = nn.Linear(width, 15)
            self.region_prior = nn.Parameter(torch.zeros(6, 15))
        # 1 active + 3 point + 4 role + 4 regional flags = one typed future token per part.
        self.execution_intent_encoder = nn.Linear(12, width)
        self.execution_intent_identity = nn.Embedding(6, width)
        self.execution_intent_type = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.execution_intent_type, std=.02)
        self.execution_query_identity = nn.Embedding(30, width)
        self.execution_relation_encoder = nn.Linear(6, width, bias=False)
        self.register_buffer('query_part_ancestor', torch.cat((torch.ones(1, 6),
            self.fk.body_joint_ancestor[1:].T.float()), 0), persistent=False)
        self.execution_read = TerrainCrossBlock(width)
        self.execution_coordination = _transformer(width, 1)
        self.execution_norm = nn.LayerNorm(width)
        self.root_pose_head = nn.Linear(width, 7)
        self.joint_pose_head = JointPoseHead(width)
        nn.init.normal_(self.root_pose_head.weight, std=.001)
        nn.init.zeros_(self.root_pose_head.bias)
        with torch.no_grad():
            self.root_pose_head.bias[3] = 1.

    def architecture_contract(self):
        contract = dict(schema=self.schema, width=self.width, layers=self.layers,
            location_width=self.location_width, independent_observation_encoders=True,
            body_geometry=self.body_geometry, part_geometry=self.part_geometry,
            region_plan=self.region_plan, unified_contact=self.unified_contact, event_roles=self.event_roles,
            parts=list(BODY_NAMES[1:]), joints=list(G1_29DOF_JOINT_ORDER),
            observation_tokens=['root_rotation', 'part_pose[6]', 'actual_contact[6]', 'joint_angle[29]',
                                'physical_geometry_by_entity_if_configured'],
            intent_tokens='future_contact_intent[6], separate from current observation',
            readout='root_query[1] + joint_query[29], independent output rows',
            coordination='attention; no across-part arithmetic pooling',
            relation='canonical URDF query/part ancestor features; no hard attention mask')
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
        for name, parameter in self.named_parameters():
            planner = name.startswith(('planner_observation.', 'role_head.', 'location_query.',
                                       'location_key.', 'location_scan_key.', 'region_head.', 'region_prior'))
            groups['planner' if planner else 'executor'].append(parameter)
        return groups

    def clip_training_gradients(self, max_norm=10.):
        return {name: nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True).to(parameters[0].device)
                for name, parameters in self.training_parameter_groups().items()}

    def compilable_modules(self):
        return (self.planner_observation, self.executor_observation, self.location_key,
                self.execution_read, self.execution_coordination)

    _decode_pose = staticmethod(DenseHeightmapInteractionPredictor._decode_pose)
    encode_pose = DenseHeightmapInteractionPredictor.encode_pose
    decode_pose = DenseHeightmapInteractionPredictor.decode_pose

    def forward(self, current_q, current_contact, current_anchor, heightmap=None, **conditioning):
        if current_q.ndim != 2 or current_q.shape[1] != 36:
            raise ValueError('Current pose must be [B,36]')
        if current_contact.shape != (len(current_q), 6) or current_contact.dtype != torch.bool:
            raise ValueError('Current actual contact must be bool [B,6]')
        if current_anchor.shape != (len(current_q), 6, 3):
            raise ValueError('Current actual contact anchors must be [B,6,3]')
        basis, yaw = _root_yaw_basis(current_q)
        position, rotation6d = self.fk(current_q[:, None])
        positions = position[:, 0, 1:7]
        rotation = _matrix_from_rotation6d(rotation6d[:, 0, 1:7])
        local_position = torch.einsum('bij,bpj->bpi', basis.transpose(1, 2), positions-current_q[:, None, :3])
        local_rotation = basis[:, None].transpose(-1, -2) @ rotation
        local_anchor = torch.einsum('bij,bpj->bpi', basis.transpose(1, 2), current_anchor-current_q[:, None, :3])
        part_pose = torch.cat((local_position, _rotation6d(local_rotation)), -1)
        root_rotation = _rotation6d(basis.transpose(1, 2) @ _quaternion_matrix_wxyz(current_q[:, 3:7]))
        features = None
        if self.region_plan:
            if self.depth_observation:
                from generator.depth_observation import depth_region_features
                features = depth_region_features(self.region_geometry, local_position, local_rotation,
                                                 current_q, basis, conditioning)
            else:
                features = self.region_geometry(local_position, local_rotation, heightmap)
        elif self.part_geometry:
            features = self.part_geometry_observation(local_position, local_rotation, heightmap)
        elif self.body_geometry:
            features = body_geometry_features(self, current_q, heightmap, basis)
        args = (root_rotation, part_pose, current_contact, local_anchor, current_q[:, 7:], heightmap, features)
        _, planned_observation, image_features = self.planner_observation(*args, depth=conditioning)
        selected = select_position_plan(self, current_q, heightmap, basis, planned_observation[:, 1:7],
            current_contact=current_contact, image_features=image_features, **conditioning)
        _, observed, _ = self.executor_observation(*args, depth=conditioning)
        contact = selected.conditioned_contact.detach()
        # Inactive positions cannot influence execution. Roles still distinguish release/idle.
        points = torch.where(contact[..., None], selected.conditioned_points_local.detach(), 0.)
        roles = F.one_hot(selected.conditioned_role.detach(), 4).to(points) if self.event_roles else points.new_zeros(len(points), 6, 4)
        regions = selected.planned_regions.detach().to(points) if self.region_plan else points.new_zeros(len(points), 6, 4)
        intent = self.execution_intent_encoder(torch.cat((contact[..., None].to(points), points, roles, regions), -1))
        intent = intent + self.execution_intent_identity.weight[None] + self.execution_intent_type
        memory = torch.cat((observed, intent), 1)
        # Observed token order: root1, part6, contact6, joint29, optional geometry.
        queries = torch.cat((observed[:, :1], observed[:, 13:42]), 1)
        queries = queries + self.execution_query_identity.weight[None]
        queries = queries + self.execution_relation_encoder(self.query_part_ancestor.to(queries))[None]
        queries = self.execution_read(queries, memory)
        queries = self.execution_norm(self.execution_coordination(queries))
        raw = torch.cat((self.root_pose_head(queries[:, 0]), self.joint_pose_head(queries[:, 1:])), -1)
        return selected.prediction(self._decode_pose(raw, current_q, basis, yaw))
