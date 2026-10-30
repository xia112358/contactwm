"""Calibrated raw depth -> world surface mesh interaction candidates.

Camera tilt is removed by full rigid world registration. Only vertices on
upward horizontal faces are selectable, and triangles never join distinct
physical surfaces. The controlled renderer uses initialized Newton terrain;
robot self-occlusion and camera noise are not modeled. Sensor visibility and
label coverage never define actual contact, which remains Newton's domain.
"""
import hashlib
import json
import math

import numpy as np
import torch

from somaforge_core.contact_face_selection import upward_face_mask

SCAN_SCHEMA = 'raw_depth_world_surface_mesh_v1'
DEPTH_OBSERVATION_SCHEMA = 'calibrated_optical_depth_image_v1'
LEGACY_DEPTH_PROFILE = 'torso_depth_87x58_fov89p5_pitch27_v1'
WIDE_DEPTH_PROFILE = 'torso_depth_128x128_fov120_pitch55_v1'
COMPACT_DEPTH_PROFILE = 'torso_depth_96x96_fov110_pitch60_v1'
STABILIZED_DEPTH_PROFILE = 'torso_depth_96x96_fov110_pitch60_yaw_v1'
ROOT_DEPTH_PROFILE = 'root_depth_96x96_fov90_pitch75_yaw_v1'
ROOT_COMPACT_DEPTH_PROFILE = 'root_depth_87x58_fov89p5_pitch85_yaw_v1'


def depth_camera_profile(name):
    """Versioned optical calibration; old checkpoints retain the legacy rig."""
    profiles = {
        LEGACY_DEPTH_PROFILE: dict(width=87, height=58, horizontal_fov_deg=89.5,
            pitch_deg=27., min_depth_m=.3, max_depth_m=3.),
        WIDE_DEPTH_PROFILE: dict(width=128, height=128, horizontal_fov_deg=120.,
            pitch_deg=55., min_depth_m=.2, max_depth_m=3.),
        COMPACT_DEPTH_PROFILE: dict(width=96, height=96, horizontal_fov_deg=110.,
            pitch_deg=60., min_depth_m=.2, max_depth_m=3.),
        STABILIZED_DEPTH_PROFILE: dict(width=96, height=96, horizontal_fov_deg=110.,
            pitch_deg=60., min_depth_m=.2, max_depth_m=3., orientation='yaw_only'),
        ROOT_DEPTH_PROFILE: dict(width=96, height=96, horizontal_fov_deg=90.,
            pitch_deg=75., min_depth_m=.2, max_depth_m=3., orientation='yaw_only',
            mount_frame='root_yaw', mount_offset_m=[.2, 0., .75]),
        ROOT_COMPACT_DEPTH_PROFILE: dict(width=87, height=58, horizontal_fov_deg=89.5,
            pitch_deg=85., min_depth_m=.2, max_depth_m=3., orientation='yaw_only',
            mount_frame='root_yaw', mount_offset_m=[.2, 0., .75]),
    }
    if name not in profiles:
        raise ValueError(f'Unknown depth camera profile: {name}')
    return dict(profiles[name])


DEPTH_OBSERVATION_KEYS = ('depth_image', 'depth_valid', 'depth_intrinsics',
    'depth_camera_rotation_world', 'depth_camera_position_world', 'scan_pixel_index')


def depth_world_mesh(depth_m, intrinsics, camera_rotation_world, camera_position_world,
                     hit_valid, surface_ids, *, spacing_m):
    """Raw optical-axis depth -> observed world mesh vertices and connectivity.

    Inputs are unresized metric depth and calibrated optical-camera extrinsics.
    Hit validity and registered physical surface identity must be supplied by
    the scanner; depth values never imply actual robot contact. No blur, height
    lookup, rounding of XYZ, or cross-surface triangle is introduced here.
    Camera tilt is removed by the full rigid transform to world coordinates.
    """
    if (depth_m.ndim != 3 or hit_valid.shape != depth_m.shape or surface_ids.shape != depth_m.shape
            or hit_valid.dtype != torch.bool or surface_ids.dtype != torch.long):
        raise ValueError('Raw depth requires matching hit-valid and registered surface maps')
    batch, height, width = depth_m.shape
    if (intrinsics.shape != (batch, 3, 3) or camera_rotation_world.shape != (batch, 3, 3)
            or camera_position_world.shape != (batch, 3) or height < 2 or width < 2):
        raise ValueError('Missing/invalid calibrated raw-depth camera geometry')
    if (bool((hit_valid & ((depth_m <= 0) | ~torch.isfinite(depth_m) | (surface_ids < 0))).any())
            or not math.isfinite(spacing_m) or spacing_m <= 0):
        raise ValueError('Unknown hit depth/surface or invalid observation sampling scale')
    dtype, device = depth_m.dtype, depth_m.device
    if any(not bool(torch.isfinite(t).all()) for t in
           (intrinsics, camera_rotation_world, camera_position_world)):
        raise ValueError('Nonfinite raw-depth camera calibration')
    identity = torch.eye(3, device=device, dtype=dtype).expand(batch, -1, -1)
    if (not torch.allclose(camera_rotation_world.transpose(-1, -2) @ camera_rotation_world,
                           identity, atol=1e-6, rtol=1e-6)
            or not torch.allclose(torch.linalg.det(camera_rotation_world),
                                  depth_m.new_ones(batch), atol=1e-6, rtol=1e-6)):
        raise ValueError('Depth camera extrinsics must be a rigid world rotation')
    if not torch.allclose(intrinsics[:, 2], depth_m.new_tensor([0., 0., 1.]).expand(batch, -1)):
        raise ValueError('Depth intrinsics must use optical-axis Z calibration')
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                           torch.arange(width, device=device, dtype=dtype), indexing='ij')
    pixel = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
    # The image contains Z-depth, not Euclidean range. K^-1[u,v,1]
    # multiplied by Z is the camera-frame 3D point (no ray normalization).
    rays = torch.einsum('bij,nj->bni', torch.linalg.inv(intrinsics), pixel)
    camera = rays*torch.where(hit_valid, depth_m, 0.).flatten(1)[..., None]
    world = camera_position_world[:, None]+torch.einsum('bij,bnj->bni', camera_rotation_world, camera)
    ids = torch.arange(height*width, device=device).reshape(height, width)
    a, b, c, d = ids[:-1, :-1].flatten(), ids[1:, :-1].flatten(), ids[:-1, 1:].flatten(), ids[1:, 1:].flatten()
    faces = torch.cat((torch.stack((a, b, c), -1), torch.stack((d, c, b), -1)))
    triangle = world[:, faces]
    valid = hit_valid.flatten(1)[:, faces].all(-1)
    sf = surface_ids.flatten(1)[:, faces]
    valid &= (sf == sf[..., :1]).all(-1)
    normal = torch.cross(triangle[:, :, 1]-triangle[:, :, 0], triangle[:, :, 2]-triangle[:, :, 0], dim=-1)
    norm = normal.norm(dim=-1)
    valid &= norm > 0
    normal /= norm.clamp_min(torch.finfo(dtype).tiny)[..., None]
    facing = (normal*(camera_position_world[:, None]-triangle.mean(-2))).sum(-1)
    normal = torch.where((facing < 0)[..., None], -normal, normal)
    normal = normal.masked_fill(~valid[..., None], 0.)
    vertex_normal = world.new_zeros(batch, height*width, 3)
    vertex_seen = torch.zeros(batch, height*width, device=device, dtype=torch.long)
    for corner in range(3):
        index = faces[:, corner][None].expand(batch, -1)
        vertex_normal.scatter_add_(1, index[..., None].expand(-1, -1, 3), normal)
        vertex_seen.scatter_add_(1, index, valid.long())
    vertex_normal = torch.nn.functional.normalize(vertex_normal, dim=-1)
    upward = torch.as_tensor(upward_face_mask(vertex_normal.detach().cpu().double().reshape(-1, 3).numpy()),
                             device=device).reshape(batch, -1)
    # Identity bins register observations; coordinates themselves are never
    # snapped or averaged across layers. Surface identity separates layers.
    registered = torch.round(world.double()*1e6).long()
    point_id = ((registered[..., 0]*73856093) ^ (registered[..., 1]*19349663)
                ^ (registered[..., 2]*83492791) ^ (surface_ids.flatten(1)*2654435761)) & ((1 << 63)-1)
    return dict(depth_image=torch.where(hit_valid, depth_m, 0.), depth_valid=hit_valid,
        depth_intrinsics=intrinsics, depth_camera_rotation_world=camera_rotation_world,
        depth_camera_position_world=camera_position_world,
        scan_pixel_index=ids.flatten()[None].expand(batch, -1),
        scan_points_world=world.masked_fill(~hit_valid.flatten(1)[..., None], 0.),
        scan_surface=surface_ids.flatten(1).masked_fill(~hit_valid.flatten(1), -1),
        scan_valid=(vertex_seen > 0) & upward, scan_point_id=point_id.masked_fill(~hit_valid.flatten(1), -1),
        scan_spacing_m=depth_m.new_full((batch,), spacing_m)), dict(
        faces=faces, face_valid=valid, normals_world=vertex_normal,
        source='raw optical-axis depth; same-surface triangles only; no interpolation')


class WorldMeshScanRouter:
    def __init__(self, metadata, **kwargs):
        self.scenes = {int(i): DepthWorldMeshScanner(m, **kwargs) for i, m in metadata.items()}

    def __call__(self, q, scene_ids, fk):
        if scene_ids.shape != (len(q),) or scene_ids.dtype != torch.long:
            raise ValueError('World depth scan requires one registered scene per state')
        result = {}
        self.last_audit = {}
        for i in scene_ids.unique().tolist():
            selected = (scene_ids == i).nonzero().flatten()
            packet = self.scenes[i].scan(q[selected], fk)
            self.last_audit[i] = dict(batch_indices=selected, current_q=q[selected],
                                      **self.scenes[i].last_audit)
            for key, value in packet.items():
                if key not in result:
                    result[key] = value.new_empty((len(q), *value.shape[1:]))
                result[key][selected] = value
        return result

    def contract(self):
        return {str(i): s.contract() for i, s in self.scenes.items()}


class DepthWorldMeshScanner:
    """Clean calibrated depth frontend against initialized Newton terrain.

    Raw image geometry is never cropped, resized, blurred, or filled. The
    complete current rigid camera pose maps optical depth into world XYZ.
    Robot self-occlusion/noise are explicit future sensor-model work; neither
    this renderer nor its surface segmentation defines actual robot contact.
    """
    def __init__(self, metadata, *, profile=LEGACY_DEPTH_PROFILE, width=None, height=None,
                 horizontal_fov_deg=None, min_depth_m=None, max_depth_m=None, pitch_deg=None):
        calibration = depth_camera_profile(profile)
        overrides = dict(width=width, height=height, horizontal_fov_deg=horizontal_fov_deg,
                         min_depth_m=min_depth_m, max_depth_m=max_depth_m, pitch_deg=pitch_deg)
        calibration.update({k: v for k, v in overrides.items() if v is not None})
        width, height = calibration['width'], calibration['height']
        horizontal_fov_deg = calibration['horizontal_fov_deg']
        min_depth_m, max_depth_m = calibration['min_depth_m'], calibration['max_depth_m']
        self.pitch = float(calibration['pitch_deg'])
        self.orientation = calibration.get('orientation', 'body')
        self.mount_frame = calibration.get('mount_frame', 'torso')
        self.mount_offset = calibration.get('mount_offset_m', [.01, .01, .44])
        if not math.isfinite(self.pitch) or not 0 <= self.pitch <= 90:
            raise ValueError('Invalid depth camera pitch')
        self.profile = profile
        self.calibration_overridden = calibration != depth_camera_profile(profile)
        if not metadata.get('provenance', {}).get('model_fingerprint'):
            raise ValueError('Depth scan requires actual initialized scene provenance')
        if width < 2 or height < 2 or not 0 < horizontal_fov_deg < 180 or not 0 < min_depth_m < max_depth_m:
            raise ValueError('Invalid depth scanner calibration')
        triangles, surface = [], []
        for face in sorted(metadata['surface_catalog'], key=lambda f: int(f['surface'])):
            if not face.get('triangles_w'):
                raise ValueError('Depth scan source is missing actual world triangles')
            triangles.extend(face['triangles_w'])
            surface.extend([int(face['surface'])]*len(face['triangles_w']))
        self.triangles = np.asarray(triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3) or not np.isfinite(self.triangles).all():
            raise ValueError('Invalid initialized depth geometry')
        self.surface = np.asarray(surface, dtype=np.int64)
        catalog = metadata['surface_catalog']
        upward = upward_face_mask([face['normal_w'] for face in catalog])
        self.upward_surface_ids = [int(face['surface']) for face, keep in zip(catalog, upward) if keep]
        self.width, self.height, self.fov = width, height, horizontal_fov_deg
        self.minimum, self.maximum = min_depth_m, max_depth_m
        self.fingerprint = metadata['provenance']['model_fingerprint']
        self.geometry_sha256 = hashlib.sha256(json.dumps(metadata['surface_catalog'], sort_keys=True).encode()).hexdigest()
        self._device = {}

    def contract(self):
        return dict(schema=SCAN_SCHEMA, observation_schema=DEPTH_OBSERVATION_SCHEMA,
            model_fingerprint=self.fingerprint,
            geometry_sha256=self.geometry_sha256, width=self.width, height=self.height,
            horizontal_fov_deg=self.fov, depth_range_m=[self.minimum, self.maximum],
            depth='raw optical-axis metres', coordinate_frame='world',
            camera=f'{self.mount_frame}; offset {self.mount_offset}; downward pitch {self.pitch:g} degrees',
            camera_profile=self.profile, calibration_overridden=self.calibration_overridden,
            orientation=self.orientation,
            mount_position=('root translation plus yaw-aligned fixed offset; body articulation ignored'
                if self.mount_frame == 'root_yaw' else 'torso rigid offset; orientation stabilization does not move the sensor'),
            pitch_deg=self.pitch,
            vertical_fov_deg=math.degrees(2*math.atan(self.height/self.width*math.tan(math.radians(self.fov)/2))),
            resize=False, blur=False, cross_surface_faces=False, robot_self_occlusion=False,
            contact_truth=False, current_contact_nodes='actual allocated Newton terrain witnesses')

    @torch.no_grad()
    def scan(self, q, fk):
        device = q.device
        if str(device) not in self._device:
            self._device[str(device)] = (torch.as_tensor(self.triangles, device=device),
                                         torch.as_tensor(self.surface, device=device))
        triangles, surface = self._device[str(device)]
        if self.mount_frame == 'root_yaw':
            from somaforge_core.g1_kinematics import _quaternion_matrix_wxyz
            position = q.double()[:, None, :3]
            body_rotation = _quaternion_matrix_wxyz(q.double()[:, 3:7])[:, None]
        else:
            position, body_rotation = fk.link_poses(q.double(), ('torso_link',))
        angle = math.radians(self.pitch)
        optical = position.new_tensor([[0., -math.sin(angle), math.cos(angle)],
                                      [-1., 0., 0.], [0., -math.cos(angle), -math.sin(angle)]])
        mount_rotation = body_rotation[:, 0]
        if self.orientation == 'yaw_only':
            # Project the mounting frame's forward axis onto the horizontal
            # plane. Optical pitch is always relative to world up.
            forward_xy = mount_rotation[:, :2, 0]
            length = forward_xy.norm(dim=-1)
            if bool((length < 1e-8).any()):
                raise ValueError('Yaw-stabilized camera has undefined horizontal mounting-frame heading')
            forward_xy = forward_xy / length[:, None]
            c, s = forward_xy.unbind(-1)
            z, one = torch.zeros_like(c), torch.ones_like(c)
            mount_rotation = torch.stack((c, -s, z, s, c, z, z, z, one), -1).reshape(-1, 3, 3)
        offset_rotation = mount_rotation if self.mount_frame == 'root_yaw' else body_rotation[:, 0]
        origin = position[:, 0]+torch.einsum('bij,j->bi', offset_rotation, position.new_tensor(self.mount_offset))
        rotation = mount_rotation@optical
        focal = self.width/(2*math.tan(math.radians(self.fov)/2))
        k = position.new_tensor([[focal, 0., (self.width-1)/2],
                                 [0., focal, (self.height-1)/2], [0., 0., 1.]])[None].expand(len(q), -1, -1)
        yy, xx = torch.meshgrid(torch.arange(self.height, device=device, dtype=torch.float64),
                               torch.arange(self.width, device=device, dtype=torch.float64), indexing='ij')
        pixel = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
        camera_ray = torch.einsum('bij,nj->bni', torch.linalg.inv(k), pixel)
        rays = torch.einsum('bij,bnj->bni', rotation, camera_ray)
        depth = position.new_full((len(q), len(pixel)), torch.inf)
        hit_surface = torch.full_like(depth, -1, dtype=torch.long)
        v0, e1, e2 = triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        delta = origin[:, None]-v0
        cross = torch.cross(delta, e1[None].expand_as(delta), dim=-1)
        for start in range(0, len(pixel), 128):
            ray = rays[:, start:start+128]
            pvec = torch.cross(ray[:, :, None].expand(-1, -1, len(triangles), -1),
                               e2[None, None].expand(len(q), ray.shape[1], -1, -1), dim=-1)
            det = (pvec*e1).sum(-1)
            nonparallel = det.abs() > 1e-12
            inv = torch.where(nonparallel, det.reciprocal(), 0.)
            u, v = (pvec*delta[:, None]).sum(-1)*inv, (ray[:, :, None]*cross[:, None]).sum(-1)*inv
            t = (cross*e2).sum(-1)[:, None]*inv
            # Numerical triangle arithmetic only; no contact classification.
            intersect = nonparallel & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (t > 0)
            nearest, face = t.masked_fill(~intersect, torch.inf).min(-1)
            depth[:, start:start+128] = nearest
            hit_surface[:, start:start+128] = surface[face]
        valid = torch.isfinite(depth) & (depth >= self.minimum) & (depth <= self.maximum)
        hit_surface = hit_surface.masked_fill(~valid, -1)
        packet, audit = depth_world_mesh(depth.reshape(len(q), self.height, self.width), k, rotation, origin,
            valid.reshape(len(q), self.height, self.width), hit_surface.reshape(len(q), self.height, self.width),
            spacing_m=self.maximum/focal)
        # Actual source-face orientation remains authoritative. In particular,
        # looking at a back face from inside a solid cannot promote its bottom.
        upward_ids = torch.tensor(self.upward_surface_ids, device=device, dtype=torch.long)
        packet['scan_valid'] &= torch.isin(packet['scan_surface'], upward_ids)
        self.last_audit = dict(raw_depth_m=depth, hit_valid=valid, surface=hit_surface,
            camera_intrinsics=k, camera_rotation_world=rotation, camera_position_world=origin, **audit)
        for key in ('scan_points_world', 'scan_spacing_m', 'depth_image', 'depth_intrinsics',
                    'depth_camera_rotation_world', 'depth_camera_position_world'):
            packet[key] = packet[key].to(q)
        return packet


def append_current_contact_nodes(packet, observed):
    """Measured current terrain witnesses retain occluded keep relations.

    These nodes are current tactile observations, not invented future points
    or hidden terrain geometry. All source pairs were Newton-validated first.
    """
    points = packet['scan_points_world']
    rows = [[] for _ in range(len(points))]
    if observed.get('schema') == 'newton_device_witness_batch_v1':
        p = observed['pairs']
        required = ('geometry_point0_w', 'geometry_point1_w', 'body_link0', 'body_link1')
        if any(k not in p for k in required):
            raise ValueError('Missing actual Newton terrain-witness geometry')
        eligible = p['eligible']
        terrain0 = (p['body_link0'] < 0) & (p['body_link1'] >= 0)
        terrain1 = (p['body_link1'] < 0) & (p['body_link0'] >= 0)
        if bool((eligible & ~(terrain0 | terrain1)).any()):
            raise ValueError('Eligible contact has no uniquely identified terrain side')
        terrain = torch.where(terrain0[:, None], p['geometry_point0_w'], p['geometry_point1_w']).to(points)
        for i in eligible.nonzero().flatten().tolist():
            rows[int(p['sample'][i])].append((terrain[i], int(p['primary_surface'][i])))
    else:
        raw = observed['pairs']
        if len(points) == 1 and (not raw or isinstance(raw[0], dict)):
            raw = [raw]
        catalogs = observed.get('surface_catalog_by_sample',
                                [observed.get('surface_catalog')] * len(points))
        if len(raw) != len(points) or len(catalogs) != len(points):
            raise ValueError('Current terrain-witness batch/catalog mismatch')
        from somaforge_core.contact_face_selection import select_contact_pairs
        for b, pairs in enumerate(raw):
            selected = select_contact_pairs([pairs], catalogs[b])
            for p in selected['contact_pairs'][0]:
                if 'terrain_position_w' not in p:
                    raise ValueError('Missing actual Newton terrain witness position')
                rows[b].append((points.new_tensor(p['terrain_position_w']), int(p['surface'])))
    capacity = 64
    for key in ('scan_points_world', 'scan_surface', 'scan_valid', 'scan_point_id'):
        value = packet[key]
        shape = (len(points), capacity, *value.shape[2:])
        extra = value.new_zeros(shape) if key in ('scan_points_world', 'scan_valid') else value.new_full(shape, -1)
        packet[key] = torch.cat((value, extra), dim=1)
    if 'scan_pixel_index' in packet:
        # Actual tactile witnesses have no image pixel. They must never borrow
        # a nearby pixel's visual feature or acquire fabricated camera depth.
        pixel = packet['scan_pixel_index']
        packet['scan_pixel_index'] = torch.cat((pixel, pixel.new_full((len(points), capacity), -1)), 1)
    first = points.shape[1]
    for b, row in enumerate(rows):
        if len(row) > capacity:
            raise ValueError('Current measured contact mesh capacity exceeded')
        for j, (point, surface) in enumerate(row):
            packet['scan_points_world'][b, first+j] = point
            packet['scan_surface'][b, first+j] = surface
            packet['scan_valid'][b, first+j] = True
            coord = torch.round(point.double()*1e6).long()
            packet['scan_point_id'][b, first+j] = ((coord[0]*73856093) ^ (coord[1]*19349663)
                ^ (coord[2]*83492791) ^ (surface*2654435761)) & ((1 << 63)-1)
    if not bool(packet['scan_valid'].any(-1).all()):
        raise ValueError('No observed depth/tactile interaction surface; state is unknown')
    return packet


def mesh_location_labels(points_world, active, surface, packet, *, radius_m=None):
    """Same-surface 3D supervision; uncovered targets remain unknown.

    Coverage tolerance derives from mesh sampling resolution. It is only an
    observation/label representability check, never a contact truth test.
    """
    candidates = packet['scan_points_world']
    allowed = packet['scan_valid'][:, None] & (packet['scan_surface'][:, None] == surface[..., None])
    distance = torch.cdist(points_world, candidates).square().masked_fill(~allowed, torch.inf)
    error, cell = distance.min(-1)
    radius = 2*packet['scan_spacing_m'] if radius_m is None else radius_m
    if radius.shape != (len(candidates),) or not bool(torch.isfinite(radius).all()) or bool((radius < 0).any()):
        raise ValueError('Location label coverage radius must be finite nonnegative [B]')
    # Unmaterialized corpus rows have zero spacing and no observed candidates.
    # They remain explicitly uncovered, never synthetic positive labels.
    valid = active & (radius[:, None] > 0) & torch.isfinite(error) & (error <= radius[:, None].square())
    return torch.where(valid, cell, torch.zeros_like(cell)), valid
