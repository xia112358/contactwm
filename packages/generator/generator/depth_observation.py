"""Ordered 2D optical-depth scan features, with raw-pixel interaction readout.

Convolutions operate on the camera image, never an unordered point cloud or
voxel volume. Feature downsampling does not resample the geometric depth.
"""
import torch
from torch import nn

from somaforge_core.world_mesh_scan import DEPTH_OBSERVATION_SCHEMA


def depth_observation_contract():
    return dict(schema=DEPTH_OBSERVATION_SCHEMA, encoder='2D image CNN; stride 2',
        channels=['optical_depth_m / 3', 'hit_valid', 'calibrated_ray_x', 'calibrated_ray_y'],
        environment='depth image only; no heightmap, point-cloud or voxel encoder',
        candidates='raw pixel world XYZ + corresponding image feature; tactile nodes have no pixel',
        region_features='projected landmark optical-depth residual; unknown pixels explicitly masked',
        geometry_resampling=False)


def predictor_observation(model, q, observed, scene, scene_ids, workers):
    """Same observation path for corpus generation, recursion and refinement."""
    from generator.role_validity import validity_inputs
    result = dict(current_q=q, current_contact=observed['contact_part_mask'],
                  current_anchor=observed['contact_position_w'])
    heightmap = None
    if not getattr(model, 'depth_observation', False):
        from generator.next_interaction_heightmap_v2 import render_root_yaw_box_heightmaps
        numpy = lambda v: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        heightmap = torch.as_tensor(render_root_yaw_box_heightmaps(*(numpy(scene[k]) for k in
            ('box_center', 'box_rotation', 'box_half_extents', 'ground_height')), numpy(q)), device=q.device)
        result['heightmap'] = heightmap
    result.update(validity_inputs(model, q, heightmap, observed,
        scan_provider=workers.scan, scene_ids=scene_ids))
    return result


class DepthScanEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(64, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.norm = nn.LayerNorm(width)

    def forward(self, depth_image, depth_valid, depth_intrinsics):
        if depth_image is None or depth_valid is None or depth_intrinsics is None:
            raise ValueError('Depth observation requires the calibrated 2D scan and validity mask')
        if (depth_image.ndim != 3 or depth_image.shape != depth_valid.shape
                or depth_valid.dtype != torch.bool or depth_intrinsics.shape != (len(depth_image), 3, 3)
                or min(depth_image.shape[-2:]) < 2):
            raise ValueError('Invalid calibrated depth image shapes/dtypes')
        if (not bool(torch.isfinite(depth_intrinsics).all())
                or bool((depth_valid & (~torch.isfinite(depth_image) | (depth_image <= 0))).any())):
            raise ValueError('Invalid observed optical depth/calibration')
        b, h, w = depth_image.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=depth_image.device, dtype=depth_image.dtype),
                               torch.arange(w, device=depth_image.device, dtype=depth_image.dtype), indexing='ij')
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1)
        rays = torch.einsum('bij,hwj->bihw', torch.linalg.inv(depth_intrinsics), pixels)
        channels = torch.cat((torch.where(depth_valid, depth_image/3., 0.)[:, None],
                              depth_valid[:, None].to(depth_image), rays[:, :2]), 1)
        feature = self.conv(channels)
        memory = self.norm(feature.flatten(2).transpose(1, 2))
        return memory, memory.transpose(1, 2).reshape_as(feature)


def image_candidate_keys(key_layer, image_features, scan_pixel_index, depth_image):
    """Read image features at each raw pixel; -1 denotes a tactile-only node."""
    b, h, w = depth_image.shape
    if (scan_pixel_index is None or scan_pixel_index.ndim != 2 or len(scan_pixel_index) != b
            or scan_pixel_index.dtype != torch.long
            or bool(((scan_pixel_index < -1) | (scan_pixel_index >= h*w)).any())):
        raise ValueError('Depth candidate is missing its raw image pixel identity')
    feature = key_layer(image_features.flatten(2).transpose(1, 2))
    # CNN centres have raw-pixel coordinates (2*y, 2*x); geometry is never
    # interpolated, even when neighbouring pixels belong to different layers.
    pixel = scan_pixel_index.clamp_min(0)
    index = (pixel.div(w, rounding_mode='floor').div(2, rounding_mode='floor') * image_features.shape[-1]
             + pixel.remainder(w).div(2, rounding_mode='floor'))
    return feature.gather(1, index[..., None].expand(-1, -1, feature.shape[-1])).masked_fill(
        (scan_pixel_index < 0)[..., None], 0.)


def depth_region_features(regions, position, rotation, current_q, basis, packet):
    """Project body landmarks into the depth image, retaining unknown evidence.

The residual is optical-axis distance, not a height or a contact definition.
Only the feature is sampled; no target point is lifted from this residual.
"""
    local = position[:, :, None]+torch.einsum('bpij,prj->bpri', rotation, regions.landmark)
    world = current_q[:, None, None, :3]+torch.einsum('bij,bprj->bpri', basis, local)
    camera = torch.einsum('bji,bprj->bpri', packet['depth_camera_rotation_world'],
                         world-packet['depth_camera_position_world'][:, None, None])
    projected = torch.einsum('bij,bprj->bpri', packet['depth_intrinsics'], camera)
    uv = projected[..., :2]/projected[..., 2:].clamp_min(1e-8)
    image, valid = packet['depth_image'], packet['depth_valid']
    h, w = image.shape[-2:]
    visible = ((camera[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= w-1)
               & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1) & regions.valid)
    pixel = uv.round().long()
    index = pixel[..., 1].clamp(0, h-1)*w+pixel[..., 0].clamp(0, w-1)
    visible &= valid.flatten(1).gather(1, index.flatten(1)).reshape_as(visible)
    measured = image.flatten(1).gather(1, index.flatten(1)).reshape_as(visible)
    gap = torch.where(visible, camera[..., 2]-measured, 0.)
    offset = local-position[:, :, None]
    feature = torch.cat((offset/.1, regions.extent[None].expand(len(local), -1, -1, -1)/.1,
                         gap[..., None]/.1, visible[..., None].to(local)), -1)
    return (feature*regions.valid[None, :, :, None]).flatten(2)
