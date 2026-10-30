import math
from types import SimpleNamespace

import pytest
import torch

from somaforge_core.world_mesh_scan import (
    depth_world_mesh, DepthWorldMeshScanner, append_current_contact_nodes, mesh_location_labels)


def camera():
    angle = math.radians(35.)
    rotation = torch.tensor([[[0., -math.sin(angle), math.cos(angle)],
                             [-1., 0., 0.], [0., -math.cos(angle), -math.sin(angle)]]], dtype=torch.float64)
    k = torch.tensor([[[8., 0., 2.5], [0., 8., 1.5], [0., 0., 1.]]], dtype=torch.float64)
    origin = torch.tensor([[0., 0., 1.5]], dtype=torch.float64)
    yy, xx = torch.meshgrid(torch.arange(4, dtype=torch.float64), torch.arange(6, dtype=torch.float64), indexing='ij')
    rays = torch.stack((xx, yy, torch.ones_like(xx)), -1) @ torch.linalg.inv(k[0]).T @ rotation[0].T
    return k, rotation, origin, rays


def test_tilted_camera_preserves_horizontal_world_planes_and_sharp_step():
    k, rotation, origin, rays = camera()
    surface = torch.zeros(1, 4, 6, dtype=torch.long); surface[:, :, 3:] = 1
    world_height = surface.double() * .7
    depth = (world_height-origin[:, None, 2:3]) / rays[..., 2]
    packet, audit = depth_world_mesh(depth, k, rotation, origin, torch.ones_like(surface, dtype=torch.bool),
                                    surface, spacing_m=.1)
    assert packet['scan_valid'].all()
    torch.testing.assert_close(packet['scan_points_world'][..., 2], world_height.flatten(1), atol=1e-14, rtol=0)
    torch.testing.assert_close(audit['normals_world'], torch.tensor([0., 0., 1.], dtype=torch.float64).expand(1, 24, 3),
                               atol=1e-13, rtol=0)
    face_surface = packet['scan_surface'][:, audit['faces']]
    assert (face_surface[audit['face_valid']] == face_surface[audit['face_valid']][:, :1]).all()
    assert (~audit['face_valid']).sum() == 6  # two triangles per boundary quad
    # Rotating the camera never turns the reconstructed floor into a ramp.
    assert not torch.any((packet['scan_points_world'][..., 2] > 1e-12)
                         & (packet['scan_points_world'][..., 2] < .7-1e-12))


def test_depth_holes_are_not_filled_or_connected():
    k, rotation, origin, rays = camera()
    depth = -origin[:, None, 2:3] / rays[..., 2]
    valid = torch.ones_like(depth, dtype=torch.bool); valid[:, :, 2] = False
    depth[:, :, 2] = torch.nan
    surface = torch.zeros_like(depth, dtype=torch.long); surface[:, :, 2] = -1
    packet, audit = depth_world_mesh(depth, k, rotation, origin, valid, surface, spacing_m=.1)
    assert torch.isfinite(packet['scan_points_world']).all()
    assert not packet['scan_valid'].reshape(1, 4, 6)[:, :, 2].any()
    assert (valid.flatten(1)[:, audit['faces']][audit['face_valid']]).all()


def test_nonrigid_camera_calibration_is_rejected():
    k, rotation, origin, rays = camera()
    rotation[:, 0] *= 2
    with pytest.raises(ValueError, match='rigid world rotation'):
        depth_world_mesh(torch.ones(1, 4, 6, dtype=torch.float64), k, rotation, origin,
                         torch.ones(1, 4, 6, dtype=torch.bool), torch.zeros(1, 4, 6, dtype=torch.long), spacing_m=.1)


def test_renderer_uses_world_triangles_and_marks_only_upward_faces_selectable():
    # Ground, box top, and a front wall all participate in actual ray occlusion.
    def plane(z, x0, x1):
        return [[[x0, -5., z], [x1, -5., z], [x0, 5., z]],
                [[x1, 5., z], [x0, 5., z], [x1, -5., z]]]
    catalog = [dict(surface=0, normal_w=[0., 0., 1.], triangles_w=plane(0., -5., 5.)),
               dict(surface=1, normal_w=[0., 0., 1.], triangles_w=plane(.7, 1., 2.)),
               dict(surface=2, normal_w=[-1., 0., 0.], triangles_w=[[[1., -5., 0.], [1., 5., 0.], [1., -5., .7]],
                                          [[1., 5., .7], [1., -5., .7], [1., 5., 0.]]])]
    scanner = DepthWorldMeshScanner(dict(surface_catalog=catalog, provenance=dict(model_fingerprint='test')),
                                    width=30, height=20)
    fk = SimpleNamespace(link_poses=lambda q, names: (q[:, None, :3], torch.eye(3, dtype=q.dtype)[None, None].expand(len(q), 1, -1, -1)))
    q = torch.zeros(1, 36); q[:, 2] = 1.; q[:, 3] = 1.
    packet = scanner.scan(q, fk)
    assert packet['scan_valid'].any()
    for surface, height in ((0, 0.), (1, .7)):
        mask = packet['scan_valid'] & (packet['scan_surface'] == surface)
        assert mask.any()
        torch.testing.assert_close(packet['scan_points_world'][..., 2][mask], torch.full_like(packet['scan_points_world'][..., 2][mask], height))
    assert not (packet['scan_valid'] & (packet['scan_surface'] == 2)).any()
    assert scanner.contract()['cross_surface_faces'] is False
    assert scanner.contract()['contact_truth'] is False


def test_wide_camera_preserves_world_ground_and_versioned_calibration():
    from somaforge_core.world_mesh_scan import WIDE_DEPTH_PROFILE, LEGACY_DEPTH_PROFILE
    triangles = [[[-10., -10., 0.], [10., -10., 0.], [-10., 10., 0.]],
                 [[10., 10., 0.], [-10., 10., 0.], [10., -10., 0.]]]
    metadata = dict(surface_catalog=[dict(surface=0, normal_w=[0., 0., 1.], triangles_w=triangles)],
                    provenance=dict(model_fingerprint='test'))
    fk = SimpleNamespace(link_poses=lambda q, names: (q[:, None, :3],
        torch.eye(3, dtype=q.dtype)[None, None].expand(len(q), 1, -1, -1)))
    q = torch.zeros(1, 36); q[:, 2] = .8; q[:, 3] = 1.
    wide = DepthWorldMeshScanner(metadata, profile=WIDE_DEPTH_PROFILE)
    packet = wide.scan(q, fk)
    assert packet['depth_image'].shape == (1, 128, 128)
    assert wide.contract()['vertical_fov_deg'] == pytest.approx(120.)
    assert wide.contract()['depth_range_m'] == [.2, 3.]
    points = packet['scan_points_world'][packet['scan_valid']]
    assert len(points) > 0
    torch.testing.assert_close(points[:, 2], torch.zeros_like(points[:, 2]), atol=1e-6, rtol=0)
    legacy = DepthWorldMeshScanner(metadata)
    assert legacy.profile == LEGACY_DEPTH_PROFILE
    assert (legacy.width, legacy.height, legacy.pitch) == (87, 58, 27.)


def test_yaw_stabilization_ignores_tilt_but_preserves_mount_position_and_turning():
    from somaforge_core.world_mesh_scan import STABILIZED_DEPTH_PROFILE, COMPACT_DEPTH_PROFILE
    triangles = [[[-10., -10., 0.], [10., -10., 0.], [-10., 10., 0.]],
                 [[10., 10., 0.], [-10., 10., 0.], [10., -10., 0.]]]
    metadata = dict(surface_catalog=[dict(surface=0, normal_w=[0., 0., 1.], triangles_w=triangles)],
                    provenance=dict(model_fingerprint='test'))
    yaw, pitch, roll = .4, .5, -.3
    rz = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.],
                       [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]], dtype=torch.float64)
    ry = torch.tensor([[math.cos(pitch), 0., math.sin(pitch)], [0., 1., 0.],
                       [-math.sin(pitch), 0., math.cos(pitch)]], dtype=torch.float64)
    rx = torch.tensor([[1., 0., 0.], [0., math.cos(roll), -math.sin(roll)],
                       [0., math.sin(roll), math.cos(roll)]], dtype=torch.float64)
    rotations = torch.stack((rz, rz @ ry @ rx, torch.eye(3, dtype=torch.float64)))
    fk = SimpleNamespace(link_poses=lambda q, names: (q[:, None, :3], rotations[:, None]))
    q = torch.zeros(3, 36, dtype=torch.float64); q[:, 2] = .8; q[:, 3] = 1.
    stable = DepthWorldMeshScanner(metadata, profile=STABILIZED_DEPTH_PROFILE).scan(q, fk)
    rigid = DepthWorldMeshScanner(metadata, profile=COMPACT_DEPTH_PROFILE).scan(q, fk)
    assert stable['depth_image'].shape == (3, 96, 96)
    rotation = stable['depth_camera_rotation_world']
    torch.testing.assert_close(rotation[0], rotation[1])
    torch.testing.assert_close(rotation[0], rz @ rotation[2])
    torch.testing.assert_close(stable['depth_camera_position_world'], rigid['depth_camera_position_world'])
    assert not torch.allclose(rotation[1], rigid['depth_camera_rotation_world'][1])
    assert not torch.allclose(stable['depth_camera_position_world'][0], stable['depth_camera_position_world'][1])
    points = stable['scan_points_world'][stable['scan_valid']]
    assert len(points) > 0
    torch.testing.assert_close(points[:, 2], torch.zeros_like(points[:, 2]), atol=1e-6, rtol=0)


@pytest.mark.parametrize('profile', ['root_depth_96x96_fov90_pitch75_yaw_v1',
                                   'root_depth_87x58_fov89p5_pitch85_yaw_v1'])
def test_root_camera_ignores_articulation_and_tilt_and_rays_share_one_center(profile):
    from somaforge_core.g1_kinematics import _rpy_matrix
    from scipy.spatial.transform import Rotation
    triangles = [[[-10., -10., 0.], [10., -10., 0.], [-10., 10., 0.]],
                 [[10., 10., 0.], [-10., 10., 0.], [10., -10., 0.]]]
    metadata = dict(surface_catalog=[dict(surface=0, normal_w=[0., 0., 1.], triangles_w=triangles)],
                    provenance=dict(model_fingerprint='test'))
    def forbidden(*args):
        raise AssertionError('Root scan must not depend on articulated link poses')
    fk = SimpleNamespace(link_poses=forbidden)
    q = torch.zeros(3, 36, dtype=torch.float64); q[:, 2] = .8
    for i, angles in enumerate(((0., 0., .4), (-.3, .5, .4), (0., 0., -.7))):
        xyzw = Rotation.from_euler('xyz', angles).as_quat()
        q[i, 3:7] = torch.tensor(xyzw[[3, 0, 1, 2]])
    q[1, 7:] = .7  # Includes all torso joints; sensor stays fixed in root-yaw frame.
    q[2, :2] = torch.tensor([.2, -.1])
    scanner = DepthWorldMeshScanner(metadata, profile=profile)
    packet = scanner.scan(q, fk)
    torch.testing.assert_close(packet['depth_image'][0], packet['depth_image'][1])
    torch.testing.assert_close(packet['depth_camera_rotation_world'][0], packet['depth_camera_rotation_world'][1])
    yaw = q.new_tensor([.4, .4, -.7])
    offset = torch.stack((.2*yaw.cos(), .2*yaw.sin(), torch.full_like(yaw, .75)), -1)
    torch.testing.assert_close(packet['depth_camera_position_world'], q[:, :3]+offset)
    torch.testing.assert_close(packet['depth_camera_rotation_world'][2],
        _rpy_matrix((0., 0., -1.1)).double() @ packet['depth_camera_rotation_world'][0], atol=1e-7, rtol=1e-7)
    # Every hit must back-project from the same center through its raw pixel.
    h, w = packet['depth_image'].shape[-2:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pixel = torch.stack((x, y, torch.ones_like(x)), -1).reshape(-1, 3).double()
    rays = pixel @ packet['depth_intrinsics'][0].inverse().T @ packet['depth_camera_rotation_world'][0].T
    expected = packet['depth_camera_position_world'][0]+rays*packet['depth_image'][0].reshape(-1, 1)
    valid = packet['scan_valid'][0]
    assert int(valid.sum()) > 1
    torch.testing.assert_close(packet['scan_points_world'][0, valid], expected[valid])
    torch.testing.assert_close(expected[valid, 2], torch.zeros_like(expected[valid, 2]), atol=1e-6, rtol=0)
    assert not torch.allclose(rays[0], rays[-1])


def test_current_tactile_nodes_retain_exact_terrain_witness_and_surface():
    packet = dict(scan_points_world=torch.zeros(1, 1, 3), scan_surface=torch.full((1, 1), -1, dtype=torch.long),
                  scan_valid=torch.zeros(1, 1, dtype=torch.bool), scan_point_id=torch.full((1, 1), -1, dtype=torch.long),
                  scan_spacing_m=torch.tensor([.05]))
    observed = dict(schema='newton_device_witness_batch_v1', pairs=dict(
        eligible=torch.tensor([True, False]), sample=torch.tensor([0, 0]), primary_surface=torch.tensor([0, 1]),
        body_link0=torch.tensor([-1, -1]), body_link1=torch.tensor([4, 5]),
        geometry_point0_w=torch.tensor([[.5, .2, 0.], [1., 1., .7]]),
        geometry_point1_w=torch.tensor([[.5, .2, -.002], [1., 1., .698]])))
    result = append_current_contact_nodes(packet, observed)
    assert result['scan_valid'].sum() == 1
    torch.testing.assert_close(result['scan_points_world'][0, 1], observed['pairs']['geometry_point0_w'][0])
    assert result['scan_surface'][0, 1] == 0


def test_labels_match_full_xyz_and_surface_and_mark_unseen_targets_unknown():
    packet = dict(scan_points_world=torch.tensor([[[1., 0., 0.], [1., 0., .7]]]),
                  scan_surface=torch.tensor([[0, 1]]), scan_valid=torch.tensor([[True, True]]),
                  scan_spacing_m=torch.tensor([.05]))
    target = torch.tensor([[[1., 0., .7], [1., 0., 0.], [2., 0., .7], [1., 0., .7]]])
    ids, valid = mesh_location_labels(target, torch.ones(1, 4, dtype=torch.bool), torch.tensor([[1, 0, 1, 0]]), packet)
    assert ids.tolist() == [[1, 0, 0, 0]]
    assert valid.tolist() == [[True, True, False, False]]
