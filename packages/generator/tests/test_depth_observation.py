import math
from types import SimpleNamespace

import pytest
import torch

from generator.depth_observation import DepthScanEncoder, image_candidate_keys, depth_region_features
from generator.predictor_architecture import build_position_predictor, load_position_predictor
from somaforge_core.robot_assets import encode_robot_asset_json
from somaforge_core.world_mesh_scan import depth_world_mesh


@pytest.mark.parametrize('schema', ['full1000_position_predictor_v1', 'full1000_position_predictor_v2'])
@pytest.mark.parametrize('profile', ['torso_depth_128x128_fov120_pitch55_v1',
                                   'torso_depth_96x96_fov110_pitch60_v1',
                                   'torso_depth_96x96_fov110_pitch60_yaw_v1',
                                   'root_depth_96x96_fov90_pitch75_yaw_v1',
                                   'root_depth_87x58_fov89p5_pitch85_yaw_v1'])
def test_camera_profile_roundtrip_and_legacy_checkpoint_guard(schema, profile):
    from somaforge_core.world_mesh_scan import LEGACY_DEPTH_PROFILE
    config = dict(architecture=schema, width=24, layers=1, location_width=4,
        body_geometry=False, part_geometry=False, region_plan=True, unified_contact=True,
        event_roles=True, role_validity=True, world_surface_scan=True, depth_observation=True)
    legacy = build_position_predictor(config, schema=schema)
    saved = dict(schema=schema, config=dict(config), model=legacy.state_dict(),
        architecture_contract=legacy.architecture_contract(), robot_asset_json=encode_robot_asset_json())
    assert load_position_predictor(saved).depth_scan_profile == LEGACY_DEPTH_PROFILE
    saved['config']['depth_scan_profile'] = profile
    with pytest.raises(ValueError, match='camera calibration'):
        load_position_predictor(saved)
    wide = build_position_predictor(saved['config'], schema=schema)
    saved['architecture_contract'] = wide.architecture_contract()
    assert load_position_predictor(saved).depth_scan_profile == profile


def observation():
    h, w = 8, 10
    a = math.radians(35)
    rotation = torch.tensor([[[0., -math.sin(a), math.cos(a)], [-1., 0., 0.],
                              [0., -math.cos(a), -math.sin(a)]]], dtype=torch.float64)
    k = torch.tensor([[[8., 0., 4.5], [0., 8., 3.5], [0., 0., 1.]]], dtype=torch.float64)
    origin = torch.tensor([[0., 0., 1.5]], dtype=torch.float64)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1).double()
    ray = pixels @ k[0].inverse().T @ rotation[0].T
    depth = -origin[:, None, 2:3]/ray[..., 2]
    packet, _ = depth_world_mesh(depth, k, rotation, origin,
        torch.ones_like(depth, dtype=torch.bool), torch.zeros_like(depth, dtype=torch.long), spacing_m=.1)
    packet = {k: v.float() if v.is_floating_point() else v for k, v in packet.items()}
    q = torch.zeros(1, 36); q[:, 2] = .8; q[:, 3] = 1.
    return dict(current_q=q, current_contact=torch.ones(1, 6, dtype=torch.bool),
        current_anchor=torch.zeros(1, 6, 3), contact_relation_points=torch.zeros(1, 6, 1, 3), current_contact_relations=torch.ones(1, 6, 1, dtype=torch.bool),
        contact_surface_ids=torch.zeros(1, 1, dtype=torch.long), **packet)


@pytest.mark.parametrize('schema', ['full1000_position_predictor_v1', 'full1000_position_predictor_v2'])
def test_depth_features_drive_role_location_and_pose_without_heightmap(schema):
    torch.manual_seed(2)
    config = dict(architecture=schema, width=24, layers=1, location_width=4,
        body_geometry=False, part_geometry=False, region_plan=True, unified_contact=True,
        event_roles=True, role_validity=True, world_surface_scan=True, depth_observation=True)
    model = build_position_predictor(config, schema=schema).eval()
    # Any remaining regional height lookup must fail, even if its contribution
    # happens to be initialized to zero and prediction equality would hide it.
    def forbidden(*args, **kwargs):
        raise AssertionError('heightmap geometry was evaluated')
    model.region_geometry.forward = forbidden
    assert not any('.height.' in '.'+key for key in model.state_dict())
    obs = observation()
    first = model(**obs)
    changed = dict(obs, depth_image=obs['depth_image'].clone())
    changed['depth_image'][:, :, 5:] += .8  # Counterfactual dependency probe; XYZ held fixed.
    second = model(**changed)
    assert not torch.equal(first.role_logits, second.role_logits)
    assert not torch.equal(first.location_logits, second.location_logits)
    assert not torch.equal(first.qpos, second.qpos)
    # Learned image descriptors participate in location supervision as well.
    loss = first.role_logits.square().mean()+first.location_logits.square().mean()+first.qpos.square().mean()
    loss.backward()
    for prefix in (('depth.',) if schema.endswith('v1') else
                   ('planner_observation.depth.', 'executor_observation.depth.')):
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                   for name, p in model.named_parameters() if name.startswith(prefix))
    assert model.location_scan_key.weight.grad.abs().sum() > 0
    # Candidate geometry stays on the raw plane, regardless of CNN features.
    assert first.candidate_points_local[..., 2].add(.8).abs().max() < 1e-6
    with pytest.raises(ValueError, match='does not accept a heightmap'):
        model(**obs, heightmap=torch.zeros(1, 4, 4))
    saved = dict(schema=schema, config=config, model=model.state_dict(),
                 architecture_contract=model.architecture_contract(), robot_asset_json=encode_robot_asset_json())
    loaded = load_position_predictor(saved).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(**obs).qpos, first.qpos)
    saved['architecture_contract'].pop('environment_observation')
    with pytest.raises(ValueError, match='2D depth observation contract'):
        load_position_predictor(saved)


def test_unknown_depth_values_cannot_change_image_encoding():
    obs = observation(); valid = obs['depth_valid'].clone(); valid[:, :, 4] = False
    a = obs['depth_image'].clone(); b = a.clone()
    a[:, :, 4] = float('nan'); b[:, :, 4] = 10000.
    encoder = DepthScanEncoder(24)
    torch.testing.assert_close(encoder(a, valid, obs['depth_intrinsics'])[0],
                               encoder(b, valid, obs['depth_intrinsics'])[0])
    with pytest.raises(ValueError, match='Invalid observed optical depth'):
        encoder(a, obs['depth_valid'], obs['depth_intrinsics'])


def test_pixel_readout_preserves_image_order_and_never_invents_tactile_pixels():
    features = torch.arange(6.).reshape(1, 1, 2, 3)
    pixels = torch.tensor([[0, 1, 2, 4, 10, 19, -1]])
    read = image_candidate_keys(torch.nn.Identity(), features, pixels, torch.ones(1, 4, 5))
    assert read[..., 0].tolist() == [[0., 0., 1., 2., 3., 5., 0.]]
    with pytest.raises(ValueError, match='pixel identity'):
        image_candidate_keys(torch.nn.Identity(), features, torch.tensor([[20]]), torch.ones(1, 4, 5))


def test_region_depth_residual_is_camera_axis_and_unknown_is_masked():
    obs = dict(depth_image=torch.full((1, 5, 5), 1.5), depth_valid=torch.ones(1, 5, 5, dtype=torch.bool),
        depth_intrinsics=torch.tensor([[[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]]]),
        depth_camera_rotation_world=torch.eye(3)[None], depth_camera_position_world=torch.zeros(1, 3))
    region = SimpleNamespace(landmark=torch.zeros(1, 1, 3), extent=torch.ones(1, 1, 3),
                             valid=torch.ones(1, 1, dtype=torch.bool))
    q = torch.zeros(1, 36); pos = torch.tensor([[[0., 0., 2.]]]); rotation = torch.eye(3)[None, None]
    features = depth_region_features(region, pos, rotation, q, torch.eye(3)[None], obs)
    assert features[0, 0, -2:].tolist() == [5., 1.]
    obs['depth_valid'][:, 2, 2] = False
    features = depth_region_features(region, pos, rotation, q, torch.eye(3)[None], obs)
    assert features[0, 0, -2:].tolist() == [0., 0.]


@pytest.mark.parametrize('schema', ['full1000_position_predictor_v1', 'full1000_position_predictor_v2'])
def test_keep_binds_actual_anchor_without_camera_candidate_in_all_branches(schema):
    from generator.planned_contact_predictor import contact_plan_objective, plan_points_in_pose_frame
    config = dict(architecture=schema, width=24, layers=1, location_width=4,
        body_geometry=False, part_geometry=False, region_plan=False, unified_contact=True,
        event_roles=True, role_validity=True, world_surface_scan=True, depth_observation=True)
    model = build_position_predictor(config, schema=schema).eval()
    with torch.no_grad():
        model.role_head.weight.zero_()
        model.role_head.bias.copy_(torch.tensor([0., 0., 20., 0.]))
    obs = observation()
    # Camera sees a different surface; held points have no corresponding pixel.
    obs['scan_surface'].fill_(7)
    anchors = torch.arange(18.).reshape(1, 6, 1, 3)/100
    obs['contact_relation_points'] = anchors
    roles = torch.full((1, 6), 2, dtype=torch.long)
    for extra in ({}, dict(teacher_mask=torch.tensor([True]), teacher_role=roles,
            teacher_contact=roles != 0, teacher_cell=torch.full_like(roles, -1)),
            dict(repair_mask=torch.tensor([True]), repair_role=roles,
                 repair_points_world=torch.full((1, 6, 3), 100.),
                 repair_surface=torch.zeros_like(roles), repair_point_id=torch.full_like(roles, -1))):
        result = model(**obs, **extra)
        torch.testing.assert_close(plan_points_in_pose_frame(obs['current_q'], result.conditioned_points_local), anchors.squeeze(2))
        assert (result.conditioned_cell == -1).all()
        assert (result.conditioned_point_id == -1).all()
        assert (result.conditioned_surface == 0).all()
    target = dict(role=roles, contact_cell=torch.full_like(roles, -1), contact_cell_valid=roles.bool())
    loss, metrics = contact_plan_objective(result, target)
    assert metrics['plan_location_loss'].item() == 0
    loss.sum().backward()
    assert model.location_query.weight.grad is None or model.location_query.weight.grad.count_nonzero() == 0
    # A different scan cannot move a keep anchor.
    obs['scan_points_world'] = obs['scan_points_world'] + 10
    changed = model(**obs)
    torch.testing.assert_close(changed.conditioned_points_local, result.conditioned_points_local)
    obs.pop('contact_relation_points')
    with pytest.raises(ValueError, match='actual current contact points'):
        model(**obs)
