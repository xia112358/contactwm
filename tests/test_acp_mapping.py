"""Temporal mapping geometry, visibility, neural mode supervision and native rays."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.acp.mapping import (
    MAPPING_SCHEMA, ElevationMap, MappingConfig, MappingNetwork, MappingPipeline,
    compose_rotation, load_mapping_network, mode_labels, rasterize, rotate,
)


def small_map(envs=1):
    return ElevationMap(envs, config=MappingConfig(resolution=0.1, map_size=12, max_age=2))


def fuse(mapper, points, time=0, sensor=None, std=0.03):
    points = torch.tensor(points, dtype=torch.float32)
    if points.ndim == 2:
        points = points[None].expand(mapper.num_envs, -1, -1)
    sensor = torch.tensor(sensor, dtype=torch.float32) if sensor is not None else torch.tensor([[0., 0., 1.]]).expand(mapper.num_envs, -1)
    mapper.fuse(points, torch.isfinite(points).all(-1), torch.full(points.shape[:2], std), sensor, time)


def test_raster_keeps_highest_and_does_not_fabricate_unknown_ground():
    cfg = MappingConfig()
    points = torch.tensor([[[0.01, 0.01, -1.], [0.02, 0.02, -0.5], [4., 0., 7.], [float("inf"), 0., 0.]]])
    height, valid = rasterize(points, torch.ones(1, 4, dtype=torch.bool), cfg)
    assert valid.sum() == 1
    assert height[0, 100, 50] == -0.5
    assert not height[~valid].any()


def test_world_history_survives_motion_without_map_wraparound():
    mapper = small_map()
    fuse(mapper, [[0.15, 0.05, 0.4]])
    mapper.recenter(torch.tensor([[0.3, 0., 1.]]))
    height, valid = mapper.query(torch.tensor([[[0.15, 0.05], [0.65, 0.05]]]), 0.5, fill=False)
    assert valid.tolist() == [[True, False]]
    assert height[0, 0] == pytest.approx(0.4)
    mapper.recenter(torch.tensor([[10., 0., 1.]]))
    assert not torch.isfinite(mapper.variance).any()


def test_fusion_precision_duplicate_frames_and_stale_history():
    mapper = small_map()
    fuse(mapper, [[0.15, 0.05, 0.4]])
    before = mapper.variance.clone()
    fuse(mapper, [[0.15, 0.05, 0.45]])
    assert torch.equal(before, mapper.variance)
    fuse(mapper, [[0.15, 0.05, 0.45]], time=0.1)
    height, valid = mapper.query(torch.tensor([[[0.15, 0.05]]]), 0.1, fill=False)
    assert valid.item() and 0.4 < height.item() < 0.45
    assert not mapper.query(torch.tensor([[[0.15, 0.05]]]), 3, fill=False)[1].any()
    with pytest.raises(ValueError, match="Out-of-order"):
        fuse(mapper, [[0.15, 0.05, 0.4]], time=0)


def test_overhang_prior_protects_rung_but_allows_shallow_view_update():
    mapper = small_map()
    fuse(mapper, [[0.15, 0.05, 0.5]])
    fuse(mapper, [[0.15, 0.05, 0.0]], time=0.1)
    xy = torch.tensor([[[0.15, 0.05]]])
    assert mapper.query(xy, 0.1, fill=False)[0].item() == 0.5
    # Lower the sensor: the new point is now seen at elevation > -40 degrees.
    fuse(mapper, [[0.15, 0.05, 0.0]], time=0.2, sensor=[[0., 0., 0.05]])
    assert mapper.query(xy, 0.2, fill=False)[0].item() == 0.0


def test_highest_incoming_point_owns_uncertainty_and_envs_reset_independently():
    mapper = small_map(2)
    points = torch.tensor([[[0.15, 0.05, 0.5], [0.15, 0.05, 0.0]]]).expand(2, -1, -1)
    mapper.fuse(points, torch.ones(2, 2, dtype=torch.bool), torch.tensor([[0.2, 0.01]]).expand(2, -1),
                torch.tensor([[0., 0., 1.]]).expand(2, -1), 0)
    assert torch.allclose(mapper.variance[torch.isfinite(mapper.variance)], torch.full((2,), 0.04))
    mapper.reset([0])
    _, valid = mapper.query(torch.tensor([[[0.15, 0.05]]]).expand(2, -1, -1), 0, fill=False)
    assert valid.tolist() == [[False], [True]]
    fuse(mapper, [[[0.15, 0.05, 0.7]], [[0.15, 0.05, 9.0]]], time=0)
    height = mapper.query(torch.tensor([[[0.15, 0.05]]]).expand(2, -1, -1), 0, fill=False)[0]
    assert height[:, 0].tolist() == pytest.approx([0.7, 0.5])


def test_rotation_and_roll_pitch_gravity_alignment():
    qz = torch.tensor([[0., 0., 2**-0.5, 2**-0.5]])
    qx = torch.tensor([[2**-0.5, 0., 0., 2**-0.5]])
    p = torch.tensor([[[1., 0., 0.]]])
    assert torch.allclose(rotate(p, qz), torch.tensor([[[0., 1., 0.]]]), atol=1e-6)
    assert torch.allclose(rotate(rotate(p, qx), qz), rotate(p, compose_rotation(qz, qx)), atol=1e-6)
    pipeline = MappingPipeline(1)
    q = compose_rotation(qz, qx)
    world_delta = torch.tensor([[[0.15, 0.15, -1.]]])
    inverse = q.clone()
    inverse[:, :3] *= -1
    local = rotate(world_delta, inverse)
    pipeline.integrate(local, torch.ones(1, 1, dtype=torch.bool), torch.tensor([[0., 0., 1.]]), q, 0)
    points = pipeline.map.height[torch.isfinite(pipeline.map.variance)]
    assert len(points) == 1
    assert points.item() == pytest.approx(0, abs=1e-6)


def test_history_scan_rotates_with_robot_and_empty_frame_keeps_evidence():
    pipeline = MappingPipeline(1)
    fuse(pipeline.map, [[0.015, 0.315, 0.5]])
    # No returns on the next scan: retain the earlier surface.
    pipeline.integrate(torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([[0., 0., 1.]]), torch.tensor([[0., 0., 0., 1.]]), 0.1)
    scan, valid = pipeline.scan(torch.tensor([[0.015, 0.015, 1.]]),
                               torch.tensor([[0., 0., 2**-0.5, 2**-0.5]]), 0.1)
    index = 11*11+5  # local x = +0.3, local y = 0 => world +y
    assert valid[0, index]
    assert scan[0, index].tolist() == pytest.approx([0.3, 0, -0.5], abs=1e-6)


def test_local_fill_does_not_invent_large_unobserved_regions():
    mapper = small_map()
    fuse(mapper, [[0.15, 0.05, 0.5]])
    xy = torch.tensor([[[0.25, 0.05], [-0.45, -0.45]]])
    assert mapper.query(xy, 0, fill=False)[1].tolist() == [[False, False]]
    h, valid = mapper.query(xy, 0)
    assert valid.tolist() == [[True, False]]
    assert h.tolist() == [[0.5, 0.0]]


def test_mode_labels_known_unknown_and_optimizer_gradients():
    height = torch.zeros(2, 9, 7)
    valid = torch.ones_like(height, dtype=torch.bool)
    height[:, 4, 3] = 9
    target = torch.zeros_like(height)
    known = valid.clone()
    known[:, 0, 0] = False
    labels = mode_labels(height, valid, target, known)
    assert labels[0, 0, 0] == -100
    assert labels[0, 4, 3] == 2  # Both measured and neighborhood mean are wrong.
    assert labels[0, 8, 6] == 0
    model = MappingNetwork()
    loss = torch.nn.functional.cross_entropy(model(height, valid), labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    prediction, mask, std = model.predict(height, torch.zeros_like(valid))
    assert not mask.any() and not prediction.any()
    assert torch.isfinite(std).all()


def test_mapping_export_records_only_valid_history_and_refuses_overwrite(tmp_path):
    mapper = small_map()
    fuse(mapper, [[0.15, 0.05, 0.5]])
    path = tmp_path/"map.npz"
    mapper.export(path)
    with np.load(path, allow_pickle=False) as data:
        assert data["points_w"].shape == (1, 3)
        assert data["valid"].sum() == 1
        assert str(data["schema"]) == MAPPING_SCHEMA
    with pytest.raises(FileExistsError):
        mapper.export(path)


def test_fit_checkpoint_resume_and_neural_reconstruction(tmp_path):
    from holosoma.acp.fit_mapping import Options, fit
    dataset = tmp_path/"train.npz"
    h = np.zeros((2, 9, 7), np.float32)
    np.savez(dataset, height=h, target=h, valid=np.ones_like(h, bool), target_valid=np.ones_like(h, bool))
    output = tmp_path/"model.pt"
    report = fit(Options(dataset, output, steps=15, batch_size=2))
    assert report["last_loss"] < report["first_loss"]
    model = load_mapping_network(output, "cpu")
    assert not model.training
    resumed = fit(Options(dataset, tmp_path/"resumed.pt", steps=1, resume=output))
    assert resumed["trained_steps"] == 16
    untrained = tmp_path/"untrained.pt"
    torch.save({"schema": MAPPING_SCHEMA, "trained_steps": 0}, untrained)
    with pytest.raises(ValueError, match="trained"):
        load_mapping_network(untrained, "cpu")


def test_offline_recording_exports_episode_maps_and_both_scans(tmp_path):
    from holosoma.acp.map_recording import Options, reconstruct
    recording = tmp_path/"recording.npz"
    np.savez(recording, points_sensor=np.tile([[[0.015, 0.015, -1.]]], (3, 1, 1)).astype(np.float32),
        valid=np.ones((3, 1), bool), position_w=np.tile([[0, 0, 1]], (3, 1)).astype(np.float32),
        quaternion_xyzw=np.tile([[0, 0, 0, 1]], (3, 1)).astype(np.float32), timestamps=[0., 0.1, 0.],
        root_position_w=np.tile([[0, 0, 1]], (3, 1)).astype(np.float32),
        root_quaternion_xyzw=np.tile([[0, 0, 0, 1]], (3, 1)).astype(np.float32), episode_start=[True, False, True])
    report = reconstruct(Options(recording, tmp_path/"result"))
    assert report["episodes"] == 2
    with np.load(tmp_path/"result/scans.npz") as data:
        assert data["planner_points"].shape == (3, 187, 3)
        assert data["tracker_heights"].shape == (3, 9, 7)
        assert data["planner_valid"].any()
    assert (tmp_path/"result/map_0001.npz").is_file()


def test_native_terrain_ray_adapter_rate_reset_and_flat_ground():
    pytest.importorskip("warp")
    from holosoma.utils.warp_utils import convert_to_wp_mesh
    from holosoma.acp.perception import ACPPerception
    vertices = np.array([[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], np.float32)
    mesh = convert_to_wp_mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]]), "cpu")
    simulator = SimpleNamespace(body_names=["torso_link"], _rigid_body_pos=torch.tensor([[[0., 0., 0.8]]]),
        _rigid_body_rot=torch.tensor([[[0., 0., 0., 1.]]]), robot_root_states=torch.tensor([[0., 0., 0.8, 0., 0., 0., 1.]]))
    env = SimpleNamespace(acp_settings={"terrain_observation": "fused_raw", "lidar_points": 2000},
        device="cpu", num_envs=1, simulator=simulator,
        terrain_manager=SimpleNamespace(get_state=lambda name: SimpleNamespace(warp_mesh=mesh)))
    perception = ACPPerception(env)
    perception.update(0)
    assert torch.isfinite(perception.pipeline.map.variance).sum() > 100
    first = perception.pipeline.map.stamp.clone()
    perception.update(0.02)
    assert torch.equal(first, perception.pipeline.map.stamp)
    perception.update(0.1)
    assert perception.last_scan.item() == pytest.approx(0.1)
    perception.reset([0])
    assert not torch.isfinite(perception.pipeline.map.variance).any()
    perception.update(0.1)
    assert torch.isfinite(perception.pipeline.map.variance).any()


def test_actor_fused_observations_never_query_privileged_terrain(monkeypatch):
    from holosoma.acp import observations as obs
    from holosoma.acp import perception
    calls = []

    def empty_scan(env, rows=17, columns=11):
        calls.append((rows, columns))
        return torch.zeros(1, rows*columns, 3), torch.zeros(1, rows*columns, dtype=torch.bool)

    monkeypatch.setattr(perception, "fused_scan", empty_scan)
    monkeypatch.setattr(obs, "goal", lambda env: torch.zeros(1, 3))
    cmd = SimpleNamespace(time_steps=torch.tensor([0]), acp_target_w=torch.zeros(1, 9, 3),
        acp_target_occupancy=torch.zeros(1, 9, dtype=torch.bool), _reference_env_origins=torch.zeros(1, 3))

    def forbidden(*args, **kwargs):
        raise AssertionError("Unknown actor scan must not call ground truth or all-masked attention")

    env = SimpleNamespace(acp_settings={"terrain_observation": "fused_raw", "stage": 2,
        "input_start": 0, "anneal_steps": 1, "observation_noise": True},
        simulator=SimpleNamespace(robot_root_states=torch.tensor([[0., 0., 1., 0., 0., 0., 1.]])),
        terrain_manager=SimpleNamespace(get_state=forbidden), command_manager=SimpleNamespace(get_state=lambda _: cmd),
        _holosoma_step_cache={}, acp_history=torch.zeros(1, 5, 93), acp_planner=forbidden,
        acp_iteration=2, num_envs=1, device="cpu", is_evaluating=False)
    assert not obs.tracker_scan(env).any()
    assert not obs.contact_targets(env).any()
    assert calls == [(9, 7), (17, 11)]


def test_replay_rejects_entirely_unknown_scans():
    from holosoma.acp.replay import HindsightReplay
    frame = {"active": np.zeros(9, bool), "force": np.zeros((9, 3), np.float32),
        "position": np.zeros((9, 3), np.float32), "known": np.ones(9, bool),
        "root": np.zeros(3, np.float32), "yaw": np.asarray(0, np.float32),
        "history": np.zeros(465, np.float32), "goal": np.zeros(3, np.float32),
        "terrain": np.zeros((187, 3), np.float32), "valid": np.zeros(187, bool)}
    replay = HindsightReplay()
    assert not replay.admit([frame]*120, success=True, initial_phase=0, final_phase=1, tile=0)


def test_mapping_configuration_requires_weights_and_valid_mount():
    from holosoma.acp.train import Options, build_config
    with pytest.raises(ValueError, match="mapping-checkpoint"):
        build_config(Options(manifest="unused.json", terrain_observation="fused_neural"))
    with pytest.raises(ValueError, match="quaternion"):
        build_config(Options(manifest="unused.json", lidar_rotation_xyzw=(0., 0., 0., 0.)))
    config = build_config(Options(manifest="unused.json"))
    settings = config.command.setup_terms["motion_command"].params["acp"]
    assert settings["terrain_observation"] == "fused_raw"
    assert settings["lidar_points"] == 20_000
