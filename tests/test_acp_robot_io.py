"""Deployment observation construction without a simulator or ground-truth body poses."""
from types import SimpleNamespace

import pytest
import torch

from holosoma.acp.robot_io import G1TrackerInput, RobotSample
from somaforge_core import G1_29DOF_JOINT_ORDER


def fixture():
    names = tuple(G1_29DOF_JOINT_ORDER)
    bodies = ("pelvis", "torso_link", "left_wrist_yaw_link")
    keys = ("actions", "base_ang_vel", "base_lin_vel", "dof_pos", "dof_vel", "motion_command",
            "motion_ref_pos_b", "motion_ref_ori_b", "pelvis_global_pos", "pelvis_global_lin_vel",
            "robot_body_pos_b", "robot_body_ori_b", "terrain_height_scan")
    terms = {k: SimpleNamespace(scale=1.0, clip=(-1., 1.) if k == "terrain_height_scan" else None) for k in keys}
    builder = G1TrackerInput(names, bodies, "torso_link", torch.zeros(1, 29), terms)
    sample = RobotSample(0., torch.tensor([[1., 2., .8]]), torch.tensor([[0., 0., 2**-.5, 2**-.5]]),
        torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 0., .2]]), torch.zeros(1, 29), torch.zeros(1, 29))
    qpos = torch.cat((sample.base_position_w, sample.base_quaternion_xyzw[:, [3, 0, 1, 2]], sample.joint_position), -1)
    anchor, _ = builder.fk.link_poses(qpos, ("torso_link",))
    inputs = dict(reference_joints=torch.zeros(1, 29), reference_velocities=torch.zeros(1, 29),
        reference_anchor_position_w=anchor[:, 0], reference_anchor_quaternion_xyzw=sample.base_quaternion_xyzw,
        previous_action=torch.zeros(1, 29), terrain_height_scan=torch.full((1, 49), 3.))
    return builder, sample, inputs


def test_fk_and_imu_odometry_build_the_tracker_observations():
    builder, sample, inputs = fixture()
    obs, evidence = builder.build(sample, **inputs)
    terms = evidence["raw_terms"]
    torch.testing.assert_close(terms["base_lin_vel"], torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(terms["motion_ref_pos_b"], torch.zeros(1, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(terms["motion_ref_ori_b"], torch.tensor([[1., 0., 0., 1., 0., 0.]]), atol=1e-6, rtol=0)
    expected = torch.cat([terms[k].clamp(-1, 1) if k == "terrain_height_scan" else terms[k]
                          for k in sorted(terms)], -1)
    torch.testing.assert_close(obs, expected)
    assert torch.isfinite(obs).all()


def test_sensor_time_is_monotonic_and_nonfinite_input_fails():
    builder, sample, inputs = fixture()
    builder.build(sample, **inputs)
    sample.timestamp = -.1
    with pytest.raises(ValueError, match="backwards"):
        builder.build(sample, **inputs)
    builder.reset()
    builder.build(sample, **inputs)
    sample.timestamp = float("nan")
    with pytest.raises(ValueError, match="finite"):
        builder.build(sample, **inputs)
    sample.timestamp = 0.
    sample.imu_angular_velocity_b[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        builder.build(sample, **inputs)


def test_joint_order_mapping_keeps_canonical_fk_consistent():
    builder, sample, inputs = fixture()
    sample.joint_position[:] = torch.linspace(-.1, .1, 29)
    _, first = builder.build(sample, **inputs)
    reversed_builder = G1TrackerInput(tuple(reversed(builder.joint_names)), builder.tracked_bodies,
        builder.anchor_body, builder.default, builder.terms)
    sample.joint_position = sample.joint_position.flip(-1)
    _, second = reversed_builder.build(sample, **inputs)
    torch.testing.assert_close(first["body_position_w"], second["body_position_w"])
    torch.testing.assert_close(first["body_rotation_w"], second["body_rotation_w"])


def test_fused_tracker_grid_uses_torso_yaw_native_xy_order_and_unknown_mask():
    builder, sample, inputs = fixture()
    captured = {}

    def query(xy, timestamp):
        captured["xy"] = xy
        assert timestamp == sample.timestamp
        valid = torch.ones(xy.shape[:-1], dtype=torch.bool)
        valid[:, 0] = False
        return xy[..., 0]+2*xy[..., 1], valid

    height, valid = builder.mapped_height_scan(sample, SimpleNamespace(map=SimpleNamespace(query=query)))
    xy = captured["xy"]
    # Root yaw is +90 degrees: the first seven x-grid steps advance world y.
    torch.testing.assert_close(xy[0, 1]-xy[0, 0], torch.tensor([0., .1]), atol=1e-6, rtol=0)
    torch.testing.assert_close(xy[0, 7]-xy[0, 0], torch.tensor([-.1, 0.]), atol=1e-6, rtol=0)
    expected = inputs["reference_anchor_position_w"][:, 2:3]-xy[..., 0]-2*xy[..., 1]-.5
    expected[:, 0] = 0
    torch.testing.assert_close(height, expected)
    assert not valid[0, 0] and valid.sum() == 48
