from types import SimpleNamespace

import torch
from holosoma.managers.observation.terms import wbt


class _MotionCommandStub:
    def __init__(self, num_envs: int = 2, num_dofs: int = 29):
        self.motion_cfg = SimpleNamespace(
            body_names_to_track=[
                "pelvis",
                "torso_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "left_knee_link",
                "right_knee_link",
            ],
            body_name_ref=["torso_link"],
        )
        self.robot_ref_pos_w = torch.zeros(num_envs, 3)
        self.robot_ref_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
        self._joint_pos = torch.arange(num_envs * num_dofs, dtype=torch.float32).reshape(num_envs, num_dofs)
        self._joint_vel = self._joint_pos + 100.0
        self.future_ref_calls = 0
        self._policy_reference_window_cache = None

    def future_joint_pos_vel(self, offsets):
        assert offsets in ((1, 2, 4, 8), (0, 1, 2, 4, 8))
        frame_count = len(offsets)
        frame_delta = torch.zeros(1, frame_count, 1)
        if offsets == (0, 1, 2, 4, 8):
            frame_delta = torch.arange(frame_count, dtype=torch.float32)[None, :, None] * 1000.0
        return self._joint_pos[:, None, :] + frame_delta, self._joint_vel[:, None, :] + frame_delta

    def future_ref_body_pos_quat_offsets(self, offsets):
        assert offsets in ((1, 2, 4, 8), (0, 1, 2, 4, 8))
        self.future_ref_calls += 1
        num_envs = self.robot_ref_pos_w.shape[0]
        pos = torch.zeros(num_envs, len(offsets), 3)
        if offsets == (1, 2, 4, 8):
            pos[:, 2] = torch.tensor([1.0, 2.0, 3.0])
        else:
            pos[..., 0] = torch.as_tensor(offsets, dtype=torch.float32)
        quat = torch.zeros(num_envs, len(offsets), 4)
        quat[..., 3] = 1.0
        return pos, quat

    def future_body_pos_quat_offsets(self, offsets):
        assert offsets == (0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50)
        num_envs = self.robot_ref_pos_w.shape[0]
        body_count = len(self.motion_cfg.body_names_to_track)
        pos = torch.zeros(num_envs, len(offsets), body_count, 3)
        pos[..., 0] = torch.arange(body_count, dtype=torch.float32)
        quat = torch.zeros(num_envs, len(offsets), body_count, 4)
        quat[..., 3] = 1.0
        return pos, quat

    def future_contact_force_part_mask_offsets(self, offsets):
        mask = torch.zeros(self.robot_ref_pos_w.shape[0], len(offsets), 8, dtype=torch.bool)
        mask[..., 1] = True
        mask[..., 5] = True
        return mask


def test_future_reference_frame_matches_current_reference_layout(monkeypatch) -> None:
    command = _MotionCommandStub()
    env = SimpleNamespace(num_envs=2)
    monkeypatch.setattr(wbt, "_get_motion_command_and_assert_type", lambda _: command)

    future_command = wbt.future_motion_command(env, offset=4)
    future_pos = wbt.future_motion_ref_pos_b(env, offset=4)
    future_ori = wbt.future_motion_ref_ori_b(env, offset=4)
    future_frame = torch.cat([future_command, future_pos, future_ori], dim=1)

    assert future_command.shape == (2, 58)
    assert future_pos.shape == (2, 3)
    assert future_ori.shape == (2, 6)
    assert future_frame.shape == (2, 67)
    torch.testing.assert_close(future_command[:, :29], command._joint_pos)
    torch.testing.assert_close(future_command[:, 29:], command._joint_vel)
    torch.testing.assert_close(future_pos, torch.tensor([[1.0, 2.0, 3.0]]).repeat(2, 1))
    torch.testing.assert_close(
        future_ori,
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]).repeat(2, 1),
    )


def test_fused_policy_window_hides_current_frame_from_actor(monkeypatch) -> None:
    command = _MotionCommandStub()
    env = SimpleNamespace(num_envs=2)
    monkeypatch.setattr(wbt, "_get_motion_command_and_assert_type", lambda _: command)

    critic = wbt.policy_reference_window(env, include_current=True, add_noise=False).reshape(2, 5, 67)
    actor = wbt.policy_reference_window(env, include_current=False, add_noise=False).reshape(2, 4, 67)

    assert critic.shape == (2, 5, 67)
    assert actor.shape == (2, 4, 67)
    torch.testing.assert_close(actor, critic[:, 1:])
    torch.testing.assert_close(critic[:, 0, :29], command._joint_pos)
    torch.testing.assert_close(actor[:, 0, :29], command._joint_pos + 1000.0)
    assert command.future_ref_calls == 1


def test_sparse_climb_window_has_no_joint_reference(monkeypatch) -> None:
    command = _MotionCommandStub()
    env = SimpleNamespace(num_envs=2, device="cpu")
    monkeypatch.setattr(wbt, "_get_motion_command_and_assert_type", lambda _: command)

    frames = wbt.sparse_climb_reference_window(env, add_noise=False).reshape(2, 11, 69)

    assert frames.shape == (2, 11, 69)
    torch.testing.assert_close(frames[..., :3], torch.ones(2, 11, 1).expand(-1, -1, 3) * torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        frames[..., 9:27:3], torch.tensor([2, 3, 4, 5, 6, 7]).float().expand(2, 11, 6)
    )
    torch.testing.assert_close(
        frames[..., 63:],
        torch.tensor([1, 0, 0, 1, 0, 0]).float().expand(2, 11, 6),
    )


def test_sparse_climb_window_uses_executed_torso_yaw_only(monkeypatch) -> None:
    command = _MotionCommandStub(num_envs=1)
    # Executed torso has 90-degree roll and zero yaw. A full-torso transform
    # would rotate targets; a yaw-only transform must leave their world axes.
    half_sqrt = 2.0**-0.5
    command.robot_ref_quat_w[:] = torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt])
    env = SimpleNamespace(num_envs=1, device="cpu")
    monkeypatch.setattr(wbt, "_get_motion_command_and_assert_type", lambda _: command)

    frames = wbt.sparse_climb_reference_window(env, add_noise=False).reshape(1, 11, 69)
    torch.testing.assert_close(frames[..., :3], torch.tensor([1.0, 0.0, 0.0]).expand(1, 11, 3))
