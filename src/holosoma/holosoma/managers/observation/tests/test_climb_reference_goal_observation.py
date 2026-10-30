from __future__ import annotations

from types import SimpleNamespace

import torch

import holosoma.managers.observation.terms.climb_reference_goal as climb_observations


def test_root_pose_goal_hides_contact_event_structure(monkeypatch) -> None:
    command = SimpleNamespace(
        robot_ref_pos_w=torch.tensor([[1.0, 2.0, 0.8]]),
        robot_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        goal_ref_pos_w=torch.tensor([[1.5, 1.0, 1.2]]),
        goal_ref_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    env = SimpleNamespace(num_envs=1)
    monkeypatch.setattr(climb_observations, "_command", lambda _: command)

    goal = climb_observations.climb_root_pose_goal(env)

    assert goal.shape == (1, 6)
    torch.testing.assert_close(goal[0, :2], torch.tensor([0.5, -1.0]))
    torch.testing.assert_close(goal[0, 2:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
