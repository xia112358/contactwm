from types import SimpleNamespace

import torch

from holosoma.managers.termination.terms import wbt as wbt_termination
from holosoma.managers.termination.terms.wbt import BadTracking


def test_probe_uses_fixed_boundary_while_normal_env_uses_adaptive_boundary(monkeypatch) -> None:
    motion_command = SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=["pelvis"]),
        ref_pos_w=torch.tensor([[0.20, 0.0, 0.0], [0.20, 0.0, 0.0]]),
        robot_ref_pos_w=torch.zeros(2, 3),
        ref_quat_w=torch.zeros(2, 4),
        robot_ref_quat_w=torch.zeros(2, 4),
        body_pos_relative_w=torch.tensor([[[0.20, 0.0, 0.0]], [[0.20, 0.0, 0.0]]]),
        robot_body_pos_w=torch.zeros(2, 1, 3),
        _probe_env_mask=torch.tensor([True, False]),
        motion=SimpleNamespace(has_object=False),
    )
    tracking_precision = SimpleNamespace(
        base_root_pos_threshold=0.50,
        base_root_ori_threshold=0.50,
        base_body_threshold=0.30,
        observe_probe_tracking_error=lambda *args, **kwargs: None,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=SimpleNamespace(get_state=lambda _: motion_command),
        curriculum_manager=SimpleNamespace(get_term=lambda _: tracking_precision),
        is_evaluating=False,
    )
    term = BadTracking.__new__(BadTracking)
    term.env = env
    term.metrics = {}
    term.body_names_to_track = ["pelvis"]
    term.bad_motion_body_pos_body_indexes = torch.tensor([0])
    term.check_motion_body_pos = True
    term.bad_ref_pos_threshold = 0.10
    term.bad_ref_ori_threshold = 0.10
    term.bad_motion_body_pos_threshold = 0.10
    term.bad_object_pos_threshold = 0.10
    term.bad_object_ori_threshold = 0.10
    term.probe_fixed_qualification_boundary = True
    term.exclude_probe_envs = False

    monkeypatch.setattr(wbt_termination, "gravity_vector", lambda _: torch.zeros(2, 3))
    monkeypatch.setattr(wbt_termination, "quat_rotate_inverse", lambda quat, vec, w_last: vec)

    assert term(env).tolist() == [False, True]

