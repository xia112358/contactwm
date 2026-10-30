"""Paper architecture, real supervision boundaries and shared contact compatibility."""
import numpy as np
import pytest
import torch

from holosoma.acp.labels import hindsight_labels
from holosoma.acp.models import ContactPlanner, contact_schedule, planner_loss
from holosoma.acp.replay import HindsightReplay
from holosoma.acp.train import Options, build_config
from somaforge_core.contact_face_selection import select_contact_pairs


def test_paper_parameter_count_and_surface_constrained_decoder():
    torch.manual_seed(11)
    model = ContactPlanner()
    assert sum(p.numel() for p in model.parameters()) == 786_993
    terrain = torch.randn(2, 187, 3)
    valid = torch.ones(2, 187, dtype=torch.bool)
    valid[:, :40] = False
    prediction = model(torch.randn(2, 375), torch.randn(2, 3), terrain, valid)
    assert prediction["positions"].shape == (2, 9, 3)
    assert not prediction["affordance"][..., :40].any()
    selected = prediction["ungated_positions"]
    assert ((selected[:, :, None]-terrain[:, None, 40:]).abs().amax(-1) == 0).any(-1).all()
    with pytest.raises(ValueError, match="observed terrain"):
        model(torch.zeros(2, 375), torch.zeros(2, 3), terrain, torch.zeros_like(valid))


def test_planner_positive_and_unknown_losses_have_finite_gradients():
    model = ContactPlanner()
    terrain = torch.randn(2, 187, 3)
    prediction = model(torch.randn(2, 375), torch.randn(2, 3), terrain)
    occupancy = torch.zeros(2, 9, dtype=torch.bool)
    occupancy[:, 3] = True
    known = torch.ones_like(occupancy)
    known[:, 7] = False
    loss = planner_loss(prediction, terrain, torch.randn(2, 9, 3), occupancy, known)
    loss["loss"].backward()
    assert torch.isfinite(loss["loss"])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.attention.in_proj_weight.grad.abs().sum() > 0
    empty = planner_loss(prediction, terrain, torch.zeros(2, 9, 3), occupancy, torch.zeros_like(known))
    assert empty["loss"] == 0


def make_contacts(length=120):
    active = np.zeros((length, 9), bool)
    force = np.zeros((length, 9, 3), np.float32)
    position = np.zeros_like(force)
    known = np.ones_like(active)
    return active, force, position, known


def test_hindsight_requires_native_activation_and_ten_step_event():
    active, force, pos, known = make_contacts()
    active[20:30, 2] = True
    force[20:30, 2, 2] = 5
    pos[20:30, 2] = [0.2, 0.1, 0.8]
    active[35:44, 4] = True  # Short transient, not a supervision event.
    force[35:44, 4, 2] = 5
    force[50:70, 5, 2] = 100  # Force alone must not invent activation.
    saved = active.copy()
    labels = hindsight_labels(active, force, pos, known)
    assert labels["occupancy"][0, 2]
    np.testing.assert_array_equal(labels["target_w"][0, 2], np.asarray([0.2, 0.1, 0.8], np.float32))
    assert not labels["occupancy"][:, 4:6].any()
    np.testing.assert_array_equal(active, saved)


def test_hindsight_unknown_and_recording_boundaries_are_not_negatives():
    active, force, pos, known = make_contacts()
    active[:20, 0] = True
    force[:20, 0, 2] = 10  # Left-censored already-active contact has no onset.
    known[10, 1] = False
    labels = hindsight_labels(active, force, pos, known)
    assert not labels["occupancy"][:, 0].any()
    assert not labels["known"][0, 1]
    assert labels["known"][0, 2]
    assert not labels["known"][-75:].any()


def test_nine_body_face_selection_preserves_default_six_body_contract():
    catalog = [{"surface": 0, "normal_w": [0, 0, 1]}, {"surface": 1, "normal_w": [1, 0, 0]}]
    top = dict(surface=0, part=8, allocated=True, position_w=[0, 0, 0], dist=-0.01, includemargin=0)
    side = {**top, "surface": 1}
    result = select_contact_pairs([[top, side]], catalog, part_count=9)
    assert result["contact_part_mask"].shape == (1, 9)
    assert result["contact_part_mask"][0, 8]
    assert len(result["abnormal_contact_pairs"][0]) == 1
    with pytest.raises(ValueError, match="Unknown surface/part"):
        select_contact_pairs([[top]], catalog)
    assert select_contact_pairs([[]], catalog)["contact_part_mask"].shape == (1, 6)


def test_stage_transition_schedule_and_saved_config_roundtrip():
    assert [contact_schedule(i, 20, 10) for i in [0, 20, 25, 30, 40]] == [0, 0, 0.5, 1, 1]
    from holosoma.utils.eval_utils import _experiment_config_from_serialized
    config = build_config(Options(manifest="reference.json"))
    restored = _experiment_config_from_serialized(config.to_serializable_dict())
    assert restored.command.setup_terms["motion_command"].params["acp"]["stage"] == 1
    assert restored.algo._target_ == "holosoma.acp.ppo.ACPPPO"
    assert restored.robot.control.action_scale == 0.25
    assert restored.algo.config.module_dict.actor.layer_config.hidden_dims == [512, 256, 128]
    assert all("reference" not in term.func for term in restored.observation.groups["actor_obs"].terms.values())


def test_planner_gradient_accumulation_matches_global_masked_objective():
    from copy import deepcopy
    torch.manual_seed(9)
    full = ContactPlanner()
    accumulated = deepcopy(full)
    history, goal, terrain = torch.randn(3, 375), torch.randn(3, 3), torch.randn(3, 187, 3)
    target = torch.randn(3, 9, 3)
    occupancy = torch.zeros(3, 9, dtype=torch.bool)
    occupancy[0, 0] = True
    occupancy[2, :7] = True
    known = torch.ones_like(occupancy)
    known[1, :5] = False
    planner_loss(full(history, goal, terrain), terrain, target, occupancy, known)["loss"].backward()
    normalizers = ((occupancy & known).sum(), known.sum())
    for index in [slice(0, 1), slice(1, 3)]:
        prediction = accumulated(history[index], goal[index], terrain[index])
        planner_loss(prediction, terrain[index], target[index], occupancy[index], known[index],
                     normalizers=normalizers)["loss"].backward()
    for left, right in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)


def test_replay_rejects_failed_and_late_start_episodes_and_restores_sampling():
    active, force, position, known = make_contacts()
    active[20:35, 0] = True
    force[20:35, 0, 2] = 10
    position[20:35, 0] = [0.2, 0, 0]
    frames = [{"active": active[i], "force": force[i], "position": position[i], "known": known[i],
               "root": np.zeros(3, np.float32), "yaw": np.asarray(0, np.float32),
               "history": np.zeros(465, np.float32), "goal": np.zeros(3, np.float32),
               "terrain": np.zeros((187, 3), np.float32), "valid": np.ones(187, bool)} for i in range(120)]
    replay = HindsightReplay(seed=3)
    assert not replay.admit(frames, success=False, initial_phase=0, final_phase=1, tile=0)
    assert not replay.admit(frames, success=True, initial_phase=0.2, final_phase=1, tile=0)
    assert replay.admit(frames, success=True, initial_phase=0, final_phase=1, tile=0)
    restored = HindsightReplay()
    restored.load_state_dict(replay.state_dict())
    first, second = replay.sample(32, "cpu"), restored.sample(32, "cpu")
    assert first["occupancy"].any()
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    bounded = HindsightReplay(capacity=4)
    assert bounded.admit(frames, success=True, initial_phase=0, final_phase=1, tile=0)
    assert bounded.size == 4


def test_task_reward_tracks_inactive_contact_bodies_and_emits_tensor_metrics(monkeypatch):
    from types import SimpleNamespace as State
    from holosoma.acp import rewards
    from holosoma.acp.labels import BODY_NAMES
    q = torch.zeros(1, 29)
    positions = torch.zeros(1, 9, 3)
    quaternions = torch.zeros(1, 9, 4)
    quaternions[..., 3] = 1
    motion = State(body_pos_w=positions.clone(), body_quat_w=quaternions,
                   body_lin_vel_w=torch.zeros_like(positions), body_ang_vel_w=torch.zeros_like(positions))
    cmd = State(time_steps=torch.zeros(1, dtype=torch.long), motion=motion,
                _reference_env_origins=torch.zeros(1, 3), joint_pos=q, joint_vel=q,
                acp_reference_active=torch.zeros(1, 9, dtype=torch.bool))
    root = torch.zeros(1, 13)
    root[:, 6] = 1
    sim = State(robot_root_states=root, dof_pos=q, dof_vel=q, body_names=list(BODY_NAMES),
                _robot=State(data=State(body_pos_w=State(torch=positions))))
    env = State(simulator=sim, command_manager=State(get_state=lambda _: cmd),
                num_envs=1, device="cpu", acp_settings={"stage": 1}, log_dict={})
    monkeypatch.setattr(rewards, "measured_contacts", lambda _: {
        "active": torch.zeros(1, 9, dtype=torch.bool), "known": torch.ones(1, 9, dtype=torch.bool),
        "force_w": torch.zeros(1, 9, 3)})
    matched = rewards.task(env)
    positions[0, 6, 0] = 1
    displaced = rewards.task(env)
    assert displaced < matched
    assert all(isinstance(value, torch.Tensor) for value in env.log_dict.values())
