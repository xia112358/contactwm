"""Stage-1 curriculum, privileged inputs, and physical regularizer contracts."""
from types import SimpleNamespace as State

import pytest
import torch

from holosoma.acp.command import ACPCommand
from holosoma.acp import observations, rewards
from holosoma.acp.train import Options, build_config
from holosoma.managers.command.terms.wbt import MotionCommand


def test_csi_uses_episode_outcomes_independently_for_envs_sharing_one_motion(monkeypatch):
    command = ACPCommand.__new__(ACPCommand)
    command.device = 'cpu'
    command.acp_settings = {'stage': 1}
    command.csi_phase = torch.tensor([0.9, 0.9, 0.9])
    command.motion_ids = torch.zeros(3, dtype=torch.long)
    command.motion = State(motion_start_idx=torch.tensor([0]), motion_end_idx=torch.tensor([101]))
    command.time_steps = torch.tensor([99, 30, 0])
    command._env = State(is_evaluating=False, acp_settings=command.acp_settings,
                         _pending_episode_lengths=torch.tensor([10, 10, 0]),
                         termination_manager=State(terminated=torch.tensor([False, True, False])))
    monkeypatch.setattr(MotionCommand, 'reset', lambda *args: None)
    command.reset(torch.arange(3))
    torch.testing.assert_close(command.csi_phase, torch.tensor([0.85, 0.95, 0.9]))
    torch.testing.assert_close(command.sample_reset_phase(torch.tensor([0, 1])), torch.tensor([0.85, 0.95]))
    command._env.is_evaluating = True
    command.reset(torch.arange(3))
    torch.testing.assert_close(command.csi_phase, torch.tensor([0.85, 0.95, 0.9]))
    assert not command.sample_reset_phase(torch.arange(3)).any()


def test_csi_restore_rejects_the_old_motion_level_state_and_preserves_env_phases(monkeypatch):
    command = ACPCommand.__new__(ACPCommand)
    command.device = 'cpu'
    command.csi_phase = torch.tensor([0.2, 0.8])
    monkeypatch.setattr(MotionCommand, 'get_checkpoint_state', lambda _: {'existing_sampler': 7})
    monkeypatch.setattr(MotionCommand, 'load_checkpoint_state', lambda *args: None)
    state = command.get_checkpoint_state()
    command.csi_phase.zero_()
    command.load_checkpoint_state(state)
    assert state['existing_sampler'] == 7
    torch.testing.assert_close(command.csi_phase, torch.tensor([0.2, 0.8]))
    with pytest.raises(ValueError, match='Legacy ACP CSI'):
        command.load_checkpoint_state({'acp_csi_phase': torch.tensor([0.9])})


def test_reference_critic_has_383_inputs_for_29_joints_without_actor_commands(monkeypatch):
    env = State(acp_settings={'interface_version': 2})
    monkeypatch.setattr(observations, 'proprio', lambda _: torch.zeros(2, 93))
    monkeypatch.setattr(observations, 'base_lin_vel', lambda _: torch.zeros(2, 3))
    monkeypatch.setattr(observations, 'terrain', lambda _: (torch.zeros(2, 187, 3), None))
    monkeypatch.setattr(observations, 'critic_reference', lambda _: torch.zeros(2, 100))
    monkeypatch.setattr(observations, 'goal', lambda _: pytest.fail('Goal is not part of the reference critic'))
    monkeypatch.setattr(observations, 'contact_targets', lambda _: pytest.fail('Contact command is actor input'))
    assert observations.critic_observation(env).shape == (2, 383)


def test_missing_regularizers_follow_table3_limits_and_reset_boundaries():
    term = State(computed_torques=torch.tensor([[10., -8.]]), torques=torch.tensor([[5., -4.]]))
    env = State(action_manager=State(get_term=lambda _: term),
                simulator=State(dof_vel=torch.tensor([[9.5, -20.]])), dof_vel_limits=torch.tensor([10., 10.]))
    torch.testing.assert_close(rewards.torque_limits(env), torch.tensor([9.]))
    torch.testing.assert_close(rewards.joint_velocity_limits(env), torch.tensor([1.5]))
    env.simulator.hard_dof_pos_limits = torch.tensor([[-1., 1.], [-1., 1.]])
    env.simulator.dof_pos = torch.tensor([[100., -100.]])
    assert rewards.joint_position_limits(env).item() == 30
    env.simulator.body_names = list(rewards.BODY_NAMES)
    env.simulator._robot = State(data=State(body_lin_vel_w=State(torch=torch.ones(1, 9, 3))))
    env.acp_previous_body_velocity = torch.zeros(1, 9, 3)
    env.acp_body_velocity_ready = torch.tensor([False])
    env.dt = 0.02
    assert rewards.contact_acceleration(env).item() == 0
    env.acp_body_velocity_ready.fill_(True)
    assert rewards.contact_acceleration(env).item() == pytest.approx(9*3**0.5/0.02)


def test_stage1_configuration_selects_the_aligned_contract_and_full_regularizers():
    config = build_config(Options(manifest='reference.json'))
    assert config.command.setup_terms['motion_command'].params['acp']['interface_version'] == 2
    assert config.reward.terms['torque_limits'].weight == -1e-3
    assert config.reward.terms['velocity_limits'].weight == -1
    assert config.reward.terms['contact_acceleration'].weight == -0.01


def test_delayed_odometry_interpolates_time_and_does_not_cross_a_reset():
    from holosoma.acp.sensing import PoseHistory
    history = PoseHistory(2, 0.02, 'cpu')
    root = torch.zeros(2, 13)
    root[:, 6] = 1
    for step in range(7):
        root[:, 0] = step
        history.update(root, step)
    xyz, _ = history.measured(torch.tensor([0.05, 0.1]), torch.zeros(2, 3))
    torch.testing.assert_close(xyz[:, 0], torch.tensor([3.5, 1.]))
    unchanged = history.pose[1].clone()
    history.ready[0] = False
    root[0, 0] = 100
    history.update(root, 6)
    xyz, _ = history.measured(torch.tensor([0.1, 0.1]), torch.zeros(2, 3))
    assert xyz[0, 0] == 100
    torch.testing.assert_close(history.pose[1], unchanged)


def test_odometry_yaw_interpolation_wraps_at_pi():
    from holosoma.acp.sensing import PoseHistory
    history = PoseHistory(1, 0.02, 'cpu')
    root = torch.zeros(1, 13)
    for step, degrees in enumerate([179., -179.]):
        angle = torch.deg2rad(torch.tensor(degrees))
        root[0, 5], root[0, 6] = (angle/2).sin(), (angle/2).cos()
        history.update(root, step)
    _, yaw = history.measured(torch.tensor([0.01]), torch.zeros(1, 3))
    assert yaw.abs().item() == pytest.approx(torch.pi)


def test_actuator_delay_preserves_zero_delay_and_resolves_substep_arrival(monkeypatch):
    from holosoma.acp.actions import ACPJointPositionAction
    from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
    term = ACPJointPositionAction.__new__(ACPJointPositionAction)
    term.env = State(acp_actuation_delay_s=torch.tensor([0., 0.0075, 0.02]), sim_dt=0.005)
    term.previous_issued_action = torch.zeros(3, 1)
    term.current_issued_action = torch.ones(3, 1)
    term._actions_after_delay = torch.zeros(3, 1)
    term._substep_idx = 0
    observed = []
    def capture(self):
        observed.append(self._actions_after_delay.clone())
        self._substep_idx += 1
    monkeypatch.setattr(JointPositionActionTerm, 'apply_actions', capture)
    for _ in range(4):
        term.apply_actions()
    actions = torch.stack(observed).squeeze(-1).T
    torch.testing.assert_close(actions, torch.tensor([[1., 1., 1., 1.], [0., 0.5, 1., 1.], [0., 0., 0., 0.]]))


def test_pd_command_observation_does_not_change_existing_torque_clipping():
    from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
    term = JointPositionActionTerm.__new__(JointPositionActionTerm)
    term._randomize_torque_rfi = False
    term.action_scales = torch.tensor([0.25])
    term._kp_scale = term._kd_scale = torch.ones(1, 1)
    term.p_gains, term.d_gains = torch.tensor([20.]), torch.tensor([1.])
    term.env = State(robot_config=State(control=State(control_type='P', clip_torques=True)),
        simulator=State(dof_pos=torch.zeros(1, 1), dof_vel=torch.zeros(1, 1)),
        default_dof_pos=torch.zeros(1, 1), torque_limits=torch.tensor([3.]))
    applied = term._compute_torques(torch.tensor([[4.]]))
    assert applied.item() == 3
    assert term.computed_torques.item() == 20


def test_startup_randomization_ranges_persist_across_episode_resets(monkeypatch):
    from holosoma.acp.randomization import PaperRandomization
    config = build_config(Options(manifest='reference.json'))
    captures = []
    term = State(update_pd_scales=lambda *args: captures.append(args))
    env = State(num_envs=512, num_dof=29, device='cpu', robot_config=config.robot,
        default_dof_pos_base=torch.zeros(1, 29), action_manager=State(get_term=lambda _: term),
        acp_settings={'domain_randomization': True}, is_evaluating=False)
    startup = []
    monkeypatch.setattr(PaperRandomization, '_randomize_native_properties', lambda _: startup.append(True))
    state = PaperRandomization(State(params={'enabled': True}), env)
    state.setup()
    parameters = {k: v.clone() for k, v in state.startup_parameters.items()}
    assert startup == [True] and len(captures) == 1
    assert ((captures[0][1] >= 0.9) & (captures[0][1] <= 1.1)).all()
    assert ((env.acp_actuation_delay_s >= 0) & (env.acp_actuation_delay_s <= 0.02)).all()
    assert ((env.acp_odometry_delay_s >= 0.05) & (env.acp_odometry_delay_s <= 0.1)).all()
    state.reset(torch.arange(512))
    assert (env.acp_odometry_drift[:, :2].norm(dim=-1) <= 0.05+1e-7).all()
    assert (env.acp_odometry_drift[:, 2].abs() <= 0.05).all()
    for name, value in parameters.items():
        torch.testing.assert_close(state.startup_parameters[name], value)


def test_evaluation_does_not_count_bad_tracking_or_a_budget_cutoff_as_completion():
    from holosoma.acp.evaluate import summarize_episode
    rows = [dict(root_error_sq=1., contact_error_sq=2., matched_contacts=1,
                 missed_contacts=2, unexpected_contacts=0),
            dict(root_error_sq=9., contact_error_sq=0., matched_contacts=1,
                 missed_contacts=0, unexpected_contacts=1)]
    result = summarize_episode(rows, completed=True, terminated=False, truncated=False)
    assert result['complete']
    assert result['root_position_rms_m'] == pytest.approx(5**0.5)
    assert result['reference_contact_position_rms_m'] == 1
    assert result['missed_contact_samples'] == 2
    assert not summarize_episode(rows, completed=True, terminated=True, truncated=False)['complete']
    assert not summarize_episode(rows, completed=True, terminated=False, truncated=True)['complete']
