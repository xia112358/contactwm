"""Only head-start attempts measure completion; RSI never improves the rate."""
from types import SimpleNamespace as State
import csv

import pytest
import torch

from holosoma.php_contact import segments
from holosoma.php_contact.statistics import SegmentStatistics


KEYS = [('motion_a', 0, 10), ('motion_b', 5, 20)]


def test_native_dimension_inference_resets_onto_task_terrain_before_observation():
    from holosoma.php_contact.distill import SegmentDistillPPO
    from holosoma.agents.ppo.distill_ppo import DistillPPO
    on_task = False

    def dimensions():
        if not on_task:
            raise RuntimeError('unobserved terrain before task reset')
        return dict(actor_obs=102, actor_proprio_history=93, critic_obs=341, expert_route=342)

    def reset_all(*, step_after_reset):
        nonlocal on_task
        assert step_after_reset is False
        on_task = True

    groups = {name:State(history_length=length) for name, length in
              [('actor_obs',1), ('actor_proprio_history',5), ('critic_obs',1), ('expert_route',1)]}
    algo = SegmentDistillPPO.__new__(SegmentDistillPPO)
    algo.env = State(reset_all=reset_all, robot_config=State(actions_dim=29),
                     observation_manager=State(get_obs_dims=dimensions, cfg=State(groups=groups)))
    algo.config = State(actor_learning_rate=.001, critic_learning_rate=.001,
        max_actor_learning_rate=None, min_actor_learning_rate=None,
        max_critic_learning_rate=None, min_critic_learning_rate=None,
        use_symmetry=False, empirical_normalization=True,
        module_dict=State(actor=State(input_dim=['actor_obs','actor_proprio_history']),
                          critic=State(input_dim=['critic_obs'])),
        distill=State(teacher_obs_keys=['expert_route']))
    with pytest.raises(RuntimeError, match='unobserved terrain'):
        DistillPPO._init_config(algo)
    algo._init_config()
    assert algo.algo_obs_dim_dict['actor_obs'] == 102
    assert algo.algo_history_length_dict['actor_proprio_history'] == 5
    assert algo.teacher_obs_keys == ['expert_route'] and algo.num_act == 29


def test_rsi_successes_never_enter_head_completion_denominators():
    stats = SegmentStatistics(KEYS, 'cpu')
    ids = torch.tensor([0, 0, 1, 1])
    heads = torch.tensor([True, False, True, False])
    stats.sampled(ids, heads)
    stats.finished(ids, heads, torch.tensor([True, True, False, True]),
        torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool),
        torch.tensor([1., 1., .4, 1.]))
    values = stats.metrics()
    assert values['segment_head_attempts'] == 2
    assert values['segment_head_completion_rate'] == .5
    assert values['segment_head_contact_success_rate'] == .5
    assert values['segment_head_mean_progress'] == pytest.approx(.7)
    assert values['segment_start_sample_fraction'] == .5
    torch.testing.assert_close(stats.counts[:, 2], torch.tensor([1, 1]))


def test_unknown_terminal_contact_is_not_success_even_if_activation_matches():
    stats = SegmentStatistics(KEYS, 'cpu')
    ids = torch.tensor([0]); yes = torch.tensor([True])
    stats.sampled(ids, yes)
    stats.finished(ids, yes, yes, yes, yes, torch.ones(1))
    values = stats.metrics()
    assert values['segment_head_completion_rate'] == 1
    assert values['segment_head_contact_success_rate'] == 0
    assert values['segment_head_unknown_end_count'] == 1


def test_boundary_failure_and_repeated_metric_calls_cannot_create_success(monkeypatch):
    cmd = segments.SegmentCommand.__new__(segments.SegmentCommand)
    cmd.segment_mode = 'independent'
    cmd._episode_open = torch.tensor([True, True])
    cmd._episode_from_head = torch.tensor([True, False])
    cmd.segment_start = torch.tensor([0, 5]); cmd.segment_end = torch.tensor([10, 20])
    cmd.time_steps = torch.tensor([9, 19]); cmd.selected_segment = torch.tensor([0, 1])
    cmd.goal_active = torch.zeros(2, 6, dtype=torch.bool); cmd.goal_active[:, 0] = True
    cmd.segment_statistics = SegmentStatistics(KEYS, 'cpu')
    cmd.segment_statistics.sampled(cmd.selected_segment, cmd._episode_from_head)
    terminal = State(terminated=torch.tensor([True, False]), time_outs=torch.tensor([True, True]),
                     term_dones={'segment_boundary':torch.tensor([True, True])})
    cmd.env = State(termination_manager=terminal, episode_length_buf=torch.tensor([10, 15]))
    monkeypatch.setattr(segments.ContactCommand, 'update_metrics', lambda _: None)
    monkeypatch.setattr(segments, 'measured_contacts', lambda _: dict(
        active=cmd.goal_active.clone(), known=torch.ones(2, 6, dtype=torch.bool)))
    cmd.update_metrics(); cmd.update_metrics()
    values = cmd.segment_statistics.metrics()
    assert values['segment_head_attempts'] == 1
    assert values['segment_head_completion_rate'] == 0
    assert values['segment_head_contact_success_rate'] == 0
    assert not cmd._episode_open.any()


def test_statistics_restore_by_motion_and_interval_without_simulator_state(tmp_path):
    stats = SegmentStatistics(KEYS, 'cpu')
    ids = torch.tensor([0]); yes = torch.tensor([True]); no = ~yes
    stats.sampled(ids, yes)
    stats.finished(ids, yes, yes, yes, no, torch.ones(1))
    state = stats.state_dict()
    assert set(state) == {'schema', 'keys', 'counts', 'progress'}
    restored = SegmentStatistics(KEYS[::-1], 'cpu')
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.counts, stats.counts.flip(0))
    assert restored.metrics()['segment_head_attempts'] == 0
    assert restored.metrics(cumulative=True)['segment_head_attempts'] == 1
    path = restored.write_csv(tmp_path, 100)
    rows = list(csv.DictReader(path.open()))
    assert rows[0]['head_completion_rate'] == ''  # No head attempts is unknown, not failure.
    assert rows[1]['head_contact_success_rate'] == '1.0'
    with pytest.raises(ValueError, match='intervals changed'):
        SegmentStatistics([('motion_a', 0, 11), KEYS[1]], 'cpu').load_state_dict(state)
    state['counts'][0, 4] = 2
    with pytest.raises(ValueError, match='Invalid'):
        restored.load_state_dict(state)


def sampling_command(n, length=100):
    cmd = segments.SegmentCommand.__new__(segments.SegmentCommand)
    cmd.device = 'cpu'; cmd.num_envs = n; cmd.start_probability = .2
    cmd.motion_ids = torch.zeros(n, dtype=torch.long)
    cmd.motion = State(motion_start_idx=torch.tensor([0]), motion_end_idx=torch.tensor([length+1]))
    cmd.segment_counts = torch.tensor([1]); cmd.segment_offsets = torch.tensor([0])
    cmd.bounds = torch.tensor([[0, length]])
    cmd.selected_segment = torch.zeros(n, dtype=torch.long)
    cmd.segment_start = torch.zeros(n, dtype=torch.long); cmd.segment_end = torch.zeros(n, dtype=torch.long)
    cmd.goal_active = torch.zeros(n, 6, dtype=torch.bool); cmd.goal_position_w = torch.zeros(n, 6, 3)
    cmd.contact_data = dict(target_frame=torch.full((length+1,), length),
        target_active=torch.zeros(length+1, 6, dtype=torch.bool),
        target_position_w=torch.zeros(length+1, 6, 3))
    return cmd


def test_twenty_percent_head_probability_with_strict_interior_rsi():
    from holosoma.php_contact.distill import Options
    assert Options(manifest='unused', teacher_spec='unused').segment_start_probability == .2
    cmd = sampling_command(20000)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        frame = (cmd.sample_reset_phase(torch.arange(cmd.num_envs))*100).long()
    fraction = (frame == 0).float().mean().item()
    assert .18 < fraction < .22
    assert ((frame[frame != 0] >= 1) & (frame[frame != 0] < 100)).all()


def test_segment_without_interior_frames_can_only_start_at_its_head():
    cmd = sampling_command(16, length=1); cmd.start_probability = 0
    frame = cmd.sample_reset_phase(torch.arange(cmd.num_envs)).long()
    assert not frame.any()


def test_resume_counts_new_reset_attempts_without_restoring_physical_episode_state(monkeypatch):
    saved = SegmentStatistics(KEYS, 'cpu')
    ids = torch.tensor([0]); yes = torch.tensor([True])
    saved.sampled(ids, yes); saved.finished(ids, yes, yes, yes, ~yes, torch.ones(1))
    cmd = segments.SegmentCommand.__new__(segments.SegmentCommand)
    cmd.segment_statistics = SegmentStatistics(KEYS, 'cpu')
    cmd._episode_open = torch.tensor([True, True])
    cmd._episode_from_head = torch.tensor([True, False])
    cmd.selected_segment = torch.tensor([0, 1])
    monkeypatch.setattr(segments.ContactCommand, 'load_checkpoint_state', lambda *args:None)
    cmd.load_checkpoint_state({'segment_head_statistics':saved.state_dict()})
    assert cmd.segment_statistics.counts[0, 1] == 2
    assert cmd.segment_statistics.counts[0, 2] == 1
    assert cmd.segment_statistics.counts[1, 0] == 1
    cmd.segment_statistics.finished(ids, yes, ~yes, ~yes, ~yes, torch.tensor([.5]))
    check = SegmentStatistics(KEYS, 'cpu')
    check.load_state_dict(cmd.segment_statistics.state_dict())
    assert check.metrics(cumulative=True)['segment_head_attempts'] == 2


def test_epoch_logs_finished_head_attempts_before_clearing_interval(monkeypatch, tmp_path):
    from holosoma.php_contact.distill import SegmentDistillPPO
    from holosoma.agents.ppo.distill_ppo import DistillPPO
    stats = SegmentStatistics(KEYS, 'cpu')
    yes = torch.tensor([True]); ids = torch.tensor([0])
    stats.sampled(ids, yes); stats.finished(ids, yes, yes, yes, ~yes, torch.ones(1))
    cmd = State(segment_statistics=stats)
    captured = []
    monkeypatch.setattr(DistillPPO, '_post_epoch_logging',
                        lambda *args: captured.append(stats.metrics()))
    algo = SegmentDistillPPO.__new__(SegmentDistillPPO)
    algo.env = State(command_manager=State(get_state=lambda _:cmd))
    algo.is_main_process = True; algo.config = State(save_interval=100); algo.log_dir = tmp_path
    algo._post_epoch_logging(100, {})
    assert captured[0]['segment_head_attempts'] == 1
    assert stats.metrics()['segment_head_attempts'] == 0
    assert stats.metrics(cumulative=True)['segment_head_attempts'] == 1
    assert (tmp_path/'segment_head_statistics.csv').exists()
