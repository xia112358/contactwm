"""Independent contact-goal episodes and a separate original-reference teacher view."""
from __future__ import annotations

import torch

from holosoma.php_contact.terms import ContactCommand, command, rotate, yaw
from holosoma.utils.wbt_contacts import measured_contacts


def episode_bounds(starts, ends, segment_ids, target_frames):
    """Recover the exact shared-boundary clock from a prepared contact stream."""
    bounds, counts = [], []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        changes = torch.where(segment_ids[start+1:end] != segment_ids[start:end-1])[0]+start+1
        first = torch.cat((starts.new_tensor([start]), changes))
        last = torch.cat((changes, starts.new_tensor([end-1])))
        if (first >= last).any() or not torch.equal(target_frames[first], last):
            raise ValueError('Prepared contact segments disagree with the authored endpoint clock')
        bounds.append(torch.stack((first, last), -1))
        counts.append(len(first))
    count = starts.new_tensor(counts)
    offsets = torch.cat((count.new_zeros(1), count.cumsum(0)[:-1]))
    return torch.cat(bounds), offsets, count


class SegmentCommand(ContactCommand):
    def setup(self):
        super().setup()
        self.segment_mode = self.cfg.params.get('segment_mode', 'independent')
        if self.segment_mode not in ('independent', 'continuous'):
            raise ValueError('Unknown contact episode mode')
        self.start_probability = float(self.cfg.params.get('segment_start_probability', .2))
        if not 0 <= self.start_probability <= 1:
            raise ValueError('Segment-start probability must be in [0, 1]')
        self.bounds, self.segment_offsets, self.segment_counts = episode_bounds(
            self.motion.motion_start_idx, self.motion.motion_end_idx,
            self.contact_data['segment_id'], self.contact_data['target_frame'])
        self.selected_segment = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.segment_start = torch.zeros_like(self.selected_segment)
        self.segment_end = torch.zeros_like(self.selected_segment)
        self.goal_active = torch.zeros(self.num_envs, 6, dtype=torch.bool, device=self.device)
        self.goal_position_w = torch.zeros(self.num_envs, 6, 3, device=self.device)
        self._teacher_view = None
        from holosoma.php_contact.statistics import SegmentStatistics
        starts = self.motion.motion_start_idx.tolist()
        files = self._checkpoint_motion_keys()
        keys = [(files[mid], int(a)-starts[mid], int(b)-starts[mid])
                for mid, count in enumerate(self.segment_counts.tolist())
                for a, b in self.bounds[self.segment_offsets[mid]:self.segment_offsets[mid]+count].tolist()]
        self.segment_statistics = SegmentStatistics(keys, self.device)
        self._episode_from_head = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_open = torch.zeros_like(self._episode_from_head)

    def _set_segments(self, ids, selected):
        self.selected_segment[ids] = selected
        self.segment_start[ids], self.segment_end[ids] = self.bounds[selected].unbind(-1)
        # At shared endpoint b the per-frame stream already targets the next
        # segment. Read b-1, then retain this episode's b command through its end.
        source = self.segment_end[ids]-1
        if not torch.equal(self.contact_data['target_frame'][source], self.segment_end[ids]):
            raise ValueError('Segment terminal goal is not bound to its own endpoint')
        self.goal_active[ids] = self.contact_data['target_active'][source]
        self.goal_position_w[ids] = self.contact_data['target_position_w'][source]

    def sample_reset_phase(self, env_ids):
        mids = self.motion_ids[env_ids]
        local = (torch.rand(len(env_ids), device=self.device)*self.segment_counts[mids]).long()
        selected = self.segment_offsets[mids]+local
        self._set_segments(env_ids, selected)
        a, b = self.segment_start[env_ids], self.segment_end[env_ids]
        # Keep the RSI branch strictly inside, so it cannot inflate head sampling.
        inside = a+1+(torch.rand(len(env_ids), device=self.device)*(b-a-1).clamp_min(0)).long()
        inside = torch.where(b-a > 1, inside, a)  # A one-step segment has no interior.
        frame = torch.where(torch.rand(len(env_ids), device=self.device) < self.start_probability, a, inside)
        start = self.motion.motion_start_idx[mids]
        length = self.motion.motion_end_idx[mids]-start-1
        # MotionCommand floors the phase into a frame. Use each frame bin's
        # center so float roundoff cannot initialize one frame before a boundary.
        return (frame-start+.5)/length.clamp_min(1)

    def reset(self, env_ids):
        ids = self._ensure_index_tensor(env_ids)
        self._teacher_view = None
        super().reset(env_ids)
        if not len(ids) or not hasattr(self, 'bounds'):
            return
        # Evaluation and explicit from-zero starts bypass the phase sampler.
        local = self.contact_data['segment_id'][self.time_steps[ids]]
        self._set_segments(ids, self.segment_offsets[self.motion_ids[ids]]+local)
        self._episode_from_head[ids] = self.time_steps[ids] == self.segment_start[ids]
        self._episode_open[ids] = True
        self.segment_statistics.sampled(self.selected_segment[ids], self._episode_from_head[ids])

    def update_metrics(self):
        super().update_metrics()
        if self.segment_mode != 'independent':
            return
        terminal = self.env.termination_manager
        ended = self._episode_open & (terminal.terminated | terminal.time_outs)
        ended &= self.env.episode_length_buf > 0
        ids = torch.where(ended)[0]
        self._episode_open[ids] = False  # Ignore duplicate logging and manual resets.
        ids = ids[self._episode_from_head[ids]]
        if not len(ids):
            return
        actual = measured_contacts(self.env)
        boundary = terminal.term_dones.get('segment_boundary', torch.zeros_like(ended))
        reached = boundary[ids] & ~terminal.terminated[ids]
        known = actual['known'][ids].all(-1)
        matched = (actual['active'][ids] == self.goal_active[ids]).all(-1)
        progress = ((self.time_steps[ids]+1-self.segment_start[ids]).float()
                    /(self.segment_end[ids]-self.segment_start[ids]).clamp_min(1))
        self.segment_statistics.finished(self.selected_segment[ids], self._episode_from_head[ids],
            reached, matched, ~known, progress)

    def get_motion_learning_progress_metrics(self):
        values = self.segment_statistics.metrics()
        values.update({'cumulative_'+key: value for key, value in
                       self.segment_statistics.metrics(cumulative=True).items()})
        return values

    def get_checkpoint_state(self):
        state = super().get_checkpoint_state()
        state['segment_head_statistics'] = self.segment_statistics.state_dict()
        return state

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        saved = state.get('segment_head_statistics') if isinstance(state, dict) else None
        if saved is not None:
            self.segment_statistics.load_state_dict(saved)
            # Native resume restores counters after resetting a fresh simulator.
            # Those current attempts were not part of the saved episode set.
            ids = torch.where(self._episode_open)[0]
            self.segment_statistics.sampled(self.selected_segment[ids], self._episode_from_head[ids])

    def step(self):
        super().step()
        self._teacher_view = None
        if self.segment_mode == 'continuous':
            ids = torch.arange(self.num_envs, device=self.device)
            local = self.contact_data['segment_id'][self.time_steps]
            self._set_segments(ids, self.segment_offsets[self.motion_ids]+local)


def future_contacts(env):
    cmd = command(env)
    root = env.simulator.robot_root_states
    points = cmd.goal_position_w+cmd._reference_env_origins[:, None]
    local = rotate(points-root[:, None, :3], -yaw(root[:, 3:7]))
    return torch.cat((torch.where(cmd.goal_active[..., None], local, 0).flatten(1), cmd.goal_active.float()), -1)


def goal(env):
    cmd = command(env)
    root = env.simulator.robot_root_states
    target = cmd.motion.body_pos_w[cmd.segment_end, 0]+cmd._reference_env_origins
    angle = yaw(root[:, 3:7])
    direction = rotate(target-root[:, :3], -angle)[:, :2]
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(.3)
    heading = yaw(cmd.motion.body_quat_w[cmd.segment_end, 0])-angle
    return torch.cat((direction, torch.atan2(heading.sin(), heading.cos())[:, None]), -1)


def segment_boundary(env):
    cmd = command(env)
    if cmd.segment_mode == 'continuous':
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # WBT checks termination after integration, before advancing its ref clock.
    return cmd.time_steps >= cmd.segment_end-1


def balanced_contact_score(active, known, target):
    expected = target & known
    unexpected = ~target & known
    positives = expected.sum(-1)
    hits = (active & expected).sum(-1)/positives.clamp_min(1)
    hits = torch.where(positives > 0, hits, torch.ones_like(hits))
    false_contact = (active & unexpected).sum(-1)/unexpected.sum(-1).clamp_min(1)
    return torch.where(known.any(-1), hits-false_contact, 0.)


def contact_reward(env):
    cmd = command(env)
    data = measured_contacts(env)
    target = cmd.contact_data['reference_active'][cmd.time_steps]
    return balanced_contact_score(data['active'], data['known'], target)


def goal_contact_reward(env, terminal_window=3):
    cmd = command(env)
    data = measured_contacts(env)
    near_end = cmd.time_steps >= cmd.segment_end-int(terminal_window)
    exact = data['known'].all(-1) & (data['active'] == cmd.goal_active).all(-1)
    head_near_end = cmd._episode_from_head & near_end
    env.log_dict['PHPContact/head_terminal_contact_satisfied'] = (
        (exact & head_near_end).sum()/head_near_end.sum().clamp_min(1))
    env.log_dict['PHPContact/head_terminal_contact_observations'] = head_near_end.sum().float()
    return balanced_contact_score(data['active'], data['known'], cmd.goal_active)*near_end


class TeacherEnvironment:
    """Share measured state/terrain, exposing only the teacher's reference command."""
    def __init__(self, env, teacher_command):
        self.env, self.teacher_command = env, teacher_command
        self.command_manager = self

    def get_state(self, name):
        return self.teacher_command if name == 'motion_command' else self.env.command_manager.get_state(name)

    def __getattr__(self, name):
        return getattr(self.env, name)


def teacher_observation(env, function, parameters):
    from holosoma.managers.utils import resolve_callable
    cmd = command(env)
    if cmd._teacher_view is None:
        from holosoma.php_contact.teacher_reference import reference_view
        cmd._teacher_view = reference_view(env, cmd)
    return resolve_callable(function, context='original teacher observation')(cmd._teacher_view, **parameters)
