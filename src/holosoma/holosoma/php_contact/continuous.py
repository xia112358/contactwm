"""Continuous execution within each source trajectory, without action resets."""
import torch

from holosoma.php_contact.segments import SegmentCommand
from holosoma.php_contact.terms import ContactCommand, command
from holosoma.php_contact.trajectory_statistics import TrajectoryStatistics
from holosoma.utils.wbt_contacts import measured_contacts


class ContinuousContactCommand(SegmentCommand):
    def setup(self):
        super().setup()
        if self.segment_mode != 'continuous': raise ValueError('Expected continuous segment mode')
        self.trajectory_statistics = TrajectoryStatistics(self._checkpoint_motion_keys(),
                                                         self.segment_statistics.keys, self.device)
        self._trajectory_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._trajectory_all_contacts = torch.ones_like(self._trajectory_open)
        self._trajectory_observed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_boundary = torch.full_like(self._trajectory_observed, -1)
        self._last_boundary_root_error = torch.zeros(self.num_envs, device=self.device)
        self._last_boundary_limb_error = torch.zeros_like(self._last_boundary_root_error)
        parts = ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link',
                 'right_wrist_yaw_link','left_knee_link','right_knee_link')
        self._endpoint_body_ids = torch.tensor([self.motion_cfg.body_names_to_track.index(p) for p in parts],
                                               device=self.device, dtype=torch.long)

    def sample_reset_phase(self, env_ids):
        mids = self.motion_ids[env_ids]
        length = self.motion.motion_end_idx[mids]-self.motion.motion_start_idx[mids]-1
        inside = 1+(torch.rand(len(env_ids),device=self.device)*(length-1).clamp_min(0)).long()
        inside = torch.where(length > 1, inside, torch.zeros_like(inside))
        head = torch.rand(len(env_ids),device=self.device)<self.start_probability
        frame = torch.where(head, torch.zeros_like(inside), inside)
        return (frame+.5)/length.clamp_min(1)

    def reset(self, env_ids):
        ids = self._ensure_index_tensor(env_ids)
        self._teacher_view = None
        ContactCommand.reset(self, ids)
        if not len(ids) or not hasattr(self, 'trajectory_statistics'): return
        local = self.contact_data['segment_id'][self.time_steps[ids]]
        self._set_segments(ids, self.segment_offsets[self.motion_ids[ids]]+local)
        self._episode_from_head[ids] = self.time_steps[ids] == self.motion.motion_start_idx[self.motion_ids[ids]]
        self._episode_open[ids] = True
        self._trajectory_open[ids] = True
        self._trajectory_all_contacts[ids] = True
        self._trajectory_observed[ids] = 0
        self._last_boundary[ids] = -1
        self.trajectory_statistics.sampled(self.motion_ids[ids], self._episode_from_head[ids])
        # A reset can jump to a new reference frame. Relative pose caches must
        # match it before the next termination check, without clock advancement.
        self._update_relative_body_targets()

    def update_metrics(self):
        ContactCommand.update_metrics(self)
        term = self.env.termination_manager
        live = self._trajectory_open & (self.env.episode_length_buf > 0)
        due = live & self._episode_from_head & (self.time_steps >= self.segment_end-1)
        due &= self._last_boundary != self.segment_end
        ids = torch.where(due)[0]
        if len(ids):
            actual = measured_contacts(self.env)
            known = actual['known'][ids].all(-1)
            matched = (actual['active'][ids] == self.goal_active[ids]).all(-1)
            success = known & matched & ~term.terminated[ids]
            frames = self.segment_end[ids]
            target = self.motion.body_pos_w[frames][:, self._body_indexes_in_motion]
            target = target+self._reference_env_origins[ids,None,:]
            root_error = (self.env.simulator.robot_root_states[ids,:3]-target[:,0]).norm(dim=-1)
            body_error = (self.robot_body_pos_w[ids][:,self._endpoint_body_ids]
                          -target[:,self._endpoint_body_ids]).norm(dim=-1).mean(-1)
            self._last_boundary_root_error[ids] = root_error
            self._last_boundary_limb_error[ids] = body_error
            self.trajectory_statistics.boundary(self.selected_segment[ids], known, success,root_error,body_error)
            self._trajectory_all_contacts[ids] &= success
            self._trajectory_observed[ids] += 1
            self._last_boundary[ids] = self.segment_end[ids]
        ended = live & (term.terminated | term.time_outs)
        ids = torch.where(ended & self._episode_from_head)[0]
        if len(ids):
            mids = self.motion_ids[ids]
            first = self.motion.motion_start_idx[mids]
            last = self.motion.motion_end_idx[mids]-1
            reached = term.term_dones['motion_ends'][ids] & ~term.terminated[ids]
            all_contacts = self._trajectory_all_contacts[ids] & (self._trajectory_observed[ids] == self.segment_counts[mids])
            progress = (self.time_steps[ids]+1-first).float()/(last-first).clamp_min(1)
            self.trajectory_statistics.finished(mids,self.selected_segment[ids],reached,all_contacts,progress)
        self._trajectory_open[ended] = False
        self._episode_open[ended] = False

    def get_motion_learning_progress_metrics(self):
        values = self.trajectory_statistics.metrics()
        values.update({'cumulative_'+k:v for k,v in self.trajectory_statistics.metrics(True).items()})
        return values

    def get_checkpoint_state(self):
        state = super().get_checkpoint_state()
        state['continuous_trajectory_statistics'] = self.trajectory_statistics.state_dict()
        return state

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        saved = state.get('continuous_trajectory_statistics') if isinstance(state,dict) else None
        if saved is not None:
            self.trajectory_statistics.load_state_dict(saved)
            ids = torch.where(self._trajectory_open)[0]
            self.trajectory_statistics.sampled(self.motion_ids[ids],self._episode_from_head[ids])


def trajectory_complete(env):
    cmd = command(env)
    return cmd.time_steps >= cmd.motion.motion_end_idx[cmd.motion_ids]-2
