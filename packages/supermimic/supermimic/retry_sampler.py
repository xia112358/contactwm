"""SomaForge recent-30 failure-window reset semantics, with statistics-only probes.

Adapted from holosoma/managers/command/terms/wbt.py. Motion/terrain allocation
stays fixed to preserve the controlled experiment's four-group proportions.
"""
import torch

SAMPLER_NAME = 'completion_recent30_failure_window'
SHORT_PROBE_PRE_FRAMES = 100
SHORT_PROBE_POST_FRAMES = 50


def failure_window(start, last, target, pre=50, post=20, before_prob=.7):
    center = torch.minimum(torch.maximum(target, start), last)
    before = torch.rand(len(start), device=start.device) < before_prob
    lower = torch.where(before, torch.maximum(start, center-pre), center)
    upper = torch.where(before, center, torch.minimum(last, center+post))
    offset = torch.floor(torch.rand(len(start), device=start.device) * (upper-lower+1).float()).long()
    return lower + offset


def training_transition(out, training_ids):
    """Select training environments before replay, normalizers, or learning metrics."""
    result = {k: (v[training_ids] if isinstance(v, torch.Tensor) and
                   k not in ('ep_len', 'ep_ret_norm', 'episode_env_ids') else v)
              for k, v in out.items()}
    if 'episode_env_ids' in out:
        mask = torch.isin(out['episode_env_ids'], training_ids)
        for key in ('ep_len', 'ep_ret_norm', 'episode_env_ids'):
            result[key] = out[key][mask]
    return result


class Recent30FailureSampler:
    def __init__(self, task_ids, training_mask, motion, probes_per_motion=3, probe_mode='all_long'):
        self.task_ids = task_ids
        self.training_mask = training_mask
        self.probe_mask = ~training_mask
        self.start = motion.clip_start
        self.end = motion.clip_end  # inclusive last reference frame
        self.device = task_ids.device
        self.C = len(self.start)
        self.probe_ids = self.probe_mask.nonzero().flatten().reshape(self.C, probes_per_motion)
        if probe_mode not in ('all_long', 'long_short') or (probe_mode == 'long_short' and probes_per_motion != 3):
            raise ValueError('long_short requires exactly one long and two short probes per motion')
        self.probe_mode = probe_mode
        self.short_probe_mask = torch.zeros_like(training_mask)
        if probe_mode == 'long_short':
            self.short_probe_mask[self.probe_ids[:, 1:].flatten()] = True
        self.recent = torch.zeros(self.C, 30, 4, device=self.device)
        self.count = torch.zeros(self.C, dtype=torch.long, device=self.device)
        self.cursor = self.count.clone()
        self.total_probes = self.count.clone()
        self.failed = torch.zeros_like(task_ids, dtype=torch.bool)
        self.failed_frame = torch.zeros_like(task_ids)
        self.retry = torch.zeros_like(task_ids, dtype=torch.bool)
        self.target = torch.zeros_like(task_ids)
        self.reset_counts = torch.zeros(4, dtype=torch.long, device=self.device)
        self.retry_successes = torch.zeros((), dtype=torch.long, device=self.device)

    def sample(self, ids):
        task = self.task_ids[ids]
        start, last = self.start[task], self.end[task]-1
        frames = start + torch.floor(torch.rand(len(ids), device=self.device) *
                                     (self.end[task]-start).clamp(min=1).float()).long()
        probes = self.probe_mask[ids]
        failed = self.failed[ids] & ~probes
        use_center = (~probes & ~failed & (self.count[task] > 0) &
                      (torch.rand(len(ids), device=self.device) >= .3))
        mean = self.recent[task, :, 0].sum(-1) / self.count[task].clamp(min=1)
        centers = start + torch.round(mean.clamp(0, 1) * (self.end[task]-start).float()).long()
        targets = torch.where(failed, self.failed_frame[ids], centers)
        windows = failed | use_center
        ix = windows.nonzero().flatten()
        frames[ix] = failure_window(start[ix], last[ix], targets[ix])
        frames[probes] = start[probes]
        self.retry[ids] = windows
        self.target[ids] = targets
        short = self.short_probe_mask[ids].nonzero().flatten()
        short_target = torch.where(self.failed[ids[short]], self.failed_frame[ids[short]], centers[short])
        short_target = torch.minimum(torch.maximum(short_target, start[short]), last[short])
        frames[short] = torch.maximum(start[short], short_target-SHORT_PROBE_PRE_FRAMES)
        self.target[ids[short]] = short_target
        self.failed[ids] = False
        self.reset_counts += torch.stack((probes.sum(), (~probes & ~windows).sum(),
                                          use_center.sum(), failed.sum()))
        return frames

    def transition(self, f1, term, timeout):
        # Probe episodes must run to failure or reference end, not the training
        # episode's 10-second cap, otherwise the completion statistic is censored.
        clip_done = f1 >= self.end[self.task_ids]
        short_limit = self.short_probe_mask & (f1 >= self.target+SHORT_PROBE_POST_FRAMES)
        timeout = torch.where(self.probe_mask, clip_done | short_limit, timeout)
        probe_done = (term | timeout)[self.probe_ids]
        for j in range(self.probe_ids.shape[1]):
            tasks = probe_done[:, j].nonzero().flatten()
            ids = self.probe_ids[tasks, j]
            completion = ((f1[ids]-self.start[tasks]).float() /
                          (self.end[tasks]-self.start[tasks]).clamp(min=1)).clamp(0, 1)
            success = (completion >= .98) & ~term[ids] & clip_done[ids]
            outcomes = torch.stack((completion, success.float(), term[ids].float(),
                                    (short_limit[ids] & ~clip_done[ids] & ~term[ids]).float()), -1)
            self.recent[tasks, self.cursor[tasks]] = outcomes
            self.cursor[tasks] = (self.cursor[tasks]+1) % 30
            self.count[tasks] = (self.count[tasks]+1).clamp(max=30)
            self.total_probes[tasks] += 1
        reached = self.training_mask & self.retry & (f1 >= self.target+50) & ~term
        self.retry_successes += reached.sum()
        self.failed.copy_(term & (self.training_mask | self.short_probe_mask))
        self.failed_frame.copy_(f1)
        return timeout | reached

    def state_dict(self):
        # Statistics only: no simulator state or in-flight episode snapshots.
        return {'schema': 'recent30_statistics_only_probes_v1', 'probe_mode': self.probe_mode,
                'short_window': [SHORT_PROBE_PRE_FRAMES, SHORT_PROBE_POST_FRAMES],
                **{key: getattr(self, key).detach().cpu() for key in
                   ('recent', 'count', 'cursor', 'total_probes', 'reset_counts', 'retry_successes')}}

    def load_state_dict(self, state):
        if state.get('schema') != 'recent30_statistics_only_probes_v1':
            raise ValueError('Incompatible reset sampler checkpoint')
        if state.get('probe_mode', 'all_long') != self.probe_mode:
            raise ValueError('Cannot change probe semantics on resume')
        if self.probe_mode == 'long_short' and state.get('short_window') != [SHORT_PROBE_PRE_FRAMES, SHORT_PROBE_POST_FRAMES]:
            raise ValueError('Cannot change short-probe window on resume')
        for key in ('recent', 'count', 'cursor', 'total_probes', 'reset_counts', 'retry_successes'):
            target = getattr(self, key)
            if target.shape != state[key].shape:
                raise ValueError(f'Incompatible reset sampler tensor: {key}')
            target.copy_(state[key].to(self.device))

    def summary(self, names):
        means = (self.recent.sum(1) / self.count.clamp(min=1)[:, None]).cpu().tolist()
        counts = self.count.cpu().tolist()
        totals = self.total_probes.cpu().tolist()
        return {'sampler': SAMPLER_NAME, 'training_envs': int(self.training_mask.sum()),
                'probe_mode': self.probe_mode,
                'short_probe_window_frames': [SHORT_PROBE_PRE_FRAMES, SHORT_PROBE_POST_FRAMES],
                'completion_semantics': ('absolute endpoint fraction, shared recent30 across long and short probes; '
                                         'sampling progress estimate, not from-start success rate' if self.probe_mode == 'long_short'
                                         else 'recent30 from-start probe completion'),
                'statistics_only_probe_envs': int(self.probe_mask.sum()),
                'reset_counts': dict(zip(('probe', 'uniform', 'completion_window', 'failure_retry'),
                                         self.reset_counts.cpu().tolist())),
                'retry_successes': int(self.retry_successes),
                'per_clip': {name: dict(recent_count=counts[i], total_probe_episodes=totals[i],
                                        mean_completion=means[i][0], success_rate=means[i][1])
                             for i,name in enumerate(names)}}
