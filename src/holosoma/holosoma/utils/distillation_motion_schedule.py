"""Delayed WBT probe scheduling for a multi-teacher student.

The command owns motion assignment, recent-30 statistics, failure windows and
checkpoint serialization. This adapter selects when those existing mechanisms
are enabled. Normal resets compete globally across all motions; teacher routing
does not impose a sampling quota.
"""
from __future__ import annotations

import dataclasses
from types import MethodType

import torch


class DistillationMotionSchedule:
    def __init__(self, command, teacher_for_motion, *, start_iteration=1000, probes_per_motion=3):
        if start_iteration < 0 or probes_per_motion < 1:
            raise ValueError("Invalid distillation schedule")
        self.command = command
        self.teacher_for_motion = teacher_for_motion
        self.start_iteration = int(start_iteration)
        self.probes_per_motion = int(probes_per_motion)
        self.active = False
        self.groups = torch.unique(teacher_for_motion, sorted=True)
        self.group_masks = [teacher_for_motion == group for group in self.groups]
        self.num_probes = len(teacher_for_motion) * self.probes_per_motion
        if self.num_probes >= command.num_envs:
            raise ValueError(f"Need more than {self.num_probes} environments for probes plus normal sampling")
        self.original_update = command.update_probe_motion_sampling_weights
        command.update_probe_motion_sampling_weights = MethodType(lambda cmd: self.update_weights(), command)

    def metadata(self):
        return dict(kind="global_motion_recent30", start_iteration=self.start_iteration,
                    probes_per_motion=self.probes_per_motion, active=self.active,
                    probe_envs=self.num_probes if self.active else 0,
                    balance_scope="global motion priority; fixed probes are motion-balanced")

    def advance(self, iteration, *, preserve_statistics=False):
        if self.active or iteration < self.start_iteration:
            return False
        cmd = self.command
        cmd.motion_cfg = dataclasses.replace(
            cmd.motion_cfg, use_start_probe_envs=True, use_group_probe_envs=False,
            probe_env_per_motion=self.probes_per_motion,
            reset_sampler="completion_recent30_failure_window")
        cmd._use_start_probe_envs = True
        cmd._use_group_probe_envs = False
        cmd._probe_env_per_motion = self.probes_per_motion
        cmd._num_probe_envs = self.num_probes
        cmd._probe_env_mask.zero_()
        cmd._probe_env_mask[:self.num_probes] = True
        cmd._probe_motion_ids.fill_(-1)
        cmd._probe_motion_ids[:self.num_probes] = torch.arange(
            len(self.teacher_for_motion), device=cmd.device).repeat_interleave(self.probes_per_motion)
        # An in-flight random-start episode must never count as a from-zero probe.
        # It adopts the fixed assignment only at its next natural reset.
        cmd._probe_episode_valid.zero_()
        cmd._reset_sampler = "completion_recent30_failure_window"
        cmd._uses_failure_window_sampler = True
        cmd._uses_completion_recent30_failure_window_sampler = True
        cmd._failure_window_retry_active.zero_()
        if not preserve_statistics:
            for suffix in ("count", "fail_bin_hist", "recent30_values", "recent30_count",
                           "recent30_cursor", "completion_recent30", "success_recent30",
                           "fail_recent30", "timeout_recent30"):
                getattr(cmd, "_probe_" + suffix).zero_()
            cmd._normal_motion_sampling_weights.fill_(1. / len(self.teacher_for_motion))
        self.active = True
        # Preserve checkpoint weights on resume; the next EMA update transitions
        # them to global priorities without clearing learned probe statistics.
        for mask in self.group_masks:
            weight = cmd._normal_motion_sampling_weights[mask]
            if not torch.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
                raise ValueError("Invalid restored motion weights")
        return True

    def update_weights(self):
        if not self.active:
            return self.original_update()
        cmd = self.command
        mix = float(cmd.motion_cfg.probe_uniform_mix)
        beta = float(cmd.motion_cfg.probe_weight_beta)
        temperature = max(float(cmd.motion_cfg.probe_priority_temperature), 1e-6)
        priority = torch.softmax((1. - cmd._probe_success_recent30) / temperature, dim=0)
        target = mix / len(self.teacher_for_motion) + (1. - mix) * priority
        weights = (1. - beta) * cmd._normal_motion_sampling_weights + beta * target
        cmd._normal_motion_sampling_weights.copy_(weights / weights.sum())

    def metrics(self):
        cmd = self.command
        seen = cmd._probe_recent30_count > 0
        return {
            **self.metadata(),
            "motions_with_probe_results": int(seen.sum()),
            "probe_episodes": int(cmd._probe_count.sum()),
            "probe_completion_mean_observed": float(cmd._probe_completion_recent30[seen].mean()) if seen.any() else None,
            "probe_success_mean_observed": float(cmd._probe_success_recent30[seen].mean()) if seen.any() else None,
            "normal_teacher_priors": [float(cmd._normal_motion_sampling_weights[m].sum()) for m in self.group_masks],
            "normal_weight_min": float(cmd._normal_motion_sampling_weights.min()),
            "normal_weight_max": float(cmd._normal_motion_sampling_weights.max()),
        }
