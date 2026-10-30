"""Whole-body tracking curricula driven by complete start-probe trajectories."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from holosoma.managers.curriculum.base import CurriculumTermBase


def quantile_or_none(values: torch.Tensor, quantile: float) -> float | None:
    """Return a scalar quantile for a non-empty finite tensor."""
    finite = values[torch.isfinite(values)].to(torch.float32).reshape(-1)
    if finite.numel() == 0:
        return None
    return float(torch.quantile(finite, float(quantile)).item())


def derive_tracking_settings(
    probe_episode_max_p80: float,
    *,
    base_body_threshold: float,
    body_threshold_bounds: tuple[float, float],
    base_root_pos_threshold: float,
    root_pos_threshold_bounds: tuple[float, float],
    base_root_ori_threshold: float,
    root_ori_threshold_bounds: tuple[float, float],
    base_position_sigma: float,
    position_sigma_bounds: tuple[float, float],
    base_reset_noise_scale: float,
    reset_noise_scale_bounds: tuple[float, float],
) -> dict[str, float]:
    """Derive every adaptive setting from the single probe-error statistic."""

    def bounded(value: float, bounds: tuple[float, float]) -> float:
        return min(max(float(value), float(bounds[0])), float(bounds[1]))

    body_threshold = bounded(probe_episode_max_p80, body_threshold_bounds)
    scale = body_threshold / float(base_body_threshold)
    return {
        "body_pos_threshold": body_threshold,
        "root_pos_threshold": bounded(base_root_pos_threshold * scale, root_pos_threshold_bounds),
        "root_ori_threshold": bounded(base_root_ori_threshold * scale, root_ori_threshold_bounds),
        "position_sigma": bounded(base_position_sigma * scale, position_sigma_bounds),
        "reset_noise_scale": bounded(base_reset_noise_scale * scale, reset_noise_scale_bounds),
        "derived_scale": scale,
    }


class TrackingPrecisionCurriculum(CurriculumTermBase):
    """Calibrate tracking bounds from P80 of complete probe-trajectory maxima.

    The only measured curriculum signal is the maximum tracked-body position
    error from each complete start-probe trajectory.  Bad-tracking bounds,
    position reward sigma, and reset randomization are all derived from its
    rolling P80.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.probe_quantile = float(params.get("probe_quantile", 0.80))
        self.probe_history_size = max(int(params.get("probe_history_size", 100)), 1)
        self.min_probe_episodes = max(int(params.get("min_probe_episodes", 10)), 1)
        if not 0.0 <= self.probe_quantile <= 1.0:
            raise ValueError("probe_quantile must be in [0, 1]")
        if self.min_probe_episodes > self.probe_history_size:
            raise ValueError("min_probe_episodes cannot exceed probe_history_size")

        self.base_body_threshold = float(params.get("base_body_threshold", 0.30))
        self.body_threshold_bounds = self._bounds(params, "body_threshold_bounds", (0.05, 0.30))
        self.base_root_pos_threshold = float(params.get("base_root_pos_threshold", 0.50))
        self.root_pos_threshold_bounds = self._bounds(params, "root_pos_threshold_bounds", (0.10, 0.50))
        self.base_root_ori_threshold = float(params.get("base_root_ori_threshold", 0.50))
        self.root_ori_threshold_bounds = self._bounds(params, "root_ori_threshold_bounds", (0.15, 0.50))
        self.base_position_sigma = float(params.get("base_position_sigma", 0.15))
        self.position_sigma_bounds = self._bounds(params, "position_sigma_bounds", (0.05, 0.15))
        self.base_reset_noise_scale = float(params.get("base_reset_noise_scale", 1.0))
        self.reset_noise_scale_bounds = self._bounds(params, "reset_noise_scale_bounds", (1.0 / 6.0, 1.0))

        for name, value in {
            "base_body_threshold": self.base_body_threshold,
            "base_root_pos_threshold": self.base_root_pos_threshold,
            "base_root_ori_threshold": self.base_root_ori_threshold,
            "base_position_sigma": self.base_position_sigma,
            "base_reset_noise_scale": self.base_reset_noise_scale,
        }.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        self.bad_tracking = None
        self.command = None
        self._probe_episode_max = None
        self._probe_episode_seen = None
        self._probe_episode_completed = None
        self._probe_episode_qualified = None
        self._history = None
        self._history_count = 0
        self._history_write_index = 0
        self._completed_probe_episode_count = 0
        self._accepted_probe_episode_count = 0
        self._rejected_probe_episode_count = 0
        self._last_completed_probe_max_mean = 0.0
        self._last_completed_probe_max_max = 0.0
        self._last_completed_probe_qualified_fraction = 0.0
        self._last_probe_error_mean = 0.0
        self._last_probe_error_max = 0.0
        self._stats_ready = False
        self._p50 = 0.0
        self._p75 = 0.0
        self._p80 = 0.0
        self._p90 = 0.0
        self._settings = self._derive(self.base_body_threshold)

    @staticmethod
    def _bounds(params: dict[str, Any], name: str, default: tuple[float, float]) -> tuple[float, float]:
        bounds = tuple(float(x) for x in params.get(name, default))
        if len(bounds) != 2 or bounds[0] <= 0.0 or bounds[1] < bounds[0]:
            raise ValueError(f"{name} must contain positive ordered bounds")
        return bounds

    def _derive(self, p80: float) -> dict[str, float]:
        return derive_tracking_settings(
            p80,
            base_body_threshold=self.base_body_threshold,
            body_threshold_bounds=self.body_threshold_bounds,
            base_root_pos_threshold=self.base_root_pos_threshold,
            root_pos_threshold_bounds=self.root_pos_threshold_bounds,
            base_root_ori_threshold=self.base_root_ori_threshold,
            root_ori_threshold_bounds=self.root_ori_threshold_bounds,
            base_position_sigma=self.base_position_sigma,
            position_sigma_bounds=self.position_sigma_bounds,
            base_reset_noise_scale=self.base_reset_noise_scale,
            reset_noise_scale_bounds=self.reset_noise_scale_bounds,
        )

    def setup(self) -> None:
        term_manager = getattr(self.env, "termination_manager", None)
        if term_manager is None:
            raise RuntimeError("TrackingPrecisionCurriculum requires a termination manager")
        self.bad_tracking = term_manager.get_term("bad_tracking")
        self.command = self.env.command_manager.get_state("motion_command")
        if not getattr(self.command, "_use_start_probe_envs", False):
            raise RuntimeError("TrackingPrecisionCurriculum requires complete start-probe environments")
        for name in (
            "motion_global_ref_position_error_exp",
            "motion_relative_body_position_error_exp",
        ):
            if name not in self.env.reward_manager.active_terms:
                raise RuntimeError(f"TrackingPrecisionCurriculum requires reward term {name!r}")

        self._probe_episode_max = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.env.device)
        self._probe_episode_seen = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self._probe_episode_completed = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self._probe_episode_qualified = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self._history = torch.zeros(self.probe_history_size, dtype=torch.float32, device=self.env.device)
        self._apply(self._settings)

    def observe_probe_tracking_error(
        self,
        body_position_error: torch.Tensor,
        ref_position_error: torch.Tensor | None = None,
        ref_orientation_error: torch.Tensor | None = None,
    ) -> None:
        """Accumulate complete probes that stay inside the fixed qualification boundary."""
        if (
            self.command is None
            or self._probe_episode_max is None
            or self._probe_episode_seen is None
            or self._probe_episode_completed is None
            or self._probe_episode_qualified is None
        ):
            return
        probe_mask = self.command._probe_env_mask & self.command._probe_episode_valid
        motion_ids = self.command.motion_ids
        start_idx = self.command.motion.motion_start_idx[motion_ids]
        end_idx = self.command.motion.motion_end_idx[motion_ids]

        # A completed motion may restart inside the command term without an
        # environment reset.  Unlatch it only after its timestep wraps to the
        # first frame, so repeated evaluation of the final frame cannot create
        # duplicate trajectory samples.
        restarted = probe_mask & self._probe_episode_completed & (self.command.time_steps <= start_idx)
        if torch.any(restarted):
            self._probe_episode_max[restarted] = 0.0
            self._probe_episode_seen[restarted] = False
            self._probe_episode_completed[restarted] = False
            self._probe_episode_qualified[restarted] = True

        # The canonical start frame is the reset target, before the policy has
        # produced an action.  Some backends still expose the pre-reset rigid-
        # body cache on that observation, so it is a baseline frame rather than
        # a valid execution-error sample.
        past_start = self.command.time_steps > start_idx
        sample_mask = probe_mask & ~self._probe_episode_completed & past_start
        values = body_position_error.detach().to(torch.float32)
        root_pos_values = (
            torch.zeros_like(values)
            if ref_position_error is None
            else ref_position_error.detach().to(torch.float32)
        )
        root_ori_values = (
            torch.zeros_like(values)
            if ref_orientation_error is None
            else ref_orientation_error.detach().to(torch.float32)
        )
        finite = torch.isfinite(values) & torch.isfinite(root_pos_values) & torch.isfinite(root_ori_values)
        within_fixed_boundary = (
            finite
            & (values <= self.base_body_threshold)
            & (root_pos_values <= self.base_root_pos_threshold)
            & (root_ori_values <= self.base_root_ori_threshold)
        )
        self._probe_episode_qualified[sample_mask] &= within_fixed_boundary[sample_mask]

        finite_mask = sample_mask & torch.isfinite(values)
        if not torch.any(finite_mask):
            self._last_probe_error_mean = 0.0
            self._last_probe_error_max = 0.0
            return
        self._probe_episode_max[finite_mask] = torch.maximum(
            self._probe_episode_max[finite_mask], values[finite_mask]
        )
        self._probe_episode_seen[finite_mask] = True
        observed = values[finite_mask]
        self._last_probe_error_mean = float(observed.mean().item())
        self._last_probe_error_max = float(observed.max().item())

        qualified_observed = finite_mask & self._probe_episode_qualified
        if torch.any(qualified_observed):
            qualified_values = values[qualified_observed]
            self._last_probe_error_mean = float(qualified_values.mean().item())
            self._last_probe_error_max = float(qualified_values.max().item())
        else:
            self._last_probe_error_mean = 0.0
            self._last_probe_error_max = 0.0

        completed = sample_mask & (self.command.time_steps >= end_idx - 2)
        if torch.any(completed):
            accepted = completed & self._probe_episode_seen & self._probe_episode_qualified
            completed_count = int(completed.sum().item())
            accepted_count = int(accepted.sum().item())
            rejected_count = completed_count - accepted_count
            accepted_values = self._probe_episode_max[accepted].detach()
            if accepted_values.numel() > 0:
                self._append_history(accepted_values)
                self._last_completed_probe_max_mean = float(accepted_values.mean().item())
                self._last_completed_probe_max_max = float(accepted_values.max().item())
            self._completed_probe_episode_count += completed_count
            self._accepted_probe_episode_count += accepted_count
            self._rejected_probe_episode_count += rejected_count
            self._last_completed_probe_qualified_fraction = accepted_count / completed_count
            self._probe_episode_completed[completed] = True
            if accepted_values.numel() > 0:
                self._refresh_statistics()

    def completed_probe_qualification(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return fixed-boundary qualification for completed probe episodes."""
        if self._probe_episode_completed is None or self._probe_episode_qualified is None:
            return torch.zeros_like(env_ids, dtype=torch.bool, device=self.env.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device).reshape(-1)
        return self._probe_episode_completed[env_ids] & self._probe_episode_qualified[env_ids]

    def reset(self, env_ids) -> None:
        """Reject failed partial probes, then clear their accumulators."""
        if (
            self.command is None
            or self._probe_episode_max is None
            or self._probe_episode_seen is None
            or self._probe_episode_completed is None
            or self._probe_episode_qualified is None
        ):
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device).reshape(-1)
        if env_ids.numel() == 0:
            return
        rejected = (
            self.command._probe_env_mask[env_ids]
            & self.command._probe_episode_valid[env_ids]
            & self._probe_episode_seen[env_ids]
            & ~self._probe_episode_completed[env_ids]
            & ~self._probe_episode_qualified[env_ids]
        )
        rejected_count = int(rejected.sum().item())
        if rejected_count > 0:
            self._rejected_probe_episode_count += rejected_count
            self._last_completed_probe_qualified_fraction = 0.0
        self._probe_episode_max[env_ids] = 0.0
        self._probe_episode_seen[env_ids] = False
        self._probe_episode_completed[env_ids] = False
        self._probe_episode_qualified[env_ids] = True

    def _append_history(self, values: torch.Tensor) -> None:
        assert self._history is not None
        for value in values.reshape(-1):
            self._history[self._history_write_index] = value
            self._history_write_index = (self._history_write_index + 1) % self.probe_history_size
            self._history_count = min(self._history_count + 1, self.probe_history_size)

    def _history_values(self) -> torch.Tensor:
        assert self._history is not None
        if self._history_count < self.probe_history_size:
            return self._history[: self._history_count]
        return self._history

    def _refresh_statistics(self) -> None:
        values = self._history_values()
        if values.numel() == 0:
            return
        self._p50 = quantile_or_none(values, 0.50) or 0.0
        self._p75 = quantile_or_none(values, 0.75) or 0.0
        self._p80 = quantile_or_none(values, self.probe_quantile) or 0.0
        self._p90 = quantile_or_none(values, 0.90) or 0.0
        if self._history_count < self.min_probe_episodes:
            return
        self._stats_ready = True
        self._settings = self._derive(self._p80)
        self._apply(self._settings)

    def _apply(self, settings: dict[str, float]) -> None:
        assert self.bad_tracking is not None
        assert self.command is not None
        self.bad_tracking.bad_ref_pos_threshold = settings["root_pos_threshold"]
        self.bad_tracking.bad_ref_ori_threshold = settings["root_ori_threshold"]
        self.bad_tracking.bad_motion_body_pos_threshold = settings["body_pos_threshold"]

        for name in (
            "motion_global_ref_position_error_exp",
            "motion_relative_body_position_error_exp",
        ):
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            self.env.reward_manager.set_term_cfg(
                name,
                replace(term_cfg, params={**term_cfg.params, "sigma": settings["position_sigma"]}),
            )
        self.command.init_pose_cfg = replace(
            self.command.init_pose_cfg,
            overall_noise_scale=settings["reset_noise_scale"],
        )

    def step(self) -> None:
        if not hasattr(self.env, "log_dict"):
            return
        values = {
            "tracking_precision/probe_episode_max_p80": self._p80,
            "tracking_precision/probe_episode_sample_count": float(self._history_count),
            "tracking_precision/probe_complete_episode_count_total": float(self._completed_probe_episode_count),
            "tracking_precision/probe_accepted_episode_count_total": float(self._accepted_probe_episode_count),
            "tracking_precision/probe_rejected_episode_count_total": float(self._rejected_probe_episode_count),
            "tracking_precision/probe_statistics_ready": float(self._stats_ready),
            "tracking_precision/body_pos_threshold": self._settings["body_pos_threshold"],
            "tracking_precision/root_pos_threshold": self._settings["root_pos_threshold"],
            "tracking_precision/root_ori_threshold": self._settings["root_ori_threshold"],
            "tracking_precision/position_sigma": self._settings["position_sigma"],
            "tracking_precision/reset_noise_scale": self._settings["reset_noise_scale"],
        }
        for name, value in values.items():
            self.env.log_dict[name] = torch.tensor(value, dtype=torch.float32, device=self.env.device)
