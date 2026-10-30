"""Coupled curriculum for climb reference-and-goal multi-task training."""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.climb_reference_goal import ClimbReferenceGoalCommand
from holosoma.managers.curriculum.base import CurriculumTermBase


class ClimbReferenceGoalCurriculum(CurriculumTermBase):
    """Anneal assistance, broaden goals/resets, and mix in generalization.

    A single difficulty value drives all three changes.  Advancement is based
    on completed contact goals, while reference tracking remains diagnostic and
    may still be used as a shaping reward.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.update_interval_steps = max(int(params.get("update_interval_steps", 24)), 1)
        self.performance_ema_alpha = float(params.get("performance_ema_alpha", 0.10))
        self.promote_threshold = float(params.get("promote_threshold", 0.80))
        self.demote_threshold = float(params.get("demote_threshold", 0.60))
        self.difficulty_step_up = float(params.get("difficulty_step_up", 0.001))
        self.difficulty_step_down = float(params.get("difficulty_step_down", 0.01))
        self.min_completed_episodes = max(int(params.get("min_completed_episodes", 128)), 1)
        if not 0.0 < self.performance_ema_alpha <= 1.0:
            raise ValueError("performance_ema_alpha must be in (0, 1]")
        if not 0.0 <= self.demote_threshold < self.promote_threshold <= 1.0:
            raise ValueError("curriculum thresholds must satisfy 0 <= demote < promote <= 1")
        if self.difficulty_step_up <= 0.0 or self.difficulty_step_down <= 0.0:
            raise ValueError("difficulty steps must be positive")
        self.step_count = 0
        self.command: ClimbReferenceGoalCommand | None = None
        self.performance_ema = 0.0
        self._has_performance = False
        self._completed = None
        self._qualified = None
        self._episode_task_success = None
        self._episode_success_seen = None

    def setup(self) -> None:
        command = self.env.command_manager.get_state("motion_command")
        if not isinstance(command, ClimbReferenceGoalCommand):
            raise TypeError(f"Expected ClimbReferenceGoalCommand, got {type(command)}")
        self.command = command
        self._completed = torch.zeros((), dtype=torch.float32, device=self.env.device)
        self._qualified = torch.zeros((), dtype=torch.float32, device=self.env.device)
        self._episode_task_success = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        self._episode_success_seen = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        self._update_command(float(command.difficulty))

    def observe_task_success(self, task_success: torch.Tensor) -> None:
        """Remember whether each environment completed its assigned contact goal."""
        if self._episode_task_success is None or self._episode_success_seen is None:
            return
        success = task_success.detach().to(device=self.env.device)
        if success.shape != self._episode_task_success.shape:
            raise ValueError(
                "task_success must have one value per environment, got "
                f"{tuple(success.shape)} for {self.env.num_envs} environments"
            )
        if success.dtype == torch.bool:
            finite = torch.ones_like(success)
        else:
            finite = torch.isfinite(success)
            success = success > 0.0
        self._episode_task_success[finite] |= success[finite]
        self._episode_success_seen |= finite

    def reset(self, env_ids) -> None:
        if self._completed is None or self._qualified is None:
            return
        episode_lengths = getattr(self.env, "_pending_episode_lengths", None)
        if (
            episode_lengths is None
            or self._episode_task_success is None
            or self._episode_success_seen is None
        ):
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device)
        if ids.numel() == 0:
            return
        valid = (episode_lengths[ids] > 0) & self._episode_success_seen[ids]
        if not torch.any(valid):
            self._episode_task_success[ids] = False
            self._episode_success_seen[ids] = False
            return
        qualified = valid & self._episode_task_success[ids]
        self._completed.add_(valid.to(torch.float32).sum())
        self._qualified.add_(qualified.to(torch.float32).sum())
        self._episode_task_success[ids] = False
        self._episode_success_seen[ids] = False

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.update_interval_steps == 0:
            self._update_from_performance()
        self._log_state()

    def _update_from_performance(self) -> None:
        if self.command is None or self._completed is None or self._qualified is None:
            return
        completed = float(self._completed.item())
        if completed < float(self.min_completed_episodes):
            return
        performance = float((self._qualified / self._completed.clamp_min(1.0)).item())
        if self._has_performance:
            alpha = self.performance_ema_alpha
            self.performance_ema = (1.0 - alpha) * self.performance_ema + alpha * performance
        else:
            self.performance_ema = performance
            self._has_performance = True

        difficulty = float(self.command.difficulty)
        if self.performance_ema >= self.promote_threshold:
            difficulty += self.difficulty_step_up
        elif self.performance_ema <= self.demote_threshold:
            difficulty -= self.difficulty_step_down
        self._update_command(min(max(difficulty, 0.0), 1.0))
        self._completed.zero_()
        self._qualified.zero_()

    def _update_command(self, difficulty: float) -> None:
        if self.command is None:
            return
        self.command.difficulty = float(difficulty)
        start = self.command.imitation_probability_start
        final = self.command.imitation_probability_final
        self.command.imitation_probability = start + float(difficulty) * (final - start)
        self.command.curriculum_performance_ema = float(self.performance_ema)
        self.command.curriculum_has_performance = bool(self._has_performance)

    def _log_state(self) -> None:
        if self.command is None:
            return
        self.env.log_dict["climb_reference_goal/difficulty"] = torch.tensor(
            self.command.difficulty, dtype=torch.float32
        )
        self.env.log_dict["climb_reference_goal/imitation_probability"] = torch.tensor(
            self.command.imitation_probability, dtype=torch.float32
        )
        self.env.log_dict["climb_reference_goal/curriculum_performance_ema"] = torch.tensor(
            self.performance_ema, dtype=torch.float32
        )
        self.env.log_dict["climb_reference_goal/curriculum_task_success_ema"] = torch.tensor(
            self.performance_ema, dtype=torch.float32
        )


class ClimbImitationTrackingCurriculum(CurriculumTermBase):
    """Anneal base assistance using dense imitation quality only.

    This is intentionally separate from the mixed-task success curriculum: it
    answers whether a goal-conditioned actor can learn one full behavior when
    the reference is available only to training losses and the critic.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.update_interval_steps = max(int(params.get("update_interval_steps", 24)), 1)
        self.performance_ema_alpha = float(params.get("performance_ema_alpha", 0.05))
        self.promote_threshold = float(params.get("promote_threshold", 0.72))
        self.demote_threshold = float(params.get("demote_threshold", 0.55))
        self.difficulty_step_up = float(params.get("difficulty_step_up", 0.002))
        self.difficulty_step_down = float(params.get("difficulty_step_down", 0.005))
        self.warmup_updates = max(int(params.get("warmup_updates", 20)), 0)
        if not 0.0 < self.performance_ema_alpha <= 1.0:
            raise ValueError("performance_ema_alpha must be in (0, 1]")
        if not 0.0 <= self.demote_threshold < self.promote_threshold <= 1.0:
            raise ValueError("curriculum thresholds must satisfy 0 <= demote < promote <= 1")
        if self.difficulty_step_up <= 0.0 or self.difficulty_step_down <= 0.0:
            raise ValueError("difficulty steps must be positive")
        self.command: ClimbReferenceGoalCommand | None = None
        self.step_count = 0
        self.update_count = 0
        self.performance_ema = 0.0
        self._has_performance = False
        self._score_sum = None
        self._score_count = 0

    def setup(self) -> None:
        command = self.env.command_manager.get_state("motion_command")
        if not isinstance(command, ClimbReferenceGoalCommand):
            raise TypeError(f"Expected ClimbReferenceGoalCommand, got {type(command)}")
        if command.imitation_probability_start != 1.0 or command.imitation_probability_final != 1.0:
            raise ValueError("ClimbImitationTrackingCurriculum requires imitation_probability=1")
        self.command = command
        self._score_sum = torch.zeros((), dtype=torch.float32, device=self.env.device)
        self._update_command(float(command.difficulty))

    def observe_tracking_score(self, tracking_score: torch.Tensor) -> None:
        if self._score_sum is None:
            return
        finite = tracking_score.detach().to(torch.float32)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            return
        self._score_sum.add_(finite.mean())
        self._score_count += 1

    def reset(self, env_ids) -> None:
        """Tracking statistics span episode boundaries; no per-env state is kept."""
        del env_ids

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.update_interval_steps == 0:
            self._update_from_tracking()
        self._log_state()

    def _update_from_tracking(self) -> None:
        if self.command is None or self._score_sum is None or self._score_count == 0:
            return
        performance = float((self._score_sum / float(self._score_count)).item())
        if self._has_performance:
            alpha = self.performance_ema_alpha
            self.performance_ema = (1.0 - alpha) * self.performance_ema + alpha * performance
        else:
            self.performance_ema = performance
            self._has_performance = True
        self.update_count += 1
        difficulty = float(self.command.difficulty)
        if self.update_count > self.warmup_updates:
            if self.performance_ema >= self.promote_threshold:
                difficulty += self.difficulty_step_up
            elif self.performance_ema <= self.demote_threshold:
                difficulty -= self.difficulty_step_down
        self._update_command(min(max(difficulty, 0.0), 1.0))
        self._score_sum.zero_()
        self._score_count = 0

    def _update_command(self, difficulty: float) -> None:
        if self.command is None:
            return
        self.command.difficulty = float(difficulty)
        self.command.imitation_probability = 1.0
        self.command.curriculum_performance_ema = float(self.performance_ema)
        self.command.curriculum_has_performance = bool(self._has_performance)

    def _log_state(self) -> None:
        if self.command is None:
            return
        values = {
            "difficulty": self.command.difficulty,
            "imitation_probability": 1.0,
            "curriculum_performance_ema": self.performance_ema,
        }
        for name, value in values.items():
            self.env.log_dict[f"climb_reference_goal/{name}"] = torch.tensor(
                value, dtype=torch.float32, device=self.env.device
            )
