from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.managers.curriculum.terms.climb_reference_goal import ClimbReferenceGoalCurriculum


class _Command:
    imitation_probability_start = 1.0
    imitation_probability_final = 0.5

    def __init__(self) -> None:
        self.difficulty = 0.0
        self.imitation_probability = 1.0


def _curriculum() -> tuple[ClimbReferenceGoalCurriculum, _Command, SimpleNamespace]:
    command = _Command()
    env = SimpleNamespace(
        device="cpu",
        num_envs=4,
        log_dict={},
        command_manager=SimpleNamespace(get_state=lambda _: command),
        _pending_episode_lengths=torch.ones(4, dtype=torch.long),
    )
    cfg = SimpleNamespace(
        params={
            "update_interval_steps": 1,
            "performance_ema_alpha": 1.0,
            "promote_threshold": 0.8,
            "demote_threshold": 0.6,
            "difficulty_step_up": 0.1,
            "difficulty_step_down": 0.25,
            "min_completed_episodes": 1,
        }
    )
    term = ClimbReferenceGoalCurriculum(cfg, env)
    # Avoid requiring the concrete command subclass in this isolated state-machine test.
    term.command = command  # type: ignore[assignment]
    term._completed = torch.zeros(())
    term._qualified = torch.zeros(())
    term._episode_task_success = torch.zeros(4, dtype=torch.bool)
    term._episode_success_seen = torch.zeros(4, dtype=torch.bool)
    return term, command, env


def test_curriculum_promotes_only_after_qualified_performance() -> None:
    term, command, _ = _curriculum()
    term._completed.fill_(10.0)
    term._qualified.fill_(9.0)
    term.step()
    assert command.difficulty == 0.1
    assert command.imitation_probability == 0.95


def test_curriculum_demotes_faster_after_failures() -> None:
    term, command, _ = _curriculum()
    command.difficulty = 0.5
    term._completed.fill_(10.0)
    term._qualified.fill_(5.0)
    term.step()
    assert command.difficulty == 0.25
    assert command.imitation_probability == 0.875


def test_reset_qualifies_only_completed_contact_tasks() -> None:
    term, _, env = _curriculum()
    term.observe_task_success(torch.tensor([True, False, False, True]))
    env._pending_episode_lengths[:] = torch.tensor([3, 3, 3, 0])
    term.reset(torch.arange(4))
    assert term._completed.item() == 3.0
    assert term._qualified.item() == 1.0


def test_task_success_is_sticky_within_episode() -> None:
    term, _, env = _curriculum()
    term.observe_task_success(torch.tensor([False, True, False, False]))
    term.observe_task_success(torch.zeros(4, dtype=torch.bool))
    env._pending_episode_lengths.fill_(3)
    term.reset(torch.arange(4))
    assert term._completed.item() == 4.0
    assert term._qualified.item() == 1.0
