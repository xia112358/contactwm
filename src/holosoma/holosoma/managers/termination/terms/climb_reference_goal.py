"""Termination terms for climb contact-goal episodes."""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.climb_reference_goal import ClimbReferenceGoalCommand


def climb_contact_goal_deadline(env) -> torch.Tensor:
    """End each episode at its touchdown boundary (or final stability tail)."""
    command = env.command_manager.get_state("motion_command")
    if not isinstance(command, ClimbReferenceGoalCommand):
        raise TypeError(f"Expected ClimbReferenceGoalCommand, got {type(command)}")
    return command.time_steps >= command.deadline_steps
