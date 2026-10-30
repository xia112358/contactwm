from __future__ import annotations

import pytest
import torch
from holosoma.agents.ppo.ppo import execution_mean_consistency_error


def test_execution_mean_consistency_only_updates_execution_mean() -> None:
    exec_mu = torch.tensor([[0.4, -0.2]], requires_grad=True)
    teacher_mu = torch.tensor([[0.0, 0.2]], requires_grad=True)
    log_std = torch.tensor(-1.0, requires_grad=True)

    loss = execution_mean_consistency_error(exec_mu, teacher_mu, fixed_scale=0.2)
    loss.backward()

    assert loss.item() == pytest.approx(4.0)
    assert exec_mu.grad is not None
    assert torch.count_nonzero(exec_mu.grad).item() == exec_mu.numel()
    assert teacher_mu.grad is None
    assert log_std.grad is None


def test_execution_mean_consistency_requires_positive_fixed_scale() -> None:
    with pytest.raises(ValueError, match="fixed_scale must be positive"):
        execution_mean_consistency_error(torch.zeros(1, 2), torch.zeros(1, 2), fixed_scale=0.0)
