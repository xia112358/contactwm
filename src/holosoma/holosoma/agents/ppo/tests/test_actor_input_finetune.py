from types import SimpleNamespace

import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_module = SimpleNamespace(
            module=nn.Sequential(nn.Linear(542, 8), nn.ELU(), nn.Linear(8, 4), nn.ELU(), nn.Linear(4, 2))
        )
        self.log_std = nn.Parameter(torch.zeros(2))


def _ppo(mode: str) -> PPO:
    algo = object.__new__(PPO)
    algo.config = SimpleNamespace(actor_finetune_mode=mode)
    algo.actor = _Actor()
    algo._actor_first_layer_gradient_mask = None
    algo._actor_first_layer_weight = None
    return algo


def test_ref_qd_finetune_masks_every_future_frame() -> None:
    algo = _ppo("ref_qd")
    algo._configure_actor_finetune()
    weight = algo._actor_first_layer_weight
    mask = algo._actor_first_layer_gradient_mask
    assert weight is not None
    assert mask is not None
    assert int(mask[0].sum().item()) == 4 * 29
    for frame in range(4):
        assert torch.all(mask[:, frame * 67 + 29 : frame * 67 + 58] == 1)
        assert torch.all(mask[:, frame * 67 : frame * 67 + 29] == 0)
    assert weight.requires_grad
    assert all(not parameter.requires_grad for name, parameter in algo.actor.named_parameters() if parameter is not weight)


def test_mask_clears_loaded_adam_momentum_and_preserves_frozen_columns() -> None:
    algo = _ppo("ref_q")
    algo._configure_actor_finetune()
    weight = algo._actor_first_layer_weight
    mask = algo._actor_first_layer_gradient_mask
    assert weight is not None
    assert mask is not None
    algo.actor_optimizer = torch.optim.AdamW([weight], lr=1e-2, weight_decay=0.0)
    weight.grad = torch.ones_like(weight)
    algo.actor_optimizer.step()
    before = weight.detach().clone()
    algo.actor_optimizer.zero_grad()
    weight.grad = torch.ones_like(weight)
    algo._mask_actor_optimizer_state()
    algo.actor_optimizer.step()
    delta = weight.detach() - before
    assert torch.all(delta[mask == 0] == 0)
    assert torch.any(delta[mask == 1] != 0)
