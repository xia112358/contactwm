from types import SimpleNamespace
from unittest.mock import patch

from holosoma.agents.ppo.ppo import PPO


def _ppo_with_simulator(name: str):
    ppo = PPO.__new__(PPO)
    ppo.env = SimpleNamespace(simulator=SimpleNamespace(simulator_config=SimpleNamespace(name=name)))
    ppo.device = "cuda:0"
    return ppo


def test_newton_rollout_boundary_synchronizes_cuda():
    ppo = _ppo_with_simulator("isaaclab3_newton")
    with (
        patch("holosoma.agents.ppo.ppo.torch.cuda.is_available", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.cuda.synchronize") as synchronize,
    ):
        ppo._synchronize_newton_rollout_boundary()
    synchronize.assert_called_once_with("cuda:0")


def test_non_newton_rollout_boundary_does_not_synchronize_cuda():
    ppo = _ppo_with_simulator("isaacsim")
    with (
        patch("holosoma.agents.ppo.ppo.torch.cuda.is_available", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.cuda.synchronize") as synchronize,
    ):
        ppo._synchronize_newton_rollout_boundary()
    synchronize.assert_not_called()
