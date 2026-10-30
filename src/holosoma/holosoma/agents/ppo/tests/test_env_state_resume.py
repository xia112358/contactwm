from types import SimpleNamespace
from unittest import mock

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.ppo.ppo import PPO


def test_pending_env_state_is_restored_once_after_reset() -> None:
    ppo = PPO.__new__(PPO)
    state = {"motion_sampler": {"schema": "test"}}
    restored = []
    ppo._pending_env_state = state
    ppo._pending_env_state_restore = True
    ppo._restore_env_state = restored.append

    ppo._restore_pending_env_state_after_reset()
    ppo._restore_pending_env_state_after_reset()

    assert restored == [state]
    assert ppo._pending_env_state is None
    assert ppo._pending_env_state_restore is False


def test_missing_env_state_reaches_environment_restore_hook() -> None:
    restored = []
    algo = BaseAlgo.__new__(BaseAlgo)
    algo.env = SimpleNamespace(load_checkpoint_state=restored.append)

    algo._restore_env_state(None)

    assert restored == [None]


def test_inference_load_does_not_restore_training_state() -> None:
    ppo = PPO.__new__(PPO)
    ppo.actor = mock.Mock()
    ppo.critic = mock.Mock()
    ppo.empirical_normalization = False
    ppo._pending_env_state = {"stale": True}
    ppo._pending_env_state_restore = True
    ppo.current_learning_iteration = 17
    ppo.actor_optimizer = mock.Mock()
    ppo.critic_optimizer = mock.Mock()
    ppo._load_checked_checkpoint = mock.Mock(
        return_value={
            "actor_model_state_dict": {"actor": "weights"},
            "critic_model_state_dict": {"critic": "weights"},
            "actor_optimizer_state_dict": {"must": "not load"},
            "critic_optimizer_state_dict": {"must": "not load"},
            "iter": 999,
            "env_state": {"must": "not restore"},
            "infos": {"ok": True},
        }
    )

    infos = ppo.load_for_inference("policy.pt")

    assert infos == {"ok": True}
    ppo.actor.load_state_dict.assert_called_once_with({"actor": "weights"})
    ppo.critic.load_state_dict.assert_called_once_with({"critic": "weights"})
    ppo.actor_optimizer.load_state_dict.assert_not_called()
    ppo.critic_optimizer.load_state_dict.assert_not_called()
    assert ppo.current_learning_iteration == 17
    assert ppo._pending_env_state is None
    assert ppo._pending_env_state_restore is False
