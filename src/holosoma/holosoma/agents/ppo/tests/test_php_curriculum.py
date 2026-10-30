from types import SimpleNamespace

import pytest

from holosoma.agents.ppo.distill_ppo import DistillPPO


def _dummy(iteration: int):
    distill = SimpleNamespace(
        distill_type="php_dagger_ppo",
        lambda_dagger_init=1.0,
        lambda_dagger_final=0.1,
        lambda_kl_init=1.0,
        lambda_kl_final=0.1,
        lambda_kl_anneal_iters=10000,
        lambda_kl_anneal_schedule="php_parkour",
        lambda_ppo_init=None,
        lambda_ppo_final=None,
    )
    return SimpleNamespace(
        config=SimpleNamespace(distill=distill, num_learning_iterations=20000),
        current_learning_iteration=iteration,
        _php_curriculum_anneal_iters=lambda: 10000,
        _current_lambda_d=lambda: DistillPPO._current_lambda_d(
            SimpleNamespace(
                config=SimpleNamespace(distill=distill),
                current_learning_iteration=iteration,
                _php_curriculum_anneal_iters=lambda: 10000,
            )
        ),
    )


@pytest.mark.parametrize("iteration, expected", [(0, 1.0), (5000, 0.55), (10000, 0.1), (20000, 0.1)])
def test_php_dagger_weight_is_linear_over_first_half(iteration: int, expected: float) -> None:
    dummy = _dummy(iteration)
    assert DistillPPO._current_lambda_d(dummy) == pytest.approx(expected)
    assert DistillPPO._current_ppo_lambda(dummy) == pytest.approx(1.0 - expected)
