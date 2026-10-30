import math

import torch
from holosoma.agents.modules.ppo_modules import CensoredNormal, PPOActor
from torch.distributions import Normal


def test_censored_normal_samples_stay_within_action_bounds() -> None:
    distribution = CensoredNormal(
        loc=torch.tensor([[20.0, -20.0, 0.0]]),
        scale=torch.ones(1, 3),
        action_clip=10.0,
    )

    samples = torch.stack([distribution.sample() for _ in range(32)])

    assert torch.all(samples <= 10.0)
    assert torch.all(samples >= -10.0)
    assert torch.all(samples[:, 0, 0] == 10.0)
    assert torch.all(samples[:, 0, 1] == -10.0)


def test_censored_normal_boundary_log_prob_uses_gaussian_tail_mass() -> None:
    loc = torch.tensor([9.5, -9.5, 0.0])
    scale = torch.tensor([1.0, 1.0, 2.0])
    distribution = CensoredNormal(loc=loc, scale=scale, action_clip=10.0)
    actions = torch.tensor([10.0, -10.0, 1.0])

    actual = distribution.log_prob(actions)
    expected = torch.stack(
        [
            torch.special.log_ndtr(torch.tensor(-0.5)),
            torch.special.log_ndtr(torch.tensor(-0.5)),
            Normal(loc[2], scale[2]).log_prob(actions[2]),
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_censored_normal_entropy_matches_uncensored_normal_for_distant_bounds() -> None:
    loc = torch.tensor([0.0])
    scale = torch.tensor([1.0])
    distribution = CensoredNormal(loc=loc, scale=scale, action_clip=10.0)

    expected = torch.tensor([0.5 * math.log(2.0 * math.pi * math.e)])

    torch.testing.assert_close(distribution.entropy(), expected, atol=1.0e-6, rtol=1.0e-6)


def test_actor_inference_exposes_raw_mean_but_executes_bounded_mean() -> None:
    actor = object.__new__(PPOActor)
    torch.nn.Module.__init__(actor)
    actor.actor_module = torch.nn.Linear(1, 1)
    actor.action_clip = 10.0
    with torch.no_grad():
        actor.actor_module.weight.zero_()
        actor.actor_module.bias.fill_(25.0)
    policy_state = {"actor_obs": torch.zeros(2, 1)}

    raw_mean = actor.act_raw_inference(policy_state)
    executed_mean = actor.act_inference(policy_state)

    torch.testing.assert_close(raw_mean, torch.full((2, 1), 25.0))
    torch.testing.assert_close(executed_mean, torch.full((2, 1), 10.0))


def test_censored_normal_has_finite_gradients_for_saturated_old_policy() -> None:
    loc = torch.tensor([25.0, -25.0, 10.0, -10.0, 0.0], requires_grad=True)
    scale = torch.tensor([0.1, 0.1, 0.5, 0.5, 1.0], requires_grad=True)
    actions = torch.tensor([10.0, -10.0, 10.0, -10.0, 0.0])
    distribution = CensoredNormal(loc=loc, scale=scale, action_clip=10.0)

    loss = distribution.log_prob(actions).sum() + distribution.entropy().sum()
    loss.backward()

    assert torch.isfinite(distribution.log_prob(actions)).all()
    assert torch.isfinite(distribution.entropy()).all()
    assert torch.isfinite(loc.grad).all()
    assert torch.isfinite(scale.grad).all()


def test_upper_boundary_negative_advantage_pushes_raw_mean_back_toward_bound() -> None:
    loc = torch.tensor([12.0], requires_grad=True)
    distribution = CensoredNormal(loc=loc, scale=torch.tensor([1.0]), action_clip=10.0)
    old_log_prob = distribution.log_prob(torch.tensor([10.0])).detach()

    negative_advantage_surrogate = torch.exp(distribution.log_prob(torch.tensor([10.0])) - old_log_prob)
    negative_advantage_surrogate.backward()

    assert loc.grad.item() > 0.0
