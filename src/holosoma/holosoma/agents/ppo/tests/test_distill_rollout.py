"""Teacher observations must not bypass PPO's on-policy validity contract."""
from types import SimpleNamespace

import torch
from holosoma.agents.ppo.distill_ppo import DistillPPO


def test_distill_rollout_keeps_teacher_clock_and_excludes_deterministic_probes():
    class Actor:
        def act(self, observations):
            self.action_mean = observations['actor_obs'] * .1
            self.action_std = torch.ones_like(self.action_mean)
            return self.action_mean + .25

        def get_actions_log_prob(self, actions):
            return -((actions-self.action_mean)**2).sum(-1)

        def reset(self, dones):
            pass

    command = SimpleNamespace(_probe_env_mask=torch.tensor([False, True]))
    executed = []

    def observations(step):
        return {'student': torch.full((2, 1), float(step)),
                'teacher': torch.full((2, 2), float(step+10))}

    def step(actions):
        executed.append(actions['actions'].clone())
        # The mask may change during reset; validity belongs to the action
        # that was just chosen, not the next transition.
        command._probe_env_mask.logical_not_()
        return observations(len(executed)), torch.tensor([1., 2.]), torch.zeros(2, dtype=torch.bool), {
            'time_outs': torch.zeros(2, dtype=torch.bool)}

    algo = object.__new__(DistillPPO)
    algo.env = SimpleNamespace(num_envs=2, step=step,
        command_manager=SimpleNamespace(get_state=lambda _: command))
    algo.config = SimpleNamespace(num_steps_per_env=2, action_clip=1., gamma=.99, lam=.95,
        exec_consistency_coef=0., distill=SimpleNamespace(enable_kl=True, distill_type='kl'))
    algo.actor_obs_keys = algo.critic_obs_keys = ['student']
    algo.teacher_obs_keys = ['teacher']
    algo._teacher_uses_student_actor_obs = False
    algo._get_obs_dim = lambda keys: 2 if keys == ['teacher'] else 1
    algo._compute_dagger_valid_mask = lambda: torch.ones(2)
    algo.device = 'cpu'
    algo.num_act = 1
    algo.empirical_normalization = False
    algo.is_multi_gpu = False
    algo.log_dir = None
    algo.actor = Actor()
    algo.critic = SimpleNamespace(evaluate=lambda obs: obs['critic_obs']*.2, reset=lambda _: None)
    algo._setup_storage()
    algo._rollout_step(observations(0))
    assert algo.storage['ppo_valid'][..., 0].tolist() == [[True, False], [False, True]]
    torch.testing.assert_close(torch.stack(executed)[..., 0], torch.tensor([[.25, 0.], [.1, .35]]))
    torch.testing.assert_close(algo.storage['teacher_actor_obs'][:, 0, 0], torch.tensor([10., 11.]))
    batch = {key: value.flatten(0, 1) for key, value in algo.storage._buffers.items()}
    filtered, excluded = algo._filter_ppo_minibatch(batch)
    assert excluded == .5 and len(filtered['actions']) == 2
    torch.testing.assert_close(filtered['advantages'].mean(), torch.tensor(0.), atol=3.e-6, rtol=0)
    # This is the actual optimizer step's input boundary, including teacher
    # observations. A zero-filled validity buffer used to leave it empty.
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    before = model.weight.detach().clone()
    (model(filtered['actor_obs'])-filtered['teacher_actor_obs'][:, :1]).square().mean().backward()
    optimizer.step()
    assert optimizer.state and not torch.equal(model.weight, before)
