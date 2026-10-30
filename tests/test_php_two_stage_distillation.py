"""Verify the stage boundary and actual gradients, including the critic."""
from dataclasses import asdict
from types import SimpleNamespace as State

import pytest
import torch

from holosoma.agents.ppo.distill_ppo import DistillPPO
from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.algo import DistillConfig
from holosoma.php_contact.distill import distillation_schedule


def config():
    return distillation_schedule(DistillConfig(enable_kl=True, distill_type='php_dagger_ppo',
        dagger_coef=10., lambda_dagger_init=1., lambda_dagger_final=.1,
        lambda_kl_anneal_iters=10000, lambda_kl_anneal_schedule='php_parkour'), 200, .1)


@pytest.mark.parametrize('iteration,dagger,ppo', [(0,1.,0.),(1,1.,0.),(199,1.,0.),
    (200,.01,.99),(201,.01,.99),(10000,.01,.99),(19999,.01,.99)])
def test_exact_stage_boundary_and_actual_mse_multiplier(iteration,dagger,ppo):
    algo=object.__new__(DistillPPO)
    algo.config=State(distill=config())
    algo.current_learning_iteration=iteration
    assert algo._current_lambda_d()==pytest.approx(dagger)
    assert algo._current_ppo_lambda()==pytest.approx(ppo)
    assert algo.config.distill.dagger_coef*algo._current_lambda_d()==pytest.approx(10. if iteration<200 else .1)


def test_schedule_roundtrips_in_normal_config_and_legacy_is_unchanged():
    native=DistillConfig()
    assert distillation_schedule(native,None,.1) is native
    restored=DistillConfig(**asdict(config()))
    assert restored==config() and restored.pure_distill_iters==200


@pytest.mark.parametrize('iters,coef',[(-1,.1),(True,.1),(200,0.),(200,float('nan')),(200,11.)])
def test_invalid_stage_settings_are_rejected(iters,coef):
    with pytest.raises(ValueError): distillation_schedule(config(),iters,coef)


@pytest.mark.parametrize('iteration',[199,200])
def test_actual_distill_loss_excludes_all_ppo_gradients_before_switch(monkeypatch,iteration):
    class Actor(torch.nn.Module):
        def __init__(self,value):
            super().__init__()
            self.mean=torch.nn.Parameter(torch.tensor(float(value)))
            self.log_std=torch.nn.Parameter(torch.tensor(0.))
        def act(self,obs):
            n=len(obs['actor_obs'])
            self.action_mean=self.mean.expand(n,1)
            self.action_std=self.log_std.exp().expand(n,1)
    algo=object.__new__(DistillPPO)
    algo.device='cpu'; algo.config=State(distill=config())
    algo.current_learning_iteration=iteration
    algo.actor=Actor(1.); algo.teacher_actor=Actor(0.)
    algo.teacher_actor.requires_grad_(False)
    algo._teacher_uses_student_actor_obs=True
    critic=torch.nn.Parameter(torch.tensor(1.))
    def native_ppo(self,batch):
        return dict(actor_loss=self.actor.mean*3+self.actor.log_std*2,
                    critic_loss=critic.square())
    monkeypatch.setattr(PPO,'_compute_ppo_loss',native_ppo)
    batch=dict(actor_obs=torch.ones(4,1),dagger_valid_mask=torch.ones(4))
    losses=algo._compute_ppo_loss(batch)
    total=losses['actor_loss']+losses['critic_loss']
    total.backward()
    if iteration<200:
        assert losses['actor_loss'].item()==pytest.approx(10.)
        assert algo.actor.mean.grad.item()==pytest.approx(20.)
        assert algo.actor.log_std.grad is None and critic.grad is None
        assert losses['php_ppo_actor_active'].item()==0
    else:
        assert losses['actor_loss'].item()==pytest.approx(.99*3+.1)
        assert algo.actor.mean.grad.item()==pytest.approx(.99*3+.2)
        assert algo.actor.log_std.grad.item()==pytest.approx(.99*2)
        assert critic.grad.item()==pytest.approx(2.)
        assert losses['php_ppo_actor_active'].item()==1
    assert all(p.grad is None for p in algo.teacher_actor.parameters())
