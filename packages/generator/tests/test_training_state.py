import io
import random

import numpy as np
import pytest
import torch

from generator.parallel_rollout_pool import ParallelRolloutPool
from generator.training_state import capture_training_state, restore_training_state, validate_resume_config


def build():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    ref = dict(current_q=torch.arange(30.).reshape(10, 3))
    pool = ParallelRolloutPool(ref, torch.arange(10), torch.arange(10)-1, 4)
    return model, optimizer, scheduler, pool


def update(model, optimizer, scheduler, pool):
    slots, indices, obs = pool.batch(3)
    x = obs['current_q'] + torch.rand(3, 3)
    loss = model(x).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
    pool.observation['current_q'][slots] += torch.rand(3, 3)
    pool.age[slots] += 1
    return indices.clone(), model.weight.detach().clone(), torch.rand(3), np.random.rand(), random.random()


def test_serialized_resume_reproduces_next_updates_and_all_rng():
    torch.manual_seed(23); np.random.seed(23); random.seed(23)
    model, opt, sched, pool = build()
    update(model, opt, sched, pool)
    config = dict(output='a', steps=20, isolate_unsupervised_execution_gradients=False)
    state = capture_training_state(step=1, model=model, optimizer=opt, scheduler=sched, pool=pool,
        pending=None, config=config, robot_asset_json='canonical-test', training_predictions=3,
        history=[], best_score=1., best_step=1, best_state=None)
    stream = io.BytesIO(); torch.save(state, stream); stream.seek(0)
    state = torch.load(stream, weights_only=False)
    expected = [update(model, opt, sched, pool) for _ in range(2)]
    other, other_opt, other_sched, other_pool = build()
    restored = restore_training_state(state, model=other, optimizer=other_opt, scheduler=other_sched,
        pool=other_pool, pending=None, config={**config, 'output':'b'}, robot_asset_json='canonical-test')
    assert restored['step'] == 1
    for reference in expected:
        actual = update(other, other_opt, other_sched, other_pool)
        for a, b in zip(actual, reference):
            if isinstance(a, torch.Tensor):torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:assert a == b
    for key, value in opt.state_dict()['state'].items():
        for name, tensor in value.items():torch.testing.assert_close(tensor, other_opt.state_dict()['state'][key][name])
    torch.testing.assert_close(pool.observation['current_q'], other_pool.observation['current_q'])


def test_resume_only_allows_documented_experiment_changes():
    saved = dict(steps=1100, learning_rate=.001, isolate_unsupervised_execution_gradients=False)
    validate_resume_config(saved, {**saved, 'isolate_unsupervised_execution_gradients':True})
    with pytest.raises(ValueError, match='configuration differs'):
        validate_resume_config(saved, {**saved, 'learning_rate':.002})
