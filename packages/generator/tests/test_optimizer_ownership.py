"""Execution-only updates preserve planning parameters and AdamW state."""
import copy

import pytest
import torch

from generator.full1000_position_predictor import Full1000PositionPredictor
from generator.optimizer_ownership import prepare_optimizer_gradients
from generator.rollout_termination import demonstration_loss
from generator.training_state import capture_training_state, restore_training_state, validate_resume_config
from somaforge_core.heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS


def model_and_optimizer(*, shared=False):
    torch.set_num_threads(2)
    model = Full1000PositionPredictor(24, 1, 4, event_roles=True,
        execution_observation_gradients=shared).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    # Populate all moments so a zero-gradient-only fix fails this regression.
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def observation():
    q = torch.zeros(2, 36)
    q[:, 2] = .8
    q[:, 3] = 1
    return dict(current_q=q, current_contact=torch.zeros(2, 6, dtype=torch.bool),
        current_anchor=torch.zeros(2, 6, 3), heightmap=torch.zeros(2, HEIGHTMAP_ROWS, HEIGHTMAP_COLS))


def planning_loss(prediction, supervised):
    # Keep zero planning gradients in the graph, as the trainer's mask does.
    value = prediction.role_logits.square().mean((1, 2)) + prediction.location_logits.square().mean((1, 2))
    return demonstration_loss(value, supervised).mean()


def assert_state_equal(a, b):
    assert a.keys() == b.keys()
    for name in a:
        if torch.is_tensor(a[name]):
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        else:
            assert a[name] == b[name]


@pytest.mark.parametrize('shared', [False, True])
def test_autonomous_execution_preserves_planner_and_moments(shared):
    model, optimizer = model_and_optimizer(shared=shared)
    obs = observation()
    groups = model.training_parameter_groups()
    protected = groups['planner'] + groups.get('shared', [])
    before = [(p.detach().clone(), copy.deepcopy(optimizer.state[p])) for p in protected]
    executor_before = model.pose_head.weight.detach().clone()
    mask = torch.zeros(2, dtype=torch.bool)
    kwargs = dict(execution_observation_supervised=mask) if shared else {}
    with torch.no_grad():
        original = model(**obs)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        pred = model(**obs, **kwargs)
        (pred.qpos.square().mean() + planning_loss(pred, mask)).backward()
        assert any(p.grad is not None for p in protected)
        prepare_optimizer_gradients(model, mask, policy='planner_supervised_v1',
            isolate_unsupervised_execution_gradients=shared)
        assert all(p.grad is None for p in protected)
        model.clip_training_gradients()
        optimizer.step()
    for parameter, (weight, state) in zip(protected, before):
        torch.testing.assert_close(parameter, weight, rtol=0, atol=0)
        assert_state_equal(optimizer.state[parameter], state)
    assert not torch.equal(model.pose_head.weight, executor_before)
    with torch.no_grad():
        after = model(**obs)
    torch.testing.assert_close(after.role_logits, original.role_logits, rtol=0, atol=0)
    torch.testing.assert_close(after.location_logits, original.location_logits, rtol=0, atol=0)
    assert not torch.equal(after.qpos, original.qpos)


def test_mixed_batch_keeps_planning_learning_without_execution_contamination():
    model, optimizer = model_and_optimizer()
    mask = torch.tensor([True, False])
    pred = model(**observation())
    plan = planning_loss(pred, mask)
    parameters = model.training_parameter_groups()['planner']
    expected = torch.autograd.grad(plan, parameters, retain_graph=True, allow_unused=True)
    (plan + pred.qpos.square().mean()).backward()
    assert prepare_optimizer_gradients(model, mask, policy='planner_supervised_v1') == ()
    for parameter, grad in zip(parameters, expected):
        if grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
    before = model.role_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(model.role_head.weight, before)


def test_legacy_policy_keeps_historical_zero_gradient_updates():
    model, optimizer = model_and_optimizer()
    pred = model(**observation())
    mask = torch.zeros(2, dtype=torch.bool)
    planning_loss(pred, mask).backward()
    before = model.role_head.weight.detach().clone()
    assert prepare_optimizer_gradients(model, mask, policy='legacy') == ()
    optimizer.step()
    assert not torch.equal(model.role_head.weight, before)


def test_shared_to_isolated_resume_requires_explicit_migration_and_preserves_other_state():
    old, old_opt = model_and_optimizer(shared=True)
    old_sched = torch.optim.lr_scheduler.CosineAnnealingLR(old_opt, T_max=20)
    config = dict(architecture=old.schema, steps=20, execution_observation_gradients=True)
    state = copy.deepcopy(capture_training_state(step=7, model=old, optimizer=old_opt,
        scheduler=old_sched, pool=None, pending=None, config=config,
        robot_asset_json='canonical-test', training_predictions=14, history=[],
        best_score=1., best_step=7, best_state=None))
    desired = dict(config, execution_observation_gradients=False,
        optimizer_gradient_ownership='planner_supervised_v1')
    with pytest.raises(ValueError, match='configuration differs'):
        validate_resume_config(config, desired)
    desired['resume_gradient_isolation'] = True
    new, opt = model_and_optimizer(shared=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    restored = restore_training_state(state, model=new, optimizer=opt, scheduler=sched,
        pool=None, pending=None, config=desired, robot_asset_json='canonical-test')
    reset = restored['optimizer_migration']['reset_parameters']
    assert reset
    for (name, old_p), (new_name, new_p) in zip(old.named_parameters(), new.named_parameters()):
        assert name == new_name
        torch.testing.assert_close(old_p, new_p, rtol=0, atol=0)
        if name.startswith(old.EXECUTION_PREFIXES + old.PLAN_HEAD_PREFIXES):
            assert name not in reset
            assert_state_equal(old_opt.state[old_p], opt.state[new_p])
        else:
            assert name in reset and new_p not in opt.state
    assert sched.state_dict() == old_sched.state_dict()
    assert restored['step'] == 7 and restored['training_predictions'] == 14
    # Ordinary resume of an already migrated state must not reset again.
    migrated = copy.deepcopy(capture_training_state(step=7, model=new, optimizer=opt,
        scheduler=sched, pool=None, pending=None, config=desired,
        robot_asset_json='canonical-test', training_predictions=14, history=[],
        best_score=1., best_step=7, best_state=None))
    result = restore_training_state(migrated, model=new, optimizer=opt, scheduler=sched,
        pool=None, pending=None, config=dict(desired, resume_gradient_isolation=False),
        robot_asset_json='canonical-test')
    assert result['optimizer_migration'] is None
    for name, parameter in new.named_parameters():
        if name in reset:
            assert parameter not in opt.state
    with pytest.raises(ValueError, match='configuration differs'):
        validate_resume_config(config, dict(desired, learning_rate=.7))


def test_historical_resume_retains_legacy_optimizer_policy():
    old = dict(execution_observation_gradients=True, steps=20)
    assert validate_resume_config(old, dict(old, optimizer_gradient_ownership='legacy')) is False
    with pytest.raises(ValueError, match='configuration differs'):
        validate_resume_config(old, dict(old, optimizer_gradient_ownership='planner_supervised_v1'))
