import inspect

import torch

from generator.full1000_position_predictor import Full1000PositionPredictor
from generator.next_interaction_heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS


def setup():
    torch.manual_seed(31)
    model = Full1000PositionPredictor(width=24, layers=1, location_width=8).eval()
    q = torch.zeros(1, 36); q[:, 2] = .8; q[:, 3] = 1
    return model, dict(current_q=q, current_contact=torch.zeros(1, 6, dtype=torch.bool),
                      current_anchor=torch.zeros(1, 6, 3),
                      heightmap=torch.full((1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS), -.8))


def test_event_roles_distinguish_retouch_and_hold_without_planner_gradients():
    _, obs = setup()
    model = Full1000PositionPredictor(24, 1, 8, event_roles=True).eval()
    active = torch.ones(1, 6, dtype=torch.bool)
    kwargs = dict(teacher_contact=active, teacher_cell=torch.zeros(1, 6, dtype=torch.long),
                  teacher_mask=torch.ones(1, dtype=torch.bool))
    # Learned role vectors must affect execution independently of contact bits.
    with torch.no_grad():
        model.execution_role_encoder.weight.normal_()
    touchdown = model(**obs, **kwargs, teacher_role=torch.ones(1, 6, dtype=torch.long))
    hold = model(**obs, **kwargs, teacher_role=torch.full((1, 6), 2, dtype=torch.long))
    assert not torch.equal(touchdown.qpos, hold.qpos)
    touchdown.qpos.square().sum().backward()
    assert model.execution_role_encoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.training_parameter_groups()['planner'])


def test_event_roles_require_teacher_roles_and_reload_exactly():
    import pytest
    _, obs = setup()
    model = Full1000PositionPredictor(24, 1, 8, event_roles=True).eval()
    with pytest.raises(ValueError, match='explicit teacher roles'):
        model(**obs, teacher_contact=torch.ones(1, 6, dtype=torch.bool),
              teacher_cell=torch.zeros(1, 6, dtype=torch.long), teacher_mask=torch.ones(1, dtype=torch.bool))
    other = Full1000PositionPredictor(24, 1, 8, event_roles=True).eval()
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(**obs).qpos, other(**obs).qpos, rtol=0, atol=0)


def test_scratch_predictor_has_no_partial_weight_loading_or_face_input():
    model, _ = setup()
    assert not hasattr(model, 'load_stage_a')
    assert not any('surface' in key for key in model.state_dict())
    # Surface identities are deterministic validity metadata, never encoder inputs.
    assert not any('surface' in key for key in inspect.signature(model._encode_points).parameters)


def test_network_executes_explicit_position_plan_and_ignores_inactive_location():
    model, observation = setup()
    active = torch.zeros(1, 6, dtype=torch.bool)
    cell = torch.zeros(1, 6, dtype=torch.long)
    mask = torch.ones(1, dtype=torch.bool)
    with torch.no_grad():
        off = model(**observation, teacher_contact=active, teacher_cell=cell, teacher_mask=mask)
        ignored = model(**observation, teacher_contact=active, teacher_cell=cell+1000, teacher_mask=mask)
        torch.testing.assert_close(off.qpos, ignored.qpos, rtol=0, atol=0)
        active[:, :2] = True
        first = model(**observation, teacher_contact=active, teacher_cell=cell, teacher_mask=mask)
        second = model(**observation, teacher_contact=active, teacher_cell=cell+1000, teacher_mask=mask)
    assert not torch.equal(first.qpos, second.qpos)
    assert torch.equal(first.conditioned_points_local[:, :, 2], torch.full((1, 6), -.8))


def test_execution_gradient_isolated_from_entire_planner():
    model, observation = setup()
    with torch.no_grad():
        model.role_head.bias[1] += 10
    prediction = model(**observation)
    prediction.qpos.square().sum().backward()
    assert all(p.grad is None for p in model.training_parameter_groups()['planner'])
    assert model.plan_position_encoder.weight.grad is not None
    assert model.pose_head.weight.grad is not None
    assert model.execution_adapter.weight.grad.abs().sum() > 0
    assert model.execution_norm.weight.grad.abs().sum() > 0


def test_teacher_rows_do_not_send_execution_gradient_to_autonomous_plan_heads():
    model, observation = setup()
    observation = {k: v.expand(2, *v.shape[1:]).clone() for k, v in observation.items()}
    out = model(**observation, teacher_mask=torch.tensor([True, False]),
        teacher_contact=torch.ones(2, 6, dtype=torch.bool), teacher_cell=torch.zeros(2, 6, dtype=torch.long))
    out.role_logits.retain_grad(); out.location_logits.retain_grad()
    out.qpos.square().sum().backward()
    assert out.role_logits.grad is None
    assert out.location_logits.grad is None


def test_shared_horizontal_shift_is_equivariant():
    model, observation = setup()
    shifted = {key: value.clone() for key, value in observation.items()}
    delta = torch.tensor([.31, -.17, 0.])
    shifted['current_q'][:, :3] += delta
    shifted['current_anchor'] += delta
    with torch.no_grad():
        a, b = model(**observation), model(**shifted)
    torch.testing.assert_close(a.role_logits, b.role_logits, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a.qpos[:, :3]+delta, b.qpos[:, :3], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a.qpos[:, 3:], b.qpos[:, 3:], rtol=1e-4, atol=1e-5)


def test_geometry_warm_start_preserves_predictions_and_adds_trainable_path():
    base, obs = setup()
    model = Full1000PositionPredictor(24, 1, 8, body_geometry=True).eval()
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith(('body_geometry_', 'execution_geometry_encoder.')) for k in result.missing_keys)
    a, b = base(**obs), model(**obs)
    torch.testing.assert_close(a.qpos, b.qpos, rtol=0, atol=0)
    torch.testing.assert_close(a.location_logits, b.location_logits, rtol=0, atol=0)
    b.qpos.square().sum().backward()
    assert model.body_geometry_output.weight.grad is None
    assert model.execution_geometry_encoder.weight.grad.abs().sum() > 0
    assert not any('surface' in k for k in model.state_dict())


def test_geometry_features_are_finite_and_translation_equivariant():
    _, obs = setup()
    model = Full1000PositionPredictor(24, 1, 8, body_geometry=True).eval()
    from generator.next_interaction_heightmap_v2 import _root_yaw_basis
    basis, _ = _root_yaw_basis(obs['current_q'])
    a = model.geometry_features(obs['current_q'], obs['heightmap'], basis)
    q = obs['current_q'].clone(); q[:, :2] += torch.tensor([[3.2, -1.7]])
    b = model.geometry_features(q, obs['heightmap'], basis)
    assert a.shape[-1] == 13 and torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)
    assert ((a[..., -1] >= 0) & (a[..., -1] <= 1)).all()


def test_region_planning_and_execution_have_disjoint_gradients():
    _, obs = setup()
    model = Full1000PositionPredictor(24, 1, 8, region_plan=True, unified_contact=True).eval()
    out = model(**obs)
    out.qpos.square().sum().backward()
    assert all(p.grad is None for p in model.training_parameter_groups()['planner'])
    assert model.execution_geometry_encoder.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    out = model(**obs)
    loss = out.role_logits.square().mean()+out.location_logits.square().mean()+out.region_logits.square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.training_parameter_groups()['executor'])
    assert model.shared.layers[0].linear1.weight.grad.abs().sum() > 0
    assert model.norm.weight.grad.abs().sum() > 0
    assert model.region_geometry_encoder[-1].weight.grad.abs().sum() > 0


def test_complete_teacher_regions_replace_only_teacher_rows_and_keep_plan_gradients_isolated():
    _, obs = setup()
    obs = {key: value.expand(2, *value.shape[1:]).clone() for key, value in obs.items()}
    model = Full1000PositionPredictor(24, 1, 8, region_plan=True, event_roles=True).eval()
    with torch.no_grad():
        model.role_head.bias[1] = 10
        model.region_plan_encoder.weight.normal_()
    autonomous = model(**obs)
    regions = torch.zeros(2, 6, 4, dtype=torch.bool)
    regions[:, :, 0] = True
    regions[:, :2, 1] = True
    mask = torch.tensor([True, False])
    out = model(**obs, teacher_mask=mask, teacher_contact=torch.ones(2, 6, dtype=torch.bool),
        teacher_cell=torch.zeros(2, 6, dtype=torch.long), teacher_role=torch.ones(2, 6, dtype=torch.long),
        teacher_regions=regions)
    assert torch.equal(out.planned_regions[0], regions[0])
    assert torch.equal(out.planned_regions[1], autonomous.planned_regions[1])
    torch.testing.assert_close(out.region_logits, autonomous.region_logits, rtol=0, atol=0)
    altered = regions.clone(); altered[0, 0, 0] = False
    changed = model(**obs, teacher_mask=mask, teacher_contact=torch.ones(2, 6, dtype=torch.bool),
        teacher_cell=torch.zeros(2, 6, dtype=torch.long), teacher_role=torch.ones(2, 6, dtype=torch.long),
        teacher_regions=altered)
    assert not torch.equal(changed.qpos[0], out.qpos[0])
    torch.testing.assert_close(changed.qpos[1], out.qpos[1], rtol=0, atol=0)
    out.qpos.square().sum().backward()
    assert all(p.grad is None for p in model.training_parameter_groups()['planner'])
    assert model.region_plan_encoder.weight.grad.abs().sum() > 0


def test_regional_teacher_requires_complete_native_evidence_without_silent_fallback():
    import pytest
    _, obs = setup()
    model = Full1000PositionPredictor(24, 1, 8, region_plan=True).eval()
    kwargs = dict(teacher_mask=torch.ones(1, dtype=torch.bool),
        teacher_contact=torch.ones(1, 6, dtype=torch.bool), teacher_cell=torch.zeros(1, 6, dtype=torch.long))
    with pytest.raises(ValueError, match='requires actual demonstrated regions'):
        model(**obs, **kwargs)
    regions = torch.zeros(1, 6, 4, dtype=torch.bool); regions[:, :, 0] = True
    regions[:, 0] = False
    with pytest.raises(ValueError, match='missing native regional evidence'):
        model(**obs, **kwargs, teacher_regions=regions)
    regions[:, :, 0] = True; regions[:, 2, 3] = True
    with pytest.raises(ValueError, match='invalid anatomical region'):
        model(**obs, **kwargs, teacher_regions=regions)


def test_legacy_migration_copies_norm_and_resets_only_new_residuals():
    model, obs = setup()
    with torch.no_grad():
        model.norm.weight.fill_(1.7); model.norm.bias.fill_(.2)
        model.execution_norm.load_state_dict(model.norm.state_dict())
    state = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('execution_')}
    reference = model(**obs).qpos.detach()
    with torch.no_grad(): model.execution_adapter.weight.fill_(1.)
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(model(**obs).qpos, reference, rtol=0, atol=0)
    assert model.execution_adapter.weight.count_nonzero() == 0
    assert model.execution_norm.weight.data_ptr() != model.norm.weight.data_ptr()
    # New checkpoints preserve trained execution branches instead of resetting.
    with torch.no_grad(): model.execution_adapter.bias.fill_(.1)
    new_state = {k: v.clone() for k, v in model.state_dict().items()}
    other, _ = setup(); other.load_state_dict(new_state)
    torch.testing.assert_close(other(**obs).qpos, model(**obs).qpos, rtol=0, atol=0)


def test_clipping_large_execution_gradients_cannot_shrink_planning_gradients():
    model, _ = setup()
    groups = model.training_parameter_groups()
    assert {id(p) for p in groups['planner']}.isdisjoint(id(p) for p in groups['executor'])
    planner, executor = groups['planner'][0], groups['executor'][0]
    outputs = []
    for magnitude in (1., 100000.):
        model.zero_grad(set_to_none=True)
        planner.grad = torch.ones_like(planner)
        executor.grad = torch.full_like(executor, magnitude)
        model.clip_training_gradients()
        outputs.append(planner.grad.clone())
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


def test_clipping_in_execution_only_mode_supports_empty_planner_gradients():
    model, obs = setup()
    model(**obs).qpos.square().sum().backward()
    norms = model.clip_training_gradients()
    assert norms['planner'] == 0 and norms['executor'] > 0
    assert norms['planner'].device == norms['executor'].device == model.pose_head.weight.device
    assert torch.isfinite(torch.stack(list(norms.values())).norm())


def test_execution_observation_gradient_switch_preserves_forward_and_plan_isolation():
    _, obs = setup()
    isolated = Full1000PositionPredictor(24, 1, 8, region_plan=True, event_roles=True).eval()
    connected = Full1000PositionPredictor(24, 1, 8, region_plan=True, event_roles=True,
                                         execution_observation_gradients=True).eval()
    connected.load_state_dict(isolated.state_dict())
    a, b = isolated(**obs), connected(**obs)
    for name in ('qpos', 'role_logits', 'location_logits', 'region_logits'):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
    b.role_logits.retain_grad(); b.location_logits.retain_grad(); b.region_logits.retain_grad()
    b.qpos.square().sum().backward()
    groups = connected.training_parameter_groups()
    assert all(p.grad is None for p in groups['planner'])
    assert b.role_logits.grad is None and b.location_logits.grad is None and b.region_logits.grad is None
    for module in (connected.shared.layers[0].linear1, connected.global_encoder,
                   connected.part_encoder, connected.norm, connected.region_geometry_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    # Explicitly audit all three observation interfaces, including terrain memory.
    connected.zero_grad(set_to_none=True)
    encode = connected._encode_points
    captured = []
    def capture(*args, **kwargs):
        result = encode(*args, **kwargs)
        captured.extend(result[i] for i in (0, 3, 4))
        for value in captured: value.retain_grad()
        return result
    connected._encode_points = capture
    connected(**obs).qpos.square().sum().backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in captured)
    norms = connected.clip_training_gradients()
    assert norms['planner'] == 0 and norms['shared'] > 0 and norms['executor'] > 0
    ids = [id(p) for parameters in groups.values() for p in parameters]
    assert len(ids) == len(set(ids)) == len(list(connected.parameters()))


def test_connected_shared_clipping_does_not_rescale_plan_heads():
    model = Full1000PositionPredictor(24, 1, 8, execution_observation_gradients=True)
    groups = model.training_parameter_groups()
    planner, shared = groups['planner'][0], groups['shared'][0]
    results = []
    for magnitude in (1., 100000.):
        model.zero_grad(set_to_none=True)
        planner.grad = torch.ones_like(planner)
        shared.grad = torch.full_like(shared, magnitude)
        model.clip_training_gradients()
        results.append(planner.grad.clone())
    torch.testing.assert_close(*results, rtol=0, atol=0)
