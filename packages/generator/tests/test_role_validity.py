from types import SimpleNamespace

import pytest
import torch
from torch import nn

from generator.position_plan import select_position_plan
from generator.role_validity import (mask_role_logits, contact_validity_observation,
    subset_observed, ROLE_VALIDITY_SCHEMA)
from generator.planned_contact_predictor import contact_plan_objective
from somaforge_core.heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS


def fixture():
    model = SimpleNamespace(role_validity=True, event_roles=True, region_plan=False,
        role_head=nn.Linear(4, 4), location_query=nn.Linear(4, 1), location_key=nn.Linear(3, 1))
    with torch.no_grad():
        model.role_head.weight.zero_(); model.role_head.bias.copy_(torch.tensor([0., 2., 20., -1.]))
        model.location_query.weight.zero_(); model.location_query.bias.fill_(1.)
        model.location_key.weight.copy_(torch.tensor([[0., 0., 1.]])); model.location_key.bias.zero_()
    q = torch.zeros(1, 36); q[:, 3] = 1.
    height = torch.zeros(1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS); height[:, HEIGHTMAP_ROWS // 2:] = 1.
    active = torch.zeros(1, 6, dtype=torch.bool); active[:, 0] = True
    relation = torch.zeros(1, 6, 2, dtype=torch.bool); relation[:, 0, 0] = True
    kwargs = dict(current_contact=active, current_contact_relations=relation,
        contact_surface_ids=torch.tensor([[0, 1]]), heightmap_surface=height.long())
    return model, q, height, torch.eye(3)[None], torch.zeros(1, 6, 4), kwargs


def test_keep_probability_zero_and_legal_roles_remain_trainable():
    raw = torch.tensor([[[1., 2., 100000., 3.], [1., 2., 4., 3.]]], requires_grad=True)
    masked = mask_role_logits(raw, torch.tensor([[False, True]]))
    assert masked.softmax(-1)[0, 0, 2] == 0
    torch.testing.assert_close(masked[0, 1], raw[0, 1])
    loss = torch.nn.functional.cross_entropy(masked.transpose(1, 2), torch.tensor([[1, 2]]))
    loss.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[0, 0, 2] == 0


def test_predicted_keep_can_only_choose_an_observed_surface():
    model, q, height, basis, part, kwargs = fixture()
    plan = select_position_plan(model, q, height, basis, part, **kwargs)
    assert plan.conditioned_role[0, 0] == 2
    assert torch.all(plan.conditioned_role[0, 1:] == 1)
    assert plan.conditioned_points_local[0, 0, 2] == 0
    assert torch.all(plan.conditioned_points_local[0, 1:, 2] == 1)
    assert plan.role_logits.softmax(-1)[0, 1:, 2].count_nonzero() == 0
    assert plan.raw_role_logits[0, 1:, 2].min() == 20


def test_establish_remains_legal_when_contact_exists_and_loss_uses_target_role_support():
    model, q, height, basis, part, kwargs = fixture()
    target_role = torch.ones(1, 6, dtype=torch.long)
    top_cell = torch.full((1, 6), (HEIGHTMAP_ROWS // 2) * HEIGHTMAP_COLS, dtype=torch.long)
    plan = select_position_plan(model, q, height, basis, part, **kwargs,
        teacher_mask=torch.ones(1, dtype=torch.bool), teacher_role=target_role,
        teacher_contact=target_role != 0, teacher_cell=top_cell)
    assert torch.equal(plan.conditioned_role, target_role)
    assert plan.conditioned_points_local[..., 2].min() == 1
    target = dict(role=target_role, contact_cell=top_cell, contact_cell_valid=target_role.bool())
    loss, _ = contact_plan_objective(plan.prediction(q), target)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(model.role_head.bias.grad).all()
    assert torch.isfinite(model.location_key.weight.grad).all()


@pytest.mark.parametrize('bad', ['absent', 'other_surface'])
def test_teacher_cannot_bypass_keep_validity(bad):
    model, q, height, basis, part, kwargs = fixture()
    roles = torch.ones(1, 6, dtype=torch.long)
    roles[0, 1 if bad == 'absent' else 0] = 2
    cells = torch.zeros(1, 6, dtype=torch.long)
    if bad == 'other_surface': cells[0, 0] = (HEIGHTMAP_ROWS // 2) * HEIGHTMAP_COLS
    with pytest.raises(ValueError, match='Teacher keep'):
        select_position_plan(model, q, height, basis, part, **kwargs,
            teacher_mask=torch.ones(1, dtype=torch.bool), teacher_role=roles,
            teacher_contact=roles != 0, teacher_cell=cells)


def test_missing_surface_metadata_fails_explicitly():
    model, q, height, basis, part, kwargs = fixture()
    with pytest.raises(ValueError, match='requires actual contact relations'):
        select_position_plan(model, q, height, basis, part, current_contact=kwargs['current_contact'])


def test_repair_cannot_keep_a_different_surface():
    model, q, height, basis, part, kwargs = fixture()
    roles = torch.zeros(1, 6, dtype=torch.long); roles[0, 0] = 2
    points = torch.zeros(1, 6, 3); points[0, 0] = torch.tensor([.5, 0., 1.])
    with pytest.raises(ValueError, match='Repair keep requires the same observed contact surface'):
        select_position_plan(model, q, height, basis, part, **kwargs,
            repair_mask=torch.ones(1, dtype=torch.bool), repair_role=roles, repair_points_world=points)


def test_role_refinement_uses_legal_distribution_and_ignores_generated_teacher_roles():
    from generator.refine_predictor_roles import role_objective
    model = SimpleNamespace(role_head=nn.Linear(4, 4), role_validity=True)
    data = dict(part=torch.randn(2, 6, 4), contact=torch.zeros(2, 6, dtype=torch.bool),
        demo=torch.tensor([True, False]), role=torch.zeros(2, 6, dtype=torch.long),
        old_logits=torch.randn(2, 6, 4))
    first = role_objective(model, data, torch.arange(2))
    data['role'][1].fill_(100)
    second = role_objective(model, data, torch.arange(2))
    torch.testing.assert_close(first, second)
    second.backward()
    assert torch.isfinite(model.role_head.weight.grad).all()


def test_unrepresentable_keep_fails_explicitly():
    model, q, height, basis, part, kwargs = fixture()
    kwargs['heightmap_surface'].fill_(1)
    with pytest.raises(ValueError, match='no observed same-surface location'):
        select_position_plan(model, q, height, basis, part, **kwargs)


def actual_observation():
    pairs = dict(sample=torch.tensor([0, 0, 1, 1]), part=torch.tensor([0, 0, 1, 1]),
        primary_surface=torch.tensor([0, 1, 0, 1]), eligible=torch.tensor([True, True, True, False]),
        active=torch.tensor([True, True, True, False]), constraint_allocated=torch.tensor([True, True, True, False]),
        task_pair=torch.ones(4, dtype=torch.bool), type=torch.ones(4, dtype=torch.long),
        dist=torch.tensor([0., 0., 0., .02]), includemargin=torch.full((4,), .01),
        efc_address=torch.arange(4)[:, None], constraint_rows=torch.ones(4, dtype=torch.long))
    return dict(schema='newton_device_witness_batch_v1', pairs=pairs,
        face_surface=torch.tensor([[0, 1], [0, 1]]),
        face_normal=torch.tensor([[[0., 0., 1.], [0., 0., 1.]]] * 2),
        face_offset=torch.tensor([[0., 1.], [0., 1.]]),
        contact_part_mask=torch.tensor([[True, False, False, False, False, False],
                                       [False, True, False, False, False, False]]))


def test_complete_multi_surface_relations_and_subset_sample_mapping():
    q = torch.zeros(2, 36); q[:, 3] = 1.
    height = torch.zeros(2, HEIGHTMAP_ROWS, HEIGHTMAP_COLS)
    observed = actual_observation()
    result = contact_validity_observation(q, height, observed)
    assert result['current_contact_relations'][0, 0].all()
    assert result['current_contact_relations'][1, 1].tolist() == [True, False]
    selected = subset_observed(observed, torch.tensor([1, 0]))
    reordered = contact_validity_observation(q.flip(0), height.flip(0), selected)
    torch.testing.assert_close(reordered['current_contact_relations'], result['current_contact_relations'].flip(0))
    observed['pairs']['constraint_allocated'][0] = False
    observed['pairs']['efc_address'][0] = -1
    with pytest.raises(ValueError, match='Unallocated active'):
        contact_validity_observation(q, height, observed)


@pytest.mark.parametrize('schema', ['full1000_position_predictor_v1', 'full1000_position_predictor_v2'])
def test_both_architectures_record_and_enforce_validity(schema):
    from generator.predictor_architecture import build_position_predictor, load_position_predictor
    from somaforge_core.robot_assets import encode_robot_asset_json
    config = dict(architecture=schema, width=24, layers=1, location_width=4,
        body_geometry=False, part_geometry=False, region_plan=False, unified_contact=False,
        event_roles=True, role_validity=True)
    model = build_position_predictor(config, schema=schema)
    assert model.architecture_contract()['role_validity'] == ROLE_VALIDITY_SCHEMA
    _, q, height, _, _, kwargs = fixture()
    with torch.no_grad():
        model.role_head.weight.zero_(); model.role_head.bias.copy_(torch.tensor([0., 2., 20., -1.]))
        prediction = model(current_q=q, heightmap=height, current_anchor=torch.zeros(1, 6, 3), **kwargs)
    assert prediction.conditioned_role[0, 0] == 2
    assert prediction.role_logits.softmax(-1)[0, 1:, 2].count_nonzero() == 0
    saved = dict(schema=schema, config=config, model=model.state_dict(),
        architecture_contract=model.architecture_contract(), robot_asset_json=encode_robot_asset_json())
    reloaded = load_position_predictor(saved)
    assert reloaded.role_validity
