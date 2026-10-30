from types import SimpleNamespace

import pytest
import torch
from torch import nn

from generator.position_plan import select_position_plan
from generator.planned_contact_predictor import contact_plan_objective


def fixture():
    model = SimpleNamespace(role_validity=True, world_surface_scan=True, event_roles=True, region_plan=False,
        role_head=nn.Linear(4, 4), location_query=nn.Linear(4, 1), location_key=nn.Linear(3, 1))
    with torch.no_grad():
        model.role_head.weight.zero_(); model.role_head.bias.copy_(torch.tensor([0., 2., 20., -1.]))
        model.location_query.weight.zero_(); model.location_query.bias.fill_(1.)
        model.location_key.weight.copy_(torch.tensor([[0., 0., 1.]])); model.location_key.bias.zero_()
    q = torch.zeros(1, 36); q[:, 3] = 1.
    active = torch.zeros(1, 6, dtype=torch.bool); active[:, 0] = True
    relation = torch.zeros(1, 6, 2, dtype=torch.bool); relation[:, 0, 0] = True
    kwargs = dict(current_contact=active, current_contact_relations=relation,
        contact_surface_ids=torch.tensor([[0, 1]]),
        scan_points_world=torch.tensor([[[1., 0., 0.], [1., 0., .7], [1., 0., 100.]]]),
        scan_surface=torch.tensor([[0, 1, -1]]), scan_valid=torch.tensor([[True, True, False]]),
        scan_point_id=torch.tensor([[10, 11, -1]]), scan_spacing_m=torch.tensor([.05]))
    return model, q, torch.zeros(1, 5, 5), torch.eye(3)[None], torch.zeros(1, 6, 4), kwargs


def test_same_xy_keeps_world_surface_identity_and_padding_is_unselectable():
    model, q, height, basis, part, kwargs = fixture()
    plan = select_position_plan(model, q, height, basis, part, **kwargs)
    assert plan.conditioned_role.tolist() == [[2, 1, 1, 1, 1, 1]]
    assert plan.conditioned_surface.tolist() == [[0, 1, 1, 1, 1, 1]]
    assert plan.conditioned_point_id.tolist() == [[10, 11, 11, 11, 11, 11]]
    assert torch.isneginf(plan.location_logits[..., 2]).all()
    torch.testing.assert_close(plan.conditioned_points_local[0, :, 2], torch.tensor([0., .7, .7, .7, .7, .7]))
    # The raster can change discontinuously; it cannot relift selected mesh XYZ.
    height.fill_(-100.)
    q[:, 1] += .001
    shifted = select_position_plan(model, q, height, basis, part, **kwargs)
    assert torch.equal(shifted.conditioned_point_id, plan.conditioned_point_id)
    world = q[:, None, :3] + shifted.conditioned_points_local
    torch.testing.assert_close(world, plan.conditioned_points_local)


def test_teacher_and_repair_cannot_keep_another_layer():
    model, q, height, basis, part, kwargs = fixture()
    roles = torch.zeros(1, 6, dtype=torch.long); roles[:, 0] = 2
    cells = torch.ones(1, 6, dtype=torch.long)
    with pytest.raises(ValueError, match='same observed contact surface'):
        select_position_plan(model, q, height, basis, part, **kwargs, teacher_mask=torch.tensor([True]),
                             teacher_role=roles, teacher_contact=roles != 0, teacher_cell=cells)
    points = torch.zeros(1, 6, 3); points[:, 0] = kwargs['scan_points_world'][:, 1]
    with pytest.raises(ValueError, match='same observed contact surface'):
        select_position_plan(model, q, height, basis, part, **kwargs, repair_mask=torch.tensor([True]),
                             repair_role=roles, repair_points_world=points, repair_surface=cells,
                             repair_point_id=torch.full_like(cells, 11))


def test_invalid_teacher_nodes_and_missing_scan_fail_explicitly():
    model, q, height, basis, part, kwargs = fixture()
    with pytest.raises(ValueError, match='fallback is forbidden'):
        select_position_plan(model, q, height, basis, part, current_contact=kwargs['current_contact'])
    roles = torch.ones(1, 6, dtype=torch.long)
    with pytest.raises(ValueError, match='not a scanned interaction surface'):
        select_position_plan(model, q, height, basis, part, **kwargs, teacher_mask=torch.tensor([True]),
                             teacher_role=roles, teacher_contact=roles != 0, teacher_cell=torch.full_like(roles, 2))


def test_location_loss_counts_vertical_error_and_has_finite_gradients():
    model, q, height, basis, part, kwargs = fixture()
    plan = select_position_plan(model, q, height, basis, part, **kwargs)
    target = dict(role=torch.ones(1, 6, dtype=torch.long), contact_cell=torch.zeros(1, 6, dtype=torch.long),
                  contact_cell_valid=torch.ones(1, 6, dtype=torch.bool))
    loss, metrics = contact_plan_objective(plan.prediction(q), target)
    assert torch.isfinite(loss).all()
    assert metrics['plan_location_error_cm'].item() == pytest.approx(70.*5/6, abs=1e-4)
    loss.sum().backward()
    assert torch.isfinite(model.location_key.weight.grad).all()
    from contact_solver.contact_layout import relative_plan_objective
    torch.testing.assert_close(relative_plan_objective(plan.prediction(q), target, height)[0], loss)


def test_repair_keeps_original_node_identity_and_rejects_unknown_identity():
    model, q, height, basis, part, kwargs = fixture()
    plan = select_position_plan(model, q, height, basis, part, **kwargs)
    repair = dict(repair_mask=torch.tensor([True]), repair_role=plan.conditioned_role,
                  repair_points_world=plan.conditioned_points_local)
    with pytest.raises(ValueError, match='stored surface and point identity'):
        select_position_plan(model, q, height, basis, part, **kwargs, **repair)
    # The new observation samples nearby points; issued world XYZ/ID persist.
    kwargs['scan_points_world'] = kwargs['scan_points_world'] + torch.tensor([.001, 0., 0.])
    kwargs['scan_point_id'] = torch.tensor([[20, 21, -1]])
    retried = select_position_plan(model, q, height, basis, part, **kwargs, **repair,
                                  repair_surface=plan.conditioned_surface, repair_point_id=plan.conditioned_point_id)
    torch.testing.assert_close(retried.conditioned_points_local, plan.conditioned_points_local)
    assert torch.equal(retried.conditioned_point_id, plan.conditioned_point_id)


@pytest.mark.parametrize('schema', ['full1000_position_predictor_v1', 'full1000_position_predictor_v2'])
def test_both_architectures_require_mesh_contract_and_roundtrip(schema):
    from generator.predictor_architecture import build_position_predictor, load_position_predictor
    from somaforge_core.robot_assets import encode_robot_asset_json
    from somaforge_core.world_mesh_scan import SCAN_SCHEMA
    from somaforge_core.heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS
    config = dict(architecture=schema, width=24, layers=1, location_width=4, body_geometry=False,
                  part_geometry=False, region_plan=False, unified_contact=False, event_roles=True,
                  role_validity=True, world_surface_scan=True)
    model = build_position_predictor(config, schema=schema)
    _, q, _, _, _, kwargs = fixture()
    with torch.no_grad():
        prediction = model(current_q=q, heightmap=torch.zeros(1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS),
                           current_anchor=torch.zeros(1, 6, 3), **kwargs)
    assert prediction.candidate_points_local.shape == (1, 3, 3)
    saved = dict(schema=schema, config=config, model=model.state_dict(),
                 architecture_contract=model.architecture_contract(), robot_asset_json=encode_robot_asset_json())
    assert saved['architecture_contract']['interaction_candidates'] == SCAN_SCHEMA
    assert load_position_predictor(saved).world_surface_scan
    saved['architecture_contract'].pop('interaction_candidates')
    with pytest.raises(ValueError, match='world mesh scan contract'):
        load_position_predictor(saved)
