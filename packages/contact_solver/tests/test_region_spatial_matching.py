import torch

from contact_solver.contact_layout import nearest_spatial_pairs
from contact_solver.device_contact_objective import DeviceWitnessRows


def fixture():
    rows = object.__new__(DeviceWitnessRows)
    rows.q = torch.zeros(1, 36)
    rows.world_frame = None
    rows.sample = torch.tensor([0, 0])
    rows.part = torch.tensor([0, 0])
    rows.group = rows.sample*6+rows.part
    rows.contact_region_ids = torch.tensor([0, 2])
    witness = torch.tensor([[0., 0., 0.], [.2, 0., 0.]], requires_grad=True)
    rows.moving = torch.stack((witness*0, witness), 1)
    rows.pair = dict(geometry_point1_w=witness.detach().clone(),
        task_pair=torch.ones(2, dtype=torch.bool), eligible=torch.ones(2, dtype=torch.bool),
        primary_surface=torch.zeros(2, dtype=torch.long), dist=torch.tensor([-.02, -.001]))
    points = torch.zeros(1, 6, 3); points[0, 0, 0] = .19
    active = torch.tensor([[True, False, False, False, False, False]])
    surface = torch.zeros(1, 6, dtype=torch.long)
    return rows, witness, points, active, surface


def test_deepest_representative_switch_does_not_change_spatial_execution():
    rows, witness, points, active, surface = fixture()
    points[0, 0, 0] = .13  # Outside the accepted range, so execution has a gradient.
    error, complete, count = rows.witness_position_statistics(points, active, surface, actual=True)
    torch.testing.assert_close(error, torch.tensor([.07**2]))
    assert complete.item() and count.item() == 1
    rows.pair['dist'] = rows.pair['dist'].flip(0)
    other, _, _ = rows.witness_position_statistics(points, active, surface, actual=True)
    torch.testing.assert_close(other, error)
    loss, _ = rows.frozen_layout_diagnostic(points, active, surface)
    loss.sum().backward()
    assert witness.grad[0].abs().sum() == 0
    assert witness.grad[1, 0] > 0
    assert points.grad is None


def test_region_filter_rejects_nearby_contact_on_wrong_part_region():
    rows, _, points, active, surface = fixture()
    regions = torch.zeros(1, 6, 4, dtype=torch.bool); regions[0, 0, 0] = True
    error, complete, _ = rows.witness_position_statistics(points, active, surface, actual=True, regions=regions)
    assert complete.item() and error.item() > .03
    regions[:] = False; regions[0, 0, 3] = True
    _, complete, count = rows.witness_position_statistics(points, active, surface, actual=True, regions=regions)
    assert not complete.item() and count.item() == 0


def test_inactive_nearby_candidate_cannot_pass_actual_acceptance():
    rows, _, points, active, surface = fixture()
    rows.pair['eligible'][1] = False
    error, complete, _ = rows.witness_position_statistics(points, active, surface, actual=True)
    assert complete.item() and error.item() > .03
    loss, _ = rows.frozen_layout_diagnostic(points, active, surface)
    assert loss.item() == 0  # Candidate XY is acceptable; activation still fails.
    rows.pair['eligible'][:] = False
    _, complete, _ = rows.witness_position_statistics(points, active, surface, actual=True)
    assert not complete.item()


def test_cpu_pair_matching_agrees_with_device_region_selection_and_preserves_raw():
    rows, _, points, active, surface = fixture()
    regions = torch.zeros(1, 6, 4, dtype=torch.bool); regions[0, 0, 2] = True
    pairs = [dict(part=0, surface=0, position_w=point.tolist(), contact_region=region)
             for point, region in zip(rows.pair['geometry_point1_w'], (0, 2))]
    selected = nearest_spatial_pairs(pairs, active[0], surface[0], points[0], regions=regions[0])
    error, complete, _ = rows.witness_position_statistics(points, active, surface, actual=True, regions=regions)
    assert complete.item() and selected[0] is pairs[1]
    residual = (torch.tensor(selected[0]['position_w'])[:2]-points[0, 0, :2]).square().sum()
    torch.testing.assert_close(error[0], residual)


def test_local_fk_and_world_actual_match_after_frame_translation():
    rows, _, points, active, surface = fixture()
    origin = torch.tensor([[10., -7., 2.]])
    rows.world_frame = (origin, torch.eye(3)[None])
    rows.pair['geometry_point1_w'] += origin
    world_points = points + origin[:, None]
    differentiable, _, _ = rows.witness_position_statistics(world_points, active, surface)
    actual, _, _ = rows.witness_position_statistics(world_points, active, surface, actual=True)
    torch.testing.assert_close(differentiable, actual)
