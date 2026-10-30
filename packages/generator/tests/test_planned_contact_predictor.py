from generator.research.frozen_contact_position import frozen_contact_layout_diagnostic, point_target_approach_diagnostic
import torch
import math

from generator.contact_location_predictor import HeightmapContactLocationPredictor
from generator.planned_contact_predictor import (
    PlannedHeightmapContactPredictor, contact_plan_objective,
    plan_points_in_pose_frame, audit_intended_surfaces,
    translated_observed_plan_cells,
)
from test_contact_location_predictor import observation
from generator.contact_location_predictor import contact_points_to_observed_heightmap_map


def model():
    return PlannedHeightmapContactPredictor(width=24, layers=1, location_width=8).eval()


def test_migration_preserves_original_pose_until_conditioning_is_trained():
    torch.manual_seed(7)
    old = HeightmapContactLocationPredictor(width=24, layers=1, location_width=8).eval()
    new = model()
    loaded, fresh = new.load_embodied(old.state_dict())
    assert loaded and fresh
    args = observation(1)[:4]
    with torch.no_grad():
        torch.testing.assert_close(new(*args).qpos, old(*args).qpos, rtol=0, atol=0)


def test_changing_contact_plan_changes_pose_and_inactive_locations_do_not():
    torch.manual_seed(8)
    net = model()
    torch.nn.init.normal_(net.plan_fusion.weight, std=0.2)
    args = observation(1)[:4]
    active = torch.zeros((1, 6), dtype=torch.bool)
    cell = torch.zeros((1, 6), dtype=torch.long)
    with torch.no_grad():
        off = net.forward_with_plan(*args, planned_contact=active, planned_cell=cell)
        unused = net.forward_with_plan(*args, planned_contact=active, planned_cell=cell + 1000)
        torch.testing.assert_close(off.qpos, unused.qpos, rtol=0, atol=0)
        active[:, 0] = True
        on = net.forward_with_plan(*args, planned_contact=active, planned_cell=cell)
        elsewhere = net.forward_with_plan(*args, planned_contact=active, planned_cell=cell + 1000)
    assert not torch.equal(on.qpos, off.qpos)
    assert not torch.equal(on.qpos, elsewhere.qpos)


def test_pose_gradient_does_not_relabel_contact_heads_and_plan_loss_trains_them():
    net = model()
    args = observation(1)[:4]
    prediction = net(*args)
    prediction.qpos.square().sum().backward()
    assert net.role_head.weight.grad is None
    assert net.location_query.weight.grad is None
    assert net.plan_fusion.weight.grad is not None
    net.zero_grad(set_to_none=True)
    prediction = net(*args)
    target = {
        "role": torch.tensor([[1, 0, 0, 0, 0, 0]]),
        "contact_cell": torch.tensor([[1000, 0, 0, 0, 0, 0]]),
        "contact_cell_valid": torch.tensor([[True, False, False, False, False, False]]),
    }
    loss, _ = contact_plan_objective(prediction, target)
    loss.sum().backward()
    assert net.role_head.weight.grad.abs().sum() > 0
    assert net.location_query.weight.grad.abs().sum() > 0
    assert net.pose_head.weight.grad is None


def test_out_of_view_locations_are_not_supervised_as_cell_zero():
    net = model()
    prediction = net(*observation(1)[:4])
    target = {
        "role": torch.ones((1, 6), dtype=torch.long),
        "contact_cell": torch.zeros((1, 6), dtype=torch.long),
        "contact_cell_valid": torch.zeros((1, 6), dtype=torch.bool),
    }
    _, metrics = contact_plan_objective(prediction, target)
    assert metrics["plan_location_loss"].item() == 0


def test_teacher_and_autonomous_spatial_points_are_distinct_and_surface_free():
    net = model()
    args = observation(1)[:4]
    auto = net(*args)
    cell = (auto.cell + 123) % auto.location_logits.shape[-1]
    teacher = net.forward_with_plan(*args, planned_contact=auto.contact, planned_cell=cell)
    assert not torch.equal(teacher.conditioned_points_local, teacher.contact_points_local)
    assert not any("surface" in name for name in net.state_dict())


def test_spatial_plan_coordinates_follow_observation_yaw_and_translation():
    q = torch.zeros(1, 36)
    q[:, :3] = torch.tensor([3., 4., 2.])
    q[:, 3] = q[:, 6] = 2 ** -0.5
    point = torch.tensor([[[1., 0., -2.]]]).expand(1, 6, 3)
    world = plan_points_in_pose_frame(q, point)
    torch.testing.assert_close(world[0, 0], torch.tensor([3., 5., 0.]))


def test_spatial_approach_has_pose_gradient_without_any_gt_contact_plan():
    net = model()
    q = observation(1)[0].clone().requires_grad_(True)
    active = torch.zeros(1, 6, dtype=torch.bool)
    active[:, 2] = True  # Require the hand, independently of any demonstration.
    points = torch.zeros(1, 6, 3, requires_grad=True)
    points.data[:, 2, :2] = torch.tensor([0.5, 0.2])
    loss, _ = point_target_approach_diagnostic(net, q, points, active, torch.tensor([0.017]))
    assert loss.item() > 0
    loss.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    assert points.grad is None  # Decoder must realize the fixed intent.
    inactive, _ = point_target_approach_diagnostic(net, q, points, ~torch.ones_like(active), torch.tensor([0.017]))
    assert inactive.item() == 0


def test_intention_audit_excludes_sides_and_does_not_require_ground_top_ids():
    points = torch.zeros(1, 6, 3)
    points[:, :, 2] = 0.8
    active = torch.tensor([[True, False, True, False, False, False]])
    observed = {"surface_catalog": [
        {"surface": 7, "normal_w": [0, 0, 1], "plane_offset": 0.0},
        {"surface": 42, "normal_w": [0, 0, 1], "plane_offset": 0.8},
        {"surface": 99, "normal_w": [1, 0, 0], "plane_offset": 0.0},
    ]}
    surface = audit_intended_surfaces(points, active, observed)
    assert surface.tolist() == [[42, -1, 42, -1, -1, -1]]


def test_3d_raster_labels_do_not_drop_a_ledge_contact_to_the_ground():
    net = model()
    q, contact, _, heightmap, _, _ = observation(1)
    # A point just on the low cell's side of a ledge, at the upper height.
    heightmap[:] = -q[0, 2]
    heightmap[:, 30:, :] += 0.7
    point = torch.zeros(1, 6, 3)
    point[:, 0, 0] = net.grid_xy.reshape(*heightmap.shape[1:], 2)[29, 25, 0] + 0.009
    point[:, 0, 1] = net.grid_xy.reshape(*heightmap.shape[1:], 2)[29, 25, 1]
    point[:, 0, 2] = 0.715
    raster, cell, valid = contact_points_to_observed_heightmap_map(q, contact, point, heightmap)
    chosen_height = heightmap.flatten(1).gather(1, cell[:, :1]) + q[:, 2:3]
    torch.testing.assert_close(chosen_height, torch.tensor([[0.7]]))
    assert valid[0, 0] and raster[0, 0].sum() == 1


def test_spatial_realization_is_invariant_to_world_yaw_and_translation():
    from generator.next_interaction_heightmap_v2 import _root_yaw_basis
    net = model()
    q = observation(1)[0].clone()
    points = torch.zeros(1, 6, 3)
    points[:, 0, :2] = torch.tensor([0.15, 0.08])
    active = torch.tensor([[True, False, False, False, False, False]])
    margin = torch.tensor([0.023])
    base, _ = point_target_approach_diagnostic(net, q, points, active, margin)
    moved = q.clone()
    moved[:, 3] = math.cos(0.37 / 2)
    moved[:, 6] = math.sin(0.37 / 2)
    moved[:, 0:2] = torch.tensor([0.7, -0.4])
    basis, _ = _root_yaw_basis(moved)
    shifted = torch.einsum("bij,bpj->bpi", basis, points) + torch.tensor([0.7, -0.4, 0.0])
    rotated, _ = point_target_approach_diagnostic(net, moved, shifted, active, margin, cell_basis=basis)
    torch.testing.assert_close(base, rotated, atol=1e-4, rtol=1e-5)


def test_counterfactual_plan_translation_preserves_observed_height_and_bounds():
    from generator.next_interaction_heightmap import HEIGHTMAP_COLS
    _, contact, _, heightmap, _, _ = observation(3)
    heightmap[:] = 0
    heightmap[:, 30:, :] = 0.7
    cells = torch.zeros(3, 6, dtype=torch.long)
    cells[:, 0] = torch.tensor([32, 29, 0]) * HEIGHTMAP_COLS + 25
    moved, eligible = translated_observed_plan_cells(
        cells, contact, heightmap, torch.tensor([1, 1, -1]), torch.zeros(3, dtype=torch.long),
    )
    assert eligible.tolist() == [True, False, False]
    assert moved[0, 0] == cells[0, 0] + HEIGHTMAP_COLS
    torch.testing.assert_close(moved[1:], cells[1:])


def test_scene_layout_is_translation_equivariant_but_penalizes_sliding():
    from types import SimpleNamespace
    class FakeFK:
        def link_poses(self, q, names):
            positions = []
            for name in names:
                part = 0 if name == "left" else 1
                positions.append(q[:, :3] + torch.stack((q[:, 7 + part], q[:, 9] * 0, q[:, 9] * 0), -1))
            return torch.stack(positions, 1), torch.eye(3).reshape(1, 1, 3, 3).expand(len(q), len(names), 3, 3)
    net = SimpleNamespace(fk=FakeFK())
    q = torch.zeros(1, 36, requires_grad=True)
    points = torch.zeros(1, 6, 3)
    points[0, 1, 0] = 0.3
    active = torch.tensor([[True, True, False, False, False, False]])
    surfaces = torch.tensor([[7, 7, -1, -1, -1, -1]])
    def rows(shift):
        return [[({"part": part, "surface": 7, "body_name": name, "dist": 0.001,
                   "position_w": [x + shift[0], shift[1], 0]}, None)
                 for part, name, x in ((0, "left", 0), (1, "right", 0.2))]]
    loss, _ = frozen_contact_layout_diagnostic(net, q, points, active, surfaces, rows((0, 0)))
    moved = q.detach().clone()
    moved[:, :2] = torch.tensor([2., -3.])
    translated_points = points + points.new_tensor([2., -3., 0.])
    translated, _ = frozen_contact_layout_diagnostic(net, moved, translated_points, active, surfaces, rows((2, -3)))
    slid, _ = frozen_contact_layout_diagnostic(net, moved, points, active, surfaces, rows((2, -3)))
    assert slid.item() > loss.item()
    torch.testing.assert_close(loss, translated)
    loss.sum().backward()
    assert q.grad[:, :2].abs().sum() > 0
    assert q.grad[:, 7:9].abs().sum() > 0


def test_missing_newton_pairs_are_not_fabricated_for_relative_layout():
    q = torch.zeros(1, 36, requires_grad=True)
    loss, metrics = frozen_contact_layout_diagnostic(None, q, torch.zeros(1, 6, 3),
        torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 6, dtype=torch.long), [[]])
    assert loss.item() == 0 and metrics["relative_layout_observed_parts"].item() == 0
