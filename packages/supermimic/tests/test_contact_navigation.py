"""Navigation geometry stays independent of solver contact truth and training."""
import math

import pytest
import torch

from supermimic.contact_navigation import WholeFootBoxNavigation, segment_visible


@pytest.fixture
def navigation():
    vertices = torch.tensor([[x, y, z] for x in [-.1, .1] for y in [-.04, .04] for z in [-.02, .02]])
    local = torch.tensor([[-.1, -.04, -.02], [-.1, .04, -.02], [.1, -.04, -.02], [.1, .04, -.02]])
    goals = local + torch.tensor([.03, .02, .57])
    metric = WholeFootBoxNavigation([[0, 0, .25, 1, 0, 0, 0, 1, 1, .6]], goals, vertices, local, 'cpu')
    return metric, goals


def test_visible_distance_is_exact_and_goal_updates_reuse_graph(navigation):
    metric, goals = navigation
    point = metric.goal_world[None] + torch.tensor([0., 0., .08])
    torch.testing.assert_close(metric.anchor_distance(point), torch.full((1, 4), .08), atol=1e-5, rtol=1e-4)
    nodes, costs = metric.road.nodes.data_ptr(), metric.road.costs.data_ptr()
    previous = metric.anchor_distance(point)
    metric.set_goal(goals + torch.tensor([.04, -.03, 0.]))
    assert metric.road.nodes.data_ptr() == nodes and metric.road.costs.data_ptr() == costs
    assert not torch.allclose(metric.anchor_distance(point), previous)


def test_blocked_queries_use_clear_routes_and_raise_the_foot(navigation):
    metric, _ = navigation
    position = torch.zeros(1, 4, 3)
    position[:, :, 0] = .037
    position[:, :, 1] = -metric.nav_half[:, 1] - .13
    position[:, :, 2] = .13
    position.requires_grad_()
    goal = metric.goals[None].clone().requires_grad_()
    distance, (free, index_i, index_j, direct) = metric.road.local_query(position, goal, return_route=True)
    assert not direct.any()
    assert (distance >= (position - goal).norm(dim=-1) - 1e-6).all()
    gp, gg = torch.autograd.grad(distance.sum(), (position, goal))
    assert torch.isfinite(gp).all() and torch.isfinite(gg).all()
    assert (gp[:, :, 2] < 0).all()
    for part in range(4):
        first = int(index_i[0, part])
        last = int(index_j[0, part, first])
        chain, cursor = [last], last
        while cursor != first:
            cursor = int(metric.road.predecessors[part, first, cursor])
            assert cursor >= 0
            chain.append(cursor)
        path = torch.stack([free[0, part]] + [metric.road.nodes[part, i] for i in reversed(chain)] + [goal[0, part]])
        assert segment_visible(path[:-1], path[1:], metric.nav_half[part]).all()
    epsilon = .0005
    plus, minus = position.detach().clone(), position.detach().clone()
    plus[0, 0, 2] += epsilon
    minus[0, 0, 2] -= epsilon
    finite_difference = (metric.road.local_query(plus, goal.detach()).sum() - metric.road.local_query(minus, goal.detach()).sum()) / (2 * epsilon)
    torch.testing.assert_close(gp[0, 0, 2], finite_difference, atol=.003, rtol=.003)


def test_envelope_covers_mesh_when_foot_is_rotated(navigation):
    metric, _ = navigation
    for angle in [0., math.pi / 3, math.pi / 2]:
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        vertices = metric.vertices @ rotation.T
        anchors = metric.material_local @ rotation.T
        radial = (vertices[None, :, :2] - anchors[:, None, :2]).norm(dim=-1)
        assert (radial <= metric.radius[:, None] + 1e-6).all()
        assert vertices[:, 2].max() - vertices[:, 2].min() <= 2 * metric.radius.min() + 1e-6


def test_rigid_pose_distance_has_finite_position_and_rotation_gradients(navigation):
    metric, goals = navigation
    position = torch.tensor([[0., -.8, .4]], requires_grad=True)
    rotation = torch.eye(3)[None].requires_grad_()
    points = position[:, None] + torch.einsum('bij,kj->bki', rotation, metric.material_local)
    distance = metric.distance(points, position, rotation)
    gp, gr = torch.autograd.grad(distance.square().mean(), (position, rotation))
    assert distance.shape == (1, 4)
    assert torch.isfinite(gp).all() and torch.isfinite(gr).all()
    assert gp.norm() > 0 and gr.norm() > 0


def test_floating_box_is_rejected_at_scene_boundary(navigation):
    metric, goals = navigation
    with pytest.raises(ValueError, match='underpasses'):
        WholeFootBoxNavigation([[0, 0, 1., 1, 0, 0, 0, 1, 1, .6]], goals, metric.vertices, metric.material_local, 'cpu')


def test_low_overlap_exits_above_floor_instead_of_through_box_bottom(navigation):
    metric, _ = navigation
    position = torch.zeros(1, 4, 3)
    position[:, :, 2] = -.15  # local coordinates: world z=.1, inside envelope
    position.requires_grad_()
    distance, (free, _, _, _) = metric.road.local_query(position, metric.goals[None], return_route=True)
    assert torch.isfinite(distance).all()
    assert (free[:, :, 2] + metric.center[2] >= 0).all()
    gradient = torch.autograd.grad(distance.sum(), position)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
