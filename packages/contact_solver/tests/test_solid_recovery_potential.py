"""Recovery gradients preserve physical evidence and follow an issued task."""
from types import SimpleNamespace

import torch

from contact_solver.solid_witness_loss import SolidSceneRouter
from somaforge_core.robot_assets import canonical_g1_asset_metadata
from somaforge_core.solid_distance import SOLID_GEOMETRY_SCHEMA


LINK = 'left_sphere_hand_link'


class TranslatingHand:
    joint_records = ()

    def link_poses(self, q, names):
        return q[:, None, :3], torch.eye(3).to(q).expand(len(q), 1, 3, 3)


def scene(height):
    return dict(schema=SOLID_GEOMETRY_SCHEMA, model_fingerprint='f'*64,
        robot_asset=canonical_g1_asset_metadata(), allowed_pairs=[[0, 1]], shapes=[
            dict(shape=0, body=None, kind='box', scale=[.5, .5, height/2],
                 transform=[0., 0., height/2, 0., 0., 0., 1.]),
            dict(shape=1, body=LINK, kind='sphere', scale=[.1, .1, .1],
                 transform=[0., 0., 0., 0., 0., 0., 1.])])


def evaluate(router, q, heights, *, world_frame=None, audit=False):
    fk = TranslatingHand()
    ids = torch.arange(len(q), device=q.device) if len(router.scenes) > 1 else torch.zeros(len(q), dtype=torch.long, device=q.device)
    solid = router(fk, q, ids, world_frame=world_frame)
    normal = q.new_tensor([0., 0., 1.]).expand(len(q), 1, 3)
    observed = dict(configured_margin=q.new_full((len(q),), .02),
        face_normal=normal, face_offset=q.new_tensor(heights)[:, None],
        face_surface=torch.ones(len(q), 1, dtype=torch.long, device=q.device))
    rows = SimpleNamespace(q=q, solid=solid, observed=observed, world_frame=world_frame)
    points = q.new_zeros(len(q), 6, 3); points[:, 2, 2] = q.new_tensor(heights)
    active = torch.zeros(len(q), 6, dtype=torch.bool, device=q.device); active[:, 2] = True
    current = q.detach().clone(); current[:, 2] = q.new_tensor(heights)+.2
    if world_frame is not None:
        origin, basis = world_frame
        points = origin[:, None]+torch.einsum('bij,bpj->bpi', basis, points)
        current[:, :3] = origin+torch.einsum('bij,bj->bi', basis, current[:, :3])
        observed['face_normal'] = torch.einsum('bij,bfj->bfi', basis, normal)
        observed['face_offset'] += (observed['face_normal']*origin[:, None]).sum(-1)
    distance, evidence = router.recovery_distances(fk, rows, ids, current, points, active,
        torch.ones_like(active, dtype=torch.long), audit=audit)
    return solid, distance, evidence


def pose(z):
    q = torch.zeros(1, 36, dtype=torch.float64); q[:, 2] = z; q[:, 3] = 1
    return q.requires_grad_()


def test_contained_shape_uses_task_exit_instead_of_closest_wrong_face():
    router = SolidSceneRouter({0: scene(1.)}, (LINK,)); q = pose(.2)
    solid, recovery, evidence = evaluate(router, q, [1.], audit=True)
    physical = solid.distances.clone()
    raw_gradient = torch.autograd.grad(-physical.sum(), q, retain_graph=True)[0]
    task_gradient = torch.autograd.grad(-recovery.sum(), q)[0]
    assert raw_gradient[0, 2] > 0  # Closest exit is through the bottom.
    assert task_gradient[0, 2] < 0  # Issued top-face task requires moving up.
    torch.testing.assert_close(recovery, q.new_tensor([-.9]))
    torch.testing.assert_close(solid.distances, physical)
    torch.testing.assert_close(solid.depths(), q.new_tensor([[.3, 0.]]))
    assert evidence[0]['normal'] == [0., 0., 1.]


def test_shallow_valid_task_face_preserves_the_physical_gradient():
    router = SolidSceneRouter({0: scene(1.)}, (LINK,)); q = pose(1.099)
    solid, recovery, _ = evaluate(router, q, [1.])
    torch.testing.assert_close(recovery, solid.distances, atol=1.e-10, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(recovery.sum(), q, retain_graph=True)[0],
        torch.autograd.grad(solid.distances.sum(), q)[0], atol=1.e-10, rtol=0)


def test_fresh_queries_agree_with_autograd_inside_the_transition_band():
    router = SolidSceneRouter({0: scene(1.)}, (LINK,)); q = pose(-.098)
    _, recovery, _ = evaluate(router, q, [1.])
    derivative = torch.autograd.grad(recovery.sum(), q)[0][0, 2]
    shift = torch.zeros_like(q); shift[:, 2] = 1.e-7
    plus = evaluate(router, q.detach()+shift, [1.])[1].sum()
    minus = evaluate(router, q.detach()-shift, [1.])[1].sum()
    torch.testing.assert_close(derivative, (plus-minus)/2.e-7, atol=1.e-6, rtol=1.e-6)


def test_physical_boundary_has_zero_loss_and_continuous_loss_gradient():
    router = SolidSceneRouter({0: scene(1.)}, (LINK,))
    for z in (-.100001, -.1000001, -.0999999):
        q = pose(z); _, distance, _ = evaluate(router, q, [1.])
        loss = distance.square().sum()+q.sum()*0
        assert loss.item() < 1.e-12
        assert torch.autograd.grad(loss, q)[0].abs().max() < 1.e-5


def test_mixed_scenes_use_each_actual_obstacle_and_contact_margin():
    router = SolidSceneRouter({0: scene(1.), 1: scene(.5)}, (LINK,))
    q = pose(.2).detach().repeat(2, 1).requires_grad_()
    _, recovery, _ = evaluate(router, q, [1., .5])
    torch.testing.assert_close(recovery, q.new_tensor([-.9, -.4]))
    torch.testing.assert_close(torch.autograd.grad(recovery.sum(), q)[0][:, 2], q.new_ones(2))


def test_recovery_value_and_gradient_are_independent_of_world_chart():
    from scipy.spatial.transform import Rotation
    router = SolidSceneRouter({0: scene(1.)}, (LINK,)); q = pose(.2)
    data = scene(1.)
    basis = torch.tensor(Rotation.from_euler('z', .8).as_matrix(), dtype=q.dtype)[None]
    origin = q.new_tensor([[7., -3., 1.]])
    data['shapes'][0]['transform'] = [7., -3., 1.5, *Rotation.from_euler('z', .8).as_quat()]
    transformed = SolidSceneRouter({0: data}, (LINK,))
    first = evaluate(router, q, [1.])[1]
    second = evaluate(transformed, q, [1.], world_frame=(origin, basis))[1]
    torch.testing.assert_close(first, second, atol=1.e-10, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(first.sum(), q)[0],
        torch.autograd.grad(second.sum(), q)[0], atol=1.e-10, rtol=0)


def test_self_collision_keeps_its_physical_value_and_derivative():
    router = SolidSceneRouter({0: scene(1.)}, (LINK,)); q = pose(.2)
    solid = router(TranslatingHand(), q, torch.zeros(1, dtype=torch.long))
    solid.pair['full_kind'] = torch.ones_like(solid.pair['full_kind'])
    rows = SimpleNamespace(q=q, solid=solid, observed=dict(configured_margin=q.new_tensor([.02]),
        face_normal=q.new_tensor([[[0., 0., 1.]]]), face_offset=q.new_tensor([[1.]]),
        face_surface=torch.ones(1, 1, dtype=torch.long)))
    current = q.detach().clone()
    points = q.new_zeros(1, 6, 3); active = torch.zeros(1, 6, dtype=torch.bool)
    recovery, evidence = router.recovery_distances(TranslatingHand(), rows,
        torch.zeros(1, dtype=torch.long), current, points, active, torch.zeros_like(active, dtype=torch.long))
    assert not evidence
    torch.testing.assert_close(recovery, solid.distances)
    torch.testing.assert_close(torch.autograd.grad(recovery.sum(), q, retain_graph=True)[0],
        torch.autograd.grad(solid.distances.sum(), q)[0])


def test_missing_native_surface_catalog_is_explicitly_rejected():
    import pytest
    router = SolidSceneRouter({0: scene(1.)}, (LINK,)); q = pose(.2)
    solid = router(TranslatingHand(), q, torch.zeros(1, dtype=torch.long))
    rows = SimpleNamespace(q=q, solid=solid, observed=dict(configured_margin=q.new_tensor([.02])))
    active = torch.zeros(1, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match='actual solver margin and surface catalog'):
        router.recovery_distances(TranslatingHand(), rows, torch.zeros(1, dtype=torch.long),
            q.detach(), q.new_zeros(1, 6, 3), active, torch.zeros_like(active, dtype=torch.long))
