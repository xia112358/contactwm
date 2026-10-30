"""Behavioral checks for solid recovery, containment, FK and native isolation."""
import copy

import numpy as np
import pytest
import torch

from contact_solver.solid_witness_loss import SolidSceneRouter
from somaforge_core.robot_assets import canonical_g1_asset_metadata
from somaforge_core.solid_distance import SOLID_GEOMETRY_SCHEMA, SolidDistanceScene


class TranslatingLinks:
    def link_poses(self, q, names):
        position = torch.stack([q[:, :3] if n == 'first' else q[:, :3]+q[:, 7:10] for n in names], 1)
        rotation = torch.eye(3, dtype=q.dtype, device=q.device).expand(len(q), len(names), 3, 3)
        return position, rotation


def shape(index, body, kind, scale, position=(0., 0., 0.)):
    return dict(shape=index, body=body, kind=kind, scale=list(scale), transform=[*position, 0., 0., 0., 1.])


def contract(shapes, allowed):
    return dict(schema=SOLID_GEOMETRY_SCHEMA, model_fingerprint='f'*64,
                robot_asset=canonical_g1_asset_metadata(), shapes=shapes, allowed_pairs=allowed)


def box_sphere():
    data = contract([shape(0, None, 'box', (.5, .5, .25), (0., 0., .25)),
                     shape(1, 'first', 'sphere', (.1, .1, .1))], [[0, 1]])
    return data, SolidSceneRouter({0: data}, ('first',))


def test_deep_crossing_uses_solid_exit_and_full_q_finite_difference():
    _, router = box_sphere()
    q = torch.zeros(1, 36, dtype=torch.float64); q[0, 2] = .45; q.requires_grad_()
    fk = TranslatingLinks(); ids = torch.zeros(1, dtype=torch.long)
    rows = router(fk, q, ids)
    torch.testing.assert_close(rows.distances, torch.tensor([-.15], dtype=q.dtype))
    gradient = torch.autograd.grad(rows.distances.sum(), q)[0]
    assert gradient[0, 2] > 0
    finite = torch.zeros_like(q)
    for index in range(36):
        shift = torch.zeros_like(q); shift[0, index] = 1.e-5
        plus, minus = router(fk, q.detach()+shift, ids), router(fk, q.detach()-shift, ids)
        finite[0, index] = (plus.distances.sum()-minus.distances.sum())/2.e-5
    torch.testing.assert_close(gradient, finite, atol=1.e-8, rtol=1.e-8)
    assert router(fk, q.detach()+.001*gradient, ids).distances[0] > rows.distances[0]


def test_fully_contained_body_does_not_require_triangle_contacts():
    _, router = box_sphere()
    q = torch.zeros(1, 36, dtype=torch.float64); q[0, 2] = .30; q.requires_grad_()
    ids = torch.zeros(1, dtype=torch.long)
    rows = router(TranslatingLinks(), q, ids)
    assert len(rows.pair['dist']) == 1 and rows.distances[0] < -.2
    gradient = torch.autograd.grad(rows.distances.sum(), q)[0]
    updated = router(TranslatingLinks(), q.detach()+.001*gradient, ids)
    assert updated.distances[0] > rows.distances[0]


@pytest.mark.parametrize('penalty', ['log1p', 'huber'])
def test_unified_loss_replaces_the_inward_triangle_penetration_gradient(penalty):
    from types import SimpleNamespace
    from contact_solver.contact_regions import unified_region_objective
    from contact_solver.device_contact_objective import DeviceWitnessRows

    class FK:
        def link_poses(self, q, names):
            return q[:, None, :3].expand(-1, len(names), -1), torch.eye(3).to(q).expand(len(q), len(names), 3, 3)

    class Regions:
        valid = torch.ones(6, 4, dtype=torch.bool)
        def witness_regions(self, fk, q, sample, part, points):
            return torch.zeros_like(part)
        def missing_region_distance(self, fk, q, surface, scene, margin):
            return q.new_zeros(len(q), 6, 4)
        def actual_mask(self, fk, rows, surface):
            return torch.zeros(len(rows), 6, 4, dtype=torch.bool)

    name = 'left_sphere_hand_link'
    data = contract([shape(0, None, 'box', (.5, .5, .25), (0., 0., .25)),
                     shape(1, name, 'sphere', (.1, .1, .1))], [[0, 1]])
    router = SolidSceneRouter({0: data}, (name,))
    q = torch.zeros(1, 36, dtype=torch.float64); q[0, 2] = .45; q.requires_grad_()
    pair = dict(sample=torch.tensor([0]), part=torch.tensor([2]), primary_surface=torch.tensor([0]),
        task_pair=torch.tensor([False]), upward=torch.tensor([False]), eligible=torch.tensor([False]),
        type=torch.tensor([0]), active=torch.tensor([False]), constraint_allocated=torch.tensor([False]),
        dist=q.new_tensor([-.02]), includemargin=q.new_tensor([.02]), full_kind=torch.tensor([0]),
        body_link0=torch.tensor([-1]), body_link1=torch.tensor([0]),
        geometry_point0_w=q.new_tensor([[0., 0., .5]]), geometry_point1_w=q.new_tensor([[0., 0., .52]]),
        normal_w=q.new_tensor([[0., 0., -1.]]))
    observed = dict(schema='newton_device_witness_batch_v1', pairs=pair,
                    configured_margin=q.new_tensor([.02]), link_names=(name,))
    rows = DeviceWitnessRows(FK(), q, observed)
    old_gradient = torch.autograd.grad((-rows.distances).relu().sum(), q, retain_graph=True)[0]
    assert old_gradient[0, 2] > 0  # descent follows the triangle downwards
    rows.solid = router(FK(), q, torch.tensor([0]))
    loss, metrics = unified_region_objective(SimpleNamespace(fk=FK(), region_geometry=Regions()),
        rows, torch.zeros(1, 6, dtype=torch.bool), torch.zeros(1, 6, dtype=torch.long), {},
        penetration_penalty=penalty)
    gradient = torch.autograd.grad(loss.sum(), q)[0]
    assert gradient[0, 2] < 0  # descent follows the complete solid upwards
    torch.testing.assert_close(metrics['solid_penetration_cm'], q.new_tensor([15.]))
    after = router(FK(), q.detach()-.001*gradient/gradient.norm(), torch.tensor([0]))
    assert after.distances[0] > rows.solid.distances[0]


def test_shallow_penetration_and_clearance_have_correct_hinge():
    _, router = box_sphere(); q = torch.zeros(2, 36, dtype=torch.float64)
    q[:, 2] = torch.tensor([.599, .601]); q.requires_grad_()
    rows = router(TranslatingLinks(), q, torch.zeros(2, dtype=torch.long))
    depth = rows.depths()
    torch.testing.assert_close(depth[:, 0], torch.tensor([.001, 0.], dtype=q.dtype), atol=1.e-7, rtol=0)
    assert rows.pair['sample'].tolist() == [0]
    gradient = torch.autograd.grad(depth.sum(), q)[0]
    assert gradient[0, 2] < 0 and gradient[1].abs().sum() == 0


def test_self_collision_common_root_translation_cancels_and_relative_motion_recovers():
    data = contract([shape(0, 'first', 'sphere', (.1, .1, .1)),
                     shape(1, 'second', 'sphere', (.1, .1, .1))], [[0, 1]])
    router = SolidSceneRouter({0: data}, ('first', 'second'))
    q = torch.zeros(1, 36, dtype=torch.float64); q[0, 7] = .15; q.requires_grad_()
    ids = torch.zeros(1, dtype=torch.long); rows = router(TranslatingLinks(), q, ids)
    assert rows.pair['full_kind'].tolist() == [1]
    gradient = torch.autograd.grad(rows.distances.sum(), q)[0]
    torch.testing.assert_close(gradient[0, :3], torch.zeros(3, dtype=q.dtype), atol=1.e-12, rtol=0)
    assert gradient[0, 7] > 0
    new = router(TranslatingLinks(), q.detach()+.001*gradient, ids)
    assert new.distances[0] > rows.distances[0]


def test_recorded_cylinder_convex_loss_gradient_exits_instead_of_deepening():
    """A native curved-shape EPA failure must not become a recovery gradient."""
    import json
    from pathlib import Path
    from scipy.spatial.transform import Rotation

    fixture = Path(__file__).parents[2]/'somaforge_core'/'tests'/'fixtures'/'g1_cylinder_convex_epa_regression.json'
    data = json.loads(fixture.read_text())
    names = tuple(s['body'] for s in data['shapes'])
    case = data['cases'][1]
    local = np.array([s['transform'] for s in data['shapes']])
    rotation = np.array(case['rotations']) @ Rotation.from_quat(local[:, 3:]).as_matrix().swapaxes(-1, -2)
    position = np.array(case['positions'])-np.einsum('bij,bj->bi', rotation, local[:, :3])

    class RecordedFK:
        def link_poses(self, q, link_names):
            assert link_names == names
            first = q[:, 7:10]
            shift = torch.stack((first, torch.zeros_like(first)), 1)
            return (q[:, None, :3]+torch.as_tensor(position).to(q)+shift,
                    torch.as_tensor(rotation).to(q).expand(len(q), 2, 3, 3))

    router = SolidSceneRouter({0: data}, names)
    q = torch.zeros(1, 36, dtype=torch.float64, requires_grad=True)
    ids = torch.zeros(1, dtype=torch.long); fk = RecordedFK()
    rows = router(fk, q, ids)
    loss = rows.distances.square().sum()/2
    gradient = torch.autograd.grad(loss, q)[0]
    torch.testing.assert_close(gradient[0, :3], q.new_zeros(3), atol=1.e-12, rtol=0)
    finite = q.new_zeros(3)
    for axis in range(3):
        shift = torch.zeros_like(q); shift[0, 7+axis] = 1.e-5
        plus = router(fk, q.detach()+shift, ids).distances.square().sum()/2
        minus = router(fk, q.detach()-shift, ids).distances.square().sum()/2
        finite[axis] = (plus-minus)/2.e-5
    torch.testing.assert_close(gradient[0, 7:10], finite, atol=1.e-6, rtol=2.e-3)
    after = router(fk, q.detach()-.001*gradient/gradient.norm(), ids)
    assert after.distances[0] > rows.distances[0]+.00099


def test_disabled_pairs_have_no_penetration_or_gradient():
    data, _ = box_sphere(); data['allowed_pairs'] = []
    router = SolidSceneRouter({0: data}, ('first',))
    q = torch.zeros(1, 36, dtype=torch.float64); q[0, 2] = .45
    rows = router(TranslatingLinks(), q, torch.zeros(1, dtype=torch.long))
    assert rows.depths().sum() == 0 and len(rows.distances) == 0


def test_rigid_world_frame_does_not_change_distance_or_q_gradient():
    from scipy.spatial.transform import Rotation
    data, router = box_sphere(); shifted = copy.deepcopy(data)
    rotation = Rotation.from_euler('z', .8); origin = np.array([7., -3., 1.])
    shifted['shapes'][0]['transform'] = [*(rotation.apply([0., 0., .25])+origin), *rotation.as_quat()]
    transformed = SolidSceneRouter({0: shifted}, ('first',))
    q = torch.zeros(1, 36, dtype=torch.float64); q[0, 2] = .45; q.requires_grad_()
    ids = torch.zeros(1, dtype=torch.long); fk = TranslatingLinks()
    first = router(fk, q, ids)
    second = transformed(fk, q, ids, world_frame=(torch.tensor(origin)[None], torch.tensor(rotation.as_matrix())[None]))
    torch.testing.assert_close(first.distances, second.distances, atol=1.e-10, rtol=1.e-10)
    torch.testing.assert_close(torch.autograd.grad(first.distances.sum(), q)[0],
                               torch.autograd.grad(second.distances.sum(), q)[0], atol=1.e-10, rtol=1.e-10)


def test_float32_fk_and_world_charts_are_rigid_in_double_geometry():
    from scipy.spatial.transform import Rotation
    data, router = box_sphere()
    chart = torch.tensor(Rotation.from_euler('z', .7).as_matrix(), dtype=torch.float32)
    body = torch.tensor(Rotation.from_euler('xyz', [.3, .7, -.9]).as_matrix(), dtype=torch.float32)

    class QuantizedFK(TranslatingLinks):
        def link_poses(self, q, names):
            position, _ = super().link_poses(q, names)
            return position, body.to(q).expand(len(q), len(names), 3, 3)

    q = torch.zeros(1, 36, dtype=torch.float64); q[0, 2] = .45; q.requires_grad_()
    ids = torch.zeros(1, dtype=torch.long)
    rows = router(QuantizedFK(), q, ids, world_frame=(torch.zeros(1,3), chart[None]))
    torch.testing.assert_close(rows.distances, q.new_tensor([-.15]), atol=1.e-10, rtol=0)
    gradient = torch.autograd.grad(rows.distances.sum(), q)[0]
    torch.testing.assert_close(gradient[0,:3], q.new_tensor([0.,0.,1.]), atol=1.e-10, rtol=0)
    # Raw nonrigid matrices are invalid solver inputs, not a reason to relax
    # the witness/unit-normal check or discard a penetrating body pair.
    scene = SolidDistanceScene(data, ('first',))
    with pytest.raises(ValueError, match='proper rigid'):
        scene.query(q.detach()[:,None,:3].numpy(), body.double()[None,None].numpy())


def test_disconnected_static_solids_are_not_replaced_by_one_enclosing_hull():
    import trimesh
    a, b = trimesh.creation.box(extents=(.2, .2, .2)), trimesh.creation.box(extents=(.2, .2, .2))
    a.apply_translation((-.5, 0., 0.)); b.apply_translation((.5, 0., 0.))
    mesh = trimesh.util.concatenate((a, b))
    static = shape(0, None, 'mesh', (1., 1., 1.)); static.update(vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist())
    data = contract([static, shape(1, 'first', 'sphere', (.05, .05, .05))], [[0, 1]])
    scene = SolidDistanceScene(data, ('first',))
    assert len(scene.records) == 3
    router = SolidSceneRouter({0: data}, ('first',))
    rows = router(TranslatingLinks(), torch.zeros(1, 36, dtype=torch.float64), torch.zeros(1, dtype=torch.long))
    assert len(rows.distances) == 0


def test_open_mesh_fails_explicitly_instead_of_inventing_a_solid():
    static = shape(0, None, 'mesh', (1., 1., 1.))
    static.update(vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], faces=[[0, 1, 2]])
    data = contract([static, shape(1, 'first', 'sphere', (.1, .1, .1))], [[0, 1]])
    with pytest.raises(ValueError, match='closed convex solid'):
        SolidDistanceScene(data, ('first',))


@pytest.mark.parametrize('kind,scale', [('sphere', (.1,.1,.1)), ('box', (.1,.1,.1)),
    ('capsule', (.1,.1,0.)), ('cylinder', (.1,.1,0.)), ('ellipsoid', (.1,.08,.06))])
def test_batched_solver_preserves_material_gradient(kind, scale):
    data = contract([shape(0, None, 'box', (.5,.5,.25), (0.,0.,.25)),
                     shape(1, 'first', kind, scale)], [[0,1]])
    values = []
    for backend in ('python', 'native_batch'):
        router = SolidSceneRouter({0:data}, ('first',), backend=backend)
        q = torch.zeros(3,36,dtype=torch.float64)
        q[:,2] = torch.tensor([.45,.3,2.]); q.requires_grad_()
        rows = router(TranslatingLinks(), q, torch.zeros(3,dtype=torch.long))
        gradient, = torch.autograd.grad(rows.depths().square().sum(), q)
        values.append((rows.pair, gradient))
    for key in values[0][0]:
        torch.testing.assert_close(values[0][0][key], values[1][0][key], atol=0, rtol=0)
    torch.testing.assert_close(values[0][1], values[1][1], atol=0, rtol=0)
