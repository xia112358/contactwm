import pytest
import torch

from contact_solver.contact_material_skin import ContactMaterialSkin
from contact_solver.constraint_penalty import constraint_penalty
from contact_solver.contact_surface_interval import finite_face_band_cost, convex_face_tangent_distance


class Regions:
    valid = torch.ones(6, 4, dtype=torch.bool)
    names = (('a', 'b', 'c', 'd'),)*6

    def witness_regions(self, fk, q, sample, part, points):
        return torch.zeros_like(part)


class RigidFK:
    def link_poses(self, q, names):
        c, s = q[:, 3].cos(), q[:, 3].sin()
        zero, one = c*0, c*0+1
        rot = torch.stack((c, zero, s, zero, one, zero, -s, zero, c), -1).reshape(-1, 1, 3, 3)
        positions = q[:, None, :3].expand(-1, len(names), -1).clone()
        if len(names) > 1:
            positions[:, 1, 2] += .2
        return positions, rot.expand(-1, len(names), -1, -1)


def observation(q, link=None):
    n = len(q)
    links = torch.zeros(n, dtype=torch.long) if link is None else link
    local = q.new_tensor([0., 0., -.1]).expand(n, 3)
    names = ('foot', 'foot_attachment')
    pos, rot = RigidFK().link_poses(q, names)
    point = pos[torch.arange(n), links]+torch.einsum('nij,nj->ni', rot[torch.arange(n), links], local)
    return dict(schema='newton_device_witness_batch_v1', link_names=names, pairs=dict(
        sample=torch.arange(n), part=torch.zeros(n, dtype=torch.long), body_link0=torch.full((n,), -1),
        body_link1=links, geometry_point1_w=point, eligible=torch.ones(n, dtype=torch.bool),
        active=torch.ones(n, dtype=torch.bool), constraint_allocated=torch.ones(n, dtype=torch.bool),
        task_pair=torch.ones(n, dtype=torch.bool), upward=torch.ones(n, dtype=torch.bool)))


def make_skin():
    skin = ContactMaterialSkin(Regions(), ('foot', 'foot_attachment'))
    q = torch.tensor([[0., 0., .1, 0.]], dtype=torch.float64)
    skin.add(RigidFK(), Regions(), q, observation(q), torch.tensor([7]), torch.tensor([True]))
    skin.finalize()
    return skin


def distance(skin, q):
    scene = dict(box_center=q.new_tensor([[0., 0., -.5]]).expand(len(q), -1),
        box_rotation=torch.eye(3).to(q).expand(len(q), 3, 3),
        box_half_extents=q.new_tensor([[1., 1., .5]]).expand(len(q), -1), ground_height=q.new_zeros(len(q)))
    return skin.region_distance(RigidFK(), q, torch.zeros(len(q), 6, dtype=torch.long),
        scene, q.new_full((len(q),), .02), q.new_zeros(len(q), 6, 4))[:, 0, 0]


def test_loss_uses_actual_contact_skin_instead_of_a_top_point_already_near_plane():
    skin = make_skin()
    # The upper surface can be at z=0 while the real sole is buried by 10cm.
    # A nearest-any-vertex interval is zero; this material residual is not.
    q = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    loss = constraint_penalty(distance(skin, q))
    assert loss.item() > 0
    gradient = torch.autograd.grad(loss.sum(), q)[0]
    assert gradient[0, 2] < 0
    # No correction is requested inside the native-width zero interval.
    above = q.detach().clone(); above[:, 2] = .11; above.requires_grad_()
    assert distance(skin, above).item() == 0
    assert torch.autograd.grad(distance(skin, above).sum(), above)[0].abs().sum() == 0


def test_finite_difference_matches_true_gradient_and_materials_do_not_switch_at_normal_zero():
    skin = make_skin()
    for angle in (.31, torch.pi/2-1e-7, torch.pi/2+1e-7):
        q = torch.tensor([[0., 0., -.1, angle]], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda x: constraint_penalty(distance(skin, x)), (q,))
    a = torch.tensor([[0., 0., -.1, torch.pi/2-1e-7]], dtype=torch.float64)
    b = a.clone(); b[:, 3] += 2e-7
    assert abs(float(distance(skin, a)-distance(skin, b))) < .001


def test_source_split_selection_excludes_validation_points_and_freezes_bank():
    skin = ContactMaterialSkin(Regions(), ('foot', 'foot_attachment'))
    q = torch.tensor([[0., 0., .1, 0.], [0., 0., .4, 0.]], dtype=torch.float64)
    obs = observation(q)
    obs['pairs']['geometry_point1_w'][1, 2] += .9
    skin.add(RigidFK(), Regions(), q, obs, torch.tensor([7, 8]), torch.tensor([True, False]))
    skin.finalize()
    assert skin.source_ids == {7} and skin.source_contact_count == 1
    torch.testing.assert_close(skin.local_0_0, q.new_tensor([[0., 0., -.1]]))
    assert not skin.state_dict() and not list(skin.parameters())
    with pytest.raises(ValueError, match='finalized'):
        skin.add(RigidFK(), Regions(), q, obs, torch.tensor([7, 8]), torch.tensor([True, False]))


def test_actual_attached_link_and_duplicate_sources_preserve_geometry():
    skin = ContactMaterialSkin(Regions(), ('foot', 'foot_attachment'))
    q = torch.zeros(2, 4, dtype=torch.float64)
    obs = observation(q, torch.ones(2, dtype=torch.long))
    skin.add(RigidFK(), Regions(), q, obs, torch.tensor([1, 2]), torch.ones(2, dtype=torch.bool))
    skin.finalize()
    assert skin.link_0_0.tolist() == [1]
    # Attachment's own position is 20cm higher, so its material is at +10cm.
    assert float(distance(skin, q[:1])) > 0
    shifted = q[:1].clone(); shifted[:, 2] = -.09
    assert distance(skin, shifted).item() == 0


def test_unallocated_or_nonterrain_source_is_rejected_and_unknown_regions_are_explicit():
    q = torch.zeros(1, 4, dtype=torch.float64)
    for field, value, message in [('constraint_allocated', False, 'Unallocated'), ('body_link0', 0, 'terrain'),
                                  ('upward', False, 'terrain')]:
        skin = ContactMaterialSkin(Regions(), ('foot', 'foot_attachment'))
        obs = observation(q); obs['pairs'][field][:] = value
        with pytest.raises(ValueError, match=message):
            skin.add(RigidFK(), Regions(), q, obs, torch.tensor([1]), torch.tensor([True]))
    skin = make_skin()
    assert skin.known.sum() == 1 and not skin.known[1:].any()
    assert skin.contract()['contact_truth_unchanged'] and not skin.contract()['network_input']


def test_native_width_band_includes_edge_contact_without_demanding_exact_footprint():
    # An actual source knee witness was 19.54mm above and 1.93mm beyond the
    # top edge: its 19.64mm face distance is inside the native 20mm margin.
    coordinate = torch.tensor([[.01954, .00193, 0.]], dtype=torch.float64, requires_grad=True)
    def cost(x):
        return finite_face_band_cost(x[:, 0], x[:, 1:].abs(), .02)
    value = cost(coordinate)
    assert value.item() == 0 and torch.autograd.grad(value.sum(), coordinate)[0].abs().sum() == 0
    separated = torch.tensor([[.025, .01, .005]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(cost(separated), (separated.norm(dim=-1)-.02).square())
    assert torch.autograd.gradcheck(cost, (separated,))


def test_band_retains_upward_recovery_below_a_finite_face_and_has_finite_zero_gradient():
    embedded = torch.tensor([[-.09, .03, .04]], dtype=torch.float64, requires_grad=True)
    def cost(x):
        return finite_face_band_cost(x[:, 0], x[:, 1:].abs(), .02)
    gradient = torch.autograd.grad(cost(embedded).sum(), embedded)[0]
    assert gradient[0, 0] < 0 and (gradient[0, 1:] > 0).all()
    assert torch.autograd.gradcheck(cost, (embedded,))
    zero = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.grad(cost(zero).sum(), zero)[0].eq(0).all()


def test_fixed_material_skin_uses_the_finite_face_edge_interval():
    skin = make_skin()
    q = torch.tensor([[1.00193, 0., .11954, 0.]], dtype=torch.float64, requires_grad=True)
    scene = dict(box_center=q.new_tensor([[0., 0., -.5]]), box_rotation=torch.eye(3).to(q)[None],
        box_half_extents=q.new_tensor([[1., 1., .5]]), ground_height=q.new_zeros(1))
    value = skin.region_distance(RigidFK(), q, torch.ones(1, 6, dtype=torch.long),
        scene, q.new_tensor([.02]), q.new_zeros(1, 6, 4))[:, 0, 0]
    assert value.item() == 0 and torch.autograd.grad(value.sum(), q)[0].eq(0).all()


def test_generic_convex_face_uses_nearest_edge_distance_and_true_fk_derivative():
    vertices = [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]
    n,e,o = [0., 0., 1.], [[0., -1., 0.], [2**-.5, 2**-.5, 0.], [-1., 0., 0.]], [0., 2**.5, 0.]
    points = torch.tensor([[2., 2., .03], [3., 0., .04], [.5, .5, .01]], dtype=torch.float64, requires_grad=True)
    def distance(p):
        return convex_face_tangent_distance(p, p.new_tensor(n), p.new_tensor(vertices),
            p.new_tensor(e), p.new_tensor(o))
    torch.testing.assert_close(distance(points).squeeze(-1), points.new_tensor([2**.5, 1., 0.]))
    # Perturb a unique nearest edge away from its endpoint/tie boundaries.
    trial = points[:1].clone().detach().requires_grad_()
    assert torch.autograd.gradcheck(lambda p: finite_face_band_cost(p[:, 2], distance(p), .02), (trial,))
