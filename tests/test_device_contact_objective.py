import pytest
import torch

from contact_solver.device_contact_objective import DeviceWitnessRows, acceptance, reduce_groups, realization
from contact_solver.interaction_acceptance import contact_acceptance
from contact_solver.newton_witness_loss import full_body_violation


class Bodies:
    def link_poses(self, q, names):
        return torch.stack([q[:, :3] if name == 'a' else q[:, 7:10] for name in names], 1), torch.eye(3).to(q).expand(len(q), len(names), 3, 3)


def observation(q, *, empty=False):
    n = 0 if empty else 2
    tensor = lambda value, dtype=None: torch.tensor(value, dtype=dtype, device=q.device)[:n]
    p = dict(sample=tensor([0, 0], torch.long), part=tensor([-1, -1], torch.long),
        task_pair=tensor([False, False]), dist=tensor([-.02, -.02], q.dtype),
        geometry_point0_w=q.new_zeros(n, 3), geometry_point1_w=q.new_zeros(n, 3),
        normal_w=tensor([[0., 0., 1.]]*2, q.dtype), body_link0=tensor([-1, 0], torch.long),
        body_link1=tensor([0, 1], torch.long), full_kind=tensor([0, 1], torch.long),
        primary_surface=tensor([0, -1], torch.long), upward=tensor([True, False]),
        active=tensor([True, True]), constraint_allocated=tensor([True, True]),
        includemargin=tensor([.02, .02], q.dtype), eligible=tensor([False, False]))
    return dict(schema='newton_device_witness_batch_v1', pairs=p, link_names=('a', 'b'),
        contact_part_mask=torch.zeros(len(q), 6, dtype=torch.bool),
        contact_surface=torch.full((len(q), 6), -1, dtype=torch.long),
        configured_margin=q.new_full((len(q),), .02))


def test_fullbody_tie_and_two_sided_gradient_match_legacy():
    q = torch.zeros(2, 36, dtype=torch.float64, requires_grad=True)
    obs = observation(q)
    rows = DeviceWitnessRows(Bodies(), q, obs)
    actual, invalid = rows.full_body()
    witness = dict(body_name0=None, body_name1='a', point0_w=[0, 0, 0], point1_w=[0, 0, 0], normal_w=[0, 0, 1], dist=-.02)
    legacy = full_body_violation(Bodies(), q, [dict(worst_terrain=witness,
        worst_self=dict(witness, body_name0='a', body_name1='b')), dict(worst_terrain=None, worst_self=None)])
    torch.testing.assert_close(actual, legacy)
    ga = torch.autograd.grad(actual.sum(), q, retain_graph=True)[0]
    gl = torch.autograd.grad(legacy.sum(), q)[0]
    torch.testing.assert_close(ga, gl)
    assert invalid.sum() == 0


def test_group_ties_split_gradients_and_missing_group_stays_zero():
    value = torch.tensor([.01, .01], requires_grad=True)
    result = reduce_groups(value, torch.tensor([0, 0]), torch.tensor([True, True]), 2, minimum=True)
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([.5, .5]))
    assert result[1] == 0


def test_material_witness_gradient_uses_requested_coordinate_frame():
    q = torch.zeros(1, 36, dtype=torch.float64, requires_grad=True)
    obs = observation(q)
    obs['pairs']['normal_w'][:] = q.new_tensor([1., 0., 0.])
    origin = q.new_tensor([[2., 3., 4.]])
    basis = q.new_tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    rows = DeviceWitnessRows(Bodies(), q, obs, world_frame=(origin, basis))
    gradient = torch.autograd.grad(rows.distances[0], q)[0]
    torch.testing.assert_close(gradient[0, :3], q.new_tensor([0., -1., 0.]))


def test_empty_query_missing_intent_and_finite_zero_gradient():
    q = torch.zeros(2, 36, requires_grad=True)
    obs = observation(q, empty=True)
    active = torch.ones(2, 6, dtype=torch.bool)
    loss, metrics = realization(DeviceWitnessRows(Bodies(), q, obs), active, torch.zeros(2, 6, dtype=torch.long))
    assert metrics['newton_missing_intended_mask'].all()
    assert not metrics['newton_contact_accepted'].any()
    assert torch.isfinite(loss).all() and loss.sum() == 0
    loss.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() == 0


def test_invalid_fullbody_keeps_distance_detaches_gradient_and_audits(tmp_path):
    q = torch.zeros(1, 36, requires_grad=True)
    obs = observation(q)
    obs['pairs']['normal_w'][:] = float('nan')
    rows = DeviceWitnessRows(Bodies(), q, obs)
    active = torch.zeros(1, 6, dtype=torch.bool)
    audit = tmp_path/'device_invalid.jsonl'
    loss, metrics = realization(rows, active, torch.full((1, 6), -1), invalid_policy='detach', audit_path=audit)
    assert metrics['newton_invalid_fullbody_witnesses'].item() == 2
    assert metrics['newton_penetration_cm'].item() == pytest.approx(2)
    loss.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() == 0
    assert 'witnesses' in audit.read_text() and 'transformed_normal' in audit.read_text()


def test_acceptance_inspects_every_extra_pair_with_actual_margin():
    q = torch.zeros(1, 36)
    obs = observation(q)
    p = obs['pairs']; p['part'][:] = torch.tensor([0, 1]); p['eligible'][:] = True
    p['primary_surface'][:] = 0; p['dist'][:] = torch.tensor([.01, .0195])
    active = torch.tensor([[True, False, False, False, False, False]])
    surface = torch.tensor([[0, -1, -1, -1, -1, -1]])
    records = [dict(part=i, surface=0, dist=float(p['dist'][i]), includemargin=float(p['includemargin'][i]), allocated=True) for i in range(2)]
    expected = contact_acceptance(records, active[0].tolist(), surface[0].tolist())
    actual = acceptance(obs, active, surface)
    assert actual['contact_accepted'].item() == expected['contact_accepted']
    assert actual['ignored'].item() == expected['ignored_extra_pairs'] == 1
    p['dist'][1] = .018
    assert not acceptance(obs, active, surface)['contact_accepted'].item()


def test_multibody_loss_supervises_smaller_collision_and_ignores_duplicate_rows():
    q = torch.zeros(1, 36, requires_grad=True)
    obs = observation(q)
    p = obs['pairs']
    p['body_link0'][:] = -1
    p['body_link1'][:] = torch.tensor([0, 1])
    p['full_kind'][:] = 0
    p['dist'][:] = torch.tensor([-.02, -.01])
    rows = DeviceWitnessRows(Bodies(), q, obs, collision_aggregation='body_mean_plus_max')
    loss = rows.body_collision_loss()
    gradient = torch.autograd.grad(loss.sum(), q)[0]
    assert gradient[0, 2] < 0 and gradient[0, 9] < 0
    duplicate = dict(obs, pairs={k: v.repeat_interleave(3, dim=0) for k, v in p.items()})
    other = DeviceWitnessRows(Bodies(), q, duplicate, collision_aggregation='body_mean_plus_max')
    torch.testing.assert_close(loss, other.body_collision_loss())
