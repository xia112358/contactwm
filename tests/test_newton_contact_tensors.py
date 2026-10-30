"""Reader semantics with different raw/solver capacities and two worlds."""
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from somaforge_core.newton_contact_tensors import NewtonContactTensorReader


def fixture(monkeypatch):
    monkeypatch.setitem(sys.modules, 'warp', NS(to_torch=lambda value: value))
    model = NS(device='cpu', world_count=2, body_count=4, shape_count=5,
        body_label=['left_ankle_roll_link', 'right_ankle_roll_link']*2,
        shape_type=torch.tensor([8, 3, 3, 3, 3]), shape_body=torch.tensor([-1, 0, 1, 2, 3]),
        shape_margin=torch.zeros(5))
    # Two native faces share an edge. A side normal must select the side even
    # when the emitting triangle belongs to the top.
    faces = [dict(surface=0, normal_w=[0, 0, 1], triangle_indices=[0],
                  triangles_w=[[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]),
             dict(surface=1, normal_w=[0, -1, 0], triangle_indices=[1],
                  triangles_w=[[[0, 0, 0], [1, 0, 0], [1, 0, -1]]])]
    contact = NS(dist=torch.tensor([.01, .01, float('nan'), float('nan')]),
        includemargin=torch.tensor([.02, .02, 0., 0.]), type=torch.tensor([1, 1, 0, 0]),
        efc_address=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [-1]*4, [-1]*4]),
        worldid=torch.tensor([0, 1, -1, -1]), geom=torch.tensor([[0, 1], [0, 2], [-1, -1], [-1, -1]]),
        dim=torch.tensor([3, 3, 0, 0]), frame=torch.eye(3).repeat(4, 1, 1))
    contact.frame[:, 0] = torch.tensor([0., 0., 1.])
    solver = NS(mjw_data=NS(contact=contact, nacon=torch.tensor([2])),
        mjw_model=NS(opt=NS(cone=0)), mjc_geom_to_newton_shape=torch.tensor([[0, 1, 2], [0, 3, 4]]),
        _contact_tid_to_cid=torch.tensor([1, 0, -1, -1, -1, -1]))
    raw = NS(rigid_contact_count=torch.tensor([2]), rigid_contact_shape0=torch.zeros(8, dtype=torch.long),
        rigid_contact_shape1=torch.tensor([4, 1, -1, -1, -1, -1, -1, -1]))
    for side in (0, 1):
        setattr(raw, f'rigid_contact_point{side}', torch.tensor([.5, 0, side*.01]).repeat(8, 1))
        setattr(raw, f'rigid_contact_offset{side}', torch.zeros(8, 3))
        setattr(raw, f'rigid_contact_margin{side}', torch.zeros(8))
    capture = NS(keys=torch.tensor([(4 << 4) | 1, (1 << 4) | 1, -1, -1, -1, -1, -1, -1]),
        shape_bits=3, sub_key_bits=4, analytic_sphere_types=True)
    reader = NewtonContactTensorReader(model, solver, {0: 0, 1: 0, 2: 1, 3: 1}, {0: faces})
    reader.fixture_state = NS(body_q=torch.tensor([0., 0., 0., 0., 0., 0., 1.]).repeat(4, 1))
    return reader, contact, raw, capture


def test_world_mapping_padding_and_source_permutation(monkeypatch):
    reader, contact, raw, capture = fixture(monkeypatch)
    result = reader.read(reader.fixture_state, raw, capture)
    reader.validate(result)
    assert result['contact_part_mask'].nonzero().tolist() == [[0, 0], [1, 1]]
    assert result['contact_surface'].tolist() == [[0, -1, -1, -1, -1, -1], [-1, 0, -1, -1, -1, -1]]
    assert result['source_key'][:2].tolist() == [(1 << 23) | 8, (4 << 23) | 8]


def test_side_primary_excluded_without_promoting_top(monkeypatch):
    reader, contact, raw, capture = fixture(monkeypatch)
    contact.frame[0, 0] = torch.tensor([0., -1., 0.])
    result = reader.read(reader.fixture_state, raw, capture)
    reader.validate(result)
    assert result['primary_surface'][0] == 1
    assert not result['contact_part_mask'][0].any()


@pytest.mark.parametrize('fault', ['unallocated', 'source', 'coverage', 'capacity', 'mapping'])
def test_invalid_solver_or_source_state_fails_closed(monkeypatch, fault):
    reader, contact, raw, capture = fixture(monkeypatch)
    if fault == 'unallocated': contact.efc_address[0, 3] = -1
    if fault == 'source': capture.keys[0] = -1
    if fault == 'coverage': reader.solver._contact_tid_to_cid[0] = 0
    if fault == 'capacity': raw.rigid_contact_count[0] = 7
    if fault == 'mapping': contact.worldid[0] = 9
    with pytest.raises(ValueError, match='Invalid device Newton snapshot'):
        reader.validate(reader.read(reader.fixture_state, raw, capture))


def test_actual_margin_boundary_and_constraint_type(monkeypatch):
    reader, contact, raw, capture = fixture(monkeypatch)
    contact.dist[0] = contact.includemargin[0]
    contact.type[1] = 0
    result = reader.read(reader.fixture_state, raw, capture)
    reader.validate(result)
    assert not result['active'].any()
    assert not result['contact_part_mask'].any()


def test_side_interior_cannot_promote_top_even_with_inward_normal(monkeypatch):
    reader, contact, raw, capture = fixture(monkeypatch)
    capture.keys[:2] = torch.tensor([(4 << 4) | 3, (1 << 4) | 3])
    raw.rigid_contact_point0[:] = torch.tensor([.7, 0., -.2])
    contact.frame[:, 0] = torch.tensor([0., 1., 1e-7])
    result = reader.read(reader.fixture_state, raw, capture)
    reader.validate(result)
    assert result['primary_surface'][:2].tolist() == [1, 1]
    assert result['active'][:2].all() and result['constraint_allocated'][:2].all()
    assert not result['contact_part_mask'].any()
    torch.testing.assert_close(result['geometry_point0_w'][:2], raw.rigid_contact_point0[:2])


def test_missing_actual_witness_fails_without_distance_fallback(monkeypatch):
    reader, contact, raw, capture = fixture(monkeypatch)
    with pytest.raises(ValueError, match='Actual terrain geometry witness required'):
        reader.read(None, raw, capture)
