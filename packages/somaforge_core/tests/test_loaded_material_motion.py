import numpy as np
import pytest
import torch

from somaforge_core.loaded_material_motion import (
    loaded_material_steps, material_path_report, unknown_material_path)


def measure(pos, rot, local, **kwargs):
    n = len(local)
    return loaded_material_steps(pos, rot, frames=np.zeros(n, int), links=np.zeros(n, int),
        parts=np.zeros(n, int), local=local, normals=np.tile([0., 0., 1.], (n, 1)),
        loads=kwargs.get('loads', np.ones(n)), surfaces=kwargs.get('surfaces', np.zeros(n, int)))


def test_loaded_pivot_rotation_is_free_but_opposing_loaded_points_do_not_cancel():
    pos = torch.zeros(2, 1, 3, dtype=torch.double, requires_grad=True)
    rot = torch.eye(3, dtype=torch.double).repeat(2, 1, 1, 1)
    rot[1, 0] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    pivot = measure(pos, rot, [[0., 0., 0.]])
    assert pivot[0, 0] == 0
    pivot[0, 0].backward()
    assert torch.isfinite(pos.grad).all() and not pos.grad.any()
    points = measure(pos, rot, [[1., 0., 0.], [-1., 0., 0.]])
    assert points[0, 0].item() == pytest.approx(2**.5)
    assert points[0, 1:].isnan().all()
    # A stationary third point must not hide other loaded points' movement.
    assert measure(pos, rot, [[0., 0., 0.], [1., 0., 0.]])[0, 0] == 1


def test_tangential_translation_has_correct_finite_gradient_and_normal_is_excluded():
    pos = torch.tensor([[[0., 0., 0.]], [[.03, .04, .9]]], dtype=torch.double, requires_grad=True)
    rot = torch.eye(3, dtype=torch.double).repeat(2, 1, 1, 1)
    value = measure(pos, rot, [[0., 0., 0.]])[0, 0]
    assert value.item() == pytest.approx(.05)
    value.backward()
    np.testing.assert_allclose(pos.grad[1, 0], [.6, .8, 0.])
    assert torch.autograd.gradcheck(lambda p: measure(p, rot, [[0., 0., 0.]])[0, 0], (pos,))


def test_path_excludes_known_unloaded_samples_and_never_passes_unknown_samples():
    steps = np.full((3, 6), np.nan)
    steps[:, 0] = [.02, np.nan, .03]
    known = np.ones_like(steps, bool)
    bearing = np.isfinite(steps)
    phase = [dict(start=0, end=3, part=0)]
    result = material_path_report(steps, phase, load_known=known, load_bearing=bearing)
    assert result['within_budget'] is True
    assert result['phases'][0]['material_tangent_path_m'] == pytest.approx(.05)
    known[1, 0] = False
    assert material_path_report(steps, phase, load_known=known, load_bearing=bearing)['within_budget'] is None
    known[1, 0] = True
    bearing[1, 0] = True
    assert material_path_report(steps, phase, load_known=known, load_bearing=bearing)['within_budget'] is None
    steps[:, 0] = [.02, .02, .03]
    assert material_path_report(steps, phase, load_known=known, load_bearing=bearing)['within_budget'] is False


def test_missing_trajectory_and_cross_surface_evidence_are_explicit():
    pos = torch.zeros(2, 1, 3, dtype=torch.double)
    rot = torch.eye(3, dtype=torch.double).repeat(2, 1, 1, 1)
    with pytest.raises(ValueError, match='adjacent-pose'):
        measure(pos[:1], rot[:1], [[0., 0., 0.]])
    with pytest.raises(ValueError, match='different terrain surfaces'):
        measure(pos, rot, [[0., 0., 0.], [0., 0., 0.]], surfaces=[0, 1])
    status = unknown_material_path()
    assert status['within_budget'] is None and status['material_tangent_path_m'] is None
    assert status['actual_support_status'] == 'unknown'
