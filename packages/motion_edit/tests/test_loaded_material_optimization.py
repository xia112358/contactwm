import numpy as np
import pytest
import torch

from somaforge_core.loaded_material_motion import (
    loaded_material_steps, tangent_material_rms, phase_added_motion, source_motion_allowances)


class FK:
    def link_poses(self, q, names):
        angle = q[:, 3]; c, s = angle.cos(), angle.sin(); z = c*0; one = z+1
        rotation = torch.stack((c, -s, z, s, c, z, z, z, one), -1).reshape(-1, 1, 3, 3)
        return q[:, None, :3], rotation


def reference(q, *, local=None, frames=None):
    frames = np.arange(len(q)-1) if frames is None else np.asarray(frames)
    local = np.zeros((len(frames), 3)) if local is None else np.asarray(local)
    n = len(frames)
    sites = dict(frame=frames, part=np.zeros(n, int), surface=np.zeros(n, int),
        body=np.asarray(['foot']*n), local=local, normal=np.tile([0., 0., 1.], (n, 1)), weight=np.ones(n))
    pos, rot = FK().link_poses(q, ['foot'])
    steps = loaded_material_steps(pos, rot, frames=frames, links=np.zeros(n, int), parts=sites['part'],
        local=local, normals=sites['normal'], loads=sites['weight'], surfaces=sites['surface']).detach().numpy()
    known = np.ones_like(steps, bool); bearing = np.isfinite(steps)
    phases = [dict(event_id='keep', start=0, end=len(q)-1, part=0, surface=0)]
    calibration = source_motion_allowances(steps, phases, load_known=known, load_bearing=bearing)
    return dict(sites=sites, phases=phases, source_steps=steps,
                load_known=known, load_bearing=bearing, calibration=calibration)


def test_dense_rms_matches_shared_material_steps_and_analytic_gradient():
    delta = np.asarray([[[0., 0, 0], [.02, 0, 0], [-.02, 0, 0]]], np.float32)
    normals = np.broadcast_to([0., 0, 1.], delta.shape).astype(np.float32)
    weights = np.asarray([[1., 2., 2.]], np.float32)
    expected = (.0004*4/5)**.5
    value = tangent_material_rms(torch.tensor(delta), torch.tensor(normals), torch.tensor(weights), xp=torch)
    assert value.item() == pytest.approx(expected)
    function = lambda x: tangent_material_rms(x, torch.tensor(normals), torch.tensor(weights), xp=torch).sum()
    assert np.isfinite(torch.autograd.functional.jacobian(function, torch.zeros_like(torch.tensor(delta))).numpy()).all()
    derivative = torch.autograd.functional.jacobian(function, torch.tensor(delta)).numpy()
    analytic = delta*weights[..., None]/(weights.sum(-1)[..., None, None]*expected)
    np.testing.assert_allclose(derivative, analytic, atol=1.e-6)
