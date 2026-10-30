"""Versioned native contact response, independent of robot collision geometry."""
import json

PROFILES = ('legacy', 'firm_v2', 'manifold_v1', 'manifold_v2', 'manifold_soft_v1', 'manifold_balanced_v1')


def contract(profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown contact response: {profile}')
    soft = profile == 'manifold_soft_v1'
    balanced = profile == 'manifold_balanced_v1'
    return dict(schema='native_contact_response_v1', profile=profile,
                control_dt=.02, substeps=4 if profile == 'legacy' or soft or balanced else 10,
                solref=None if profile=='legacy' else [.02 if soft else .01 if balanced else .004, 1.],
                solimp=None if profile=='legacy' else ([.9,.95,.001,.5,2.] if soft else [.95,.99,.001,.5,2.] if balanced else [.99,.999,.001,.5,2.]),
                enable_multiccd=profile.startswith('manifold'),
                manifold_normal='epa_witness_manifold_normal_v1' if profile in ('manifold_v2','manifold_soft_v1','manifold_balanced_v1') else 'upstream',
                geometry='unchanged', margins='unchanged', gaps='unchanged')



def metadata(profile):
    return json.dumps(contract(profile), sort_keys=True)


def apply(env, profile):
    """Persist through Newton model updates; never patch only a GPU mirror."""
    import torch
    import warp as wp
    from newton import ModelFlags
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
    from .contact_manifold_normals import validate_runtime
    spec = contract(profile)
    validate_runtime(profile)
    manager = env.sim.backend_manager
    if abs(float(manager.get_solver_dt()) - spec['control_dt']/spec['substeps']) > 1e-9:
        raise ValueError('Contact response and actual solver step disagree')
    model, solver = manager.get_model(), manager._solver
    import mujoco
    multiccd = not bool(solver.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    if multiccd != spec['enable_multiccd']:
        raise ValueError('Contact response and native multi-contact flag disagree')
    if profile == 'legacy':
        env.contact_response = profile
        return
    wp.to_torch(model.mujoco.solref_mode).fill_(SOLREF_MODE_RAW)
    wp.to_torch(model.mujoco.solref)[:] = torch.tensor(spec['solref'], device=env.device)
    wp.to_torch(model.mujoco.geom_solimp)[:] = torch.tensor(spec['solimp'], device=env.device)
    manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
    solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
    for field, value in [('geom_solref', spec['solref']), ('geom_solimp', spec['solimp'])]:
        actual = wp.to_torch(getattr(solver.mjw_model, field))
        expected = torch.tensor(value, device=actual.device, dtype=actual.dtype).expand_as(actual)
        if not torch.allclose(actual, expected, atol=1e-7, rtol=0):
            raise RuntimeError(f'Native solver did not apply {field}')
    env.contact_response = profile


def validate_identity(checkpoint, expected):
    """Missing historical metadata means legacy; check both transfer directions."""
    if checkpoint.get('production_compatible') is False or 'contact_response_diagnostic' in checkpoint:
        raise ValueError('Diagnostic contact-response transfer is not a production training input')
    old = checkpoint.get('contact_response_contract', metadata('legacy'))
    new = expected.get('contact_response_contract', metadata('legacy'))
    if old != new:
        raise ValueError('Contact-response physics changed: old checkpoints/data cannot resume '
                         'as matching physics; use a fresh run or an explicit diagnostic transfer')
