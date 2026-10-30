"""A shared C2 pose curve and constrained, motion-aware population registration.

Interpolation transports observations; it never supplies contact truth. The
unit-quaternion lift is continuous through heading wraps and commutes with a
rigid change of world frame. All body channels use exactly the same clock.
"""
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline


class ContinuousMotion:
    def __init__(self, motion):
        q = np.asarray(motion, float)
        if q.ndim != 2 or q.shape[1] != 36 or len(q) < 3 or not np.isfinite(q).all():
            raise ValueError('Expected finite canonical pose sequence')
        if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-4):
            raise ValueError('Expected unit root quaternion')
        lifted = q.copy()
        for i in range(1, len(q)):
            if lifted[i-1, 3:7] @ lifted[i, 3:7] < 0:
                lifted[i, 3:7] *= -1
        self.curve = CubicSpline(np.arange(len(q)), lifted)
        self.n = len(q)

    def evaluate(self, times, rate=None, acceleration=None):
        t = np.asarray(times, float)
        if not np.isfinite(t).all() or np.any((t < 0) | (t > self.n-1)):
            raise ValueError('Pose clock outside source duration')
        q, v, a = [self.curve(t, order) for order in range(3)]
        u, du, ddu = [x[..., 3:7].copy() for x in (q, v, a)]
        norm = np.linalg.norm(u, axis=-1, keepdims=True)
        if np.any(norm < .5):
            raise ValueError('Ambiguous quaternion interpolation; source sampling is insufficient')
        dot = (u*du).sum(-1, keepdims=True)
        q[..., 3:7] = u/norm
        v[..., 3:7] = du/norm-u*dot/norm**3
        a[..., 3:7] = (ddu/norm-2*du*dot/norm**3
            -u*((du*du).sum(-1, keepdims=True)+(u*ddu).sum(-1, keepdims=True))/norm**3
            +3*u*dot**2/norm**5)
        if rate is not None:
            r = np.asarray(rate)[..., None]
            bending = 0 if acceleration is None else np.asarray(acceleration)[..., None]
            a = a*r*r+v*bending
            v = v*r
        return q, v, a


def clock_basis(frame_count, resolution):
    """Quintic displacements: fixed endpoints, unit rate and zero clock bend."""
    if frame_count < 3 or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('Invalid clock resolution')
    knots = np.unique(np.r_[np.arange(0, frame_count-1, max(1, int(resolution))), frame_count-1.])
    values = np.eye(len(knots))
    zero = np.zeros(len(knots))
    curve = make_interp_spline(knots, values, k=5,
        bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
    return curve(np.arange(frame_count))[:, 1:-1], knots, curve


def clock_rate_certificate(curve):
    """Bernstein coefficients bound the quartic rate throughout every span.

    Checking frame differences alone can miss a reversal between samples. A
    polynomial stays within the convex hull of these coefficients everywhere.
    """
    from math import comb
    knots = np.unique(curve.t)
    u = np.linspace(0, 1, 5)
    bernstein = np.array([[comb(4,k)*t**k*(1-t)**(4-k) for k in range(5)] for t in u])
    grid = knots[:-1,None]+np.diff(knots)[:,None]*u
    values = curve(grid, nu=1)[...,1:-1]
    return np.einsum('ij,sjk->sik', np.linalg.inv(bernstein), values).reshape(-1,values.shape[-1])


def pose_correction_basis(frame_count, resolution):
    """Constrain the final fit at the same boundaries as signal processing.

    Continuous endpoint derivatives and first/last sampled velocities remain
    fixed. Null-space constraints preserve the spline itself rather than crop
    individual samples of an otherwise unconstrained correction.
    """
    from scipy.linalg import null_space
    basis, knots, _ = clock_basis(frame_count,resolution)
    if basis.shape[1]:
        basis = basis @ null_space(basis[[1,-2]],rcond=1e-10)
    return basis, knots


def certify_clocks(mappings, knots, controls, rate_bounds):
    """Independently reconstruct and certify the exact clocks used for transport."""
    mappings, knots, controls = map(lambda a: np.asarray(a,float), (mappings,knots,controls))
    e,n = mappings.shape
    if (controls.shape != (e,len(knots)) or knots[0]!=0 or knots[-1]!=n-1
            or not np.isfinite(controls).all() or np.any(controls[:,[0,-1]] != 0)):
        raise ValueError('Invalid continuous clock evidence')
    zero=np.zeros(len(knots))
    basis=make_interp_spline(knots,np.eye(len(knots)),k=5,
        bc_type=([(1,zero),(2,zero)],[(1,zero),(2,zero)]))
    reconstructed=np.arange(n)[None]+controls@basis(np.arange(n)).T
    if not np.allclose(reconstructed,mappings,atol=1e-8,rtol=0):
        raise ValueError('Continuous certificate belongs to different sampled clocks')
    rates=1+controls[:,1:-1]@clock_rate_certificate(basis).T
    low,high=rate_bounds
    if np.any(rates<low-1e-8) or np.any(rates>high+1e-8):
        raise ValueError('Continuous clock rate exceeds its certified bounds')
    return dict(passed=True,schema='continuous_transport_validation_v1',
        rate_bounds=list(rate_bounds),continuous_rate_ranges=[[float(r.min()),float(r.max())] for r in rates],
        endpoint_state='fixed time, unit rate, zero clock acceleration',
        certificate='exact displacement reconstruction and Bernstein derivative convex hull')


def joint_motion_clocks(motions, mappings, anchors, *, resolution, steps=100, rate_bounds=(.25, 4.)):
    """Joint robust pose/clock registration before native contact validation.

    The latent pose and all source clocks are optimized together. Soft event
    evidence anchors the population time gauge; no original reference is used.
    Normalized-quaternion cubics are evaluated differentiably on the same curve
    used by final transport. Rate constraints and endpoint derivatives are built
    into the parameterization, rather than repaired after pose resampling.
    """
    import torch
    from .g1_kinematics import _quaternion_matrix_wxyz
    motions, mappings = np.asarray(motions), np.asarray(mappings, float)
    e, n, d = motions.shape
    if e < 3 or d != 36 or mappings.shape != (e, n) or len(anchors) != e or steps < 2:
        raise ValueError('Invalid population registration evidence')
    low, high = rate_bounds
    if not 0 < low < 1 < high:
        raise ValueError('Clock bounds must straddle identity')
    curves = [ContinuousMotion(q) for q in motions]
    basis, knots, clock_curve = clock_basis(n, resolution)
    if not basis.shape[1]:
        return np.tile(np.arange(n), (e, 1)), dict(schema='joint_continuous_registration_v1', steps=0,
            clock_control_knots=knots.tolist(),clock_displacement_controls=np.zeros((e,len(knots))).tolist(),
            endpoint_state='fixed time, unit rate, zero clock acceleration',
            rate_constraint='identity clock; rate one and acceleration zero everywhere')
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
    b, frames = tensor(basis), torch.arange(n, dtype=torch.float64)
    rate_certificate = tensor(clock_rate_certificate(clock_curve))
    # Fit the old event proposal inside the boundary-constrained clock space.
    initial = np.linalg.lstsq(basis, (mappings-np.arange(n)).T, rcond=None)[0].T
    controls = torch.nn.Parameter(tensor(initial))
    coefficients = tensor(np.stack([c.curve.c for c in curves]))
    env = torch.arange(e)[:, None]

    def bounded_maps():
        delta = controls @ b.T
        rates = 1+controls @ rate_certificate.T
        minimum, maximum = rates.amin(1), rates.amax(1)
        blend = torch.minimum(torch.ones(e, dtype=b.dtype), torch.minimum(
            (1-low)/(1-minimum).clamp_min(1e-12), (high-1)/(maximum-1).clamp_min(1e-12)))
        return frames[None]+blend[:, None]*delta

    def sample(clock):
        index = clock.floor().long().clamp(0, n-2)
        t = (clock-index)[..., None]
        c = coefficients[env, :, index, :]
        pose = ((c[:, :, 0]*t+c[:, :, 1])*t+c[:, :, 2])*t+c[:, :, 3]
        rotation = pose[:, :, 3:7]
        rotation = rotation/rotation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.cat((pose[:, :, :3], rotation, pose[:, :, 7:]), -1)

    def features(pose):
        return torch.cat((pose[..., :3]/.1,
            _quaternion_matrix_wxyz(pose[..., 3:7]).flatten(-2)/.3, pose[..., 7:]/.3), -1)

    with torch.no_grad():
        first = sample(bounded_maps())
        # Quaternion signs only select a continuous lift, never an angular loss.
        signs = torch.where((first[..., 3:7]*first[:1, :, 3:7]).sum(-1, keepdim=True)<0, -1., 1.)
        first[:, :, 3:7] *= signs
        center_seed = first.mean(0).clone()
        scale = torch.diff(features(first), n=2, dim=1).square().mean((0, 1)).sqrt().clamp_min(1e-4)
    center_controls = torch.nn.Parameter(torch.zeros((b.shape[1],36),dtype=b.dtype))
    optimizer = torch.optim.Adam([dict(params=[controls],lr=.015),
                                  dict(params=[center_controls],lr=.00005)])
    event_terms = []
    for i, (x, y) in enumerate(anchors):
        x, y = np.asarray(x, float), np.asarray(y, float)
        inside = (x > 0) & (x < n-1)
        event_terms.append((i, tensor(x[inside]), tensor(y[inside])))
    history = []
    best, best_pose, best_controls, best_value = None, None, None, float('inf')
    for step in range(steps):
        optimizer.zero_grad()
        clock = bounded_maps(); warped = sample(clock)
        observed, latent = features(warped), features(center_seed+b@center_controls)
        error = (observed-latent).square().mean(-1)
        data = (torch.sqrt(1+error)-1).mean()
        event_loss = torch.zeros((), dtype=b.dtype)
        for i, x, y in event_terms:
            if not len(x): continue
            ix = x.floor().long(); fraction = x-ix
            predicted = clock[i, ix]*(1-fraction)+clock[i, (ix+1).clamp_max(n-1)]*fraction
            event_loss = event_loss+(torch.sqrt(1+((predicted-y)/resolution).square())-1).mean()/e
        bend = torch.diff(clock, n=2, dim=1).square().mean()
        # Motion costs use actual transported samples, rather than assuming
        # that a smooth scalar clock makes every body channel smooth.
        transported = torch.diff(observed, n=2, dim=1)/scale
        motion = (torch.sqrt(1+transported.square())-1).mean()
        smooth = (torch.diff(latent, n=2, dim=0)/scale).square().mean()
        gauge = ((clock.mean(0)-tensor(mappings).mean(0))/resolution).square().mean()
        loss = data+event_loss+.01*motion+.001*smooth+.1*bend+.05*gauge
        if not torch.isfinite(loss): raise ValueError('Nonfinite joint pose/clock objective')
        value = float(loss.detach())
        if value < best_value:
            best, best_value = clock.detach().numpy().copy(), value
            best_pose = (center_seed+b@center_controls).detach().numpy().copy()
            best_pose[:,3:7] /= np.linalg.norm(best_pose[:,3:7],axis=1)[:,None]
            rates=1+controls.detach()@rate_certificate.T
            minimum,maximum=rates.amin(1),rates.amax(1)
            blend=torch.minimum(torch.ones(e,dtype=b.dtype),torch.minimum(
                (1-low)/(1-minimum).clamp_min(1e-12),(high-1)/(maximum-1).clamp_min(1e-12)))
            best_controls=np.pad((controls.detach()*blend[:,None]).numpy(),((0,0),(1,1)))
        if step in (0, steps-1): history.append(dict(step=step, loss=value, event_loss=float(event_loss.detach())))
        loss.backward(); optimizer.step()
    best[:, [0, -1]] = [0, n-1]
    if np.any(np.diff(best, axis=1)<low-1e-8) or np.any(np.diff(best, axis=1)>high+1e-8):
        raise AssertionError('Clock constraints violated')
    return best, dict(schema='joint_continuous_registration_v1', steps=steps, history=history,best_loss=best_value,
        clock_control_knots=knots.tolist(),clock_displacement_controls=best_controls.tolist(),
        consensus_pose=best_pose.tolist(),
        endpoint_state='fixed time, unit rate, zero clock acceleration',
        objective='robust population pose + observed events + transported derivatives + latent smoothness',
        source_interpolation='C2 normalized quaternion and Euclidean cubic',
        rate_constraint='Bernstein convex-hull certificate over every continuous polynomial span',
        original_reference_used=False, native_contact_validation_required=True,
        final_rate_ranges=[[float(r.min()), float(r.max())] for r in np.diff(best, axis=1)])
