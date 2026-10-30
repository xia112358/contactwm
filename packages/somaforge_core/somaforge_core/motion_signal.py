"""Whole-motion signal smoothing; source observations are never modified."""
import numpy as np


def constrained_signal(values, strength, movable):
    """Solve the signal itself with fixed boundary states; never crop a filter.

    Outside samples enter the same derivative stencils as inside samples. This
    avoids creating a new bend at the edge of a multiplied correction window.
    The first/last velocity is retained whenever the support reaches an endpoint.
    """
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve
    n = len(values)
    free = np.asarray(movable, bool).copy()
    free[[0, 1, n-2, n-1]] = False
    fixed = ~free
    result = np.asarray(values, float).copy()
    if not free.any(): return result
    d = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n))
    a = (eye(n)+strength*(d.T@d)).tocsc()
    rhs = result[free]-a[free][:, fixed]@result[fixed]
    result[free] = spsolve(a[free][:, free], rhs)
    return result


def smooth_motion_signal(motion, *, fps, time_scale=.08):
    """Reuse second-difference smoothing on the fused signal, not just residuals.

    A continuous quaternion lift preserves arbitrary heading evolution. Fixed
    endpoint states enter the linear solve, retaining position and first sample
    velocity. Quaternion normalization restores SO(3).
    Native contact must be requeried; this is not a feasibility projection.
    """
    q = np.asarray(motion)
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 3 or not np.isfinite(q).all():
        raise ValueError('Expected finite canonical motion')
    if not np.isfinite([fps, time_scale]).all() or fps <= 0 or time_scale <= 0: raise ValueError('Invalid smoothing time scale')
    if not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-4):
        raise ValueError('Expected normalized root orientation')
    lifted = q.astype(np.float64).copy()
    for i in range(1,len(q)):
        if np.dot(lifted[i-1,3:7],lifted[i,3:7]) < 0: lifted[i,3:7] *= -1
    smooth = constrained_signal(lifted, (time_scale*fps)**4, np.ones(len(q), bool))
    norms=np.linalg.norm(smooth[:,3:7],axis=1)
    if np.any(norms < np.finfo(float).eps):
        raise ValueError('Ambiguous root orientation during smoothing')
    smooth[:,3:7] /= norms[:,None]
    smooth[[0,1,-2,-1]]=lifted[[0,1,-2,-1]]
    return smooth.astype(np.result_type(q.dtype, np.float32))


def local_smooth_motion_signal(motion, frames, *, fps, time_scale=.04, support_time_scale=.12):
    """Solve denoising on compact support with shared boundary conditions.

    The same window transports all body channels. Outside it the original pose
    is exact; orientation uses a consistent quaternion hemisphere. The C2
    envelope bounds subsequent spline corrections, but never crops a filter.
    """
    q = np.asarray(motion)
    # Validate and obtain the continuous quaternion lift used by global fitting.
    smooth_motion_signal(q, fps=fps, time_scale=time_scale)
    frames = np.asarray(frames, int)
    if frames.ndim != 1 or np.any((frames <= 0) | (frames >= len(q)-1)):
        raise ValueError('Local refinement needs interior violation frames')
    if not np.isfinite(support_time_scale) or support_time_scale <= 0:
        raise ValueError('Invalid local correction support')
    window = np.zeros(len(q), dtype=np.float64)
    for center in np.unique(frames):
        radius = 3*support_time_scale*fps
        bump = np.maximum(0., 1-((np.arange(len(q))-center)/radius)**2)**3
        window = 1-(1-window)*(1-bump)
    window[[0, -1]] = 0
    lifted = q.astype(float).copy()
    for i in range(1, len(q)):
        if lifted[i-1,3:7] @ lifted[i,3:7] < 0: lifted[i,3:7] *= -1
    result = constrained_signal(lifted, (time_scale*fps)**4, window > 0)
    result[:,3:7] /= np.linalg.norm(result[:,3:7],axis=1)[:,None]
    result[window == 0] = q[window == 0]
    result[[0,1,-2,-1]]=q[[0,1,-2,-1]]
    return result.astype(np.result_type(q.dtype,np.float32)), window
