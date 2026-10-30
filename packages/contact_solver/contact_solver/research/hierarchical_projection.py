"""Research-only lexicographic feasibility restoration and posture QP."""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import osqp


def lexicographic_step(g, jacobian, residual, residual_jacobian, weights, lower, upper):
    """Prioritize g+J dx>=0 over posture; trust/joint bounds are always hard.

    Stage 1 minimizes the maximum normalized constraint violation t>=0.
    Stage 2 minimizes weighted posture error with t capped at the stage-1
    optimum (plus numerical solver tolerance). No loss weight trades feasible
    contact/collision constraints for a smaller posture correction.
    """
    g, J, r, B = map(lambda x: np.asarray(x, dtype=np.float64), (g, jacobian, residual, residual_jacobian))
    n = J.shape[1]
    restoration = linprog(np.r_[np.zeros(n), 1.], A_ub=np.c_[-J, -np.ones(len(g))], b_ub=g,
                          bounds=list(zip(lower, upper))+[(0., None)], method='highs')
    if not restoration.success:
        raise RuntimeError('Feasibility LP failed: '+restoration.message)
    slack = max(0., float(restoration.x[-1]))
    W = np.diag(weights)
    H = B.T@W@B + 1e-5*np.eye(n)
    linear = B.T@W@r
    A = sp.vstack((sp.csc_matrix(J), sp.eye(n)), format='csc')
    solver = osqp.OSQP()
    solver.setup(P=sp.triu(sp.csc_matrix(H), format='csc'), q=linear, A=A,
                 l=np.r_[-g-slack-1e-7, lower], u=np.r_[np.full(len(g), np.inf), upper],
                 eps_abs=1e-8, eps_rel=1e-8, max_iter=30000, polishing=True, verbose=False)
    result = solver.solve()
    if result.x is None or result.info.status_val not in (1, 2):
        raise RuntimeError('Posture QP failed: '+result.info.status)
    violation = max(0., float((-g-J@result.x).max()))
    if violation > slack+2e-6:
        raise RuntimeError('Posture QP degraded higher-priority feasibility')
    return result.x, dict(minimum_linearized_violation=slack, achieved_linearized_violation=violation,
                          qp_status=result.info.status, qp_iterations=int(result.info.iter))
