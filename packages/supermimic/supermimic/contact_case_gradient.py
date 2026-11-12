"""Reduce conflicting per-segment gradients without rejecting the whole update."""
import torch


def guarded_case_gradient(gradients,rotation=0):
    """Cap dominant case norms at the median, then project conflicting components.

    Every row is one sampled segment's full loss gradient. A zero row contributes
    no direction; it occurs when that sampled segment has no actor derivative.
    """
    norms=gradients.norm(dim=-1);nonzero=norms>0
    if not bool(nonzero.any()):
        return gradients.new_zeros(gradients.shape[-1]),dict(cases=len(norms),nonzero=0,conflicts_before=0,conflicts_after=0)
    median=norms[nonzero].median();tiny=torch.finfo(gradients.dtype).tiny
    scale=(median/norms.clamp_min(tiny)).clamp(max=1);scaled=gradients*scale[:,None];direction=scaled.mean(0)
    initial=scaled@direction;before=int((initial<0).sum())
    for j in range(len(norms)):
        i=(j+rotation)%len(norms);ref=scaled[i];dot=direction.dot(ref);denom=ref.square().sum().clamp_min(tiny);direction=direction-(dot/denom).clamp(max=0)*ref
    after=int(((scaled@direction)<-1e-8).sum())
    return direction,dict(cases=len(norms),nonzero=int(nonzero.sum()),conflicts_before=before,conflicts_after=after,median_case_gradient_norm=float(median),max_case_gradient_norm=float(norms.max()))


def guard_adam_displacement(displacement,gradients):
    """Project the actual Adam step onto sampled per-case descent half-spaces.

    Adam preconditioning and saved momentum can rotate a protected raw gradient.
    Solve the dual of the minimum-distance parameter-step projection so the
    feasible component remains available instead of rejecting the whole update.
    """
    import numpy as np
    from scipy.optimize import minimize
    norms=gradients.norm(dim=-1);nonzero=norms>0
    if not bool(nonzero.any()):
        return displacement,dict(cases=len(norms),nonzero=0,conflicts_before=0,conflicts_after=0,step_ratio=1.)
    unit=gradients[nonzero]/norms[nonzero,None];descent=-displacement;step_norm=descent.norm();dots=unit@descent;before=int((dots<0).sum())
    if before==0:
        return displacement,dict(cases=len(norms),nonzero=int(nonzero.sum()),conflicts_before=0,conflicts_after=0,step_ratio=1.)
    gram=(unit@unit.T).double().cpu().numpy();b=(dots/step_norm).double().cpu().numpy();gram=(gram+gram.T)*.5
    def objective(alpha):return .5*alpha@gram@alpha+b@alpha,gram@alpha+b
    solution=minimize(objective,np.zeros(len(b)),jac=True,bounds=[(0,None)]*len(b),method='L-BFGS-B',options=dict(ftol=1e-14,gtol=1e-11,maxiter=10000))
    if not solution.success:raise RuntimeError('Per-case Adam step projection did not converge: '+str(solution.message))
    alpha=torch.as_tensor(solution.x,device=unit.device,dtype=unit.dtype);projected=descent+step_norm*(alpha@unit);residual=unit@projected
    tolerance=max(float(projected.norm())*1e-5,1e-12)
    if float(residual.min()) < -tolerance:raise RuntimeError('Per-case Adam step projection violates a sampled descent constraint')
    return -projected,dict(cases=len(norms),nonzero=int(nonzero.sum()),conflicts_before=before,conflicts_after=int((residual < -tolerance).sum()),step_ratio=float(projected.norm()/descent.norm().clamp_min(torch.finfo(descent.dtype).tiny)))


# Numerical outlier handling only; no segment/source labels or norm equalization.
ACTOR_GRADIENT_GROUP_SIZE = 128
ACTOR_GRADIENT_OUTLIER_RATIO = 10.


def cap_gradient_outliers(gradients, weights, ratio=ACTOR_GRADIENT_OUTLIER_RATIO):
    """Cap only >ratio times the nonzero median, then preserve sample weights.

    Rows are unweighted small-batch means. No small row is amplified and no
    projection is performed. Zero gradients retain their original zero weight
    contribution; a wholly zero update remains zero.
    """
    norms=gradients.norm(dim=-1)
    if not bool(torch.isfinite(norms).all()):
        raise FloatingPointError('Nonfinite actor gradient before sample aggregation')
    nonzero=norms[norms>0]
    median=nonzero.median() if len(nonzero) else norms.new_zeros(())
    cap=ratio*median
    scale=(cap/norms.clamp_min(torch.finfo(norms.dtype).tiny)).clamp(max=1.)
    direction=(gradients*(weights*scale)[:,None]).sum(0)
    return direction,dict(groups=len(norms),clipped_groups=int((norms>cap).sum()),
        median_norm=float(median),max_norm=float(norms.max()),cap=float(cap),
        clipped_sample_fraction=float(weights[norms>cap].sum()/weights.sum()),
        min_scale=float(scale.min()))
