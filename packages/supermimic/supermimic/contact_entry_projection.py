"""Project the actual optimizer step onto sampled entrance descent constraints."""
import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

@torch.no_grad()
def project_entry_step(parameters, before, gradients):
    """Consume the gradient buffer in place after ordinary gradient aggregation."""
    proposed=torch.cat([p.detach().flatten() for p in parameters])-before
    unit=gradients.div_(gradients.norm(dim=1,keepdim=True).clamp_min(1e-20))
    linear_tensor=unit@proposed
    if unit.is_cuda:
        from .contact_projection_gpu import solve_nonnegative_qp
        gram=unit@unit.T
        multipliers,solver_audit=solve_nonnegative_qp(gram,linear_tensor,tolerance=5e-8)
        actual=proposed-unit.T@multipliers
        iterations=solver_audit['iterations']
        extra=dict(solver='gpu_full',active_constraints=len(gradients),active_passes=1)
    else:
        gram=(unit@unit.T).double().cpu().numpy()
        linear=linear_tensor.double().cpu().numpy()
        def dual(lam):
            value=gram@lam-linear
            return .5*lam@(value-linear),value
        with threadpool_limits(limits=1,user_api='blas'):
            result=minimize(dual,np.zeros(len(gradients)),jac=True,bounds=[(0.,None)]*len(gradients),method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-14,gtol=1e-10,maxcor=20))
        if not result.success:
            raise RuntimeError('Entrance projection QP failed: '+str(result.message))
        actual=proposed-unit.T@torch.as_tensor(result.x,device=gradients.device,dtype=gradients.dtype)
        iterations=int(result.nit)
        extra=dict(solver='cpu_lbfgsb',active_constraints=len(gradients),active_passes=1)
    for p,v in zip(parameters,(before+actual).split([p.numel() for p in parameters])):p.copy_(v.view_as(p))
    response=unit@actual
    return dict(entrances=len(gradients),retained_step_ratio=float(actual.norm()/proposed.norm().clamp_min(1e-20)),proposed_step_norm=float(proposed.norm()),actual_step_norm=float(actual.norm()),predicted_harmed_before=int((linear_tensor>1e-8).sum()),predicted_harmed_after=int((response>1e-8).sum()),max_linear_increase=float(response.max()),solver_iterations=iterations,**extra)
