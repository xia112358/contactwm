"""Local descent direction constrained by protected halfspaces.

This changes the optimizer, not the scalar loss. Dual coordinate descent projects
onto A d <= bounds (zero by default) without a pose target or scene-specific contact semantics.
"""
import numpy as np
import torch


def project_direction(direction, normals, *, bounds=None, sweeps=4000, tolerance=1e-10, algorithm='coordinate'):
    if normals.ndim != 2 or normals.shape[1] != direction.numel():
        raise ValueError('Direction and constraint dimensions disagree')
    if not torch.isfinite(direction).all() or not torch.isfinite(normals).all():
        raise ValueError('Nonfinite direction or normals')
    if sweeps<1 or tolerance<=0:raise ValueError('Invalid projection convergence settings')
    # Parameter gradients can be large and nearly cancel. Use double precision
    # and relative scaling; the small Gram solve runs on CPU to avoid GPU syncs.
    d=direction.double();lengths=normals.double().norm(dim=1)
    bound=direction.new_zeros(len(normals)).double() if bounds is None else bounds.double()
    if bound.shape != lengths.shape or not torch.isfinite(bound).all():raise ValueError('Invalid halfspace bounds')
    if bool(((lengths<=1e-12)&(bound<0)).any()):raise ValueError('Infeasible constant halfspace')
    a=normals[lengths>1e-12].double()/lengths[lengths>1e-12,None]
    if not len(a):return d,dict(rows=0,converged=True,max_violation=0.,relative_violation=0.)
    magnitude=max(float(d.norm()),1e-30)
    b=bound[lengths>1e-12]/lengths[lengths>1e-12]
    if algorithm=='rowspace':magnitude=max(magnitude,float(b.norm()))
    gram=(a@a.T).cpu().numpy();rhs=(a@(d/magnitude)-b/magnitude).cpu().numpy();dual=np.zeros_like(rhs)
    if algorithm=='rowspace':
        from scipy.optimize import minimize, nnls, linprog
        # Whiten the small row space. Optimize a primal Euclidean projection
        # there rather than iterating a poorly conditioned dual Gram system.
        eigen,u=np.linalg.eigh(gram)
        keep=eigen>np.finfo(float).eps*max(1,len(a))*max(float(eigen[-1]),1.)
        u=u[:,keep];root=np.sqrt(eigen[keep]);basis=u*root
        ad=(a@(d/magnitude)).cpu().numpy();bn=(b/magnitude).cpu().numpy()
        facial=None
        if np.all(bn==0):
            # Moreau decomposition: projection on a homogeneous cone is an
            # NNLS fit to its polar cone. This also handles feasible sets with
            # no strict interior (opposing/dependent active constraints).
            from types import SimpleNamespace
            y0=(u.T@ad)/root
            try:
                lam,_=nnls(basis.T,y0,maxiter=max(1000,10*sweeps))
                solve=SimpleNamespace(x=-basis.T@lam,success=True,nit=0,message='polar-cone NNLS')
                if np.max(basis@(y0+solve.x),initial=0)>10*tolerance:
                    # Detect the cone's implicit equalities with bounded LPs.
                    # Eliminate these before NNLS: opposing rows otherwise
                    # produce huge cancelling dual coefficients.
                    from scipy.linalg import null_space
                    forced=[]
                    for i in range(len(basis)):
                        check=linprog(basis[i],A_ub=basis,b_ub=np.zeros(len(basis)),
                            bounds=[(-1,1)]*len(root),method='highs',
                            options=dict(primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10))
                        if check.success and check.fun>=-1e-10:forced.append(i)
                    if forced:
                        null=null_space(basis[forced],rcond=1e-12)
                        reduced=basis@null;use=np.linalg.norm(reduced,axis=1)>1e-12
                        target=null.T@y0
                        weights=nnls(reduced[use].T,target,maxiter=max(1000,10*sweeps))[0] if len(target) and use.any() else np.zeros(int(use.sum()))
                        z=target-reduced[use].T@weights
                        solve=SimpleNamespace(x=null@z-y0,success=True,nit=0,message='facially reduced polar-cone NNLS')
                        facial=dict(equalities=forced,reduced_rank=null.shape[1],
                            stationarity=float(np.linalg.norm(z-target+reduced[use].T@weights)),
                            complementarity=float(np.max(np.abs(weights*(reduced[use]@z)),initial=0)))
            except RuntimeError as exc:
                solve=SimpleNamespace(x=np.zeros(len(root)),success=False,nit=0,message=str(exc))
        else:
            constraint=dict(type='ineq',fun=lambda w:bn-ad-basis@w,jac=lambda w:-basis)
            solve=minimize(lambda w:.5*float(w@w),np.zeros(len(root)),jac=lambda w:w,
                           constraints=constraint,method='SLSQP',
                           options=dict(ftol=min(tolerance,1e-12),maxiter=sweeps))
        correction=torch.as_tensor(u@(solve.x/root),device=d.device,dtype=torch.double)
        result=d+magnitude*(a.T@correction)
        violation=float((a@result-b).clamp_min(0).max())/magnitude
        slack=bn-ad-basis@solve.x
        active=slack<=max(1e-8,100*tolerance)
        try:
            multipliers=nnls(basis[active].T,-solve.x,maxiter=max(100,10*len(a)))[0] if active.any() else np.zeros(0)
            stationarity=float(np.linalg.norm(solve.x+basis[active].T@multipliers))
            complementarity=float(np.max(np.abs(multipliers*slack[active]),initial=0))
        except RuntimeError:
            stationarity=complementarity=float('inf')
        if facial is not None:
            stationarity=facial['stationarity'];complementarity=facial['complementarity']
        converged=bool(solve.success and violation<=10*tolerance and stationarity<=1e-7 and complementarity<=1e-7)
        feasibility=None
        if not converged:
            check=linprog(np.zeros(len(root)),A_ub=basis,b_ub=bn,
                          bounds=[(None,None)]*len(root),method='highs',
                          options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            feasibility=dict(status=int(check.status),message=str(check.message))
        return result,dict(algorithm=algorithm,rows=len(a),rank=len(root),iterations=int(solve.nit),
            converged=converged,max_violation=violation*magnitude,relative_violation=violation,
            stationarity=stationarity,complementarity=complementarity,message=str(solve.message),direction_norm=magnitude,feasibility=feasibility,
            facial_reduction=facial,failed_problem=None if converged else dict(gram=gram.tolist(),rhs=rhs.tolist(),bounds=bn.tolist()))
    if algorithm!='coordinate':raise ValueError('Unknown projection algorithm')
    for iteration in range(sweeps):
        previous=dual.copy()
        for i in range(len(a)):
            dual[i]=max(0.,dual[i]+(rhs[i]-gram[i]@dual)/gram[i,i])
        if np.max(np.abs(dual-previous))<=tolerance:break
    result=d-magnitude*(a.T@torch.as_tensor(dual,device=d.device))
    violation=float((a@result-b).clamp_min(0).max())
    return result,dict(rows=len(a),iterations=iteration+1,converged=violation/magnitude<=10*tolerance,max_violation=violation,relative_violation=violation/magnitude,direction_norm=magnitude)
