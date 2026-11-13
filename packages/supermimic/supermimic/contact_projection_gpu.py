"""GPU solution of the entrance-protection QP; all entrance constraints retained."""
import torch

@torch.no_grad()
def solve_nonnegative_qp(gram,linear,initial=None,*,tolerance=1e-8,max_iterations=32768,graph=True):
    lam=torch.zeros_like(linear) if initial is None else initial.clone()
    z=lam.clone();t=linear.new_ones(());lipschitz=gram.abs().sum(-1).max()
    def steps():
        for _ in range(128):
            nxt=(z-(gram@z-linear)/lipschitz).clamp_min(0)
            nt=(1+torch.sqrt(1+4*t*t))*.5
            z.copy_(nxt+((t-1)/nt)*(nxt-lam));lam.copy_(nxt);t.copy_(nt)
    # A captured block removes thousands of small Python/kernel launches.
    if graph:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):steps()
        torch.cuda.current_stream().wait_stream(stream)
        lam.zero_() if initial is None else lam.copy_(initial)
        z.copy_(lam);t.fill_(1)
        captured=torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):steps()
        # Capture executes a block; count it as part of the solve.
    iterations=128 if graph else 0
    for _ in range(max_iterations//128-int(graph)):
        if graph:captured.replay()
        else:steps()
        iterations+=128
        gradient=gram@lam-linear
        residual=((lam-(lam-gradient/lipschitz).clamp_min(0))*lipschitz).abs().max()
        if float(residual)<=tolerance:
            return lam,dict(iterations=iterations,kkt_residual=float(residual))
    raise RuntimeError(f'GPU QP did not converge: {float(residual)}')

@torch.no_grad()
def solve_active(gram,linear,*,tolerance=1e-8,graph=True):
    # Begin with violated constraints, then add every newly violated entrance.
    active=(linear>0).nonzero().flatten()
    lam=torch.zeros_like(linear);passes=[]
    while len(active):
        solution,audit=solve_nonnegative_qp(gram[active[:,None],active[None,:]].contiguous(),linear[active],lam[active],tolerance=tolerance,graph=graph)
        lam.zero_();lam[active]=solution
        response=linear-gram@lam
        included=torch.zeros_like(linear,dtype=torch.bool);included[active]=True
        added=((response>tolerance)&~included).nonzero().flatten()
        passes.append(dict(active=len(active),added=len(added),max_violation=float(response.max()),**audit))
        if not len(added):return lam,dict(passes=passes,active=len(active),full_constraints=len(linear))
        active=torch.cat((active,added)).sort().values
    return lam,dict(passes=passes,active=0,full_constraints=len(linear))

@torch.no_grad()
def solve_active_from_gradients(unit,proposed,*,tolerance=5e-8):
    linear=unit@proposed
    # Begin with violated constraints, then add every newly violated entrance.
    active=(linear>0).nonzero().flatten()
    lam=torch.zeros(len(active),device=unit.device,dtype=unit.dtype)
    passes=[]
    if not len(active):return proposed.clone(),dict(passes=passes,active=0,full_constraints=len(unit))
    active_unit=unit[active];gram=active_unit@active_unit.T
    while True:
        lam,audit=solve_nonnegative_qp(gram,linear[active],lam,tolerance=tolerance,graph=unit.is_cuda)
        actual=proposed-active_unit.T@lam
        response=unit@actual
        included=torch.zeros(len(unit),device=unit.device,dtype=torch.bool);included[active]=True
        added=((response>tolerance)&~included).nonzero().flatten()
        passes.append(dict(active=len(active),added=len(added),max_violation=float(response.max()),**audit))
        if not len(added):return actual,dict(passes=passes,active=len(active),full_constraints=len(unit))
        # Reuse the previous Gram block; compute only new rows/columns.
        new_unit=unit[added];all_unit=torch.cat((active_unit,new_unit))
        bottom=new_unit@all_unit.T
        gram=torch.cat((torch.cat((gram,bottom[:,:len(active)].T),1),bottom),0)
        active_unit=all_unit;active=torch.cat((active,added));lam=torch.cat((lam,lam.new_zeros(len(added))))
