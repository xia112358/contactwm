"""Differentiable trajectory collision sampling and duration-aware budgets."""
import torch
from torch import Tensor
from somaforge_core.g1_kinematics import _quaternion_matrix_wxyz


def boundary_cubic_reference(start: Tensor, end: Tensor, phase: Tensor) -> Tensor:
    """Kinematic budget witness from immutable endpoints, never motion labels.

    It has zero analytic endpoint derivatives; no collision, support or
    realized contact is claimed for this reference.
    """
    aligned=end.clone()
    aligned[:,3:7]=torch.where((start[:,3:7]*end[:,3:7]).sum(-1,keepdim=True)<0,
                              -end[:,3:7],end[:,3:7])
    blend=phase.square()*(3-2*phase)
    q=(1-blend)*start[:,None]+blend*aligned[:,None]
    q=torch.cat((q[...,:3],torch.nn.functional.normalize(q[...,3:7],dim=-1),q[...,7:]),-1)
    return torch.where(phase==0,start[:,None],torch.where(phase==1,end[:,None],q))


def densify_qpos(q: Tensor, factor: int) -> Tensor:
    """Nlerp each quaternion on its shortest arc; preserve input sample count at factor=1."""
    if factor < 1 or q.ndim != 3 or q.shape[1] < 2:
        raise ValueError('expected [B,T,Q], T>=2 and positive factor')
    if factor == 1:
        return q
    a, b = q[:, :-1, None], q[:, 1:, None]
    u = torch.arange(factor, device=q.device, dtype=q.dtype)[None,None,:,None] / factor
    linear = (1-u)*a + u*b
    qb = torch.where((a[...,3:7]*b[...,3:7]).sum(-1,keepdim=True)<0, -b[...,3:7], b[...,3:7])
    quat = torch.nn.functional.normalize((1-u)*a[...,3:7]+u*qb, dim=-1)
    result = torch.cat((linear[...,:3],quat,linear[...,7:]),-1).flatten(1,2)
    return torch.cat((result,q[:,-1:]),1)


def derivative_metrics(q: Tensor, duration: Tensor) -> dict[str, Tensor]:
    """Per-segment discrete derivative peak/RMS, normalized to 50 Hz."""
    if q.shape[1] < 4 or bool((duration <= 0).any()):
        raise ValueError('need >=4 samples and positive durations')
    factor = (q.shape[1]-1)/(50*duration.reshape(-1))
    signals = {'root':q[...,:3], 'joint':q[...,7:],
               'rotation_matrix':_quaternion_matrix_wxyz(q[...,3:7]).flatten(-2)}
    result = {}
    for name, value in signals.items():
        for order in (1,2,3):
            delta = torch.diff(value,n=order,dim=1)
            magnitude = delta.abs().amax(-1) if name=='joint' else delta.norm(dim=-1)
            magnitude = magnitude*factor[:,None].pow(order)
            result[f'{name}_d{order}_peak'] = magnitude.amax(-1)
            # vector_norm has a finite derivative at an all-zero trajectory.
            result[f'{name}_d{order}_rms'] = magnitude.norm(dim=-1)/magnitude.shape[1]**0.5
    return result


def smooth_budget_loss(q: Tensor, duration: Tensor, reference: Tensor,
                       reference_duration: Tensor, *, ratio: float = 1.5,
                       root_acceleration_cm: float = 0.53,
                       minimum_limits: dict[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
    """Soft budget, not projection. Root cap absolute, other limits reference-relative."""
    if ratio <= 0 or root_acceleration_cm <= 0:
        raise ValueError('budgets must be positive')
    current = derivative_metrics(q,duration)
    with torch.no_grad():
        target = derivative_metrics(reference,reference_duration)
        limits = {k:v.clamp_min(1e-6)*ratio for k,v in target.items()}
        if minimum_limits is not None:
            limits = {k:torch.maximum(v, minimum_limits[k].to(v)) for k,v in limits.items()}
        limits['root_d2_peak'] = torch.full_like(duration.reshape(-1),root_acceleration_cm/100)
    ratios = torch.stack([current[k]/limits[k] for k in current],-1)
    excess = (ratios-1).relu().square()
    return (excess.mean(-1)+excess.amax(-1)).mean(), ratios.amax(-1)
