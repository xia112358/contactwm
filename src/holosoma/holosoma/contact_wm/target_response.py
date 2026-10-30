"""Target sampling and evaluation diagnostics; no additional training objective."""
import torch


def target_grid(span,device='cpu'):
    axis=torch.tensor([-span,0.,span],device=device)
    x,y=torch.meshgrid(axis,axis,indexing='ij')
    return torch.stack((x.flatten(),y.flatten(),torch.zeros(9,device=device)),-1)


def response_metrics(points,targets,exclude_center=False):
    x=(targets-targets[4])[:,:2];y=(points-points[4])[:,:2]
    jacobian=torch.linalg.lstsq(x,y).solution.T
    mask=torch.ones(9,dtype=torch.bool,device=points.device)
    if exclude_center:mask[4]=False
    return dict(target_to_endpoint_jacobian_xy=jacobian.detach().cpu().tolist(),
                mean_target_error_m=float((points-targets).norm(dim=-1)[mask].mean()),
                response_error_m=float((y-x).norm(dim=-1)[mask].mean()),
                target_positions=targets.detach().cpu().tolist(),endpoints=points.detach().cpu().tolist(),
                evaluated_indices=mask.nonzero().flatten().cpu().tolist())
