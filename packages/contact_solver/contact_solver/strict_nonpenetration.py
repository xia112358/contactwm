"""Zero-tolerance Newton-reported penetration priority and rejection gate.

Passing this gate means collision clearance under the configured solver's
detected pairs, not proof about excluded collisions or continuous trajectories.
Contact/topology/task acceptance is a separate requirement.
"""
import math
import torch

SCHEMA='newton_reported_zero_penetration_priority_v1'


def clearance_status(report):
    if not isinstance(report,dict) or report.get('schema')!='newton_full_robot_signed_separation_v1':
        return 'unknown'
    depths=[]
    for kind in ('terrain','self'):
        if any(kind+suffix not in report for suffix in ('_rows','_penetration_m')) or 'worst_'+kind not in report:
            return 'unknown'
        try:
            value=float(report[kind+'_penetration_m'])
            count=report[kind+'_rows']
        except (KeyError,TypeError,ValueError):return 'unknown'
        if not math.isfinite(value) or value<0 or not isinstance(count,int) or count<0:return 'unknown'
        worst=report.get('worst_'+kind)
        if value>0:
            if not isinstance(worst,dict):return 'unknown'
            try:distance=float(worst['dist'])
            except (KeyError,TypeError,ValueError):return 'unknown'
            if count==0 or not math.isfinite(distance) or distance>=0 or not math.isclose(-distance,value,rel_tol=1e-6,abs_tol=0.):return 'unknown'
        elif worst is not None:return 'unknown'
        depths.append(value)
    return 'penetrating' if max(depths)>0 else 'clear'


def generation_gate(reports):
    """No epsilon, margin subtraction, clipping or unknown-as-success fallback."""
    statuses=[clearance_status(r) for r in reports]
    return dict(statuses=statuses,collision_pass=[s=='clear' for s in statuses],
                all_frames_collision_pass=bool(statuses) and all(s=='clear' for s in statuses))


def priority_loss(task_loss,depth,reports):
    if task_loss.shape!=depth.shape or depth.ndim!=1 or len(reports)!=len(depth):
        raise ValueError('Batch/penetration shape mismatch')
    gate=generation_gate(reports)
    if 'unknown' in gate['statuses']:raise ValueError('Unknown collision result cannot supervise successful generation')
    if not torch.isfinite(depth).all() or (depth<0).any():raise ValueError('Invalid differentiable penetration')
    expected=depth.new_tensor([max(r['terrain_penetration_m'],r['self_penetration_m']) for r in reports])
    torch.testing.assert_close(depth.detach(),expected,rtol=1e-5,atol=1e-9)
    if not all(gate['collision_pass']):
        # Linear term remains informative for arbitrarily small positive depths.
        # Whole-batch priority avoids task gradients from other samples competing
        # through shared network parameters with this batch's feasibility step.
        normalized=depth/.02
        return normalized.mean()+normalized.max(),'feasibility',gate
    return task_loss.mean(),'task',gate
