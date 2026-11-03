"""Experimental native contact-process measurement. Not a training default.

Coverage is a bounded kernel on REAL allocated witnesses, not an invented
contact point or surface-distance activation. The process score integrates a
whole arrival process; it is not a centimetre error or a success threshold.
"""
import torch


def native_patch(positions, groups, target, radius):
    """Set-to-target coverage; invariant to ordering and duplicate solver rows."""
    delta=(positions-target.flatten(0,1)[groups])/radius.reshape(-1)[groups,None]
    quality=1/(1+delta.square().sum(-1))
    coverage=target.new_zeros(target.shape[0]*target.shape[1])
    coverage.scatter_reduce_(0,groups,quality,reduce='amax',include_self=True)
    return coverage.reshape(target.shape[:2])


def material_patch(local, groups, nodes, group_count, bandwidth=.01):
    """Fixed material-node evidence, not calibrated area/pressure/occupancy.

    Max splatting avoids contact count and duplicate-row dependence. Nodes
    only describe measured witnesses; they do not activate absent contacts.
    """
    weights=torch.exp(-(local[:,None]-nodes[groups%len(nodes)]).square().sum(-1)/(2*bandwidth**2))
    result=nodes.new_zeros(group_count,nodes.shape[1])
    result.scatter_reduce_(0,groups[:,None].expand_as(weights),weights,reduce='amax',include_self=True)
    return result


def process_loss(coverage, active, initial_active, *, dt=.002, dwell_scale=.04, live=None, close_on_dropout=False):
    """Integrate the measured arrival process through the supplied task horizon.

    Initial support is excluded until an actual off->on transition. Once the
    process starts, gaps count in duration but earn ZERO coverage. Recontact
    adds evidence without clearing earlier deficits. There is no gap threshold.
    loss = 1 - integral(actual coverage dt)/(elapsed process time + dwell_scale).
    close_on_dropout is the rejected strict-first-episode diagnostic baseline.
    This bounded score is not a centimetre error or calibrated success rule.
    Event onset remains discrete; missing contact has no positional derivative.
    """
    if dt<=0 or dwell_scale<=0:raise ValueError('Positive physical time scales required')
    if live is None:live=torch.ones_like(active)
    previous=initial_active.clone();started=torch.zeros_like(previous);closed=started.clone()
    duration=coverage.new_zeros(coverage.shape[1:]);mass=duration.clone()
    losses=[];phases=[];used=[]
    for t in range(len(coverage)):
        on=active[t]&live[t]
        begin=on&~previous&~started&~closed
        started|=begin
        if close_on_dropout:closed|=started&(~on)
        else:closed|=started&~live[t]
        running=started&~closed&live[t]
        include=running&on
        duration=duration+(include if close_on_dropout else running)*dt
        mass=mass+torch.where(include,coverage[t],0)*dt
        losses.append(1-mass/(duration+dwell_scale))
        # 0 awaiting onset; 1 measured contact; 2 measured gap; 3 horizon closed.
        phases.append(torch.where(closed,3,torch.where(include,1,torch.where(started,2,0))))
        used.append(include);previous=on
    return dict(loss=torch.stack(losses),phase=torch.stack(phases),used=torch.stack(used),
                duration=duration,coverage_time=mass,score=mass/(duration+dwell_scale))
