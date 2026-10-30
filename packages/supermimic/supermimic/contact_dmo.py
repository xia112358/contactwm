"""On-policy simulator forward values with original tracking WM derivatives.

DMO gradient swapping: arXiv:2509.00215v2, Appendix A.1.3.
The simulator rollout is collected before rebuilding the graph, with no actor
update in between. Every participating trajectory starts at its real reset.
"""
import torch
import warnings
from .contact_control import observation, frozen
from .mdp import unflatten_state
from .rotations import rot_angle_sq


class SimulatorForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predicted, actual):
        return actual.detach().clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def swap(predicted, actual):
    return SimulatorForward.apply(predicted, actual)


def _replay_events(data, clock, event, stop, truncated, completion_mode, stage_plan):
    """Resolve immutable native event/terminal indices before rebuilding the graph.

    One compact CPU transfer replaces per-event nonzero/readbacks inside BPTT.
    A contact contributes at its first eligible hit, or at its stage endpoint.
    """
    T=int(stop.max());device=stop.device
    frames=torch.arange(T,device=device)[:,None];live=frames<stop[None]
    raw=data['contact_goal' if completion_mode=='accumulated_target' else 'contact_first'][:T]
    evaluate=torch.zeros_like(raw);terminal=torch.zeros_like(live)
    length=1 if stage_plan is None else stage_plan['length']
    for k in range(length):
        if stage_plan is None:
            selected=torch.ones_like(live);enabled=torch.ones_like(stop,dtype=torch.bool)
            endpoint=stop;cut=truncated
        else:
            selected=data['goal_index'][:T]==event[None]+k
            enabled=stage_plan['enabled'][k];endpoint=stage_plan['ends'][k];cut=stage_plan['truncated'][k]
        end=clock.event_ends[event+k]-1
        new=clock.target_masks[end]&~clock.keep_masks[end]
        eligible=live&selected&enabled[None]&(frames<endpoint[None])
        hits=raw&new[None]&eligible[...,None]&~cut[None,:,None]
        seen=hits.to(torch.int8).cummax(0).values.bool()
        before=torch.cat((torch.zeros_like(seen[:1]),seen[:-1]),0)
        final=(frames+1==endpoint[None])&enabled[None]&live&selected
        terminal|=final
        evaluate|=(hits|final[...,None]&new[None])&~before
    # Part 6 denotes a root endpoint. Coordinates are sorted time/world/part.
    coordinates=torch.cat((evaluate,terminal[...,None]),-1).nonzero().cpu().tolist()
    schedule={};indices=[]
    for t,b,p in coordinates:
        item=schedule.setdefault(t,dict(contact=[],parts=set(),root=[]))
        if p==6:item['root'].append(b)
        else:
            if not item['contact'] or item['contact'][-1]!=b:item['contact'].append(b)
            item['parts'].add(p)
    for item in schedule.values():
        for name in ('contact','root'):
            begin=len(indices);indices.extend(item[name]);item[name]=slice(begin,len(indices))
        item['parts']=sorted(item['parts'])
    packed=torch.tensor(indices,device=device,dtype=torch.long)
    for item in schedule.values():
        for name in ('contact','root'):item[name]=packed[item[name]]
    return evaluate,terminal,schedule


def replay_loss(actor,wm,clock,geometry,context,normalize,default_q,data,*,
                prior_actor=None,prior_weight=1.,pose_weight=1.,check=False,completion_mode='onset',component_losses=None,endpoint_probe=None,detailed_losses=None,per_attempt_losses=None,stop_steps=None,stage_plan=None,initial_history=None,goal_normals=None,normal_weight=1.,tangent_weight=1.,contact_surface=None):
    """Rebuild complete first-attempt graphs; input tensors are [time,batch,...]."""
    assert initial_history is not None or (data['ep_step'][0]==0).all(), 'DMO requires reset or explicit real entry history'
    assert data['done'].any(0).all(), 'Incomplete native attempts'
    event=data['goal_index'][0];B=len(event);device=event.device
    from .contact_library import segment_ranges
    start=segment_ranges(clock)[0][event];end=clock.event_ends[event]
    native_stop=data['done'].long().argmax(0)+1
    stop=(stage_plan['stop'] if stage_plan is not None else native_stop if stop_steps is None else stop_steps)
    assert stop.shape==native_stop.shape and bool(((stop>=1)&(stop<=native_stop)).all())
    truncated=stop<native_stop
    S=unflatten_state(data['s'][0]);xi=data['xi'][0];previous=data['a_prev'][0]
    hist=observation(S,previous,xi,default_q,context)[:,None].expand(-1,actor.hist,-1).clone()
    if initial_history is not None:hist=initial_history.detach().clone()
    target=clock.data['point'][end-1]
    new=clock.target_masks[end-1]&~clock.keep_masks[end-1]
    reference=unflatten_state(clock.data['s1'][end-1])
    expected=new.long();stage_divisor=target.new_ones(B)
    if stage_plan is not None:
        stage_divisor=stage_plan['enabled'].sum(0).clamp_min(1)
        expected=torch.zeros_like(new,dtype=torch.long)
        for k in range(stage_plan['length']):
            endpoint=clock.event_ends[event+k]-1
            expected+=(clock.target_masks[endpoint]&~clock.keep_masks[endpoint]).long()*stage_plan['enabled'][k,:,None]
    surface_parts=((clock.target_masks&~clock.keep_masks).any(0) if stage_plan is not None else new.any(0)).nonzero().flatten().tolist() if contact_surface is not None and not hasattr(contact_surface,'region_loss') else None
    initial_event=event.clone()
    pair_probe=None;pair_root=None
    if check and stage_plan is not None:
        eligible=stage_plan['enabled'][-1]&((data['goal_index']==initial_event[None]+stage_plan['length']-1)&(torch.arange(len(data['done']),device=device)[:,None]<stop[None])).any(0)
        if eligible.any():pair_probe=int(eligible.nonzero()[0])
    evaluations,terminals,schedule=_replay_events(data,clock,event,stop,truncated,completion_mode,stage_plan)
    count=evaluations.sum(0)
    if not torch.equal(count,expected):raise AssertionError('Each target part must contribute once')
    contact_total=target.new_zeros(B);body_total=contact_total.clone();regular_total=contact_total.clone()
    diagnostic_totals={k:target.new_zeros(B) for k in ('root_position','root_orientation','root_velocity','dagger_prior','action_smoothness','action_magnitude')} if detailed_losses is not None or per_attempt_losses is not None else None
    action_error=target.new_zeros(());forward_error=target.new_zeros(())
    normal_total=target.new_zeros(B);tangent_total=normal_total.clone()
    missing=(evaluations&~(data['known'][:len(evaluations)]&data['active'][:len(evaluations)])).sum()
    early_action=None;last_root=None
    unroll=int(stop.max())
    pair_stop=int(stop[pair_probe]) if pair_probe is not None else None
    with frozen(wm):
        for t in range(unroll):
            live=t<stop;terminal=terminals[t]
            if stage_plan is not None:
                event=data['goal_index'][t]
                start=torch.as_tensor(clock.bounds[:-1],device=device)[event];end=clock.event_ends[event]
                target=clock.data['point'][end-1]
                new=clock.target_masks[end-1]&~clock.keep_masks[end-1]
                reference=unflatten_state(clock.data['s1'][end-1])
            condition=clock.condition(S,start,ends=end)
            action_pred=actor(hist,condition)
            if check and t==0:
                early_action=action_pred;early_action.retain_grad()
            action_error=torch.maximum(action_error,(action_pred.detach()-data['a'][t]).abs().masked_fill(~live[:,None],0).max())
            # Preserve exactly the action used by the simulator and actor derivatives.
            action=swap(action_pred,data['a'][t])
            xi_n=normalize(xi)
            if callable(getattr(wm,'joint_step',None)) and hasattr(wm,'contact_head'):
                # State and sensor heads share the exact same encoded inputs.
                predicted,predicted_xi=wm.joint_step(S,action,xi,xi_n)
            else:
                predicted=wm.step(S,action,xi_n)
                predicted_xi=wm.next_context(S,action,xi,xi_n)
            actual=unflatten_state(data['s1'][t])
            S1=tuple(swap(p,r) for p,r in zip(predicted,actual))
            xi1=swap(predicted_xi,data['next_xi'][t])
            if endpoint_probe is not None and t==endpoint_probe['frame']:
                endpoint_probe['state']=S1
            if check:
                for p,r in zip(S1,actual):forward_error=torch.maximum(forward_error,(p.detach()-r).abs().max())
                forward_error=torch.maximum(forward_error,(xi1.detach()-data['next_xi'][t]).abs().max())
            evaluate=evaluations[t];scheduled=schedule.get(t)
            if scheduled is not None and len(scheduled['contact']):
                # A measured contact uses its measured body-local material point.
                # With no contact, retain the established FK endpoint proxy.
                observed=data['known'][t]&data['active'][t]
                region=None
                if contact_surface is not None and hasattr(contact_surface,'region_loss'):
                    # Sparse native event times need only their participating
                    # worlds/parts, not B x six dense surface fields per frame.
                    slots=scheduled['contact'];parts=scheduled['parts']
                    selected_normals=None if goal_normals is None else torch.broadcast_to(goal_normals,target.shape)[slots]
                    selected=contact_surface.region_loss(tuple(v[slots] for v in S1),target[slots],parts,normals=selected_normals,
                        normal_weight=normal_weight,tangent_weight=tangent_weight)
                    region={key:target.new_zeros(target.shape[:2]).index_copy(0,slots,value)
                            for key,value in selected.items()}
                elif contact_surface is not None:
                    point=contact_surface.contact_points(S1,target,surface_parts)
                else:
                    p,R=geometry.poses(S1)
                    measured=data['point'][t]
                    local=(R.detach().transpose(-1,-2)@(measured-p.detach()).unsqueeze(-1)).squeeze(-1)
                    measured_proxy=p+(R@local.unsqueeze(-1)).squeeze(-1)
                    point=torch.where(observed[...,None],swap(measured_proxy,measured),geometry(S1,end-1))
                if region is not None:
                    errors=region['loss']
                    normal_total+=(region['normal']*evaluate).sum(-1)/new.sum(-1).clamp_min(1)
                    tangent_total+=(region['tangent']*evaluate).sum(-1)/new.sum(-1).clamp_min(1)
                else:
                    delta=(point-target)/.05
                    errors=delta.square().sum(-1)
                if goal_normals is not None and region is None:
                    normal_error=(delta*goal_normals).sum(-1).square()
                    tangent_error=(errors-normal_error).clamp_min(0.)
                    normal_total+=(normal_error*evaluate).sum(-1)/new.sum(-1).clamp_min(1)
                    tangent_total+=(tangent_error*evaluate).sum(-1)/new.sum(-1).clamp_min(1)
                    errors=normal_weight*normal_error+tangent_weight*tangent_error
                contact_total=contact_total+(errors*evaluate).sum(-1)/new.sum(-1).clamp_min(1)
            if scheduled is not None and len(scheduled['root']):
                slots=scheduled['root']
                root_position=((reference[0][slots]-S1[0][slots])/.1).square().mean(-1)
                root_orientation=rot_angle_sq(S1[1][slots],reference[1][slots])/.4**2
                root_velocity=.1*((S1[2][slots]-reference[2][slots]).square().mean(-1)+(S1[3][slots]-reference[3][slots]).square().mean(-1))
                body_total=body_total.index_add(0,slots,root_position+root_orientation+root_velocity)
                if diagnostic_totals is not None:
                    for name,value in [('root_position',root_position),('root_orientation',root_orientation),('root_velocity',root_velocity)]:
                        diagnostic_totals[name]=diagnostic_totals[name].index_add(0,slots,value)
            if check and t==unroll-1:last_root=S1[0][stop.argmax()].sum()
            if pair_probe is not None and t+1==pair_stop:pair_root=S1[0][pair_probe].sum()
            smoothness=.01*(action-previous).square().mean(-1)
            magnitude=.001*action.square().mean(-1)
            reg=smoothness+magnitude
            if prior_actor is not None and prior_weight:
                with torch.no_grad():prior=prior_actor(hist,condition)
                prior_loss=prior_weight*(action-prior).square().sum(-1)
                reg=reg+prior_loss
            regular_total+=torch.where(live,reg,0.)
            if diagnostic_totals is not None:
                components=dict(action_smoothness=smoothness,action_magnitude=magnitude,
                    dagger_prior=prior_loss if prior_actor is not None and prior_weight else reg.new_zeros(B))
                for name,value in components.items():
                    diagnostic_totals[name]+=torch.where(live,value,0.)
            if t+1==unroll:continue
            next_hist=torch.cat((hist[:,1:],observation(S1,action,xi1,default_q,context)[:,None]),1)
            S=tuple(torch.where(live.reshape((-1,)+(1,)*(p.ndim-1)),p,old) for p,old in zip(S1,S))
            xi=torch.where(live[:,None],xi1,xi);hist=torch.where(live[:,None,None],next_hist,hist)
            previous=torch.where(live[:,None],action,previous)
    contact_total=contact_total/stage_divisor;body_total=body_total/stage_divisor
    if diagnostic_totals is not None:
        for name in diagnostic_totals:
            if name.startswith('root_'):diagnostic_totals[name]=diagnostic_totals[name]/stage_divisor
    c=contact_total.mean();b=body_total.mean();r=(regular_total/stop).mean()
    loss=c+pose_weight*b+r
    if component_losses is not None:component_losses.update(contact=c,body=pose_weight*b,regularization=r)
    if detailed_losses is not None:
        detailed_losses.update(contact=c,**{name:(pose_weight*value.mean() if name.startswith('root_') else (value/stop).mean()) for name,value in diagnostic_totals.items()})
    if per_attempt_losses is not None:
        per_attempt_losses.update(contact=contact_total,**{name:(pose_weight*value if name.startswith('root_') else value/stop) for name,value in diagnostic_totals.items()})
    clipped=truncated.sum() if stage_plan is None else (stage_plan['truncated']&stage_plan['enabled']).sum()
    metrics=[c.detach(),b.detach(),r.detach(),action_error,forward_error,missing,clipped]
    if goal_normals is not None:metrics.extend([(normal_total/stage_divisor).mean().detach(),(tangent_total/stage_divisor).mean().detach()])
    values=torch.stack(metrics).cpu().tolist()
    # Diagnostic only: regrouped FP32 GEMMs can differ slightly from collection.
    # Exact simulator actions still supply every forward value through swap().
    if values[3]>1e-4:
        warnings.warn(f"Action replay numerical difference {values[3]:.8g}; continuing with recorded simulator actions",RuntimeWarning)
    if check and values[4]!=0:raise AssertionError('Forward state differs from simulator')
    details=dict(terminal_contact=values[0],terminal_body=values[1],
        process_regularization=values[2],unroll_steps=unroll,
        action_replay_max_error=values[3],forward_state_max_error=values[4] if check else None,
        missing_contact_proxy_terms=int(values[5]) if contact_surface is None else 0,
        geometric_unobserved_terms=int(values[5]) if contact_surface is not None else 0,
        trajectories=B,truncated_trajectories=int(values[6]))
    if goal_normals is not None:
        details.update(contact_normal=values[7],contact_tangent=values[8],normal_weight=normal_weight,tangent_weight=tangent_weight)
    if check:
        grad=torch.autograd.grad(last_root,early_action,retain_graph=True)[0]
        value=float(grad[stop.argmax()].abs().sum())
        if not value>0 or not torch.isfinite(grad).all():raise AssertionError('Broken or nonfinite full-trajectory gradient')
        details['terminal_to_first_action_gradient']=value
        if pair_root is not None:
            pair_gradient=torch.autograd.grad(pair_root,early_action,retain_graph=True)[0][pair_probe]
            value=float(pair_gradient.abs().sum())
            if not value>0 or not torch.isfinite(pair_gradient).all():raise AssertionError('Gradient disconnected at command handoff')
            details['handoff_terminal_to_first_action_gradient']=value
    return loss,details


@torch.no_grad()
def native_attempt_metrics(data,clock,target_faces_by_segment):
    """Real first-attempt signals; never turn missing contact coordinates into error."""
    end_index=data['done'].long().argmax(0)
    B=len(end_index);device=end_index.device;batch=torch.arange(B,device=device)
    live=torch.arange(len(data['done']),device=device)[:,None]<=end_index
    event=data['goal_index'][0];ends=clock.event_ends[event]
    new=clock.target_masks[ends-1]&~clock.keep_masks[ends-1]
    target=clock.data['point'][ends-1]
    matching=torch.ones_like(data['active']) if target_faces_by_segment is None else (data['surface_local']==target_faces_by_segment[event][None])&(target_faces_by_segment[event][None]>=0)
    valid=data['active']&data['known']&matching&live[...,None]
    distance=(data['point']-target[None]).norm(dim=-1)
    best=distance.masked_fill(~valid,float('inf')).amin(0)
    fully_observed=((~new)|torch.isfinite(best)).all(-1)
    worst_part=best.masked_fill(~new,0).max(-1).values
    final=unflatten_state(data['s1'][end_index,batch]);reference=unflatten_state(clock.data['s1'][ends-1])
    final_root=(final[0]-reference[0]).norm(dim=-1)
    final_angle=rot_angle_sq(final[1],reference[1]).sqrt()
    success=~data['term'][end_index,batch]
    # Reduce all segments together and read back one small statistics table.
    observed_error=worst_part.masked_fill(~fully_observed,0.)
    fields=torch.stack((torch.ones_like(final_root),success.float(),fully_observed.float(),
                        observed_error*100,final_root*100,(end_index+1).float()),-1).double()
    sums=fields.new_zeros(len(clock.event_ends),fields.shape[-1]).index_add_(0,event,fields)
    totals=fields.sum(0)
    extra=torch.stack((torch.nanmedian(worst_part.masked_fill(~fully_observed,float('nan')))*100,final_angle.mean()))
    packed=torch.cat((sums.flatten(),totals,extra)).cpu().tolist()
    segments=[];width=fields.shape[-1]
    for segment in range(len(clock.event_ends)):
        count,passed,observed,distance,root,steps=packed[segment*width:(segment+1)*width]
        if not count:continue
        segments.append(dict(segment=segment+1,attempts=int(count),success_rate=passed/count,
            contact_rate=observed/count,best_contact_error_cm=distance/observed if observed else None,
            root_error_cm=root/count,mean_terminal_step=steps/count))
    count,passed,observed,distance,root,steps,median,angle=packed[len(clock.event_ends)*width:]
    return dict(segments=segments,macro_success_rate=sum(s['success_rate'] for s in segments)/len(segments),attempts=B,success_rate=passed/count,
        target_face_contact_rate=observed/count,
        best_contact_error_cm=distance/observed if observed else None,
        best_contact_error_median_cm=median if observed else None,
        contact_error_samples=int(observed),root_error_cm=root/count,
        orientation_error_rad=angle,
        mean_terminal_step=steps/count,
        scope='fresh first attempts before update; closest observed native contact per target part',face_matching=target_faces_by_segment is not None)
