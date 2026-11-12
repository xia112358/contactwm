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


def stable_contact_state(active, known, frames):
    """True after a continuously observed contact has been established."""
    if frames <= 0:
        return torch.zeros_like(active, dtype=torch.bool)
    live = active & known
    streak = torch.zeros_like(live, dtype=torch.long)
    run = torch.zeros_like(live[0], dtype=torch.long)
    for t in range(live.shape[0]):
        run = torch.where(live[t], run + 1, torch.zeros_like(run))
        streak[t] = run
    return live & (streak >= frames)


def stable_contact_start(active, known, frames):
    """Rising edge of a continuously observed, established contact.

    ``frames`` is an establishment confirmation window, not a contact
    threshold.  A part contributes only once per active interval; an interval
    shorter than the window contributes nothing.
    """
    established = stable_contact_state(active, known, frames)
    previous = torch.cat((torch.zeros_like(established[:1]), established[:-1]), 0)
    return established & ~previous


def stable_target(points, evaluate, fallback):
    """Gather each part's point at its own stable-start frame."""
    has = evaluate.any(0)
    index = evaluate.long().argmax(0)
    part = torch.arange(points.shape[1], device=points.device)[None, :].expand_as(index)
    taken = points[index, part]
    return torch.where(has[..., None], taken, fallback), has


def stable_demo_target(clock, event, frames, fallback):
    """Gather the demonstration point at the DEMO's own stable start.

    The student attempt and the demonstration need not establish contact on the
    same frame.  Using the student index into demonstration time silently
    compares the wrong two instants.
    """
    from .contact_library import segment_ranges
    starts = segment_ranges(clock)[0]
    result = fallback.clone()
    has = torch.zeros(event.shape + (6,), dtype=torch.bool, device=event.device)
    for value in event.unique().tolist():
        gap = event == value
        begin = int(starts[value])
        end = int(clock.event_ends[value])
        active = clock.data['active'][begin:end]
        known = clock.data['known'][begin:end]
        stable = stable_contact_start(active, known, frames)
        index = stable.long().argmax(0)
        present = stable.any(0)
        part = torch.arange(6, device=stable.device)
        values = clock.data['point'][begin:end][index, part]
        result[gap] = torch.where(present[None, :, None], values[None], fallback[gap])
        has[gap] = present[None]
    return result, has


def _replay_events(data, clock, event, stop, truncated, completion_mode, stage_plan, first_contact_targets=False, contact_window_frames=1, native_events=None, stable_contact_frames=0):
    """Resolve immutable native event/terminal indices before rebuilding the graph.

    One compact CPU transfer replaces per-event nonzero/readbacks inside BPTT.
    A contact contributes at its first eligible hit, or at its stage endpoint.
    """
    T=int(stop.max());device=stop.device
    frames=torch.arange(T,device=device)[:,None];live=frames<stop[None]
    if stable_contact_frames and getattr(clock,'fixed_goal_mode',False):
        from .contact_loss_sampling import stable_loss_samples
        if getattr(clock,'material_goal_mode',False):
            from .contact_loss_sampling import completed_stable_samples, material_witness_errors
            subset={key:value[:T] for key,value in data.items()}
            errors=material_witness_errors(subset,clock)
            raw=subset['contact_stable']
        else:
            raw=stable_loss_samples(data['contact_stable'][:T],live)
    elif stable_contact_frames:
        from .contact_phase import landing_phase_events
        end=clock.event_ends[event]-1
        required=clock.target_masks[end]&~clock.keep_masks[end]
        raw=landing_phase_events(data['active'][:T],data['known'][:T],
            clock.contact_phase_release_starts[event],required,stable_contact_frames,live,data['contact_stable'][:T])
    else:
        raw=data['contact_first' if first_contact_targets or completion_mode!='accumulated_target' else 'contact_goal'][:T]
    evaluate=torch.zeros_like(raw);terminal=torch.zeros_like(live)
    required_terms=torch.zeros_like(raw[0],dtype=torch.long)
    length=1 if stage_plan is None else stage_plan['length']
    for k in range(length):
        if stage_plan is None:
            selected=torch.ones_like(live);enabled=torch.ones_like(stop,dtype=torch.bool)
            endpoint=stop;cut=truncated
        else:
            selected=data['goal_index'][:T]==event[None]+k
            enabled=stage_plan['enabled'][k];endpoint=stage_plan['ends'][k];cut=stage_plan['truncated'][k]
        interrupted=torch.zeros_like(enabled)
        initial_guidance=torch.zeros_like(enabled)
        if 'contact_interrupted' in data:
            native_endpoint=data['done'].long().argmax(0)+1 if stage_plan is None else stage_plan['ends'][k]
            interrupted=data['contact_interrupted'][native_endpoint-1,torch.arange(len(stop),device=device)]
            # Failed contact guidance uses the actual collected stop pose.
            # Interruption still excludes a successful body endpoint below.
        end=clock.event_ends[event+k]-1
        new=clock.target_masks[end]&~clock.keep_masks[end]
        required_terms+=new.long()*enabled[:,None]
        eligible=live&selected&enabled[None]&(frames<endpoint[None])&~initial_guidance[None]
        if stable_contact_frames and getattr(clock,'material_goal_mode',False):
            stage_witness=completed_stable_samples(raw,eligible,errors,
                clock.material_contact_tolerances[subset['goal_index']][...,None])
        else:
            stage_witness=raw
        hits=stage_witness&new[None]&eligible[...,None]
        if first_contact_targets and contact_window_frames > 1 and not stable_contact_frames:
            from .contact_arrival import arrival_indices
            subset={key:data[key][:T] for key in ('done','known','active','point','contact_first')}
            index,_=arrival_indices(subset,clock.data['point'][end],frames=contact_window_frames,live=eligible)
            hits=(frames[...,None]==index[None])&new[None]&eligible[...,None]
        if not first_contact_targets and not stable_contact_frames:hits &= ~cut[None,:,None]
        if native_events is not None:
            native_events['mask']=native_events.get('mask',torch.zeros_like(raw))|hits
        guidance=torch.zeros_like(hits)
        if stable_contact_frames and hasattr(clock,'paired_material_local'):
            from .contact_loss_sampling import stable_loss_samples
            raw_guidance=stable_loss_samples(data['contact_raw_stable'][:T],eligible)
            knees=torch.tensor([False,False,False,False,True,True],device=device)
            guidance=raw_guidance&new[None]&knees&~hits.any(0)[None]
        choices=hits|guidance
        seen=choices.to(torch.int8).cummax(0).values.bool()
        before=torch.cat((torch.zeros_like(seen[:1]),seen[:-1]),0)
        final=(frames+1==endpoint[None])&enabled[None]&live&selected
        # Failed attempts keep the contact goal at the safe prefix, but never
        # acquire a successful body endpoint from a collision.
        body_final=final&~interrupted[None]
        terminal|=body_final if 'body_endpoint_valid' not in data else body_final&data['body_endpoint_valid'][:T]
        # A matched witness wins; otherwise the latest stable native row can
        # guide material alignment without becoming a success witness.
        evaluate|=(choices|final[...,None]&new[None])&~before
    if native_events is not None:native_events['required_terms']=required_terms
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
                prior_actor=None,prior_weight=1.,pose_weight=1.,check=False,completion_mode='onset',component_losses=None,endpoint_probe=None,detailed_losses=None,per_attempt_losses=None,stop_steps=None,stage_plan=None,initial_history=None,goal_normals=None,normal_weight=1.,tangent_weight=1.,contact_surface=None,body_position_tolerance=0.,body_angle_tolerance=0.,body_angle_weight=1.,first_contact_targets=False,contact_window_frames=1,continuous_arrival=None,initial_contact_active=None,stable_contact_frames=0):
    """Rebuild complete first-attempt graphs; input tensors are [time,batch,...]."""
    assert initial_history is not None or (data['ep_step'][0]==0).all(), 'DMO requires reset or explicit real entry history'
    assert data['done'].any(0).all(), 'Incomplete native attempts'
    if continuous_arrival is not None and initial_contact_active is None:
        raise ValueError('Continuous arrival needs native reset support, before the first action')
    if continuous_arrival is not None and stage_plan is not None:
        raise ValueError('Continuous arrival currently requires independent single-segment attempts')
    if stable_contact_frames and (stage_plan is not None or continuous_arrival is not None):
        raise ValueError('Stable contact start currently requires independent single-segment attempts')
    if stable_contact_frames and not callable(getattr(contact_surface,'region_loss',None)):
        raise ValueError('Stable contact loss requires collision-surface region guidance for unconfirmed parts')
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
    arrival_state=None if continuous_arrival is None else continuous_arrival.begin(S,initial_contact_active)
    initial_event=event.clone()
    pair_probe=None;pair_root=None
    if check and stage_plan is not None:
        eligible=stage_plan['enabled'][-1]&((data['goal_index']==initial_event[None]+stage_plan['length']-1)&(torch.arange(len(data['done']),device=device)[:,None]<stop[None])).any(0)
        if eligible.any():pair_probe=int(eligible.nonzero()[0])
    native_events={}
    evaluations,terminals,schedule=_replay_events(data,clock,event,stop,truncated,completion_mode,stage_plan,first_contact_targets,contact_window_frames,native_events,stable_contact_frames)
    expected=native_events['required_terms']
    if check and stable_contact_frames and not getattr(clock,'fixed_goal_mode',False):
        live=torch.arange(len(evaluations),device=device)[:,None]<stop[None]
        collected=data['contact_first'][:len(evaluations)]&live[...,None]&new[None]
        if not torch.equal(native_events['mask'],collected):
            raise AssertionError('Online and replay landing-phase events disagree')
    count=evaluations.sum(0)
    if not torch.equal(count,expected):raise AssertionError('Each target part must contribute once')
    stable_evaluations=evaluations&native_events['mask'] if stable_contact_frames else torch.zeros_like(evaluations)
    guidance_evaluations=evaluations&~stable_evaluations if stable_contact_frames else torch.zeros_like(evaluations)
    stable_total=target.new_zeros(B);guidance_total=target.new_zeros(B)
    latent_kl_total=target.new_zeros(B)
    collision_total=target.new_zeros(B)
    contact_total=target.new_zeros(B);body_total=contact_total.clone();regular_total=contact_total.clone()
    diagnostic_totals={k:target.new_zeros(B) for k in ('root_position','root_orientation','root_velocity','dagger_prior','action_smoothness','action_magnitude')} if detailed_losses is not None or per_attempt_losses is not None else None
    action_error=target.new_zeros(());forward_error=target.new_zeros(())
    normal_total=target.new_zeros(B);tangent_total=normal_total.clone()
    measured_mask=data['known'][:len(evaluations)]&data['active'][:len(evaluations)]
    if first_contact_targets:measured_mask &= native_events['mask']
    missing=(evaluations&~measured_mask).sum()
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
            action_pred,latent_aux=actor(hist,condition,epsilon=data['latent_epsilon'][t],return_aux=True)
            latent_kl_total+=latent_aux['kl_per_sample']*live/stop
            if 'action_held' in data:
                # Confirmation repeats the last executed setpoint. Its gradient
                # belongs to that earlier decision, not a new policy output.
                held=data['action_held'][t]
                if held.ndim==1:held=held[:,None]  # archived whole-action hold recordings
                action_pred=torch.where(held,previous,action_pred)
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
            if arrival_state is not None:
                points=continuous_arrival.material_points(S1)
                if arrival_state.phase_mode=='load':
                    # Measured forces and fixed nodes belong to the same solve,
                    # not the post-integration state. Preserve those forward
                    # coordinates while using WM/FK position derivatives.
                    points=swap(points,data['node_position'][t])
                gap=continuous_arrival.gaps(points)
                arrival_result=arrival_state.update(points,gap,target,dt=continuous_arrival.dt,
                    live=live[:,None]&new,
                    normal_load=data['normal_load'][t] if 'normal_load' in data else None,
                    node_load=data['node_load'][t] if 'node_load' in data else None)
                contact_total=(arrival_result['loss']*new).sum(-1)/new.sum(-1).clamp_min(1)
            if arrival_state is None and scheduled is not None and len(scheduled['contact']):
                # A measured contact uses its measured body-local material point.
                # With no contact, retain the established FK endpoint proxy.
                pre_action=native_events.get('pre_action_guidance')
                use_initial=torch.zeros(B,device=device,dtype=torch.bool) if pre_action is None else pre_action[t]
                contact_state=tuple(torch.where(use_initial.reshape((-1,)+(1,)*(a.ndim-1)),a,b) for a,b in zip(S,S1))
                observed=measured_mask[t]
                region=None
                if contact_surface is not None and hasattr(contact_surface,'region_loss'):
                    # Sparse native event times need only their participating
                    # worlds/parts, not B x six dense surface fields per frame.
                    slots=scheduled['contact'];parts=scheduled['parts']
                    selected_normals=None if goal_normals is None else torch.broadcast_to(goal_normals,target.shape)[slots]
                    selected=contact_surface.region_loss(tuple(v[slots] for v in contact_state),target[slots],parts,normals=selected_normals,
                        normal_weight=normal_weight,tangent_weight=tangent_weight)
                    region={key:target.new_zeros(target.shape[:2]).index_copy(0,slots,value)
                            for key,value in selected.items()}
                elif contact_surface is not None:
                    point=contact_surface.contact_points(contact_state,target,surface_parts)
                else:
                    p,R=geometry.poses(contact_state)
                    measured=data['point'][t]
                    local=(R.detach().transpose(-1,-2)@(measured-p.detach()).unsqueeze(-1)).squeeze(-1)
                    measured_proxy=p+(R@local.unsqueeze(-1)).squeeze(-1)
                    point=torch.where(observed[...,None],swap(measured_proxy,measured),geometry(contact_state,end-1))
                if region is not None:
                    if stable_contact_frames:
                        # Stable native witness can precede the control endpoint.
                        # Completion keeps its separate control-end observation;
                        # the fixed body-local derivative still comes from FK/WM.
                        p,R=geometry.poses(contact_state)
                        measured=data['contact_loss_point' if clock.fixed_goal_mode else 'point'][t]
                        local=(R.detach().transpose(-1,-2)@(measured-p.detach()).unsqueeze(-1)).squeeze(-1)
                        point=swap(p+(R@local.unsqueeze(-1)).squeeze(-1),measured)
                        delta=(point-target)/.05
                        measured_cost=delta.square().sum(-1)
                        if goal_normals is None:
                            measured_normal=measured_cost;measured_tangent=torch.zeros_like(measured_cost)
                        else:
                            measured_normal=(delta*goal_normals).sum(-1).square()
                            measured_tangent=(measured_cost-measured_normal).clamp_min(0.)
                            measured_cost=normal_weight*measured_normal+tangent_weight*measured_tangent
                        region={key:torch.where(stable_evaluations[t],value,region[key])
                            for key,value in [('loss',measured_cost),('normal',measured_normal),('tangent',measured_tangent)]}
                    if getattr(clock,'material_goal_mode',False):
                        from .contact_material import world_points, point_error_sq
                        p,R=geometry.poses(contact_state)
                        predicted=world_points(p,R,clock.material_local[event])
                        actual=world_points(data['contact_loss_position'][t],data['contact_loss_rotation'][t],clock.material_local[event])
                        points=torch.where(stable_evaluations[t,:,:,None,None],swap(predicted,actual),predicted)
                        cost=point_error_sq(points,clock.material_targets[event])/.05**2
                        from navigation_objective import foot_navigation_cost
                        use=stable_evaluations[t]
                        nav_p=torch.where(use[...,None],swap(p,data['contact_loss_position'][t]),p)
                        nav_R=torch.where(use[...,None,None],swap(R,data['contact_loss_rotation'][t]),R)
                        cost=foot_navigation_cost(clock,contact_surface,context,points,nav_p,nav_R,event,new,cost)
                        if hasattr(clock,'paired_material_local'):
                            from .contact_paired_material import paired_material_error_sq
                            use=stable_evaluations[t]
                            position=torch.where(use[...,None],swap(p,data['contact_loss_position'][t]),p)
                            rotation=torch.where(use[...,None,None],swap(R,data['contact_loss_rotation'][t]),R)
                            paired,_=paired_material_error_sq(position,rotation,data['contact_loss_point'][t],use,clock.paired_material_local[event],clock.paired_material_targets[event],clock.paired_material_valid[event],data['contact_loss_candidate'][t])
                            raw_use=guidance_evaluations[t]&data['contact_raw_stable'][t]&~use_initial[:,None]
                            raw_position=swap(p,data['contact_raw_position'][t]);raw_rotation=swap(R,data['contact_raw_rotation'][t])
                            from .contact_paired_material import paired_raw_guidance_error_sq
                            raw_cost=paired_raw_guidance_error_sq(raw_position,raw_rotation,data['contact_raw_point'][t],clock.paired_material_local[event],clock.paired_material_targets[event],clock.paired_material_valid[event])
                            paired=torch.where(raw_use,raw_cost,paired)
                            cost=torch.cat((cost[:,:4],paired[:,4:]/.05**2),dim=-1)
                        hand=torch.tensor([False,False,True,True,False,False],device=cost.device)
                        region={key:torch.where(hand,region[key],value) for key,value in [('loss',cost),('normal',cost),('tangent',torch.zeros_like(cost))]}
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
                if stable_contact_frames:
                    stable_total+=(errors*stable_evaluations[t]).sum(-1)/new.sum(-1).clamp_min(1)
                    guidance_total+=(errors*guidance_evaluations[t]).sum(-1)/new.sum(-1).clamp_min(1)
            if scheduled is not None and len(scheduled['root']):
                slots=scheduled['root']
                position_distance=(reference[0][slots]-S1[0][slots]).norm(dim=-1)
                root_position=(position_distance.sub(body_position_tolerance).clamp_min(0)/.1).square()/3
                angle=rot_angle_sq(S1[1][slots],reference[1][slots]).sqrt()
                root_orientation=body_angle_weight*(angle.sub(body_angle_tolerance).clamp_min(0)/.4).square()
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
    # Kept as a diagnostic field: continuous clearance replaces the old prefix term.
    prefix_per_attempt=target.new_zeros(B)
    contact_total=contact_total/stage_divisor;body_total=body_total/stage_divisor
    if diagnostic_totals is not None:
        for name in diagnostic_totals:
            if name.startswith('root_'):diagnostic_totals[name]=diagnostic_totals[name]/stage_divisor
    c=contact_total.mean();b=body_total.mean();r=(regular_total/stop).mean()
    collision=(collision_total/stage_divisor).mean()
    loss=c+pose_weight*b+r+collision+1e-3*latent_kl_total.mean()
    if component_losses is not None:component_losses.update(contact=c,body=pose_weight*b,regularization=r,process_collision=collision,collision_prefix=(prefix_per_attempt/stage_divisor).mean())
    if detailed_losses is not None:
        detailed_losses.update(contact=c,process_collision=collision,**{name:(pose_weight*value.mean() if name.startswith('root_') else (value/stop).mean()) for name,value in diagnostic_totals.items()})
    if per_attempt_losses is not None:
        per_attempt_losses.update(contact=contact_total,process_collision=collision_total/stage_divisor,**{name:(pose_weight*value if name.startswith('root_') else value/stop) for name,value in diagnostic_totals.items()})
    clipped=truncated.sum() if stage_plan is None else (stage_plan['truncated']&stage_plan['enabled']).sum()
    metrics=[c.detach(),b.detach(),r.detach(),action_error,forward_error,missing,clipped]
    if goal_normals is not None:metrics.extend([(normal_total/stage_divisor).mean().detach(),(tangent_total/stage_divisor).mean().detach()])
    values=torch.stack(metrics).cpu().tolist()
    # Diagnostic only: regrouped FP32 GEMMs can differ slightly from collection.
    # Exact simulator actions still supply every forward value through swap().
    if values[3]>1e-4:
        warnings.warn(f"Action replay numerical difference {values[3]:.8g}; continuing with recorded simulator actions",RuntimeWarning)
    if check and values[4]!=0:raise AssertionError('Forward state differs from simulator')
    details=dict(process_collision=float(collision.detach()),collision_prefix=float(prefix_per_attempt.mean().detach()),terminal_contact=values[0],terminal_body=values[1],
        process_regularization=values[2],unroll_steps=unroll,
        action_replay_max_error=values[3],forward_state_max_error=values[4] if check else None,
        missing_contact_proxy_terms=int(values[5]) if contact_surface is None else 0,
        geometric_unobserved_terms=int(values[5]) if contact_surface is not None else 0,
        trajectories=B,truncated_trajectories=int(values[6]))
    if stable_contact_frames:
        details.update(required_contact_terms=int(expected.sum()),stable_contact_terms=int(stable_evaluations.sum()),
            terminal_guidance_terms=int(guidance_evaluations.sum()),
            contact_supervision_coverage=float(count.sum()/expected.sum().clamp_min(1)),
            stable_contact_loss=float(stable_total.mean().detach()),
            terminal_guidance_loss=float(guidance_total.mean().detach()))
    if arrival_state is not None:
        details.update(continuous_landing_process_load_v2=arrival_state.phase_mode=='load',
            native_contact_metric_unchanged=True,
            soft_arrival_mass=float((((1-arrival_state.survival)*new).sum()/new.sum().clamp_min(1)).detach()),
            soft_landing_process_seconds=float(((arrival_state.duration*new).sum()/new.sum().clamp_min(1)).detach()),
            soft_landing_proximity_seconds=float(((arrival_state.proximity_time*new).sum()/new.sum().clamp_min(1)).detach()),
            missing_contact_proxy_terms=0,geometric_unobserved_terms=0)
    if goal_normals is not None and arrival_state is None:
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
    details['bottleneck_kl']=float(latent_kl_total.mean().detach())
    return loss,details


@torch.no_grad()
def loss_aligned_attempt_metrics(data,clock,contact_tolerances,root_tolerance,root_angle_tolerance):
    """Report all supplied attempts using the exact completion witnesses used by loss.

    A missing stable witness is unsuccessful, never a synthetic contact. Error
    is conditional on every target part having a witness and uses the same
    equal-per-part, equal-per-attempt weighting as the contact objective.
    """
    from .contact_loss_sampling import stable_loss_samples
    from .contact_task import has_fallen
    event=data['goal_index'][0];B=len(event);device=event.device
    end_index=data['done'].long().argmax(0);ends=clock.event_ends[event]
    live=torch.arange(len(data['done']),device=device)[:,None]<=end_index
    if getattr(clock,'material_goal_mode',False):
        from .contact_loss_sampling import completed_stable_samples, material_witness_errors
        errors=material_witness_errors(data,clock)
        selected=completed_stable_samples(data['contact_stable'],live,errors,contact_tolerances[event][None,:,None])
    else:
        selected=stable_loss_samples(data['contact_stable'],live)
    has=selected.any(0);indices=selected.long().argmax(0)
    batch=torch.arange(B,device=device)[:,None];parts=torch.arange(6,device=device)[None]
    point=data['contact_loss_point'][indices,batch,parts]
    target=clock.data['point'][ends-1]
    required=clock.target_masks[ends-1]&~clock.keep_masks[ends-1]
    complete=((~required)|has).all(-1)
    distance=(point-target).norm(dim=-1)
    if getattr(clock,'material_goal_mode',False):
        from .contact_material import world_points, mixed_error_sq
        position=data['contact_loss_position'][indices,batch,parts]
        rotation=data['contact_loss_rotation'][indices,batch,parts]
        actual=world_points(position,rotation,clock.material_local[event])
        distance=mixed_error_sq(actual,clock.material_targets[event],point,target).sqrt()
        if hasattr(clock,'paired_material_local'):
            from .contact_paired_material import paired_material_error_sq
            paired,_=paired_material_error_sq(position,rotation,point,has,clock.paired_material_local[event],clock.paired_material_targets[event],clock.paired_material_valid[event],data['contact_loss_candidate'][indices,batch,parts])
            distance=torch.cat((distance[:,:4],paired[:,4:].sqrt()),dim=-1)
    mean_distance=(distance*required).sum(-1)/required.sum(-1).clamp_min(1)
    contact_ok=((~required)|(has&(distance<=contact_tolerances[event,None]))).all(-1)
    terminal=data['s1'][end_index,torch.arange(B,device=device)]
    final=unflatten_state(terminal);reference=unflatten_state(clock.data['s1'][ends-1])
    root=(final[0]-reference[0]).norm(dim=-1);angle=rot_angle_sq(final[1],reference[1]).sqrt()
    angle_ok=torch.ones_like(contact_ok) if root_angle_tolerance is None else angle<root_angle_tolerance
    failed=(~torch.isfinite(terminal).all(-1))|has_fallen(final)
    success=contact_ok&(root<root_tolerance)&angle_ok&~failed
    if 'contact_interrupted' in data:
        success &= ~data['contact_interrupted'][end_index,torch.arange(B,device=device)]
    worst=distance.masked_fill(~required,0.).amax(-1)
    missing=~complete
    over=complete&~contact_ok
    root_bad=root>=root_tolerance
    angle_bad=~angle_ok
    reason_masks={'missing_stable_contact':missing&~failed,
        'contact_over_tolerance':over&~failed,
        'root_position':contact_ok&root_bad&~failed,
        'root_orientation':contact_ok&~root_bad&angle_bad&~failed,
        'fallen_or_nonfinite':failed}
    diagnostic_rows=[]
    for i in range(len(clock.event_ends)):
        mask=event==i
        if not bool(mask.any()):continue
        observed=mask&complete
        diagnostic_rows.append(dict(
            case_id=int(clock.case_ids[i]) if hasattr(clock,'case_ids') else i+1,
            contact_error_samples=int(observed.sum()),missing_witness_attempts=int((mask&missing).sum()),
            worst_part_error_cm=float(worst[observed].mean()*100) if bool(observed.any()) else None,
            failure_reasons={name:int((mask&value).sum()) for name,value in reason_masks.items()},
            part_error_cm=[float(distance[mask&has[:,part],part].mean()*100) if bool(required[mask,part].any()) and bool((mask&has[:,part]).any()) else None for part in range(6)],
            part_missing_attempts=[int((mask&required[:,part]&~has[:,part]).sum()) for part in range(6)]))
    observed_error=mean_distance.masked_fill(~complete,0.)
    fields=torch.stack((torch.ones_like(root),success.float(),complete.float(),observed_error*100,root*100,(end_index+1).float()),-1).double()
    sums=fields.new_zeros(len(clock.event_ends),fields.shape[-1]).index_add_(0,event,fields)
    totals=fields.sum(0);median=torch.nanmedian(mean_distance.masked_fill(~complete,float('nan')))*100
    packed=torch.cat((sums.flatten(),totals,median[None],angle.mean()[None])).cpu().tolist()
    width=fields.shape[-1];segments=[]
    for segment in range(len(clock.event_ends)):
        count,passed,observed,distance_sum,root_sum,steps=packed[segment*width:(segment+1)*width]
        if not count:continue
        segments.append(dict(segment=segment+1,attempts=int(count),success_rate=passed/count,
            contact_rate=observed/count,best_contact_error_cm=distance_sum/observed if observed else None,
            root_error_cm=root_sum/count,mean_terminal_step=steps/count))
    for row,diagnostics in zip(segments,diagnostic_rows):row.update(diagnostics)
    count,passed,observed,distance_sum,root_sum,steps,med,ang=packed[len(clock.event_ends)*width:]
    return dict(segments=segments,macro_success_rate=sum(s['success_rate'] for s in segments)/len(segments),
        attempts=int(count),success_rate=passed/count,target_face_contact_rate=observed/count,
        best_contact_error_cm=distance_sum/observed if observed else None,
        best_contact_error_median_cm=med if observed else None,contact_error_samples=int(observed),
        root_error_cm=root_sum/count,orientation_error_rad=ang,mean_terminal_step=steps/count,
        contact_metric=('stable_hand_contact_fixed_foot_knee_material' if getattr(clock,'material_goal_mode',False) else 'loss_stable_witness_mean_part_error'),face_matching=False,
        missing_witness_attempts=int(count-observed),
        failure_reasons={name:int(value.sum()) for name,value in reason_masks.items()},
        worst_part_error_cm=float(worst[complete].mean()*100) if bool(complete.any()) else None,
        scope='all collected first attempts; first qualified completion per part, latest qualified witness while incomplete; missing witnesses are failures, excluded from conditional error',
        completion_contract='first_stable_completion_latched_v1')


@torch.no_grad()
def native_attempt_metrics(data,clock,target_faces_by_segment,*,first_contact=False,contact_window_frames=1):
    """Real first-attempt signals; never turn missing contact coordinates into error."""
    end_index=data['done'].long().argmax(0)
    B=len(end_index);device=end_index.device;batch=torch.arange(B,device=device)
    live=torch.arange(len(data['done']),device=device)[:,None]<=end_index
    event=data['goal_index'][0];ends=clock.event_ends[event]
    new=clock.target_masks[ends-1]&~clock.keep_masks[ends-1]
    target=clock.data['point'][ends-1]
    matching=torch.ones_like(data['active']) if target_faces_by_segment is None else (data['surface_local']==target_faces_by_segment[event][None])&(target_faces_by_segment[event][None]>=0)
    valid=data['active']&data['known']&matching&live[...,None]
    if first_contact:
        if contact_window_frames > 1:
            from .contact_arrival import arrival_indices
            index,_=arrival_indices(data,target,frames=contact_window_frames,eligible=matching,live=live)
            valid &= torch.arange(len(live),device=device)[:,None,None]==index[None]
        else:valid &= data['contact_first']
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
        contact_window_frames=contact_window_frames,
        contact_metric=('first_native_arrival_window' if contact_window_frames>1 else 'first_native_onset') if first_contact else 'closest_observed_contact',
        first_contact_error_cm=distance/observed if first_contact and observed else None,
        scope=f'fresh first attempts before update; first contiguous native contact episode, up to {contact_window_frames} frames per target part' if first_contact else 'fresh first attempts before update; closest observed native contact per target part',face_matching=target_faces_by_segment is not None)
