"""Adjacent-window supervision and native metrics, without handoff resets."""
import torch
from .mdp import unflatten_state
from .rotations import rot_angle_sq


@torch.no_grad()
def sequence_plan(data, length, history=None, halfwidth=2, min_success=8, min_mass=.6):
    T,B=data['done'].shape;device=data['done'].device
    native_stop=data['done'].long().argmax(0)+1
    times=torch.arange(T,device=device)[:,None]
    live=times<native_stop[None]
    first=data['goal_index'][0]
    # One reduction over the actual sequence, rather than rescanning T x B
    # for each of the 19 possible stages. Sentinel defaults match argmax on
    # the original all-false masks (start=0, native_end=1).
    relative=data['goal_index']-first[None]
    valid=live&(relative>=0)&(relative<length)
    columns=torch.arange(B,device=device)
    index=(relative.clamp(0,length-1)*B+columns[None]).flatten()
    starts=torch.full((length*B,),T,device=device,dtype=torch.long)
    starts.scatter_reduce_(0,index,times.expand(-1,B).masked_fill(~valid,T).flatten(),reduce='amin')
    last=torch.full_like(starts,T)
    last.scatter_reduce_(0,index,times.expand(-1,B).masked_fill(~(valid&data['stage_done']),T).flatten(),reduce='amin')
    starts=starts.reshape(length,B);last=last.reshape(length,B)
    reached=last<T
    starts=starts.masked_fill(starts==T,0)
    last=last.masked_fill(~reached,0);native_ends=last+1
    success=data['stage_success'][last,columns[None]]&reached
    local=(native_ends-starts).clamp_min(1)
    enabled=reached.clone();local_stops=local.clone();report=None
    if history is not None:
        stages,worlds=reached.nonzero(as_tuple=True)
        groups=first[worlds]+stages;local_frames=local[stages,worlds]
        if hasattr(history,'select_durations'):
            retained,cutoffs,report=history.select_durations(local_frames,success[stages,worlds],groups,halfwidth,min_success,min_mass)
        else:
            pseudo_done=times==local_frames[None]-1
            pseudo=dict(done=pseudo_done,term=pseudo_done&~success[stages,worlds][None],
                        goal_index=groups[None].expand(T,-1))
            retained,cutoffs,report=history.select(pseudo,halfwidth,min_success,min_mass)
        enabled.zero_();enabled[stages[retained],worlds[retained]]=True
        local_stops[stages,worlds]=cutoffs
        report['scope']='stage-local duration selection; simulator handoffs remain continuous; gradient boundaries follow the training mode'
    supervision_ends=starts+local_stops
    global_stop=supervision_ends.masked_fill(~enabled,0).max(0).values.clamp_min(1)
    selected=enabled.any(0).nonzero().flatten()
    if report is not None:
        report.update(windows=B,retained_windows=len(selected),discarded_windows=B-len(selected))
    return selected,dict(length=length,starts=starts,native_ends=native_ends,ends=supervision_ends,enabled=enabled,
        truncated=supervision_ends<native_ends,stop=global_stop),report


def slice_plan(plan,slots):
    return {k:(v[:,slots] if v.ndim==2 else v[slots]) if isinstance(v,torch.Tensor) else v for k,v in plan.items()}


@torch.no_grad()
def sequence_attempt_metrics(data,clock,length,continuous=False):
    T,B=data['done'].shape;device=data['done'].device;batch=torch.arange(B,device=device)
    stop=data['done'].long().argmax(0);first=data['goal_index'][0];last=data['goal_index'][stop,batch]
    success=~data['term'][stop,batch]
    ends=clock.event_ends[last];new=clock.target_masks[ends-1]&~clock.keep_masks[ends-1]
    live=(torch.arange(T,device=device)[:,None]<=stop)&(data['goal_index']==last[None])
    observed=data['active']&data['known']&live[...,None]
    distance=(data['point']-clock.data['point'][ends-1][None]).norm(dim=-1)
    best=distance.masked_fill(~observed,float('inf')).amin(0)
    if getattr(clock,'material_goal_mode',False):
        from .contact_loss_sampling import completed_stable_samples, material_witness_errors
        errors=material_witness_errors(data,clock)
        chosen=completed_stable_samples(data['contact_stable'],live,errors,
            clock.material_contact_tolerances[data['goal_index']][...,None])
        index=chosen.long().argmax(0);parts=torch.arange(6,device=device)[None]
        best=errors[index,batch[:,None],parts].sqrt().masked_fill(~chosen.any(0),float('inf'))
    covered=((~new)|torch.isfinite(best)).all(-1)
    error=best.masked_fill(~new,0).max(-1).values
    if getattr(clock,'material_goal_mode',False):error=best.masked_fill(~new,0).sum(-1)/new.sum(-1).clamp_min(1)
    final=unflatten_state(data['s1'][stop,batch]);ref=unflatten_state(clock.data['s1'][ends-1])
    root=(final[0]-ref[0]).norm(dim=-1);angle=rot_angle_sq(final[1],ref[1]).sqrt()
    first_pass=(data['stage_success']&(data['goal_index']==first[None])).any(0)
    progress=data['stage_success'].sum(0)
    # Reduce all segment statistics on device, then read one compact table.
    # Avoid one masked reduction and CUDA synchronization per field/segment.
    groups=len(clock.event_ends)
    values=torch.stack((torch.ones_like(root),success,progress,first_pass,covered,
                        error.masked_fill(~covered,0)*100,root*100,stop+1),-1).double()
    totals=values.new_zeros(groups,8).scatter_add_(0,first[:,None].expand(-1,8),values)
    completed=data['stage_done']
    stage_values=torch.stack((completed,completed&data['stage_success'],
                              (data['stage_step']+1)*completed),-1).reshape(-1,3).double()
    stage_totals=values.new_zeros(groups,3).scatter_add_(0,data['goal_index'].reshape(-1,1).expand(-1,3),stage_values)
    summary=torch.cat((totals.flatten(),stage_totals.flatten(),
                       torch.bincount(progress,minlength=groups+1).double(),angle.double().sum()[None])).cpu().tolist()
    rows=[];stage_outcomes=[]
    for g in range(groups):
        n,passed,steps,first_ok,observed,err,pos,duration=summary[8*g:8*g+8]
        if n:
            rows.append(dict(segment=g+1,label=f'{g+1}→{groups if continuous else g+length}',attempts=int(n),success_rate=passed/n,
                mean_completed_segments=steps/n,first_stage_success_rate=first_ok/n,contact_rate=observed/n,
                best_contact_error_cm=err/observed if observed else None,root_error_cm=pos/n,mean_terminal_step=duration/n))
        n,passed,duration=summary[8*groups+3*g:8*groups+3*g+3]
        if n:stage_outcomes.append(dict(segment=g+1,attempts=int(n),success_rate=passed/n,mean_duration=duration/n))
    aggregate=[sum(summary[8*g+j] for g in range(groups)) for j in range(8)]
    _,passed,steps,first_ok,observed,err,pos,duration=aggregate
    histogram=[int(v) for v in summary[11*groups:12*groups+1]]
    return dict(stage_outcomes=stage_outcomes,segments=rows,macro_success_rate=sum(r['success_rate'] for r in rows)/len(rows),attempts=B,
        mean_completed_segments=steps/B,max_completed_segments=max(i for i,n in enumerate(histogram) if n),progress_histogram=histogram,success_rate=passed/B,first_stage_success_rate=first_ok/B,
        target_face_contact_rate=observed/B,best_contact_error_cm=err/observed if observed else None,
        root_error_cm=pos/B,orientation_error_rad=summary[-1]/B,
        mean_terminal_step=duration/B,sequence_length=length,face_matching=False,
        scope='random segment starts through failure or trajectory end; success requires completing the remaining suffix; endpoint errors refer to final attempted stage' if continuous else 'fresh adjacent-window attempts; pair success requires both stages; contact/root metrics refer to the stage where the attempt ends')


def stage_batches(buffer,rows,plan,actor,context,default_q,batch_size,*,initial_history,sort_by_duration=True):
    """Independent BPTT graphs from actual continuous stage entries and histories.

    Replay is detached native data. A stage never has an autograd edge to its
    predecessor, while the observation history retains its exact real values.
    """
    from .contact_control import observation
    stages,worlds=plan['enabled'].nonzero(as_tuple=True)
    total=len(worlds)
    if sort_by_duration:
        order=(plan['ends'][stages,worlds]-plan['starts'][stages,worlds]).argsort(stable=True)
        stages=stages[order];worlds=worlds[order]
    for offset in range(0,total,batch_size):
        ks=stages[offset:offset+batch_size];ws=worlds[offset:offset+batch_size]
        starts=plan['starts'][ks,ws];native_ends=plan['native_ends'][ks,ws]
        duration=native_ends-starts;cutoffs=plan['ends'][ks,ws]-starts
        times=torch.arange(int(duration.max()),device=ws.device)[:,None]
        indices=starts[None]+torch.minimum(times,duration[None]-1)
        replay_rows=rows[indices]
        # DMO uses only these fields; do not copy reward, force, contact-face,
        # and collection bookkeeping through every padded stage batch.
        needed=('latent_epsilon','s','s1','xi','next_xi','a','a_prev','ep_step','contact_goal','active','known','point')
        for extra in ('contact_first','contact_stable','contact_loss_point','contact_loss_position','contact_loss_rotation','contact_loss_candidate','contact_raw_stable','contact_raw_point','contact_raw_position','contact_raw_rotation','action_held','body_endpoint_valid','contact_interrupted','contact_entry_blocked','collision_mask','collision_local','collision_point','collision_normal','collision_gap','collision_speed','collision_in_region'):
            if extra in buffer.data:
                needed += (extra,)
        batch={k:buffer.data[k][replay_rows,ws[None]] for k in needed}
        batch['done']=times==duration[None]-1
        batch['goal_index']=buffer.data['goal_index'][rows[starts],ws][None].expand_as(batch['done'])
        history_indices=starts[:,None]+torch.arange(-actor.hist+1,1,device=ws.device)
        hr=rows[history_indices.clamp_min(0)];hw=ws[:,None]
        state=buffer.data['s'][hr,hw];prev=buffer.data['a_prev'][hr,hw];xi=buffer.data['xi'][hr,hw]
        with torch.no_grad():
            obs=observation(unflatten_state(state.flatten(0,1)),prev.flatten(0,1),xi.flatten(0,1),default_q,context)
            history=obs.reshape(len(ws),actor.hist,-1)
            prefix=initial_history[ws[:,None],(history_indices+actor.hist-1).clamp(0,actor.hist-1)]
            history=torch.where((history_indices<=0)[...,None],prefix,history)
        yield batch,cutoffs,history,len(ws)/total,total
