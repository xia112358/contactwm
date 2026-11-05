"""Independent whole contact transfers from multiple native motion recordings.

Clocks are built inside each source trajectory before concatenation. A motion
boundary never becomes a synthetic contact event, and each case is assigned to
its actual paired terrain. Single-motion recordings retain their existing API.
"""
import torch


def load_slice_clock(recording, path):
    """Load audited explicit native cuts and their versioned reference goals."""
    from .contact_control import ContactClock
    slices = torch.load(path, map_location=recording['data']['s'].device, weights_only=False)
    if slices.get('schema') not in ('native_contact_slices_v1','native_region_slices_v1','native_stable_contact_slices_v1','native_evidence_contact_slices_v2','native_motion_contact_slices_v3'):
        raise ValueError('Unsupported contact slice schema')
    for key in ('task_manifest_sha256','source_checkpoint_sha256','selection_manifest_sha256'):
        if slices['metadata'][key] != recording['metadata'][key]:
            raise ValueError(f'Slice/recording identity mismatch: {key}')
    data=recording['data']; dev=data['s'].device
    clock=ContactClock.__new__(ContactClock)
    clock.fixed_goal_mode='fixed_goal_points' in slices
    clock.reference_support_contract=None
    clock.source_evidence_contract=None
    motion_v3=slices['schema']=='native_motion_contact_slices_v3'
    evidence_v2=slices['schema'] in ('native_evidence_contact_slices_v2','native_motion_contact_slices_v3')
    if evidence_v2:
        if not bool(slices['source_valid'].all()):
            raise ValueError('Contact slices contain source-invalid intervals')
        clock.source_evidence_contract=slices['metadata']['source_evidence_admission']
    if slices['schema'] in ('native_stable_contact_slices_v1','native_evidence_contact_slices_v2','native_motion_contact_slices_v3'):
        from .contact_stability import CONTRACT
        from .contact_phase import VERSION
        if not (slices['metadata']['stable_support_contract']==recording['metadata']['stable_support_contract']==CONTRACT):
            raise ValueError('Stable reference and runtime support contracts differ')
        if not (slices['metadata']['contact_phase_version']==recording['metadata']['contact_phase_version']==VERSION):
            raise ValueError('Stable reference and runtime phase versions differ')
        new=slices['target_masks']&~slices['keep_masks']
        if not clock.fixed_goal_mode and not bool((data['contact_stable'][slices['goals']]|~new).all()):
            raise ValueError('A stable reference goal contains an unqualified required part')
        clock.reference_support_contract=CONTRACT
    if clock.fixed_goal_mode:
        points=slices['fixed_goal_points']
        if points.shape!=(len(slices['goals']),6,3) or not torch.isfinite(points).all():raise ValueError('Invalid fixed segment targets')
        clock.fixed_goal_contract=slices['metadata']['fixed_goal_contract']
        data=dict(data,point=data['point'].clone())
        data['point'][slices['goals']]=points
    clock.data=data; clock.length=len(data['s']); clock.hold=1
    clock.segment_starts=slices['starts']; clock.segment_stops=slices['stops']
    clock.event_ends=slices['goals']+1; clock.segment_motion=slices['motion_ids']
    clock.case_ids=slices['case_ids']; clock.bounds=None
    clock.ends=torch.zeros(clock.length,device=dev,dtype=torch.long)
    clock.frame_source_floor=torch.empty(clock.length,device=dev,dtype=torch.long)
    source_spans=recording['metadata']['recording_spans']
    spans={s['motion']:s for s in source_spans}
    for span in source_spans:
        floor=span['history_begin'] if motion_v3 else span['begin']
        if not span['begin']<=floor<=span['end']:raise ValueError('Trimmed history floor is outside its source trace')
        clock.frame_source_floor[span['begin']:span['end']]=floor
    if evidence_v2:
        if not bool(((slices['trace_ids']>=0)&(slices['trace_ids']<len(source_spans))).all()):
            raise ValueError('Slice source trace identity is out of range')
        clock.segment_trace_ids=slices['trace_ids']
    clock.segment_local_bounds=[]
    for case,(start,stop,goal,motion) in enumerate(zip(slices['starts'].tolist(),slices['stops'].tolist(),slices['goals'].tolist(),slices['motion_ids'].tolist())):
        span=source_spans[int(slices['trace_ids'][case])] if evidence_v2 else spans[motion]
        if evidence_v2 and span['motion']!=motion:
            raise ValueError('Source trace motion mismatch')
        if not span['begin']<=start<stop<=span['end'] or not start<=goal<stop:
            raise ValueError('Slice crosses source trajectory boundary')
        if not bool((data['xi'][start:stop,33].long()==motion).all()):raise ValueError('Slice terrain identity mismatch')
        if evidence_v2 and not bool(data['source_supervision_valid'][start:stop].all()):
            raise ValueError('History-only context cannot become a supervised source interval')
        clock.ends[start:stop]=goal+1
        clock.segment_local_bounds.append((start-span['begin'],stop-span['begin']))
    for key in ('target_masks','keep_masks','support_masks'):
        value=torch.zeros_like(data['active']);value[slices['goals']]=slices[key]
        setattr(clock,key,value)
    return clock


def segment_ranges(clock):
    if hasattr(clock,'segment_starts'):
        return clock.segment_starts,clock.segment_stops
    return torch.tensor(clock.bounds[:-1],device=clock.data['s'].device),clock.event_ends


def history_indices(clock, starts, length, *, include_current=False):
    """True source history, clamped at the admissible source start, not each cut."""
    if hasattr(clock,'frame_source_floor'):
        floor=clock.frame_source_floor[starts]
    else:
        floor=torch.zeros_like(starts)
        if hasattr(clock,'segment_motion'):
            bounds=torch.as_tensor(clock.bounds[:-1],device=starts.device)
            local=torch.tensor([a for a,_ in clock.segment_local_bounds],device=starts.device)
            case=torch.searchsorted(bounds,starts,right=True)-1;floor=bounds[case]-local[case]
    offsets=torch.arange(-length+int(include_current),int(include_current),device=starts.device)
    return torch.maximum(starts[:,None]+offsets,floor[:,None])


def case_world_counts(clock, task_count, total, segment_index=None):
    """Reserve each case's own terrain world; distribute extras over cases equally."""
    counts=[1]*task_count
    if segment_index is not None:
        motion=int(clock.segment_motion[segment_index])
        counts[motion]=total-task_count+1
        if counts[motion]<1:raise ValueError('Insufficient worlds for selected segment')
        return counts
    motions=clock.segment_motion.cpu().tolist()
    for motion in set(motions):counts[motion]=motions.count(motion)
    minimum=sum(counts)
    if total<minimum:raise ValueError(f'Need at least {minimum} worlds for all cases and background tasks')
    for i in range(total-minimum):counts[motions[i%len(motions)]]+=1
    return counts


def collect_motion_library(args,env,motion,mdp,policy,reader,hist,goal_frames,metadata,checkpoint,transfer):
    """Capture independent native trajectories; completed worlds are never appended again."""
    import json
    import shutil
    import numpy as np
    from contextlib import nullcontext
    from .contact_substep_capture import SubstepCapture
    sources={}
    for task in getattr(args,'selected_motion_ids',range(len(motion.clip_len))):
        ids=(env.task_ids==task).nonzero().flatten()
        if len(ids)!=1:raise ValueError('Library capture requires one native world per paired motion')
        sources[task]=int(ids[0])
    recorded={task:[] for task in sources};final_xi={};pending=set(sources);history=None
    limit=min(args.steps,int(motion.clip_len.max())-1)
    capture=SubstepCapture(env,args.output/'substeps') if args.capture_substeps else nullcontext(None)
    with capture as substeps:
        for step in range(limit):
            live_worlds=[sources[task] for task in pending]
            state=env.state();obs=mdp.obs(state,env.a_prev,env.xi,env.f)
            history=(obs[:,None].expand(-1,hist,-1).clone() if history is None else torch.cat((history[:,1:],obs[:,None]),1))
            action=policy(history,motion.goal(env.f,goal_frames))
            if args.noise:action=(action+args.noise*torch.randn_like(action)).clamp(-20,20)
            out=env.step(action)
            if substeps is not None:substeps.flush(step,live_worlds)
            evidence=reader.read()
            if substeps is not None:evidence.update(substeps.monitor.summary())
            exported={key:out[key].cpu() for key in ('s','s1','a','a_prev','xi','f','term','done','ep_step','r')}
            exported.update({key:value.cpu() for key,value in evidence.items()})
            for task in list(pending):
                world=sources[task]
                row={key:value[world].clone() for key,value in exported.items()}
                recorded[task].append(row)
                if bool(row['done']) or len(recorded[task])>=int(motion.clip_len[task])-1:
                    final_xi[task]=env.xi[world:world+1].cpu().clone();pending.remove(task)
            if step%100==0:
                raw=reader.last_evidence.export()
                raw.update({key:reader.last_snapshot[key] for key in ('eligible','groups','robot_side')})
                np.savez_compressed(args.output/f'raw-{step:04d}.npz',**raw)
                print(json.dumps(dict(step=step,pending_motions=len(pending))),flush=True)
            if not pending:break
            env.f=torch.minimum(env.f,motion.clip_end[env.task_ids]-1)
    if args.capture_substeps:metadata=dict(metadata,substep_support_contract=capture.monitor.contract)
    reports=[]
    for task,world in sources.items():
        directory=args.output/f'motion_{task:03d}';directory.mkdir()
        data={key:torch.stack([row[key] for row in recorded[task]]) for key in recorded[task][0]}
        last=final_xi.get(task,env.xi[world:world+1].cpu().clone())
        data['next_xi']=torch.cat((data['xi'][1:],last),0)
        expected=int(motion.clip_len[task])-1
        report=dict(schema='supermimic_native_contact_rollout_v1',motion=task,env_id=world,seed=args.seed,
            frames=len(recorded[task]),expected_frames=expected,
            successful_rollout=len(recorded[task])==expected and not bool(data['term'].any()),
            source_checkpoint=str(args.checkpoint),noise=args.noise,amp=checkpoint['args'].get('amp','none'),
            dt=.02,state_quaternion='wxyz',php_policy_used=False,initialization_transfer=transfer,
            raw_directory=str(args.output.resolve()),contact_timing='last solver solve paired with post-integration pose; not strictly same-state',**metadata)
        torch.save(dict(data=data,metadata=report),directory/'rollout.pt')
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for name in ('solver-collision.mjb','solver-collision.mjb.json'):shutil.copyfile(args.output/name,directory/name)
        reports.append(report)
    (args.output/'library_capture.json').write_text(json.dumps(dict(schema='native_contact_library_capture_v1',
        successful_motions=sum(r['successful_rollout'] for r in reports),total_motions=len(reports),motions=reports),indent=2)+'\n')


def build_contact_clock(recording,**kwargs):
    from .contact_control import ContactClock
    spans=recording.get('metadata',{}).get('recording_spans')
    if not spans:return ContactClock(recording['data'],**kwargs)
    data=recording['data'];total=len(data['s']);children=[];offset=0
    for span in spans:
        begin,end=int(span['begin']),int(span['end'])
        if begin!=offset or end<=begin or end>total:raise ValueError('Noncontiguous native recording spans')
        child_data={k:v[begin:end] for k,v in data.items()}
        if not torch.all(child_data['xi'][:,33].long()==int(span['motion'])):
            raise ValueError('Source trajectory and paired terrain disagree')
        child=ContactClock(child_data,**kwargs);children.append((span,child));offset=end
    if offset!=total:raise ValueError('Incomplete native recording spans')
    result=ContactClock.__new__(ContactClock)
    result.data=data;result.length=total;result.hold=children[0][1].hold
    result.boundary_audits=[dict(motion=int(span["motion"]),**child.boundary_audit) for span,child in children]
    result.bounds=[0];result.segment_local_bounds=[];motions=[]
    for span,clock in children:
        result.bounds.extend(int(span['begin'])+end for end in clock.bounds[1:])
        result.segment_local_bounds.extend(zip(clock.bounds[:-1],clock.bounds[1:]))
        motions.extend([int(span['motion'])]*len(clock.event_ends))
    device=data['s'].device
    result.event_ends=torch.tensor(result.bounds[1:],device=device)
    result.segment_motion=torch.tensor(motions,device=device,dtype=torch.long)
    result.ends=torch.cat([clock.ends+int(span['begin']) for span,clock in children])
    for key in ('target_masks','keep_masks','support_masks'):
        setattr(result,key,torch.cat([getattr(clock,key) for _,clock in children]))
    return result


def case_world_assignment(clock,task_ids,segment_index=None):
    """Allocate independent case episodes only to their matching native terrain."""
    selected=[];segments=[]
    for motion in clock.segment_motion.unique().tolist():
        cases=(clock.segment_motion==motion).nonzero().flatten()
        if segment_index is not None:cases=cases[cases==segment_index]
        if not len(cases):continue
        worlds=(task_ids==motion).nonzero().flatten()
        if len(worlds)<len(cases):raise ValueError('Insufficient matching worlds to cover every whole transfer')
        selected.append(worlds)
        segments.append(cases[torch.arange(len(worlds),device=worlds.device)%len(cases)])
    if not selected:raise ValueError('No native worlds match the selected contact transfers')
    ids=torch.cat(selected);assigned=torch.cat(segments)
    background=torch.ones(len(task_ids),device=task_ids.device,dtype=torch.bool);background[ids]=False
    return ids,assigned,background.nonzero().flatten()


def initial_contact_indices(clock,starts):
    """Previous native evidence must stay inside its own source trajectory."""
    if hasattr(clock,'frame_source_floor'):
        return torch.maximum(starts-1,clock.frame_source_floor[starts])
    floor=torch.zeros_like(starts)
    if hasattr(clock,'segment_motion'):
        bounds=torch.as_tensor(clock.bounds[:-1],device=starts.device)
        local=torch.tensor([a for a,_ in clock.segment_local_bounds],device=starts.device)
        case=torch.searchsorted(bounds,starts,right=True)-1
        floor=bounds[case]-local[case]
    return torch.maximum(starts-1,floor)


def dagger_case_worlds(clock,task_ids,chosen,segment_ids,repeat=0):
    """Use distinct matching worlds within each student rollout batch."""
    desired=clock.segment_motion[segment_ids];answer=torch.empty_like(segment_ids)
    for motion in desired.unique().tolist():
        slots=(desired==motion).nonzero().flatten()
        worlds=chosen[task_ids[chosen]==motion]
        if len(worlds)<len(slots):raise ValueError('DAgger batch exceeds matching terrain allocation')
        answer[slots]=worlds[(torch.arange(len(slots),device=chosen.device)+repeat*len(slots))%len(worlds)]
    return answer


def require_source_replay_capacity(clock,capacity,grace_steps,segment=None):
    """A complete source transfer must fit before a ring buffer overwrites it."""
    starts,stops=segment_ranges(clock);durations=stops-starts
    longest=int(durations.max() if segment is None else durations[segment-1])
    required=longest+grace_steps
    if capacity<required:
        raise ValueError(f'Native source transfers require buffer_steps >= {required} (selected source duration plus grace); configured {capacity}')
    return required
