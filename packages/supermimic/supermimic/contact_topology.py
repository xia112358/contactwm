"""Contact truth, established support topology, and support quality are separate.

Support, transfer identity and main form share one offline contact timeline.
Native activation remains distinct from sustained bearing and point quality.
Row switching alone never invents a departure; unknown evidence is explicit.
"""
import torch
from .contact_stability import CONTRACT

PARTS=('LF','RF','LH','RH','LK','RK')
NAMES=('左脚','右脚','左手','右手','左膝','右膝')
ADMISSION=dict(version='native_successful_source_evidence_v3',
    source='finite continuous physical source interval and allocated native primary-surface evidence',
    contact_goal='fixed actual native target from qualified observations within one support placement',
    trajectory_only='retained for trajectory review; not admitted as a next-contact target task',
    collector_success='required: whole tracking trajectory completed without termination',tracking_restart='diagnostic only')
SCHEMA='native_motion_contact_slices_v3'
TOPOLOGY_CONTRACT=dict(version='native_continuous_bearing_support_v7',
    entry='native bearing and measured motion; substep independent events require the recorded event_bearing_s, excluding gaps; keep physical onset and fast confirmation separately',
    exit='use recorded exit_persistence_s and gap_motion_m for substeps, two samples for control-only records; unknown evidence and face changes end support; loaded internal rolling/sliding preserves support',
    attachments='positive-force native terrain shape sets; updated while bearing load',
    zero_state='no observed bearing support; not a statement about limb participation',
    raw_contact='unaltered native active/known masks; row stability gates point loss only')


def mask_value(bits):return sum(int(x)<<i for i,x in enumerate(bits))


def mask_name(mask):
    return '初始未观测' if mask is None else '+'.join(PARTS[p] for p in range(6) if mask&(1<<p)) or '无已确认支撑'


def physical_intervals(data):
    """Only physical/numerical validity; tracking term/done is not rejection."""
    from .rotations import quat_to_mat
    finite=torch.stack([torch.isfinite(data[k]).flatten(1).all(-1) for k in ('s','s1','a','a_prev','xi','next_xi')]).all(0)
    R0=quat_to_mat(data['s'][:,3:7]);R1=quat_to_mat(data['s1'][:,3:7])
    physical=(data['s'][:,2]>=.15)&(R0[:,2,2]>0)&(data['s1'][:,2]>=.15)&(R1[:,2,2]>0)
    valid=finite&physical
    # A reset/discontinuity creates a new source interval, not a synthetic edge.
    continuous=(data['f'][1:]==data['f'][:-1]+1)&(data['ep_step'][1:]==data['ep_step'][:-1]+1)
    state_contiguous=(data['s'][1:]-data['s1'][:-1]).abs().amax(-1)<=1e-4
    continuous &= state_contiguous
    intervals=[];start=None
    for t in range(len(valid)):
        if start is not None and (not bool(valid[t]) or not bool(continuous[t-1])):
            intervals.append((start,t));start=None
        if start is None and bool(valid[t]):start=t
    if start is not None:intervals.append((start,len(valid)))
    return intervals,valid,finite,physical


def support_topology(data,attachments,temporal=None):
    from .contact_temporal import segmentation_contact_state
    if temporal is None:temporal=segmentation_contact_state(data)
    surfaces=[tuple() for _ in range(6)];states=[]
    for t,member in enumerate(temporal['support']):
        known=data['known'][t];active=data['active'][t]
        for p in range(6):
            if not member[p]:surfaces[p]=tuple()
            elif temporal['bearing'][t,p]:surfaces[p]=tuple(attachments[t][p])
        states.append(dict(mask=mask_value(member),attachments=[list(s) for s in surfaces],
            known=bool(known.all()),qualified_mask=mask_value(temporal['bearing'][t]),raw_mask=mask_value(active&known)))
    return states


def phase_ranges(states):
    result=[];start=0
    for t in range(1,len(states)+1):
        if t==len(states) or any(states[t][k]!=states[start][k] for k in ('mask','attachments','known')):
            result.append(dict(start=start,stop=t,**{k:states[start][k] for k in ('mask','attachments','known')}));start=t
    return result


LIMBS=('LL','RL','LA','RA')
LIMB_NAMES=('左下肢','右下肢','左上肢','右上肢')
LIMB_PARTS=((0,4),(1,5),(2,),(3,))
ROLE_CONTRACT=dict(version='support_allocation_roles_v8',
    support='one measured support placement without independent participation change; a qualified same-placement unload/resume remains support role; measured bearing frames and placement-continuity frames are reported separately',
    transfer='support participation changes: engage from idle, disengage, or reallocate; a previous contact is not required',
    inactive='no established support or support participation change; idle motion and incidental contact alone are insufficient',
    lower_limb='actual observed established form only; no destination form assigned during unsupported motion',
    classification='four-limb role vector; global left-right mirror merged; foot/knee form is an attribute',
    motion_excursion_m=.02,
    motion_evidence='native part poses; arms relative to recorded torso; settled windows bound idle periods, direction reversals do not end participation; motion never defines native activation',
    attributes='row stability controls point loss only, never role or partition; contact form and quality do not split classes')
ROLE_NAMES={'S':'支撑','T':'参与变化','N':'不参与'}


def classify_limb_roles(states,data,start,stop,new,temporal=None):
    """Read whole-segment roles from the common sustained-support timeline."""
    from .contact_temporal import segmentation_contact_state,TEMPORAL_CONTRACT,participation_interval
    if temporal is None:temporal=segmentation_contact_state(data)
    interval=participation_interval(temporal['transfers'],start,stop)
    arrival_ids={e['id'] for e in interval['arrivals']}
    roles=[];forms=[];evidence=[]
    for limb,(name,parts) in enumerate(zip(LIMBS,LIMB_PARTS)):
        present=temporal['placement_forms'][start:stop,limb].unique().tolist()
        present=[p for p in present if p>=0]
        if len(present)>1:raise ValueError(f'{name}: multiple main contact forms in one segment; partition at the shared form boundary')
        p=present[0] if present else None
        bearing=bool(temporal['placement_support'][start:stop,list(parts)].any())
        events=[e for e in interval['active'] if e['limb']==limb]
        transferring=[]
        for e in events:
            source_here=e['source_support_start'] is not None and max(start,e['source_support_start'])<min(stop,e['source_support_stop'])
            arrived=e['id'] in arrival_ids
            # Unloading at a handoff remains support in the departing segment;
            # the following segment reads the same ongoing transfer event.
            other_arrival=any(x['limb']!=limb for x in interval['arrivals'])
            unfinished=False
            if e['terminal'] and not other_arrival:
                position=data['native_part_position'][max(start,e['start']-1):stop,e['part']]
                unfinished=float((position-position[:1]).norm(dim=-1).max())>TEMPORAL_CONTRACT['motion_excursion_m']
            if arrived or not source_here or unfinished:transferring.append(e)
        role='T' if transferring else 'S' if bearing else 'N'
        form=('foot' if p in (0,1) else 'knee' if p in (4,5) else 'hand') if role!='N' and p is not None else 'none'
        roles.append(role);forms.append(form)
        evidence.append(dict(limb=name,role=role,main_part=PARTS[p] if role!='N' and p is not None else None,form=form,
            established_support_frames=int(temporal['support'][start:stop,list(parts)].any(-1).sum()),
            placement_continuity_frames=int(temporal['placement_support'][start:stop,list(parts)].any(-1).sum()),
            transfer_ids=[e['id'] for e in transferring],
            participation_kinds=sorted({e['participation_kind'] for e in transferring}),
            observed_support_engagement=any(e['id'] in arrival_ids for e in transferring),
            transfer_started_before_segment=any(e['start']<start for e in transferring),
            unfinished_departure=any(e['terminal'] for e in transferring),
            evidence_complete=bool(data['known'][start:stop,list(parts)].all()),
            incidental_parts=[PARTS[q] for q in parts if (q!=p or role=='N') and bool((data['active'][start:stop,q]&data['known'][start:stop,q]).any())]))
    key=''.join(roles);mirrored=''.join(roles[p^1] for p in range(4))
    return dict(limb_roles=dict(zip(LIMBS,roles)),contact_forms=dict(zip(LIMBS,forms)),
        role_key=key,mirror_role_key=min(key,mirrored),role_evidence=evidence,
        part_roles={part:next((e['role'] for e in evidence if e['main_part']==part),'N') for part in PARTS},
        role_evidence_complete=all(x['evidence_complete'] for x in evidence))


def classify_transition(states,data,start,stop,new,temporal=None):
    """Keep path diagnostics and classify whole-segment part roles independently."""
    entry=states[start-1] if start else None;exit=states[stop-1];path=[]
    for state in ([entry] if entry is not None else [])+states[start:stop]:
        key=(state['mask'],tuple(tuple(s) for s in state['attachments']),state['known'])
        if not path or key!=path[-1][0]:path.append((key,state))
    a=None if entry is None else entry['mask'];b=exit['mask'];arrived=mask_value(new)
    added=b if a is None else b&~a;removed=0 if a is None else a&~b
    changed_surface=[]
    if entry is not None:
        changed_surface=[p for p in range(6) if (a&b)&(1<<p) and entry['attachments'][p]!=exit['attachments'][p]]
    uncertain=any(not s['known'] for _,s in path)
    if uncertain:family='unknown_evidence'
    elif a is None:family='initial_support_establishment'
    elif added and removed:family='support_exchange'
    elif added:family='support_expansion'
    elif removed:family='support_reduction'
    elif arrived:family='same_part_recontact'
    elif changed_surface:family='surface_transfer'
    elif len(path)>1:family='support_cycle'
    else:family='support_hold'
    raw=[]
    for t in range(start,stop):
        m=states[t]['raw_mask']
        if not raw or m!=raw[-1]:raw.append(m)
    tags=[]
    if any(m==0 and bool(data['known'][t].all()) for t,m in zip(range(start,stop),[s['raw_mask'] for s in states[start:stop]])):tags.append('six_parts_no_valid_contact')
    if changed_surface:tags.append('surface_transfer')
    if a is not None and arrived&a:tags.append('recontact_existing_part')
    if any(bool(data['active'][t,p] and data['known'][t,p] and data['contact_material_speed'][t,p]>.5) for t in range(start,stop) for p in range(6)):tags.append('material_motion_during_contact')
    if bool(((data['active'][start:stop]&data['known'][start:stop])&~data['contact_row_stable'][start:stop]).any()):tags.append('row_switch_or_unestablished_rows')
    part_path=[s['mask'] for _,s in path]
    structural=[]
    for m in part_path:
        if not structural or structural[-1]!=m:structural.append(m)
    if a is None:structural=[None]+structural
    mirror=lambda m:None if m is None else sum(((m>>(i^1))&1)<<i for i in range(6))
    text=' → '.join(mask_name(m) for m in structural)
    mirrored=' → '.join(mask_name(mirror(m)) for m in structural)
    continuous_mask=0 if a is None else a
    for _,state in path:continuous_mask &= state['mask']
    return dict(**classify_limb_roles(states,data,start,stop,new,temporal),maintained_parts=[PARTS[p] for p in range(6) if continuous_mask&(1<<p)],recontact_parts=[PARTS[p] for p in range(6) if a is not None and (arrived&a)&(1<<p)],family=family,entry_mask=a,exit_mask=b,entry=mask_name(a),exit=mask_name(b),
        added_parts=[PARTS[p] for p in range(6) if added&(1<<p)],removed_parts=[PARTS[p] for p in range(6) if removed&(1<<p)],
        goal_parts=[PARTS[p] for p in range(6) if bool(new[p])],surface_changed_parts=[PARTS[p] for p in changed_surface],
        support_path=structural,topology_key=text,mirror_invariant_key=min(text,mirrored),
        raw_contact_path=raw,attachment_path=[dict(mask=s['mask'],attachments=s['attachments'],known=s['known']) for _,s in path],tags=tags)


def annotate_surface_transitions(data,spans,cases,phases):
    """Compare goal attachments with the last qualified support before release.

    Search only within this native trace/run, including execution context that
    precedes a logical topology boundary. Native shape IDs stay model-scoped.
    """
    from bisect import bisect_right
    groups={i:[] for i in range(len(spans))}
    for phase in phases:groups[phase['trace_id']].append(phase)
    starts={i:[x['start'] for x in group] for i,group in groups.items()}
    def surface(trace,frame,part):
        i=bisect_right(starts[trace],frame-spans[trace]['begin'])-1
        return groups[trace][i]['attachments'][part]
    for case in cases:
        trace=case['trace_id'];span=spans[trace];goal=case['goal'];relations=[]
        for part,new in enumerate(case['new']):
            if not new:continue
            events=[e for e in span['transfer_events'] if e['id'] in case['transfer_ids'] and e['part']==part and e['arrival'] is not None and case['topology_floor']+e['arrival']<=goal]
            event=events[-1]
            destination=surface(trace,goal,part)
            release=case['topology_floor']+event['start']
            if event['source_support_stop'] is not None:
                previous=case['topology_floor']+event['source_support_stop']-1
                source=surface(trace,previous,event['source_part'])
                relation='same_surface' if source==destination else 'surface_set_change' if set(source)&set(destination) else 'cross_surface'
            else:previous=None;source=None;relation='first_observed_support'
            relations.append(dict(part=PARTS[part],part_index=part,transfer_id=event['id'],release_frame=release-span['begin'],previous_qualified_frame=None if previous is None else previous-span['begin'],source_shapes=source,target_shapes=destination,relation=relation,model_fingerprint=span['model_fingerprint']))
        topology=case['topology'];topology['landing_surfaces']=relations
        topology['surface_pattern']=';'.join(f"{x['part']}:{x['relation']}" for x in relations)
        tags=[x['relation'] for x in relations]
        if any(x['relation'] in ('cross_surface','surface_set_change') for x in relations):tags.append('surface_transfer')
        topology['tags']=sorted(set(topology['tags']+tags))
        topology['mirror_topology_class']=topology['mirror_role_key']
        union=mask_value(case['new'])
        for state in topology['support_path']:
            if state is not None:union|=state
        topology['support_modality']='+'.join(label for bits,label in ((3,'F'),(12,'H'),(48,'K')) if union&bits) or 'none'


def native_contact_phases(data,surfaces):
    """Exact allocated primary contact topology, including unqualified touches."""
    states=[dict(mask=mask_value(a&k),known_mask=mask_value(k),attachments=[list(x) for x in surf]) for a,k,surf in zip(data['active'],data['known'],surfaces)]
    phases=[];start=0
    for t in range(1,len(states)+1):
        if t==len(states) or states[t]!=states[start]:
            phases.append(dict(start=start,stop=t,**states[start]));start=t
    return phases
