"""Partition once from support establishments and independent posture phases.

Order: preserve actual contact forms; coalesce neighboring establishment cuts;
retain new activity after completion; absorb only settling/inherited activity.
A continuous posture phase owns its internal confirmation latency. Withdrawal
can preserve an existing completion boundary, but creates no cut by itself.
Reviewed edge exclusions are a separate dataset operation in the rebuild tool.
No case IDs, minimum duration filters, or maximum-duration cuts belong here.
"""
import torch


def _same_contact_surface(data,part,start,stop):
    observed=data['active'][start:stop,part]&data['known'][start:stop,part]
    return bool(data['known'][start:stop,part].all() and
        (data['surface_local'][start:stop,part][observed]==data['surface_local'][start,part]).all())


def _boundary_decision(data,temporal,events,posture,form_cuts,arrivals_at,start,boundary,stop,source_stop,nearby):
    """One source of truth for joining events or absorbing their continuation.

    Scheduling decides which original neighbors may be considered. This
    function alone decides whether their evidence permits removing the cut.
    """
    from .contact_temporal import participation_interval,TEMPORAL_CONTRACT
    decision=dict(join_events=None,absorb_continuation=None,inherited_phase=None)
    single_form=all(len([p for p in temporal['forms'][start:stop,limb].unique().tolist() if p>=0])<=1 for limb in range(4))
    if not single_form:return decision

    arrivals=[e for e in events if start<=e['arrival']<stop]
    parts=[e['part'] for e in arrivals]
    posture_boundary=any(boundary in (p['start'],p['stop']) for p in posture)
    if not posture_boundary and len(parts)==len(set(parts)):
        redundant_form=boundary in form_cuts and boundary not in arrivals_at
        neighboring=any(e['arrival']<boundary for e in arrivals) and any(e['arrival']>=boundary for e in arrivals)
        coherent=all(_same_contact_surface(data,e['part'],e['arrival'],arrivals[-1]['arrival']+1) for e in arrivals[:-1])
        if redundant_form:decision['join_events']='redundant_observed_form_cut'
        elif neighboring and nearby and coherent:decision['join_events']='adjacent_support_allocation_events'

    right=participation_interval(temporal['transfers'],boundary,stop)
    own_posture=any(boundary<=p['start']<stop for p in posture)
    new_activity=any(boundary<=a<stop and e['motion_m']>TEMPORAL_CONTRACT['motion_excursion_m']
        for e in right['active'] for a,b in e['activity_ranges'])
    same_posture=any(p['start']<=start<boundary<stop<=p['stop'] for p in posture)
    independent_right=bool(right['arrivals']) or own_posture or (new_activity and not same_posture)
    if independent_right:return decision
    decision['inherited_phase']=next((p['kind'] for p in posture if p['start']<=start<p['stop']==boundary),None)
    decision['absorb_continuation']=('participation_without_support_establishment' if right['active'] else
        'terminal_settling_without_independent_event' if stop==source_stop else 'settling_without_independent_event')
    return decision


def load_transfer_slices(data,temporal=None,source_motion=None):
    """Partition once from shared transfer arrivals and main-form boundaries."""
    from .contact_temporal import segmentation_contact_state,TEMPORAL_CONTRACT
    if temporal is None:temporal=segmentation_contact_state(data)
    hits=torch.zeros_like(data['active']);groups=[]
    form_cuts={x['frame'] for x in temporal['form_changes']}
    events=sorted((e for e in temporal['transfers'] if e['arrival'] is not None),key=lambda e:(e['arrival'],e['part']))
    for event in events:
        t,p=event['arrival'],event['part'];hits[t,p]=True
        if not event['boundary_eligible']:continue
        coherent=bool(groups and t-groups[-1][0]['arrival']<=TEMPORAL_CONTRACT['merge_window_frames'])
        if coherent:
            coherent=not any(groups[-1][0]['arrival']<b<=t for b in form_cuts)
            for earlier in groups[-1]:
                f,q=earlier['arrival'],earlier['part']
                coherent &= q!=p and bool(temporal['placement_support'][f:t+1,q].all()) and _same_contact_surface(data,q,f,t+1)
        if coherent:groups[-1].append(event)
        else:groups.append([event])
    arrivals_at={g[-1]['arrival']+1:g for g in groups}
    # Independent arrivals are never erased merely for being close to a
    # form change. Posture phases are actual motion phases, not max-length cuts.
    first,last=(0,len(hits)) if source_motion is None else (source_motion['start'],source_motion['stop'])
    posture=[] if source_motion is None else [dict(p) for p in source_motion['posture_phases']]
    # A form onset and a posture onset inside the SAME support confirmation
    # interval describe one transition. Keep the actual form boundary; do not
    # carve its confirmation latency into a separate motion segment.
    for phase in posture:
        overlaps=[e for e in events if e['arrival'] in form_cuts and e['arrival']<phase['start']<=e['confirmed']]
        if overlaps:phase['start']=max(e['arrival'] for e in overlaps)

    pose_cuts={v for phase in posture for v in (phase['start'],phase['stop'])}
    boundaries=sorted({first,last,*[b for b in form_cuts|pose_cuts|set(arrivals_at) if first<b<last]})
    # Resolve each cluster of close candidate cuts against its full interval.
    # Clusters are anchored to the FIRST original cut, never extended through
    # a chain of neighboring cuts. They cannot consume a neighbor already
    # changed by a previous cluster.
    clusters=[]
    for b in boundaries[1:-1]:
        if b in pose_cuts:continue
        if clusters and b-clusters[-1][0]<=TEMPORAL_CONTRACT['merge_window_frames'] and not any(clusters[-1][-1]<p<b for p in pose_cuts):clusters[-1].append(b)
        else:clusters.append([b])
    original=boundaries[:];removed=[];merge_reasons={}
    for cluster in clusters:
        outer_left=original[original.index(cluster[0])-1]
        if outer_left in removed:continue
        for b in cluster:
            i=boundaries.index(b);a,z=boundaries[i-1],boundaries[i+1]
            close=len(cluster)>1 or min(b-a,z-b)<=TEMPORAL_CONTRACT['merge_window_frames'] or any(abs(b-f)<=TEMPORAL_CONTRACT['merge_window_frames'] for f in form_cuts)
            decision=_boundary_decision(data,temporal,events,posture,form_cuts,arrivals_at,a,b,z,last,close)
            reason=decision['join_events']
            if reason is not None:
                boundaries.remove(b);removed.append(b);merge_reasons[b]=reason
    # A new withdrawal/reach may preserve an existing completion boundary.
    # Settling and inherited activity cannot create a separate tail.
    tail_phases={};settling_removed=[]
    for i in range(1,len(boundaries)-1):
        previous,a,z=boundaries[i-1:i+2]
        if previous in settling_removed:continue
        decision=_boundary_decision(data,temporal,events,posture,form_cuts,arrivals_at,previous,a,z,last,False)
        reason=decision['absorb_continuation']
        if reason is not None:
            phase=decision['inherited_phase']
            if phase is not None:tail_phases[z]=phase
            settling_removed.append(a);merge_reasons[a]=reason
    boundaries=[b for b in boundaries if b not in settling_removed]
    removed+=settling_removed
    cuts=[]
    for begin,end in zip(boundaries,boundaries[1:]):
        arrivals=[e for e in events if begin<=e['arrival']<end];new=torch.zeros(6,dtype=torch.bool,device=hits.device);releases=[]
        for e in arrivals:
            t,p=e['arrival'],e['part'];new[p]=True
            # This is only compatibility with the old endpoint trainer. The
            # event itself retains its true source interval across boundaries.
            inactive=data['known'][begin:t,p]&~data['active'][begin:t,p]
            pairs=(inactive.unfold(0,2,1).all(-1).nonzero().flatten() if len(inactive)>=2 else torch.empty(0,dtype=torch.long))
            releases.append(begin+int(pairs[0]) if len(pairs) else -1)
        keep=(data['active'][begin:end]&data['known'][begin:end]).all(0)&~new
        incomplete=not arrivals and any(
            e['arrival'] is None and not e['completed'] and begin<=e['start']<end
            and e['motion_m']>TEMPORAL_CONTRACT['motion_excursion_m'] for e in temporal['transfers'])
        phase=next((x['kind'] for x in posture if x['start']<=begin and end<=x['stop']),
            'incomplete_transfer' if incomplete else 'contact_motion')
        if end in tail_phases:phase=tail_phases[end]
        cuts.append(dict(start=begin,stop=end,goal=end-1,new=new,keep=keep,
            support=keep&temporal['support'][begin:end].all(0),
            arrivals=[(e['arrival'],e['part']) for e in arrivals],releases=releases,
            transfer_ids=[e['id'] for e in temporal['transfers'] if e['start']<end and e['stop']>begin],
            phase_kind=phase,
            boundary_reasons=(['main_form_change'] if end in form_cuts else [])+(['sustained_arrival'] if end in arrivals_at else [])+(['posture_phase'] if end in pose_cuts else [])+(['effective_end'] if end==last else []),
            contact_goal_complete=bool(arrivals) and all(t>=0 for t in releases) and all(bool(temporal['support'][e['arrival']:end,e['part']].all()) for e in arrivals),
            merged_arrival_boundaries=[dict(arrival_boundary=b,reason=merge_reasons[b]) for b in removed if begin<b<end],
            kind='load_contact_transfer' if arrivals else 'root_motion_tail'))
    return cuts,hits
