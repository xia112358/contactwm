"""Partition complete trajectories from the shared offline contact timeline."""
import torch


def load_transfer_slices(data,temporal=None,source_motion=None):
    """Partition once from shared transfer arrivals and main-form boundaries."""
    from .contact_temporal import segmentation_contact_state,TEMPORAL_CONTRACT
    if temporal is None:temporal=segmentation_contact_state(data)
    hits=torch.zeros_like(data['active']);groups=[]
    form_cuts={x['frame'] for x in temporal['form_changes']}
    events=sorted((e for e in temporal['transfers'] if e['arrival'] is not None),key=lambda e:(e['arrival'],e['part']))
    for event in events:
        t,p=event['arrival'],event['part'];hits[t,p]=True
        coherent=bool(groups and t-groups[-1][0]['arrival']<=TEMPORAL_CONTRACT['merge_window_frames'])
        if coherent:
            coherent=not any(groups[-1][0]['arrival']<b<=t for b in form_cuts)
            for earlier in groups[-1]:
                f,q=earlier['arrival'],earlier['part']
                coherent &= q!=p and bool(data['known'][f:t+1,q].all() and temporal['support'][f:t+1,q].all())
                observed=data['active'][f:t+1,q]&data['known'][f:t+1,q]
                coherent &= bool((data['surface_local'][f:t+1,q][observed]==data['surface_local'][f,q]).all())
        if coherent:groups[-1].append(event)
        else:groups.append([event])
    arrivals_at={g[-1]['arrival']+1:g for g in groups}
    # Independent arrivals are never erased merely for being close to a
    # form change. Posture phases are actual motion phases, not max-length cuts.
    first,last=(0,len(hits)) if source_motion is None else (source_motion['start'],source_motion['stop'])
    posture=[] if source_motion is None else source_motion['posture_phases']
    pose_cuts={v for phase in posture for v in (phase['start'],phase['stop'])}
    boundaries=sorted({first,last,*[b for b in form_cuts|pose_cuts|set(arrivals_at) if first<b<last]})
    # Coalesce an adjacent event boundary only if the resulting phase cannot
    # contain two separate arrivals of the same part. Work on the original
    # boundary list once; do not recursively expand segments.
    removed=[]
    mandatory=form_cuts|pose_cuts|{first,last}
    for i,b in enumerate(boundaries[1:-1],1):
        if b in mandatory or boundaries[i-1] in removed or boundaries[i+1] in removed:continue
        if not any(abs(b-f)<=TEMPORAL_CONTRACT['merge_window_frames'] for f in form_cuts):continue
        a,z=boundaries[i-1],boundaries[i+1]
        parts=[e['part'] for e in events if a<=e['arrival']<z]
        if len(parts)==len(set(parts)):removed.append(b)
    boundaries=[b for b in boundaries if b not in removed]
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
        cuts.append(dict(start=begin,stop=end,goal=end-1,new=new,keep=keep,
            support=keep&temporal['support'][begin:end].all(0),
            arrivals=[(e['arrival'],e['part']) for e in arrivals],releases=releases,
            transfer_ids=[e['id'] for e in temporal['transfers'] if e['start']<end and e['stop']>begin],
            phase_kind=next((x['kind'] for x in posture if x['start']<=begin and end<=x['stop']),'contact_motion'),
            boundary_reasons=(['main_form_change'] if end in form_cuts else [])+(['sustained_arrival'] if end in arrivals_at else [])+(['posture_phase'] if end in pose_cuts else [])+(['effective_end'] if end==last else []),
            contact_goal_complete=bool(arrivals) and all(t>=0 for t in releases) and all(bool(temporal['support'][e['arrival']:end,e['part']].all()) for e in arrivals),
            merged_arrival_boundaries=[dict(arrival_boundary=b,reason='same_transition_unique_part_arrivals') for b in removed if begin<b<end],
            kind='load_contact_transfer' if arrivals else 'root_motion_tail'))
    return cuts,hits
