"""One offline contact timeline for support, transfers, forms and partitioning.

Native observations and point-loss validity are never rewritten. Sustained
support is retrospective: its first bearing sample and its confirmation time
are both retained. Segment boundaries never borrow frames from another segment.
"""
import torch
from .contact_stability import CONTRACT

LIMB_PARTS=((0,4),(1,5),(2,),(3,))
TEMPORAL_CONTRACT=dict(version='native_release_contact_motion_v5',
    enter_load_n=30.,exit_load_n=15.,enter_frames=5,exit_frames=2,
    bridge_gap_frames=6,merge_window_frames=6,max_bridge_motion_m=.02,
    motion_excursion_m=.02,release_frames=2,dt=.02,
    settle_position_m=.012,settle_angle_rad=.15,settle_material_speed_m_s=.15,
    departure_speed_m_s=.12,departure_angular_speed_rad_s=.8,
    support='five bearing samples with converged same-solve pose and observed material motion; onset distinct from confirmation',
    bridge='known same-surface short gaps without measured motion; cannot create bearing evidence',
    transfer='actual part motion between settled supports, after measured native release; within-part rolling and row changes are continuous contact',
    forms='destination form during a form-changing flight; boundary at old support end',
    merge='bounded nearby arrivals while prior support persists; never across a main-form boundary',
    raw='raw contact, rows and point-loss masks are unchanged')


def _runs(bits):
    start=None
    for t,on in enumerate(bits.tolist()+[False]):
        if on and start is None:start=t
        elif not on and start is not None:
            yield start,t
            start=None


def segmentation_contact_state(data):
    active,known=data['active'],data['known'];T=len(active)
    bridge=torch.zeros_like(active);audit=[]
    for p in range(6):
        indices=(active[:,p]&known[:,p]).nonzero().flatten().tolist()
        for previous,current in zip(indices,indices[1:]):
            if not 0<current-previous-1<=TEMPORAL_CONTRACT['bridge_gap_frames']:continue
            if not bool(known[previous:current+1,p].all()):continue
            if int(data['surface_local'][previous,p])!=int(data['surface_local'][current,p]):continue
            position=data['native_part_position'][previous:current+1,p]
            motion=float((position-position[:1]).norm(dim=-1).max())
            if motion>TEMPORAL_CONTRACT['max_bridge_motion_m']:continue
            bridge[previous+1:current,p]=True
            audit.append(dict(part=p,start=previous+1,stop=current,motion_m=motion))
    effective=active|bridge
    bearing=known&active&(data['contact_load']>TEMPORAL_CONTRACT['enter_load_n'])&(data['contact_material_speed']<=CONTRACT['max_material_speed_m_s'])
    position=data['native_part_position'];rotation=data['native_part_rotation']
    speed=torch.zeros_like(data['contact_load']);angular=torch.zeros_like(speed)
    speed[1:]=(position[1:]-position[:-1]).norm(dim=-1)/TEMPORAL_CONTRACT['dt']
    relative=rotation[1:]@rotation[:-1].transpose(-1,-2)
    angular[1:]=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))/TEMPORAL_CONTRACT['dt']
    moving=(speed>TEMPORAL_CONTRACT['departure_speed_m_s'])|(angular>TEMPORAL_CONTRACT['departure_angular_speed_rad_s'])
    material_known=torch.zeros_like(active)
    loaded=known&active&(data['contact_load']>0)
    material_known[1:]=loaded[1:]&loaded[:-1]
    support=torch.zeros_like(active);episodes=[];width=TEMPORAL_CONTRACT['enter_frames']
    for p in range(6):
        off=0;episode=None;surface=None
        for t in range(T):
            observed=bool(known[t,p] and active[t,p])
            face=int(data['surface_local'][t,p]) if observed else None
            changed=observed and surface is not None and face!=surface
            if not bool(known[t,p]) or changed:
                if episode is not None:
                    episode['stop']=t;episodes.append(episode);episode=None
                off=0;surface=None
            if not bool(known[t,p]):continue
            if observed:surface=face
            absent=(not bool(bridge[t,p])) and (not observed or float(data['contact_load'][t,p])<=TEMPORAL_CONTRACT['exit_load_n'])
            off=off+1 if absent else 0
            if episode is not None and off>=TEMPORAL_CONTRACT['exit_frames']:
                episode['stop']=t-off+1;episodes.append(episode);episode=None
            if episode is None and t+1>=width:
                begin=t-width+1;window=slice(begin,t+1)
                if not bool(bearing[window,p].all() and material_known[begin+1:t+1,p].all()):continue
                if not bool((data['surface_local'][window,p]==face).all()):continue
                if not bool((data['contact_material_speed'][begin+1:t+1,p]<=TEMPORAL_CONTRACT['settle_material_speed_m_s']).all()):continue
                displacement=float((position[window,p]-position[begin,p]).norm(dim=-1).max())
                R=rotation[window,p]@rotation[begin,p].T
                angle=float(torch.acos(((R.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).max())
                if displacement<=TEMPORAL_CONTRACT['settle_position_m'] and angle<=TEMPORAL_CONTRACT['settle_angle_rad']:
                    episode=dict(part=p,start=begin,confirmed=t,surface=face)
        if episode is not None:
            episode['stop']=T;episodes.append(episode)
    episodes.sort(key=lambda x:(x['start'],x['part']))
    for ep in episodes:support[ep['start']:ep['stop'],ep['part']]=True
    for foot,knee in LIMB_PARTS[:2]:
        simultaneous=(support[:,foot]&support[:,knee]).nonzero().flatten()
        if len(simultaneous):raise ValueError(f'Simultaneous sustained foot/knee main contacts at frame {int(simultaneous[0])}, parts {foot}/{knee}')
    forms=torch.full((T,4),-1,dtype=torch.long,device=active.device)
    transfers=[];form_changes=[]
    for limb,parts in enumerate(LIMB_PARTS):
        for lo,hi in _runs(known[:,list(parts)].all(-1)):
            observed=[ep for ep in episodes if ep['part'] in parts and lo<=ep['start']<hi]
            if not observed:continue
            previous=None
            for ep in observed:
                begin=lo if previous is None else previous['stop']
                part=ep['part'];forms[begin:ep['stop'],limb]=part
                changed=previous is not None and previous['part']!=part
                if changed:form_changes.append(dict(frame=begin,limb=limb,source_part=previous['part'],target_part=part))
                end=ep['start']+1
                if previous is None:
                    # Locate the last local quiet bout before an initial reach.
                    for a,b in _runs(~moving[lo:ep['start'],part]):
                        if b-a>=width:begin=lo+b
                release_floor=lo if previous is None else previous['stop']
                inactive=known[release_floor:ep['start'],part]&~effective[release_floor:ep['start'],part]
                releases=[release_floor+a for a,b in _runs(inactive) if b-a>=TEMPORAL_CONTRACT['release_frames']]
                released=bool(releases)
                if released and not changed:begin=max(begin,releases[0])
                position=data['native_part_position'][max(lo,begin-1):end,part]
                motion=float((position-position[:1]).norm(dim=-1).max())
                if (released and motion>TEMPORAL_CONTRACT['motion_excursion_m']) or changed:
                    transfers.append(dict(limb=limb,part=part,start=begin,stop=end,arrival=ep['start'],confirmed=ep['confirmed'],
                        source_part=None if previous is None else previous['part'],source_support_start=None if previous is None else previous['start'],
                        source_support_stop=None if previous is None else previous['stop'],target_start=ep['start'],
                        form_change=changed,motion_m=motion,initial=previous is None,terminal=False))
                previous=ep
            begin=previous['stop'];part=previous['part'];forms[begin:hi,limb]=part
            if begin<hi:
                inactive=known[begin:hi,part]&~effective[begin:hi,part]
                releases=[begin+a for a,b in _runs(inactive) if b-a>=TEMPORAL_CONTRACT['release_frames']]
                released=bool(releases)
                if released:begin=releases[0]
                position=data['native_part_position'][max(lo,begin-1):hi,part]
                motion=float((position-position[:1]).norm(dim=-1).max())
                if released and motion>TEMPORAL_CONTRACT['motion_excursion_m']:
                    transfers.append(dict(limb=limb,part=part,start=begin,stop=hi,arrival=None,
                        source_part=part,source_support_start=previous['start'],source_support_stop=begin,
                        target_start=None,form_change=False,motion_m=motion,initial=False,terminal=hi==T))
    # Unsupported arms may simply travel with the torso. Only motion bouts
    # adjoining a real support release/arrival participate in that transfer.
    from .rotations import quat_to_mat
    root_rotation=quat_to_mat(data['s1'][:,3:7])
    local=(root_rotation[:,None].transpose(-1,-2)@(data['native_part_position']-data['s1'][:,None,:3])[...,None]).squeeze(-1)
    local_moving=torch.zeros_like(active)
    local_moving[1:]=(local[1:]-local[:-1]).norm(dim=-1)/TEMPORAL_CONTRACT['dt']>.12
    for event in transfers:
        a,z,p=event['start'],event['stop'],event['part']
        if p not in (2,3):event['activity_ranges']=[[a,z]];continue
        bouts=list(_runs(local_moving[a:z,p]))
        joined=[]
        for x,y in bouts:
            if joined and x-joined[-1][1]<=3:joined[-1][1]=y
            else:joined.append([x,y])
        event['activity_ranges']=[[a+x,a+y] for x,y in joined if y-x>=2 and (x<=5 and event['source_part'] is not None or z-(a+y)<=5 and event['arrival'] is not None)]
        if event['arrival'] is not None:event['activity_ranges'].append([event['arrival'],event['arrival']+1])
    transfers.sort(key=lambda x:(x['start'],x['part']))
    for i,event in enumerate(transfers):event['id']=i
    return dict(active=effective,support=support,bearing=bearing,material_motion_known=material_known,moving=moving,bridged=bridge,bridged_gaps=audit,
        episodes=episodes,transfers=transfers,forms=forms,form_changes=form_changes)
