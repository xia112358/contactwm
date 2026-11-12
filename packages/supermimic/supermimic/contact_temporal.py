"""One offline contact timeline for support, transfers, forms and partitioning.

Native observations and point-loss validity are never rewritten. Sustained
support is retrospective: its first bearing sample and its confirmation time
are both retained. Segment boundaries never borrow frames from another segment.
"""
import torch
from .contact_stability import CONTRACT

LIMB_PARTS=((0,4),(1,5),(2,),(3,))
TEMPORAL_CONTRACT=dict(version='support_establishment_events_v16',
    enter_load_n=30.,exit_load_n=15.,enter_frames=5,exit_frames=2,
    bridge_gap_frames=6,merge_window_frames=6,max_bridge_motion_m=.02,
    motion_excursion_m=.02,dt=.02,
    settle_position_m=.012,settle_angle_rad=.15,settle_material_speed_m_s=.15,
    departure_speed_m_s=.12,departure_angular_speed_rad_s=.8,
    support='substep recordings use their versioned support_sampling_contract: fast observed support is distinct from independent bearing events; control-only recordings use five bearing samples with converged same-solve pose; material motion remains below limit or measurably decays from initial excess without recrossing and with below-limit window mean; onset distinct from confirmation',
    bridge='known same-surface short gaps without measured motion; cannot create bearing evidence',
    transfer='observed support engagement, disengagement or repositioning; no previous contact or native release required for loading; internal rolling remains continuous support',
    forms='observed established main contact only; no form during flight; changes at actual target support onset',
    merge='fixed clusters anchored to first original candidate cut; preserve observed main forms and independent posture phases; settling and inherited activity may join the preceding action; a new measured departure or reach after its completion preserves that completion boundary',
    raw='raw contact, rows and point-loss masks are unchanged')


def _runs(bits):
    start=None
    for t,on in enumerate(bits.tolist()+[False]):
        if on and start is None:start=t
        elif not on and start is not None:
            yield start,t
            start=None


def participation_interval(events,start,stop):
    """Support establishments define events; withdrawal/reach describe activity.

    A future target is never assigned before its observed support onset.
    Activity alone creates no cut. A newly starting activity may preserve
    an existing establishment-completion boundary; inherited activity cannot.
    """
    active=[];arrivals=[]
    for event in events:
        if any(a<stop and b>start for a,b in event['activity_ranges']):active.append(event)
        if event['arrival'] is not None and start<=event['arrival']<stop:arrivals.append(event)
    return dict(active=active,arrivals=arrivals)


def segmentation_contact_state(data):
    active,known=data['active'],data['known'];T=len(active)
    if 'substep_support_state' in data:
        active=data['substep_contact_active'].any(1)
        known=data['substep_contact_known'].all(1)
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
    if 'substep_support_state' in data:
        from .contact_substep_support import control_support_episodes
        episodes=control_support_episodes(data)
        if 'support_placement_resume' not in data:
            # Preserve the same measured short-gap debounce at the control grid.
            # Bridging cannot create an establishment or qualify an unobserved part.
            joined=[]
            for part in range(6):
                previous=None
                for ep in (e for e in episodes if e['part']==part):
                    if previous is not None and previous['surface']==ep['surface'] and bool(bridge[previous['stop']:ep['start'],part].all()):
                        previous['stop']=ep['stop']
                    else:
                        previous=ep;joined.append(ep)
            episodes=joined
    else:
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
                    material=data['contact_material_speed'][window,p]
                    limit=TEMPORAL_CONTRACT['settle_material_speed_m_s']
                    below=bool((material[1:]<=limit).all())
                    above=material>limit
                    # An observed impact can settle within the load/pose window.
                    # Late spikes, renewed scraping, or sustained sliding do not
                    # satisfy this decay pattern. A missing first motion sample
                    # cannot stand in for measured deceleration.
                    settling=bool(material_known[begin,p] and above[0] and material.mean()<=limit
                        and not (above[1:]&~above[:-1]).any()
                        and ((material[1:]<material[:-1])|~above[1:]).all())
                    if not (below or settling):continue
                    displacement=float((position[window,p]-position[begin,p]).norm(dim=-1).max())
                    R=rotation[window,p]@rotation[begin,p].T
                    angle=float(torch.acos(((R.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).max())
                    if displacement<=TEMPORAL_CONTRACT['settle_position_m'] and angle<=TEMPORAL_CONTRACT['settle_angle_rad']:
                        episode=dict(part=p,start=begin,confirmed=t,surface=face)
            if episode is not None:
                episode['stop']=T;episodes.append(episode)
    result=describe_support_episodes(data,episodes,active,known,bridge,moving)
    return dict(result,bearing=bearing,material_motion_known=material_known,bridged_gaps=audit)


def describe_support_episodes(data,episodes,active,known,bridge,moving):
    """Describe participation from already qualified events; never infer loads."""
    T=len(active);width=TEMPORAL_CONTRACT['enter_frames']
    effective=active|bridge;position=data['native_part_position']
    support=torch.zeros_like(active)
    episodes.sort(key=lambda x:(x['start'],x['part']))
    for ep in episodes:support[ep['start']:ep['stop'],ep['part']]=True
    for foot,knee in LIMB_PARTS[:2]:
        simultaneous=(support[:,foot]&support[:,knee]).nonzero().flatten()
        if len(simultaneous):raise ValueError(f'Simultaneous sustained foot/knee main contacts at frame {int(simultaneous[0])}, parts {foot}/{knee}')
    placements=episodes;resumptions=[]
    if 'support_placement_resume' in data:
        from .contact_placement import placement_episodes
        placements,resumptions=placement_episodes(data,episodes)
    placement_support=torch.zeros_like(support);placement_forms=torch.full((T,4),-1,dtype=torch.long,device=active.device)
    for ep in placements:
        placement_support[ep['start']:ep['stop'],ep['part']]=True
        limb=next(i for i,parts in enumerate(LIMB_PARTS) if ep['part'] in parts)
        placement_forms[ep['start']:ep['stop'],limb]=ep['part']
    # Motion delimits participation, never native contact. Arms carried by the
    # torso are idle; direction reversals within a reach do not end the reach.
    from .rotations import quat_to_mat
    root_rotation=quat_to_mat(data['s1'][:,3:7])
    local=(root_rotation[:,None].transpose(-1,-2)@(data['native_part_position']-data['s1'][:,None,:3])[...,None]).squeeze(-1)
    participation_position=position.clone()
    participation_position[:,2:4]=local[:,2:4]
    # Use the existing settled-pose window for idle detection. Per-frame
    # velocity spikes in resting arms must not join long stretches of free motion.
    participation_quiet=torch.zeros_like(active)
    for end in range(width,T+1):
        window=participation_position[end-width:end]
        settled=((window-window[:1]).norm(dim=-1).amax(0)<=TEMPORAL_CONTRACT['settle_position_m'])&known[end-width:end].all(0)
        participation_quiet[end-width:end] |= settled
    forms=torch.full((T,4),-1,dtype=torch.long,device=active.device)
    transfers=[];form_changes=[]
    for limb,parts in enumerate(LIMB_PARTS):
        for lo,hi in _runs(known[:,list(parts)].all(-1)):
            observed=[ep for ep in placements if ep['part'] in parts and lo<=ep['start']<hi]
            previous=None
            for ep in observed:
                part=ep['part'];goal=ep['start']
                observed_frames=active[goal:ep['stop'],part]&known[goal:ep['stop'],part]&support[goal:ep['stop'],part]
                forms[goal:ep['stop'],limb][observed_frames]=part
                changed=previous is not None and previous['part']!=part
                if changed:form_changes.append(dict(frame=goal,limb=limb,source_part=previous['part'],target_part=part))
                floor=lo if previous is None else previous['stop']
                # A sustained idle bout separates an earlier withdrawal from a
                # later engagement. No requirement to approach monotonically.
                quiet=[(floor+a,floor+b) for a,b in _runs(participation_quiet[floor:goal,part]) if b-a>=width]
                reach=quiet[-1][1] if quiet else floor
                ranges=[[reach,goal+1]]
                # Initial support can be shorter than the confirmation window.
                # A recorded native departure still establishes an observed
                # event onset; lack of an old settled episode must not erase it.
                initial_departure=previous is None and any(
                    a>0 and b-a>=TEMPORAL_CONTRACT['exit_frames'] and bool(active[lo+a-1,part])
                    for a,b in _runs(~effective[lo:goal,part]))
                if previous is not None:
                    old=previous['part']
                    rest=[floor+a for a,b in _runs(participation_quiet[floor:goal,old]) if b-a>=width]
                    departure_stop=rest[0] if rest else goal
                    if floor<goal:ranges.append([floor,max(floor+1,departure_stop)])
                # First establishment needs observed departure, a reach after
                # idle, or measured motion to justify a completion boundary.
                # Stationary initial qualification alone does not. Same-placement
                # resumptions have already been grouped above.
                if goal>lo:
                    begin=min(a for a,b in ranges)
                    poses=participation_position[max(lo,begin-1):goal+1,part]
                    motion=float((poses-poses[:1]).norm(dim=-1).max())
                    transfers.append(dict(limb=limb,part=part,start=begin,stop=goal+1,arrival=goal,confirmed=ep['confirmed'],
                        source_part=None if previous is None else previous['part'],source_support_start=None if previous is None else previous['start'],
                        source_support_stop=None if previous is None else previous['stop'],target_start=goal,
                        form_change=changed,motion_m=motion,initial=previous is None,terminal=False,
                        boundary_eligible=previous is not None or reach>lo or initial_departure or motion>TEMPORAL_CONTRACT['motion_excursion_m'],
                        participation_kind='engage' if previous is None else 'reallocate',completed=True,activity_ranges=sorted(ranges)))
                previous=ep
            if previous is not None and previous['stop']<hi:
                begin=previous['stop'];part=previous['part']
                quiet=[begin+a for a,b in _runs(participation_quiet[begin:hi,part]) if b-a>=width]
                end=max(begin+1,quiet[0]) if quiet else hi
                poses=participation_position[max(lo,begin-1):end,part]
                motion=float((poses-poses[:1]).norm(dim=-1).max())
                transfers.append(dict(limb=limb,part=part,start=begin,stop=end,arrival=None,
                    source_part=part,source_support_start=previous['start'],source_support_stop=begin,
                    target_start=None,form_change=False,motion_m=motion,initial=False,terminal=hi==T,
                    participation_kind='disengage',completed=bool(quiet),activity_ranges=[[begin,end]]))
    transfers.sort(key=lambda x:(x['start'],x['part']))
    for i,event in enumerate(transfers):event['id']=i
    return dict(active=effective,support=support,moving=moving,bridged=bridge,
        episodes=episodes,placements=placements,placement_support=placement_support,placement_forms=placement_forms,resumptions=resumptions,transfers=transfers,forms=forms,form_changes=form_changes)
