"""Reference-free actor inputs and measured-data WM learning for one demonstration.

No tracking-policy weights enter the contact actor. Optional DAgger regresses
expert actions on student states before WM optimization. The actor receives
current sensors and spatial contact/root goals, without a deadline by default.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from .networks import RunningNorm, mlp
from .mdp import flatten_state, unflatten_state
from .rotations import mat_to_quat, yaw_mat, mat_to_6d, rot_angle_sq

PARTS = ('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee')
LINKS = ('left_ankle_roll_link','right_ankle_roll_link','left_sphere_hand_link',
         'right_sphere_hand_link','left_knee_link','right_knee_link')
OBS_DIM = 232
GOAL_DIM = 24 + 3 + 6 + 6  # target XYZ/mask, pelvis position/orientation, keep-contact roles


class BCPlateau:
    """Stop a warm start after several insignificant full-dataset improvements."""
    def __init__(self,min_steps=200,patience=4,absolute=.005,relative=.02):
        if min_steps<0 or patience<1 or absolute<0 or relative<0:
            raise ValueError('Invalid BC plateau configuration')
        self.min_steps,self.patience,self.absolute,self.relative=min_steps,patience,absolute,relative
        self.reference=None;self.stale=0

    def update(self,step,value):
        import math
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite BC validation score')
        threshold=max(self.absolute,self.relative*self.reference) if self.reference is not None else 0.
        if self.reference is None or value<self.reference-threshold:
            self.reference=value;self.stale=0
        elif step>=self.min_steps:
            self.stale+=1
        return step>=self.min_steps and self.stale>=self.patience


class ContactActor(nn.Module):
    def __init__(self, hist=5, hidden=(512,256,128), include_time=False):
        super().__init__()
        self.hist = hist
        self.include_time = include_time
        self.norm = RunningNorm(OBS_DIM)
        self.net = mlp(hist*OBS_DIM+GOAL_DIM+int(include_time),list(hidden),29)
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.zero_()

    def forward(self, history, condition):
        return self.net(torch.cat((self.norm(history).flatten(-2),condition),-1)).clamp(-20,20)


def observation(S, previous_action, xi, default_q, context):
    _,R,_,w,q,qd = S
    # Deliberately omit the tracker's reference-relative orientation channel.
    omega = (R.transpose(-1,-2) @ w.unsqueeze(-1)).squeeze(-1)
    return torch.cat((q-default_q-xi[...,1:30],qd*.1,omega*.25,-R[...,2,:],
                      previous_action,context.features(S,xi)),-1)


class ContactClock:
    """Original demonstrated contact endpoints; no perturbation or target offset."""
    def __init__(self, data, hold=8, minimum_segment=20, stable_support=False, role_aware=False, boundary_policy="legacy_v1"):
        if boundary_policy not in ('legacy_v1','event_preserving_v2'):
            raise ValueError('Unknown contact boundary policy')
        if boundary_policy=='event_preserving_v2' and not role_aware:
            raise ValueError('Event-preserving boundaries require role-aware targets')
        suppressed=[]
        active = data['active'].cpu().numpy()
        surface=data['surface'].cpu().numpy() if 'surface' in data else None
        n = len(active)
        candidates = []
        records=[]
        for part in range(6):
            changed=np.zeros(n-1,bool) if surface is None else surface[1:,part]!=surface[:-1,part]
            onsets=np.flatnonzero(active[1:,part]&(~active[:-1,part]|changed))+1
            for onset in onsets:
                transfer=False
                previous=np.flatnonzero(active[:onset,part])
                if len(previous) and surface is not None:
                    previous=int(previous[-1])
                    if onset-previous-1<=6 and surface[onset,part]==surface[previous,part]:
                        # Boundary selection only: suppress a short same-face
                        # interruption, without smoothing the observed truth.
                        suppressed.append(dict(frame=int(onset),part=part,gap_frames=int(onset-previous-1),
                            reason='short same-face interruption; raw observations unchanged'))
                        continue
                    transfer=surface[onset,part]!=surface[previous,part]
                if not stable_support:
                    if onset+hold<=n and active[onset:onset+hold,part].all():
                        candidates.append(int(onset)+1)
                        records.append((int(onset)+1,part,2 if transfer else 1 if part>=2 else 0))
                    continue
                stop=int(onset)
                while stop<n and active[stop,part]:
                    stop+=1
                for t in range(int(onset),stop-hold+1):
                    load=data['load'][t:t+hold,part]
                    slip=data['slip'][t:t+hold,part]
                    if bool((load>=1).all() and torch.isfinite(slip).all() and (slip<=.05).all()):
                        candidates.append(t+1)
                        records.append((t+1,part,2 if transfer else 1 if part>=2 else 0))
                        break
        candidates=sorted(set(candidates))
        bounds = [0]
        for t in candidates:
            if t-bounds[-1]>=minimum_segment and n-t>=hold:
                bounds.append(t)
        bounds.append(n)
        groups=[]
        if role_aware:
            for record in sorted(records):
                t,p,_=record
                coherent=bool(groups and t-groups[-1][0][0]<=6)
                if coherent:
                    for first,part,_ in groups[-1]:
                        coherent&=bool(active[first-1:t,part].all())
                        if surface is not None:
                            coherent&=bool((surface[first-1:t,part]==surface[first-1,part]).all())
                if coherent:
                    groups[-1].append(record)
                else:
                    groups.append([record])
            bounds=[0,n]
            for group in sorted(groups,key=lambda g:(-max(r[2] for r in g),g[-1][0])):
                end=group[-1][0]
                if all(abs(end-b)>=minimum_segment for b in bounds):
                    bounds.append(end)
            bounds.sort()
            if boundary_policy=='event_preserving_v2':
                # Every coherent arrival is a boundary, even when adjacent
                # transfers are shorter than the old twenty-frame preference.
                bounds=sorted({0,n,*(group[-1][0] for group in groups)})
        self.boundary_audit=dict(policy=boundary_policy,
            minimum_segment_frames=None if boundary_policy=='event_preserving_v2' else minimum_segment,
            suppressed_recontacts=suppressed,
            arrival_groups=[dict(boundary=int(group[-1][0]),retained=group[-1][0] in bounds,
                events=[dict(frame=int(t-1),part=int(part),surface_transfer=priority==2) for t,part,priority in group]) for group in groups],
            departures=[dict(frame=int(t),part=part) for part in range(6)
                for t in np.flatnonzero(active[:-1,part]&~active[1:,part])+1])
        # Consecutive endpoint goals span the whole example, including its tail.
        end_for_frame = np.empty(n,np.int64)
        previous = 0
        for end in bounds[1:]:
            stop = min(end+hold-1,n)
            end_for_frame[previous:stop] = end
            previous = stop
        self.ends = torch.tensor(end_for_frame,device=data['s'].device)
        self.event_ends = torch.tensor(bounds[1:],device=data['s'].device)
        self.bounds, self.hold, self.length = bounds, hold, n
        self.data = data
        self.target_masks=data['active'].clone()
        self.keep_masks=torch.zeros_like(data['active'])
        self.support_masks=torch.zeros_like(data['active'])
        for begin,end in zip(bounds,bounds[1:]):
            # Keep means observed continuous contact over THIS transition. It
            # does not pin a centroid or require a dwell after the new arrival.
            first=max(0,begin-1)
            self.keep_masks[end-1]=data['active'][first:end].all(0)&data['known'][first:end].all(0)
            if surface is not None:
                self.keep_masks[end-1]&=(data['surface'][first:end]==data['surface'][end-1]).all(0)
            self.support_masks[end-1]=self.keep_masks[end-1]&(data['load'][first:end]>=1).all(0)
        if role_aware:
            # Only the observed boundary arrivals are new targets. Other
            # endpoint contacts are free unless explicitly classified as keep.
            self.target_masks.zero_()
            by_end={group[-1][0]:group for group in groups}
            for end in bounds[1:]:
                for _,part,_ in by_end.get(end,[]):
                    self.target_masks[end-1,part]=True
            if boundary_policy=='legacy_v1':
                self.target_masks[n-1]=data['active'][n-1]&~self.keep_masks[n-1]
        if stable_support:
            for end in bounds[1:]:
                first=end-1;stop=min(first+hold,n)
                begin=max(0,n-hold) if end==n else first
                eligible=(data['active'][begin:stop]&data['known'][begin:stop]
                    &(data['load'][begin:stop]>=1)&torch.isfinite(data['slip'][begin:stop])
                    &(data['slip'][begin:stop]<=.05)).all(0)
                if not bool(eligible.any()):
                    raise ValueError('Demonstrated endpoint has no observed stable supporting contact')
                self.target_masks[first]=eligible

    def condition(self, S, frame, *, include_time=False, ends=None):
        f = frame.clamp(0,self.length-1)
        end = self.ends[f] if ends is None else ends
        index = end-1
        target = self.data['point'][index]
        mask = self.target_masks[index]
        Ht = yaw_mat(S[1]).transpose(-1,-2)
        relative = (Ht.unsqueeze(-3) @ (target-S[0].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        root = unflatten_state(self.data['s1'][index])
        root_relative = (Ht @ (root[0]-S[0]).unsqueeze(-1)).squeeze(-1)
        orientation = mat_to_6d(S[1].transpose(-1,-2) @ root[1])
        remaining = (end-frame).to(S[0].dtype).unsqueeze(-1)*.02
        fields=[(relative*mask.unsqueeze(-1)).flatten(-2),mask.float(),root_relative,orientation,
                self.keep_masks[index].float()]
        if include_time:
            fields.append(remaining)
        return torch.cat(fields,-1)

    def goals(self, frame):
        end = self.ends[frame.clamp(0,self.length-1)]
        i = end-1
        return self.data['point'][i],self.target_masks[i],end


class ContactGeometry(nn.Module):
    """Native-calibrated material witnesses on the canonical robot."""
    def __init__(self, data, device):
        super().__init__()
        from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics
        from somaforge_core import G1_29DOF_JOINT_ORDER
        from .robot import G1_JOINT_NAMES
        self.fk = BatchedG1ForwardKinematics().to(device)
        self.register_buffer('order',torch.tensor([G1_JOINT_NAMES.index(n) for n in G1_29DOF_JOINT_ORDER],device=device))
        with torch.no_grad():
            position,rotation = self.poses(unflatten_state(data['s1']))
            offset = (rotation.transpose(-1,-2) @ (data['point']-position).unsqueeze(-1)).squeeze(-1)
            self.register_buffer('local_by_frame',torch.where(data['active'][...,None],offset,0.))
            calibrated = []
            for p in range(6):
                valid = data['active'][:,p] & data['known'][:,p] & (data['load'][:,p]>1)
                # One material point per part, not a freely learned contact head.
                calibrated.append(offset[valid,p].median(0).values if valid.any() else offset.new_zeros(3))
            self.register_buffer('local',torch.stack(calibrated))

    def poses(self,S):
        qpos = torch.cat((S[0],mat_to_quat(S[1]),S[4].index_select(-1,self.order)),-1)
        return self.fk.link_poses(qpos,LINKS)

    def forward(self,S,frame=None):
        p,R = self.poses(S)
        local=self.local if frame is None else self.local_by_frame[frame]
        return p + (R @ local.unsqueeze(-1)).squeeze(-1)


def assess_arrival(evidence,target,new,keep,support=None):
    """Arrive once; only explicitly retained contacts carry continuity duties.

    Loaded support is optional and separately specified. New-contact arrival
    neither requires normal load nor invents a zero-slip/dwell condition.
    """
    close=(evidence['point']-target).norm(dim=-1)<=.05
    observed=evidence['known']&evidence['active']
    arrived=((~new)|(observed&close)).all(-1)
    kept=((~keep)|observed).all(-1)
    if support is not None:
        kept&=((~support)|(evidence['known']&(evidence['load']>=1))).all(-1)
    return arrived,kept


def new_part_contact(observed,previous,recontact,new):
    """Track contact onset of requested parts, independent of face/distance."""
    onset=observed&~previous
    recontact=recontact|onset
    ended=((~new)|recontact).all(-1)&new.any(-1)
    return ended,recontact


def load_contact_tolerances(clock,metadata,path):
    """Resolve explicit recording-scoped completion rules, never loss scales."""
    import json,math
    from pathlib import Path
    profile=json.loads(Path(path).read_text())
    default=float(profile['default_tolerance_m'])
    if not math.isfinite(default) or default<=0:raise ValueError('Invalid default contact tolerance')
    values=[default]*len(clock.event_ends)
    for rule in profile['overrides']:
        radius=float(rule['tolerance_m'])
        if not math.isfinite(radius) or radius<=0:raise ValueError('Invalid contact tolerance override')
        if metadata.get('task_manifest_sha256')!=rule['task_manifest_sha256']:continue
        from .contact_library import segment_ranges
        starts,stops=segment_ranges(clock)
        for i,(start,end) in enumerate(zip(starts.tolist(),stops.tolist())):
            source_motion=int(clock.segment_motion[i]) if hasattr(clock,'segment_motion') else metadata.get('motion')
            if source_motion!=rule['motion']:continue
            if hasattr(clock,'segment_local_bounds'):start,end=clock.segment_local_bounds[i]
            if [start,end]!=rule['segment_bounds']:continue
            endpoint=int(clock.event_ends[i])-1
            new=clock.target_masks[endpoint]&~clock.keep_masks[endpoint]
            if [PARTS[j] for j in new.nonzero().flatten().tolist()]!=rule['new_parts']:continue
            values[i]=radius
    return values


def goal_contact_mask(evidence,target,target_faces=None,tolerance=.05):
    """Per-part physical contact inside a scalar or broadcastable target radius."""
    observed=evidence['known']&evidence['active']
    same_face=torch.ones_like(observed) if target_faces is None else (evidence['surface_local']==target_faces)&(target_faces>=0)
    return observed&same_face&((evidence['point']-target).norm(dim=-1)<=tolerance)


def assess_region_arrival(evidence,target,target_faces,new,keep,support=None,*,tolerance=.05):
    """Actual contact within the target region; optional legacy face constraint.

    Retention is an instantaneous condition for stages with no new transfer;
    a past interruption cannot invalidate a later completed transfer.
    """
    observed=evidence['known']&evidence['active']
    arrived=((~new)|goal_contact_mask(evidence,target,target_faces,tolerance)).all(-1)
    kept=((~keep)|observed).all(-1)
    if support is not None:
        kept&=((~support)|(observed&(evidence['load']>=1))).all(-1)
    return arrived,kept


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(module.parameters(),flags):
            p.requires_grad_(flag)


def sample_window(data,batch,window):
    # Each collection is one episode, never stitch across reset boundaries.
    valid = torch.arange(len(data['s'])-window+1,device=data['s'].device)
    if 'window_allowed' in data:
        valid = valid[data['window_allowed'].unfold(0,window,1).all(-1)]
    if 'done' in data and window>1:
        bad = data['done'].unfold(0,window,1)[:,:-1].any(-1)
        valid = valid[~bad[valid]]
    if not len(valid):
        raise ValueError('No episode-bounded WM windows')
    start = valid[torch.randint(len(valid),(batch,),device=valid.device)]
    index = start[:,None]+torch.arange(window,device=start.device)
    return {k:v[index] for k,v in data.items() if isinstance(v,torch.Tensor) and v.shape[:1]==data['s'].shape[:1]}


def wm_one_step_loss(wm,w,normalize):
    """DMO deterministic branch: real (s,a,s1), one-step residual MSE.

    Preserve tracking coordinates/normalization and its existing force6 head.
    No autoregressive model states enter these supervised inputs or targets.
    """
    from .networks import run_mlp
    assert w['s'].shape[1]==1
    S=unflatten_state(w['s'][:,0]);S1=unflatten_state(w['s1'][:,0])
    inputs=wm.in_norm(wm.inputs(S,w['a'][:,0],normalize(w['xi'][:,0])))
    predicted=run_mlp(wm.net,inputs)*wm.out_norm.std()+wm.out_norm.mean
    target=wm.targets(S,S1)
    floor=predicted.new_tensor([.003]*3+[.005]*3+[.02]*3+[.05]*3+[.003]*29+[.05]*29)
    scale=torch.maximum(wm.out_norm.std(),floor)
    residual=((predicted-target)/scale).square().mean()
    window=dict(w,contact_next=w['next_xi'][...,34:52])
    contact=(wm.contact_loss_from_inputs(window,inputs[:,None])
             if callable(getattr(wm,'contact_loss_from_inputs',None)) else wm.contact_loss(window))
    return residual+.1*contact


def wm_loss(wm,w,normalize):
    """Original tracking state/force fit, with the existing replay episode mask."""
    S=unflatten_state(w['s'][:,0])
    floor=S[0].new_tensor([.003]*3+[.005]*3+[.02]*3+[.05]*3+[.003]*29+[.05]*29)
    scale=torch.maximum(wm.out_norm.std(),floor)
    mask=w.get('mask',torch.ones(w['a'].shape[:2],device=w['a'].device))
    total=S[0].new_zeros(())
    for t in range(w['a'].shape[1]):
        S=wm.step(S,w['a'][:,t],normalize(w['xi'][:,t]))
        error=(wm.state_error(S,unflatten_state(w['s1'][:,t]))/scale).square().mean(-1)
        total=total+(error*mask[:,t]).sum()/mask[:,t].sum().clamp_min(1)
    window=dict(w,mask=mask)
    if 'contact_next' not in window:
        window['contact_next']=w['next_xi'][...,34:52]
    return total/w['a'].shape[1]+.1*wm.contact_loss(window)


def event_policy_loss(actor,wm,clock,geometry,context,normalize,default_q,*,
                      terminal_steps,pose_weight=1.,prior_actor=None,prior_weight=1.,
                      segment_indices=None,return_details=False,contact_surface=None):
    """Full-segment BPTT ending at the observed native stopping time.

    Goal errors exist only on that terminal state. Intermediate state/action
    derivatives remain connected; no imagined contact classifier is required.
    """
    if actor.include_time:
        raise ValueError('Timeless actor required')
    data=clock.data;device=data['s'].device
    event=(torch.arange(len(clock.event_ends),device=device) if segment_indices is None
           else torch.as_tensor(segment_indices,device=device,dtype=torch.long))
    if not len(event) or bool(((event<0)|(event>=len(clock.event_ends))).any()):
        raise ValueError('Invalid segment selection')
    start=torch.tensor(clock.bounds[:-1],device=device)[event]
    end=clock.event_ends[event]
    stop=torch.as_tensor(terminal_steps,device=device,dtype=torch.long)[event]
    if bool((stop<1).any()):raise ValueError('Positive native terminal steps required')
    S=unflatten_state(data['s'][start]);xi=data['xi'][start];previous=data['a_prev'][start]
    obs=observation(S,previous,xi,default_q,context)
    hist=obs[:,None].expand(-1,actor.hist,-1).clone()
    target=data['point'][end-1];new=clock.target_masks[end-1]&~clock.keep_masks[end-1]
    surface_parts=new.any(0).nonzero().flatten().tolist()
    reference=unflatten_state(data['s1'][end-1])
    target_total=S[0].new_zeros(len(event));body_total=target_total.clone();regular_total=target_total.clone()
    max_steps=int(stop.max())
    with frozen(wm):
        for step in range(1,max_steps+1):
            live=step<=stop;terminal=step==stop
            condition=clock.condition(S,start,ends=end)
            action=actor(hist,condition)
            xi_n=normalize(xi)
            xi1=wm.next_context(S,action,xi,xi_n)
            S1=wm.step(S,action,xi_n)
            # No endpoint penalty before arrival, including root and velocity.
            if bool(terminal.any()):
                if contact_surface is not None and hasattr(contact_surface,'region_loss'):
                    placement=contact_surface.region_loss(S1,target,surface_parts)['loss']
                else:
                    point=(contact_surface.contact_points(S1,target,surface_parts)
                           if contact_surface is not None else geometry(S1,end-1))
                    placement=((point-target)/.05).square().sum(-1)
                contact=(placement*new).sum(-1)/new.sum(-1).clamp_min(1)
                body=((reference[0]-S1[0])/.1).square().mean(-1)+rot_angle_sq(S1[1],reference[1])/.4**2
                body=body+.1*((S1[2]-reference[2]).square().mean(-1)+(S1[3]-reference[3]).square().mean(-1))
                target_total=target_total+torch.where(terminal,contact,0.)
                body_total=body_total+torch.where(terminal,body,0.)
            regular=.01*(action-previous).square().mean(-1)+.001*action.square().mean(-1)
            if prior_actor is not None and prior_weight:
                with torch.no_grad():prior=prior_actor(hist,condition)
                regular=regular+prior_weight*(action-prior).square().sum(-1)
            regular_total=regular_total+torch.where(live,regular,0.)
            obs=observation(S1,action,xi1,default_q,context)
            hist1=torch.cat((hist[:,1:],obs[:,None]),1)
            # Freeze expired trajectories without severing any live gradient path.
            S=tuple(torch.where(live.reshape((-1,)+(1,)*(x.ndim-1)),x,old) for x,old in zip(S1,S))
            xi=torch.where(live[:,None],xi1,xi)
            hist=torch.where(live[:,None,None],hist1,hist)
            previous=torch.where(live[:,None],action,previous)
    contact_loss=target_total.mean()
    body_loss=body_total.mean()
    regular_loss=(regular_total/stop).mean()
    loss=contact_loss+pose_weight*body_loss+regular_loss
    if return_details:
        return loss,dict(terminal_contact=float(contact_loss.detach()),terminal_body=float(body_loss.detach()),
            process_regularization=float(regular_loss.detach()),unroll_steps=max_steps,
            terminal_steps=stop.tolist(),terminal_counts=[1]*len(event),segments=(event+1).tolist())
    return loss


def execution_report(data,clock,*,placement=.05,slip=.05,minimum_load=1.):
    """Native evidence only. Terminations or missing endpoint holds fail."""
    result = []
    for end in clock.bounds[1:-1]:
        start,stop = end-1,end-1+clock.hold
        expected = clock.data['active'][start]
        if stop>len(data['active']):
            result.append(dict(end=end,success=False,reason='episode ended before hold'))
            continue
        known = data['known'][start:stop][:,expected]
        active = data['active'][start:stop][:,expected]
        error = (data['point'][start:stop][:,expected]-clock.data['point'][start,expected]).norm(dim=-1)
        load = data['load'][start:stop][:,expected]
        speed = data['slip'][start:stop][:,expected]
        ok = bool(expected.any() and known.all() and active.all() and (error<=placement).all()
                  and (load>=minimum_load).all() and torch.isfinite(speed).all() and (speed<=slip).all())
        result.append(dict(end=end,success=ok,max_placement_m=float(error.max()) if error.numel() else None,
                           max_slip_m_s=float(speed.max()) if speed.numel() and torch.isfinite(speed).all() else None,
                           unknown_slip_samples=int((~torch.isfinite(speed)).sum()),
                           active=bool(active.all()),loaded=bool((load>=minimum_load).all())))
    complete = len(data['s'])>=clock.length and not bool(data['term'].any())
    return dict(completed=complete,contact_segments=result,
                success=complete and bool(result) and all(r['success'] for r in result),
                frames=len(data['s']),expected_frames=clock.length,
                thresholds=dict(placement_m=placement,slip_m_s=slip,minimum_load_n=minimum_load,hold=clock.hold))
