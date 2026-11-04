"""Native-solve support state and lossless control-interval event summaries.

Five observations mean five actual physics solves, not five policy frames.
Pose convergence is a speed bound: its displacement tolerance scales with dt.
Native activation and point-loss qualification remain separate observations.
"""
from collections import deque
import math
import torch
from .contact_temporal import TEMPORAL_CONTRACT
from .contact_stability import NativeContactStability,CONTRACT


def substep_support_contract(dt,substeps):
    width=TEMPORAL_CONTRACT['enter_frames']
    return dict(version='native_substep_support_v2',physics_dt=dt,substeps_per_control=substeps,
        enter_observations=width,exit_observations=math.ceil(TEMPORAL_CONTRACT['exit_frames']*TEMPORAL_CONTRACT['dt']/dt),
        exit_persistence_s=TEMPORAL_CONTRACT['exit_frames']*TEMPORAL_CONTRACT['dt'],
        gap_motion_m=TEMPORAL_CONTRACT['max_bridge_motion_m'],
        event_bearing_s=TEMPORAL_CONTRACT['exit_frames']*TEMPORAL_CONTRACT['dt'],
        confirmation_span_s=(width-1)*dt,min_load_n=TEMPORAL_CONTRACT['enter_load_n'],
        exit_load_n=TEMPORAL_CONTRACT['exit_load_n'],
        position_window_m=TEMPORAL_CONTRACT['settle_position_m']*dt/TEMPORAL_CONTRACT['dt'],
        angle_window_rad=TEMPORAL_CONTRACT['settle_angle_rad']*dt/TEMPORAL_CONTRACT['dt'],
        events='all establishment and exit events retained at actual episode-relative solver ticks',
        sampling='every constraint solve with its own constraint-evaluation part poses')


class SubstepSupport:
    def __init__(self,dt):
        self.dt=dt;self.width=TEMPORAL_CONTRACT['enter_frames']
        self.exit_steps=math.ceil(TEMPORAL_CONTRACT['exit_frames']*TEMPORAL_CONTRACT['dt']/dt)
        self.history=deque(maxlen=self.width);self.previous_step=None

    def update(self,observation,features,poses,step):
        active,known=observation['active'],observation['known'];load=features['contact_load']
        p,R=poses;surface=observation['surface_local'];speed=features['contact_material_speed']
        if self.previous_step is None:
            self.support=torch.zeros_like(active);self.off=torch.zeros_like(active,dtype=torch.long)
            self.previous_loaded=torch.zeros_like(active);self.surface=surface.clone()
            self.last_bearing_position=p.clone();self.bearing_steps=torch.zeros_like(self.off)
        else:
            reset=step!=self.previous_step+1
            self.support[reset]=False;self.off[reset]=0;self.previous_loaded[reset]=False
            self.bearing_steps[reset]=0;self.last_bearing_position[reset]=p[reset]
            for sample in self.history:
                sample['bearing'][reset]=False;sample['material_known'][reset]=False
        loaded=known&active&(load>0)
        changed=known&active&(surface!=self.surface)
        absent=~active|(load<=TEMPORAL_CONTRACT['exit_load_n'])
        gap_steps=torch.where(absent,self.off+1,self.off)
        moved=(p-self.last_bearing_position).norm(dim=-1)>TEMPORAL_CONTRACT['max_bridge_motion_m']
        # Time and measured displacement qualify a gap, never contact itself.
        ended=~known|changed|(gap_steps>=self.exit_steps)|((gap_steps>0)&moved)
        exited=self.support&ended
        physical_exit=step[:,None]-gap_steps+absent.long()
        exit_step=torch.where(exited,torch.where(~known|changed,step[:,None],physical_exit),-1)
        self.support &= ~ended
        self.bearing_steps=torch.where(ended,0,self.bearing_steps)
        self.off=torch.where(absent,gap_steps,0)
        self.last_bearing_position=torch.where((known&~absent)[...,None],p,self.last_bearing_position)
        self.surface=torch.where(known&active,surface,self.surface)
        self.history.append(dict(bearing=known&active&(load>TEMPORAL_CONTRACT['enter_load_n'])&(speed<=CONTRACT['max_material_speed_m_s']),
            material_known=loaded&self.previous_loaded,position=p.clone(),rotation=R.clone(),surface=surface.clone(),speed=speed.clone()))
        established=torch.zeros_like(active)
        if len(self.history)==self.width:
            stack=lambda key:torch.stack([x[key] for x in self.history])
            bearing=stack('bearing').all(0);motion_known=stack('material_known')
            P=stack('position');rotations=stack('rotation');material=stack('speed')
            displacement=(P-P[:1]).norm(dim=-1).amax(0)
            relative=rotations@rotations[0].transpose(-1,-2)
            angle=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).amax(0)
            limit=TEMPORAL_CONTRACT['settle_material_speed_m_s'];above=material>limit
            quiet=(material[1:]<=limit).all(0)
            settling=motion_known[0]&above[0]&(material.mean(0)<=limit)&~(above[1:]&~above[:-1]).any(0)&((material[1:]<material[:-1])|~above[1:]).all(0)
            scale=self.dt/TEMPORAL_CONTRACT['dt']
            established=(~self.support)&bearing&motion_known[1:].all(0)&(stack('surface')==surface).all(0)&(quiet|settling)&(displacement<=TEMPORAL_CONTRACT['settle_position_m']*scale)&(angle<=TEMPORAL_CONTRACT['settle_angle_rad']*scale)
        self.support |= established
        self.bearing_steps=torch.where(established,self.width,self.bearing_steps+(self.support&known&~absent).long())
        # Gaps contribute no bearing time. A rapid online confirmation can be
        # retained as evidence without promoting a short impact to a cut.
        event_ready=self.support&(self.bearing_steps>=self.exit_steps)
        self.previous_loaded=loaded.clone();self.previous_step=step.clone()
        return dict(state=self.support.clone(),event_ready=event_ready,established=established,exited=exited,
            onset_step=torch.where(established,step[:,None]-self.width+1,-1),
            confirmed_step=torch.where(established,step[:,None],-1),exit_step=exit_step)


class SubstepContactMonitor:
    """Observe each solver invocation; deliberately bypass CUDA graph replay.

    Python native-row snapshots cannot execute inside a captured Warp graph.
    The owned hook restores the original solver method and graph setting on close.
    """
    def __init__(self,env,reader,on_sample=None):
        if not (reader.direct and reader.primary_faces):
            raise ValueError('Substep support requires allocated native primary-surface contacts')
        self.env=env;self.reader=reader;self.manager=env.sim.backend_manager;self.on_sample=on_sample
        self.dt=float(self.manager.get_solver_dt());self.substeps=int(self.manager._num_substeps)
        self.contract=substep_support_contract(self.dt,self.substeps)
        self.point_quality=NativeContactStability(self.dt);self.support=SubstepSupport(self.dt)
        self.tick=0;self.samples=[];self.control_summary=None

    def __enter__(self):
        self.original=self.manager._step_solver;self.graph=self.manager._cfg.use_cuda_graph
        self.manager._cfg.use_cuda_graph=False;self.manager._invalidate_graph()
        self.manager._step_solver=self._step
        return self

    def _step(self,*args,**kwargs):
        result=self.original(*args,**kwargs)
        substep=self.tick%self.substeps
        if substep==0:self.samples=[]
        observation=self.reader.read_device(substep=substep,same_solve=True)
        step=self.env.ep_step*self.substeps+substep
        features=self.point_quality.update(self.reader.last_view,self.reader.last_part_poses,step)
        support=self.support.update(observation,features,self.reader.last_part_poses,step)
        self.samples.append(dict(active=observation['active'].clone(),known=observation['known'].clone(),support=support))
        if self.on_sample is not None:self.on_sample(self.tick,substep,observation,features,support)
        if substep==self.substeps-1:
            stack=lambda key:torch.stack([x['support'][key] for x in self.samples],dim=1)
            contact=torch.stack([x['active'] for x in self.samples],dim=1)
            self.control_summary=dict(substep_contact_active=contact,
                substep_contact_known=torch.stack([x['known'] for x in self.samples],dim=1),
                substep_support_state=stack('state'),substep_support_event_ready=stack('event_ready'),substep_support_established=stack('established'),
                substep_support_exited=stack('exited'),substep_support_onset_step=stack('onset_step'),
                substep_support_confirmed_step=stack('confirmed_step'),substep_support_exit_step=stack('exit_step'),
                support_state=support['state'].clone(),contact_active_any=contact.any(1))
        self.tick+=1
        return result

    def summary(self):
        if self.tick==0 or self.tick%self.substeps:
            raise ValueError('A control summary requires all actual solver substeps')
        return self.control_summary

    def __exit__(self,*args):
        self.manager._step_solver=self.original
        self.manager._cfg.use_cuda_graph=self.graph;self.manager._invalidate_graph()


def control_support_episodes(data, *, physics_bounds=False):
    """Project recorded solver events to a control-frame partition.

    A boundary belongs to the control interval containing its solver tick.
    Entirely intra-frame episodes stay in the substep evidence, not zero-frame
    training segments. Adjacent projected episodes of the same part are joined.
    No extra control-frame confirmation is applied.
    """
    state=data['substep_support_state']
    T,S,P=state.shape
    base=int(data['ep_step'][0])*S
    if not torch.equal(data['ep_step'],data['ep_step'][0]+torch.arange(T,device=state.device)):
        raise ValueError('Substep episode projection requires one contiguous source trace')
    established=data['substep_support_established'].reshape(-1,P)
    exited=data['substep_support_exited'].reshape(-1,P)
    onset=data['substep_support_onset_step'].reshape(-1,P)
    confirmed=data['substep_support_confirmed_step'].reshape(-1,P)
    exits=data['substep_support_exit_step'].reshape(-1,P)
    ready=data['substep_support_event_ready'].reshape(-1,P) if 'substep_support_event_ready' in data else torch.ones_like(established)
    episodes=[]
    for part in range(P):
        live=dict(start=base,confirmed=base,ready=bool(ready[0,part])) if bool(state[0,0,part]) and not bool(established[0,part]) else None
        projected=[]
        def close(stop):
            if not live['ready']:return
            begin=max(0,(live['start']-base)//S)
            end=min(T,max(0,(stop-base)//S))
            if end<=begin:return
            confirm=max(0,(live['confirmed']-base)//S)
            item=dict(part=part,start=begin,stop=end,confirmed=confirm,surface=int(data['surface_local'][begin,part]))
            if physics_bounds:item.update(physics_start=max(0,live['start']-base),physics_stop=min(T*S,stop-base))
            if projected and projected[-1]['stop']>=begin:
                projected[-1]['stop']=end
                if physics_bounds:projected[-1]['physics_stop']=item['physics_stop']
            else:projected.append(item)
        for i in range(T*S):
            if bool(exited[i,part]) and live is not None:
                close(int(exits[i,part]));live=None
            if bool(established[i,part]):
                live=dict(start=int(onset[i,part]),confirmed=int(confirmed[i,part]),ready=False)
            if live is not None:live['ready'] |= bool(ready[i,part])
        if live is not None:close(base+T*S)
        episodes.extend(projected)
    return sorted(episodes,key=lambda e:(e['start'],e['part']))
