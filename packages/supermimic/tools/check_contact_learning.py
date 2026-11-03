"""Check the no-distillation learner against real recorded transitions on CPU/GPU."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task_manifest',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--recording',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    import torch
    from supermimic.lowstep import install
    spec=install(args.task_manifest,contact_model='force6')
    from supermimic.networks import WorldModel
    from supermimic.lowstep_observations import HeightForceContext
    from supermimic.mdp import unflatten_state,normalize_xi
    from supermimic.robot import default_joint_pos_list
    from supermimic.contact_control import (ContactActor,ContactClock,ContactGeometry,
        observation,event_policy_loss,wm_loss,sample_window,execution_report,assess_arrival,assess_region_arrival,BCPlateau)
    torch.set_num_threads(4);torch.manual_seed(16)
    dev=args.device
    from supermimic.buffer import ReplayBuffer
    replay=ReplayBuffer(8,2,dev,extra_fields={'goal_index':((),torch.long)})
    for step in range(6):
        tr={k:torch.zeros_like(v[0]) for k,v in replay.data.items()}
        tr['ep_step'][:]=torch.tensor([step%2,step%3],device=dev)
        tr['done'][:]=torch.tensor([step%2==1,step%3==2],device=dev)
        tr['goal_index'][:]=torch.tensor([step//2,step//3],device=dev)
        tr['s'][:,0]=tr['goal_index'].float()
        replay.add(tr)
    start,history=replay.sample_starts(128,5)
    assert (history['s'][...,0]==start['goal_index'][:,None]).all(), 'History crossed reset'
    windows=replay.sample_windows(128,4)
    assert ((windows['goal_index']==windows['goal_index'][:,:1])|~windows['mask'].bool()).all(), 'WM mask crossed reset'
    data=torch.load(args.recording,map_location=dev,weights_only=False)['data']
    source=torch.load(args.checkpoint,map_location=dev,weights_only=False)
    dynamics=WorldModel().to(dev);dynamics.load_state_dict(source['wm'])
    context=HeightForceContext(spec['tasks'])
    wm=dynamics
    assert not hasattr(wm,"native_head")
    assert all(torch.equal(v,source["wm"][k]) for k,v in wm.state_dict().items())
    actor=ContactActor().to(dev)
    default=torch.tensor(default_joint_pos_list(),device=dev)
    clock=ContactClock(data)
    geometry=ContactGeometry(data,dev)
    endpoint_points=geometry(unflatten_state(data['s1']),torch.arange(len(data['s']),device=dev))
    assert (endpoint_points-data['point'])[data['active']].abs().max()<2e-6
    before={k:v.clone() for k,v in data.items()}
    obs=observation(unflatten_state(data['s']),data['a_prev'],data['xi'],default,context)
    assert obs.shape==(len(data['s']),232)
    actor.norm.update(obs)
    # The actor API cannot consume the source's 730-dimensional future-motion goal.
    condition=clock.condition(unflatten_state(data['s'][:2]),torch.arange(2,device=dev))
    assert condition.shape==(2,39)
    torch.testing.assert_close(condition,clock.condition(unflatten_state(data['s'][:2]),torch.arange(10,12,device=dev)))
    a=actor(obs[:2,None].expand(-1,actor.hist,-1),condition)
    assert a.shape==(2,29) and torch.isfinite(a).all()
    event_clock=ContactClock(data,hold=1,role_aware=True)
    terminal_steps=torch.tensor([b-a for a,b in zip(event_clock.bounds,event_clock.bounds[1:])],device=dev)
    selected=[4,6]  # Mixed full segment lengths, 21 and 44 control frames.
    seen=[];actions=[];points=[];geometry_steps=[]
    original_condition=event_clock.condition
    def record_condition(*args,**kwargs):
        seen.append(kwargs['ends'].clone())
        return original_condition(*args,**kwargs)
    def capture_action(module,inputs,output):
        output.retain_grad();actions.append(output)
    def capture_points(module,inputs,output):
        geometry_steps.append(len(actions));points.append(output)
    event_clock.condition=record_condition
    ha=actor.register_forward_hook(capture_action);hp=geometry.register_forward_hook(capture_points)
    objective,details=event_policy_loss(actor,wm,event_clock,geometry,context,normalize_xi,default,
        terminal_steps=terminal_steps,segment_indices=selected,prior_weight=0.,return_details=True)
    ha.remove();hp.remove();event_clock.condition=original_condition
    assert geometry_steps==sorted(set(terminal_steps[selected].tolist())), 'Goal evaluated before terminal state'
    assert details['terminal_counts']==[1,1] and len(actions)==int(terminal_steps[selected].max())
    assert len(seen)==len(actions) and all(torch.equal(x,seen[0]) for x in seen)
    terminal_gradient=torch.autograd.grad(points[-1][1].sum(),actions[0],retain_graph=True)[0]
    assert torch.isfinite(terminal_gradient).all() and terminal_gradient[1].abs().sum()>0, 'Terminal loss cannot reach first action'
    objective.backward()
    assert all(p.grad is None for p in wm.parameters()),'Policy backward wrote WM gradients'
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.parameters())
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    import copy
    prior=copy.deepcopy(actor).eval().requires_grad_(False);prior.zero_grad(set_to_none=True)
    anchored=event_policy_loss(actor,wm,event_clock,geometry,context,normalize_xi,default,
        terminal_steps=terminal_steps,segment_indices=[4],prior_actor=prior,prior_weight=1.)
    anchored.backward()
    assert torch.isfinite(anchored) and all(p.grad is None for p in prior.parameters())
    # A genuinely recurrent state/action gradient must reach an early action.
    action=data['a'][:1].detach().clone().requires_grad_(True)
    S=unflatten_state(data['s'][:1]);xi=data['xi'][:1]
    for step in range(4):
        a=action if step==0 else data['a'][step:step+1];xi_n=normalize_xi(xi)
        next_xi=wm.next_context(S,a,xi,xi_n);S=wm.step(S,a,xi_n);xi=next_xi
    grad=torch.autograd.grad(S[0].sum(),action)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    for p in wm.parameters():
        p.grad=None
    fit=wm_loss(wm,sample_window(data,2,4),normalize_xi)
    fit.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in wm.contact_head.parameters())
    assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)
    # No label or reference pose may be rewritten by goal construction or fitting.
    for k,v in before.items():
        torch.testing.assert_close(data[k],v,equal_nan=True)
    # Force alone cannot make native contact or a completed task successful.
    fake={k:v.clone() for k,v in data.items()}
    fake['active'].zero_();fake['force'].fill_(10000);fake['load'].fill_(10000)
    fake['term'].zero_()
    assert not execution_report(fake,clock)['success']
    # Missing rollout coverage always fails, including when every recorded contact agrees.
    partial={k:v[:5] for k,v in data.items()}
    assert not execution_report(partial,clock)['success']
    import json
    json.dumps(execution_report(fake,clock),allow_nan=False)
    reserved=(torch.arange(len(data['s']),device=dev)//32)%4==3
    training={**data,'window_allowed':~reserved}
    validation={**data,'window_allowed':reserved}
    assert sample_window(training,64,8)['window_allowed'].all()
    assert sample_window(validation,64,8)['window_allowed'].all()
    assert not (training['window_allowed']&validation['window_allowed']).any()
    # A new touch advances immediately even while unloaded/sliding. Only a
    # separately commanded keep contact has a continuity duty.
    new=torch.zeros(6,dtype=torch.bool,device=dev);new[2]=True
    keep=torch.zeros_like(new);keep[0]=True
    evidence=dict(known=torch.ones_like(new),active=new|keep,point=torch.zeros(6,3,device=dev),
                  load=torch.zeros(6,device=dev),slip=torch.full((6,),10.,device=dev))
    target=torch.zeros(6,3,device=dev);target[0]=100  # retained foot is not pinned to XYZ
    arrived,retained=assess_arrival(evidence,target,new,keep)
    assert arrived and retained
    evidence['active'][0]=False
    arrived,retained=assess_arrival(evidence,target,new,keep)
    assert arrived and not retained
    evidence['active'][0]=True;evidence['active'][2]=False;evidence['load'].fill_(10000)
    assert not assess_arrival(evidence,target,new,keep)[0]
    # A target-face landing may move its centroid; wrong-face or force-only
    # evidence must not turn into an arrival. No historical veto is consulted.
    evidence['active'][2]=True
    evidence['surface_local']=torch.full((6,),7,device=dev,dtype=torch.long)
    face=torch.full((6,),7,device=dev,dtype=torch.long)
    evidence['point'][2]=torch.tensor([.04,0.,0.],device=dev)
    assert assess_region_arrival(evidence,target,face,new,keep)[0]
    evidence['point'][2,0]=.2
    assert not assess_region_arrival(evidence,target,face,new,keep)[0]
    evidence['surface_local'][2]=8
    evidence['point'][2]=0
    assert not assess_region_arrival(evidence,target,face,new,keep)[0]
    evidence['surface_local'][2]=7;evidence['active'][2]=False
    assert not assess_region_arrival(evidence,target,face,new,keep)[0]
    evidence['active'][2]=True;evidence['active'][0]=False
    assert assess_region_arrival(evidence,target,face,new,keep)[0]
    assert not assess_region_arrival(evidence,target,face,new,keep,keep)[1]
    # Same-body contact onto a different native face is an arrival, not a keep.
    active=torch.zeros(60,6,dtype=torch.bool,device=dev);active[:,0]=True;active[1:,1]=True
    surfaces=torch.full((60,6),-1,dtype=torch.long,device=dev);surfaces[:30,0]=0;surfaces[30:,0]=1
    surfaces[1:,1]=0
    role_data=dict(s=torch.zeros(60,71,device=dev),active=active,surface=surfaces,
                   known=torch.ones_like(active),load=active.float())
    role_clock=ContactClock(role_data,hold=1,role_aware=True)
    assert role_clock.bounds==[0,31,60] and not role_clock.keep_masks[30,0]
    assert role_clock.target_masks[30,0] and not role_clock.target_masks[30,1]
    role_data['surface'][:,0]=0;role_data['active'][20:25,0]=False
    chatter_clock=ContactClock(role_data,hold=1)
    assert chatter_clock.bounds==[0,60] and not chatter_clock.keep_masks[-1,0]
    plateau=BCPlateau(min_steps=4,patience=2,absolute=.01,relative=0.)
    assert not plateau.update(1,1.) and not plateau.update(2,.8)
    assert not plateau.update(3,.7) and not plateau.update(4,.699)
    assert plateau.update(5,.698)
    plateau=BCPlateau(min_steps=1,patience=2,absolute=.01,relative=0.)
    assert not plateau.update(1,1.) and not plateau.update(2,.999)
    assert not plateau.update(3,.8) and plateau.stale==0
    assert not any('php' in name for name in actor.state_dict())
    print(dict(result='PASS',device=dev,frames=len(data['s']),contact_bounds=clock.bounds,
        policy_loss=float(objective.detach()),wm_loss=float(fit.detach()),early_action_gradient=float(grad.norm()),
        checks=['sensor-only actor contract','real-checkpoint recurrent BPTT','WM frozen in policy backward',
                'original tracking force head supervised by original sensor labels','immutable demonstration',
                'force cannot create native contact','truncated execution rejected',
                'immediate arrival without dwell/slip/load requirement','keep roles without XYZ pinning',
                'surface transfer is a new arrival']),flush=True)


if __name__=='__main__':
    main()
