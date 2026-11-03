"""Reference landing identity must exclude earlier supports and flicker."""
from types import SimpleNamespace
import torch
from supermimic.contact_phase import LandingPhaseTracker,landing_phase_events,reference_release_starts
from supermimic.contact_dmo import _replay_events


def events(sequence,release_start=0):
    active=torch.zeros(len(sequence),1,6,dtype=torch.bool);active[:,0,0]=torch.tensor(sequence,dtype=torch.bool)
    required=torch.zeros(1,6,dtype=torch.bool);required[:,0]=True
    phases=torch.zeros(1,6,dtype=torch.long);phases[:,0]=release_start
    return landing_phase_events(active,torch.ones_like(active),phases,required,2,torch.ones(len(sequence),1,dtype=torch.bool),torch.ones_like(active))


def test_initial_support_and_one_frame_flicker_do_not_become_target_landing():
    hit=events([1,1,0,1,1,0,0,1,1])
    assert hit[:,0,0].nonzero().flatten().tolist()==[8]


def test_reference_boundary_skips_an_earlier_complete_landing():
    hit=events([0,0,1,1,0,0,1,1],release_start=4)
    assert hit[:,0,0].nonzero().flatten().tolist()==[7]


def test_unknown_frames_do_not_count_as_release_or_stability():
    tracker=LandingPhaseTracker(torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),2)
    for _ in range(3):assert not tracker.update(torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool)).any()
    for _ in range(3):assert not tracker.update(torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool)).any()


def test_reset_clears_phase_identity():
    tracker=LandingPhaseTracker(torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),2)
    for a in [0,0,1,1]:hit=tracker.update(torch.tensor([[a]],dtype=torch.bool),torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool))
    assert hit.all();tracker.reset(torch.tensor([0]))
    assert not tracker.update(torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool)).any()


def clock_for(active):
    T=active.shape[0];new=torch.zeros(T,6,dtype=torch.bool);new[:,0]=True
    reference=active[:,0].clone();reference[-1,0]=True
    clock=SimpleNamespace(segment_starts=torch.tensor([0]),segment_stops=torch.tensor([T]),event_ends=torch.tensor([T]),target_masks=new,keep_masks=torch.zeros_like(new),data=dict(active=reference,known=torch.ones_like(reference)))
    clock.contact_phase_release_starts=reference_release_starts(clock,2)
    return clock


def test_segment93_regression_does_not_compare_start_support_to_next_footstep():
    active=torch.zeros(28,1,6,dtype=torch.bool);active[:3,0,0]=True;active[-2:,0,0]=True
    clock=clock_for(active);assert clock.contact_phase_release_starts[0,0]==3
    data=dict(active=active,known=torch.ones_like(active),contact_stable=torch.ones_like(active),contact_first=torch.zeros_like(active),contact_goal=torch.zeros_like(active))
    native={}
    evaluate,_,_=_replay_events(data,clock,torch.tensor([0]),torch.tensor([28]),torch.tensor([False]),'accumulated_target',None,native_events=native,stable_contact_frames=2)
    assert evaluate[:,0,0].nonzero().flatten().tolist()==[27]
    assert native['mask'][27,0,0] and not native['mask'][:3].any()
    points=torch.zeros(28,1,6,3);points[:3,0,0,0]=1.1745
    assert (points[evaluate].square().sum(-1)==0).all()


def test_confirmed_and_missing_parts_are_supervised_once_without_goal_filter():
    T,B=5,3
    active=torch.zeros(T,B,6,dtype=torch.bool);active[2:4,0,0]=True;active[-1,1,0]=True
    template=active[:,:1].clone();template[-1,0,0]=True;clock=clock_for(template)
    native={};evaluate,_,_=_replay_events(dict(active=active,known=torch.ones_like(active),contact_stable=torch.ones_like(active),contact_first=torch.zeros_like(active),contact_goal=torch.zeros_like(active)),clock,torch.zeros(B,dtype=torch.long),torch.full((B,),T),torch.zeros(B,dtype=torch.bool),'accumulated_target',None,native_events=native,stable_contact_frames=2)
    assert evaluate.sum(0)[:,0].tolist()==[1,1,1]
    assert evaluate[3,0,0] and evaluate[-1,1,0] and evaluate[-1,2,0]
    assert (evaluate&native['mask']).sum()==1
    assert (evaluate&~native['mask']).sum()==2


def test_same_native_witness_used_for_stable_loss_and_missing_has_geometry_gradient(monkeypatch):
    from torch import nn
    import supermimic.contact_dmo as dmo
    class Actor(nn.Module):
        hist=1
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.zeros(2))
        def forward(self,history,condition):return self.weight[:,None].expand(-1,29)
    class Model(nn.Module):
        def step(self,state,action,xi):return (state[0]+action[:,:1]*.01,*state[1:])
        def next_context(self,state,action,xi,xi_n):return xi
    class Region:
        def region_loss(self,state,target,parts,**kwargs):
            loss=((state[0][:,None]-target)/.05).square().sum(-1)
            return dict(loss=loss,normal=loss,tangent=torch.zeros_like(loss))
    monkeypatch.setattr(dmo,'observation',lambda state,*a:state[0][:,:1]*0)
    T,B=5,2
    state=torch.zeros(T,B,71);state[...,3]=1;state[...,:3]=.2
    active=torch.zeros(T,B,6,dtype=torch.bool);active[2:4,0,0]=True
    done=torch.zeros(T,B,dtype=torch.bool);done[-1]=True
    points=torch.full((T,B,6,3),1e6);points[3,0,0]=.2
    data=dict(s=state,s1=state,a=torch.zeros(T,B,29),a_prev=torch.zeros(T,B,29),xi=torch.zeros(T,B,52),next_xi=torch.zeros(T,B,52),ep_step=torch.arange(T)[:,None].expand(T,B),goal_index=torch.zeros(T,B,dtype=torch.long),done=done,active=active,known=torch.ones_like(active),contact_stable=torch.ones_like(active),contact_first=torch.zeros_like(active),contact_goal=torch.zeros_like(active),point=points)
    clock=clock_for(active[:,:1]);clock.data.update(point=torch.full((T,6,3),.1),s1=state[:,0]);clock.condition=lambda state,*a,**kw:state[0][:,:1]*0
    geometry=SimpleNamespace(poses=lambda state:(state[0][:,None].expand(-1,6,-1),torch.eye(3).expand(B,6,3,3)))
    actor=Actor();loss,details=dmo.replay_loss(actor,Model(),clock,geometry,None,lambda x:x,None,data,contact_surface=Region(),pose_weight=0,completion_mode='accumulated_target',stable_contact_frames=2)
    loss.backward()
    assert all(v is None or isinstance(v,(int,float)) for v in details.values())
    assert details['required_contact_terms']==2 and details['stable_contact_terms']==1 and details['terminal_guidance_terms']==1
    assert details['contact_supervision_coverage']==1
    assert abs(details['terminal_contact']-12.)<1e-5
    assert abs(details['terminal_contact']-details['stable_contact_loss']-details['terminal_guidance_loss'])<1e-5
    assert torch.isfinite(actor.weight.grad).all() and actor.weight.grad.gt(0).all()


def test_segment157_two_frame_student_flicker_before_target_release_is_ignored():
    # Demo target flight starts at t=6; student's earlier two-frame flicker
    # must not arm a landing back onto the old support at t=4/5.
    hit=events([1,1,0,0,1,1,0,0,0,0,1,1],release_start=6)
    assert hit[:,0,0].nonzero().flatten().tolist()==[11]
