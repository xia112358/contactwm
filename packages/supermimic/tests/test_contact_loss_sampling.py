import torch
from supermimic.contact_loss_sampling import stable_loss_samples
from supermimic.contact_dmo import _replay_events
from types import SimpleNamespace


def test_unstable_switching_and_post_terminal_samples_are_not_loss_witnesses():
    a=torch.ones(6,1,6,dtype=torch.bool);s=torch.zeros_like(a);s[[0,2,5],0,0]=True
    live=torch.tensor([[1],[1],[1],[1],[0],[0]],dtype=torch.bool)
    hit=stable_loss_samples(s,live)
    assert hit[:,0,0].nonzero().flatten().tolist()==[2]


def test_no_release_needed_and_unstable_target_retains_guidance():
    a=torch.ones(5,2,6,dtype=torch.bool);stable=torch.zeros_like(a);stable[1,0,0]=True
    targets=torch.zeros(5,6,dtype=torch.bool);targets[:,0]=True
    c=SimpleNamespace(fixed_goal_mode=True,event_ends=torch.tensor([5]),target_masks=targets,keep_masks=torch.zeros_like(targets),data={"point":torch.zeros(len(targets),6,3)},contact_tolerances_by_segment=torch.tensor([.05]))
    d=dict(active=a,known=a,contact_stable=stable,contact_goal=torch.zeros_like(a),contact_first=torch.zeros_like(a),contact_loss_point=torch.zeros(5,2,6,3))
    n={};e,terminal,_=_replay_events(d,c,torch.zeros(2,dtype=torch.long),torch.tensor([5,5]),torch.tensor([False,False]),'accumulated_target',None,native_events=n,stable_contact_frames=2)
    assert e.sum()==2 and (e&n['mask']).sum()==1
    assert e[1,0,0] and e[4,1,0]
    assert terminal[4].all()


def test_substep_witness_survives_unstable_or_absent_control_endpoint():
    from supermimic.contact_loss_sampling import SubstepLossWitness
    w=SubstepLossWitness()
    for substep in range(4):
        mask=torch.full((1,6),substep==1,dtype=torch.bool)
        obs=dict(active=mask,known=torch.ones_like(mask),point=torch.full((1,6,3),float(substep)))
        w(substep,substep,obs,dict(contact_stable=mask),{})
    assert w.summary()['contact_stable'].all()
    assert (w.summary()['contact_loss_point']==1).all()
    # A new interval must not retain an earlier interval's witness.
    w(4,0,obs,dict(contact_stable=mask),{})
    assert not w.summary()['contact_stable'].any()


def test_qualified_substep_is_not_gated_by_control_end_activation():
    a=torch.zeros(1,1,6,dtype=torch.bool);s=a.clone();s[0,0,0]=True
    targets=torch.zeros(1,6,dtype=torch.bool);targets[0,0]=True
    c=SimpleNamespace(fixed_goal_mode=True,event_ends=torch.tensor([1]),target_masks=targets,keep_masks=torch.zeros_like(targets),data={"point":torch.zeros(len(targets),6,3)},contact_tolerances_by_segment=torch.tensor([.05]))
    n={};e,_,_=_replay_events(dict(active=a,known=a,contact_stable=s,contact_loss_point=torch.zeros(1,1,6,3)),c,torch.tensor([0]),torch.tensor([1]),torch.tensor([False]),'accumulated_target',None,native_events=n,stable_contact_frames=2)
    assert (e&n['mask']).sum()==1
