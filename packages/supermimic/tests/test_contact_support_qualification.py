from types import SimpleNamespace
import torch
from supermimic.contact_stability import NativeContactStability
from supermimic.contact_phase import LandingPhaseTracker


def sample(x=0.,local=((0.,0.,0.),),shape_ids=None,load=100.,active=True):
    points=torch.tensor(local,dtype=torch.float32).reshape(-1,3) if active else torch.empty(0,3)
    n=len(points);p=torch.tensor([[[x,0.,0.]]]);R=torch.eye(3)[None,None]
    ids=torch.full((n,),7,dtype=torch.long) if shape_ids is None else torch.tensor(shape_ids)
    rows=dict(position_w=points+p[0,0],shape0=ids,shape1=torch.full((n,),11,dtype=torch.long))
    view=SimpleNamespace(evidence=SimpleNamespace(known=torch.ones(1,1,dtype=torch.bool),rows=rows),
        groups=torch.zeros(n,dtype=torch.long),rows=torch.arange(n),robot_side=torch.zeros(n,dtype=torch.long),
        force_w=torch.tensor([[0.,0.,load/max(1,n)]]).expand(n,3))
    return view,(p,R)


def update(tracker,step,**kwargs):
    v,poses=sample(**kwargs)
    return tracker.update(v,poses,torch.tensor([step]))


def test_scrape_then_settle_does_not_end_on_scrape_and_can_confirm_without_new_release():
    support=NativeContactStability(.02)
    phase=LandingPhaseTracker(torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),2)
    hits=[]
    for t,(a,x) in enumerate([(False,0),(False,0),(True,0),(True,.05),(True,.1),(True,.1),(True,.1)]):
        f=update(support,t,x=x,active=a)
        hits.append(bool(phase.update(torch.tensor([[a]]),torch.ones(1,1,dtype=torch.bool),f['contact_stable'])))
    assert hits==[False,False,False,False,False,True,False]


def test_row_reordering_and_duplicate_positions_preserve_bijection():
    s=NativeContactStability(.02)
    update(s,0,local=((0.,0.,0.),(.02,0.,0.),(0.,0.,0.)))
    result=update(s,1,local=((0.,0.,0.),(0.,0.,0.),(.02,0.,0.)))
    assert result['contact_row_stable'].all() and result['contact_stable'].all()


def test_count_identity_local_change_and_reset_each_invalidate_row_stability():
    for kwargs in [dict(local=((0.,0.,0.),(.02,0.,0.))),dict(shape_ids=[8]),dict(local=((.02,0.,0.),))]:
        s=NativeContactStability(.02);update(s,0)
        assert not update(s,1,**kwargs)['contact_row_stable'].any()
    s=NativeContactStability(.02);update(s,7)
    assert not update(s,1)['contact_stable'].any()


def test_high_load_is_not_enough_for_scrape_and_force_jump_is_not_stable_support():
    s=NativeContactStability(.02);update(s,0,load=400.)
    result=update(s,1,x=.05,load=400.)
    assert result['contact_row_stable'].all() and result['contact_load_stable'].all()
    assert torch.allclose(result['contact_material_speed'],torch.tensor([[2.5]]))
    assert not result['contact_stable'].any()
    for before,after in [(20.,20.),(100.,200.)]:
        s=NativeContactStability(.02);update(s,0,load=before)
        assert not update(s,1,load=after)['contact_load_stable'].any()


def test_representative_changes_do_not_create_material_motion():
    s=NativeContactStability(.02);update(s,0,local=((0.,0.,0.),(.2,0.,0.)))
    result=update(s,1,local=((.2,0.,0.),(0.,0.,0.)))
    assert result['contact_stable'].all() and not result['contact_material_speed'].any()


def test_unknown_contact_never_confirms_despite_stable_feature():
    p=LandingPhaseTracker(torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),2)
    p.released[:]=True
    for _ in range(4):assert not p.update(torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool)).any()


def test_online_and_replay_resolve_identical_support_events():
    from supermimic.contact_phase import landing_phase_events
    s=NativeContactStability(.02);p=LandingPhaseTracker(torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),2)
    active=[];stable=[];online=[]
    for t,(a,x) in enumerate([(False,0),(False,0),(True,0),(True,.05),(True,.1),(True,.1),(True,.1)]):
        active.append(torch.tensor([[a]]));stable.append(update(s,t,x=x,active=a)['contact_stable'])
        online.append(p.update(active[-1],torch.ones(1,1,dtype=torch.bool),stable[-1]))
    active=torch.stack(active);stable=torch.stack(stable)
    replay=landing_phase_events(active,torch.ones_like(active),torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),2,torch.ones(len(active),1,dtype=torch.bool),stable)
    assert torch.equal(torch.stack(online),replay)
