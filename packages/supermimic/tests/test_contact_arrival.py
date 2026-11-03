from types import SimpleNamespace
import torch
from supermimic.contact_arrival import ArrivalWindow, arrival_indices
from supermimic.contact_dmo import _replay_events


def trace(distances, active=None, first=0):
    T=len(distances)
    point=torch.zeros(T,1,6,3);point[:,0,0,0]=torch.tensor(distances)
    observed=torch.zeros(T,1,6,dtype=torch.bool)
    observed[:,0,0]=True if active is None else torch.tensor(active)
    onset=torch.zeros_like(observed)
    if first is not None:onset[first,0,0]=True
    done=torch.zeros(T,1,dtype=torch.bool);done[-1]=True
    return dict(point=point,active=observed,known=torch.ones_like(observed),contact_first=onset,done=done)


def test_real_jump_selects_actual_next_frame_not_average_or_later_hit():
    d=trace([.2032,.0011,0.])
    i,e=arrival_indices(d,torch.zeros(1,6,3))
    assert i[0,0]==1 and torch.isclose(e[0,0],torch.tensor(.0011))


def test_dropout_closes_window_and_recontact_cannot_repair():
    d=trace([.2,0.,.001],[True,False,True])
    i,e=arrival_indices(d,torch.zeros(1,6,3),frames=3)
    assert i[0,0]==0 and e[0,0]>.19


def test_initial_active_without_new_onset_is_missing():
    i,e=arrival_indices(trace([0.,0.],first=None),torch.zeros(1,6,3))
    assert i[0,0]==-1 and torch.isinf(e[0,0])


def test_wrong_face_is_not_promoted_and_inactive_candidate_not_used():
    d=trace([.2,0.]);eligible=torch.ones_like(d['active']);eligible[1]=False
    i,e=arrival_indices(d,torch.zeros(1,6,3),eligible=eligible)
    assert i[0,0]==0 and e[0,0]>.19
    d['active'][1]=False
    i,e=arrival_indices(d,torch.zeros(1,6,3))
    assert i[0,0]==0


def test_terminal_does_not_read_next_reset_and_frames_one_is_legacy():
    d=trace([.2,.001]);d['done'][0]=True
    assert arrival_indices(d,torch.zeros(1,6,3))[0][0,0]==0
    d['done'][0]=False
    assert arrival_indices(d,torch.zeros(1,6,3),frames=1)[0][0,0]==0


def test_online_closes_before_acceptance_and_partial_reset():
    d=trace([.2,.001]);w=ArrivalWindow((1,6),'cpu',2)
    for t in range(2):
        w.update(d['contact_first'][t],d['active'][t],d['known'][t],d['point'][t].norm(dim=-1),t,d['done'][t])
        assert bool(w.closed[0,0])==(t==1)
    i,e=arrival_indices(d,torch.zeros(1,6,3))
    assert torch.equal(w.index,i) and torch.equal(w.distance,e)
    w.reset(torch.tensor([0]));assert (w.index==-1).all() and torch.isinf(w.distance).all()


def test_loss_event_and_online_metric_pick_exact_same_witness():
    d=trace([.2032,.0011,0.]);new=torch.zeros(1,6,dtype=torch.bool);new[0,0]=True
    c=SimpleNamespace(event_ends=torch.tensor([1]),target_masks=new,keep_masks=torch.zeros_like(new),data={'point':torch.zeros(1,6,3)})
    native={}
    evaluate,terminal,schedule=_replay_events(d,c,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,True,2,native)
    assert evaluate[:,0,0].tolist()==[False,True,False]
    assert torch.equal(evaluate,native['mask'])
    d['contact_first'].zero_();native={}
    evaluate,_,_=_replay_events(d,c,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,True,2,native)
    assert evaluate[:,0,0].tolist()==[False,False,True]
    assert not native['mask'].any() # fallback cannot call an initial support a new arrival


def test_missing_contact_report_uses_null_and_explicit_sample_count():
    import json
    from supermimic.contact_task import ContactTask
    t=ContactTask.__new__(ContactTask)
    t.sequential=False;t.count=1;t.segment_index=None;t.grace_steps=10;t.N=1
    t.env=SimpleNamespace(N=1);t.clock=SimpleNamespace();t._report_ranges=[(0,2)]
    for key in ('attempts','successes','root_sum','point_sum','point_count'):
        setattr(t,key,torch.zeros(1))
    t.attempts.fill_(1);t.contact_tolerances_by_segment=torch.tensor([.05])
    t.last_terminal_steps=torch.tensor([2]);t.current_root=torch.tensor(0.)
    t.current_contact=torch.tensor(0.);t.current_contact_count=torch.tensor(0)
    report=t.report(0,0)
    assert report['current_contact_error_m'] is None
    assert report['segments'][0]['final_contact_error_m'] is None
    assert report['segments'][0]['contact_error_samples']==0
    json.dumps(report,allow_nan=False)
