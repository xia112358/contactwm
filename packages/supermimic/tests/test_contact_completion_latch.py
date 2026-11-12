import torch
from types import SimpleNamespace
from supermimic.contact_loss_sampling import completed_stable_samples
from supermimic.contact_dmo import loss_aligned_attempt_metrics, _replay_events


def fixture():
    T=3;state=torch.zeros(T,1,71);state[...,2]=.8;state[...,3]=1
    required=torch.zeros(1,6,dtype=torch.bool);required[0,2:4]=True
    clock=SimpleNamespace(material_goal_mode=True,fixed_goal_mode=True,
        material_local=torch.zeros(1,6,4,3),material_targets=torch.zeros(1,6,4,3),
        material_contact_tolerances=torch.tensor([.05]),event_ends=torch.tensor([1]),
        target_masks=required,keep_masks=torch.zeros_like(required),case_ids=torch.tensor([116]),
        data={'point':torch.zeros(1,6,3),'s1':state[0]})
    stable=torch.zeros(T,1,6,dtype=torch.bool);stable[0,0,2]=True;stable[2,0,2:4]=True
    point=torch.zeros(T,1,6,3);point[0,0,2,0]=.02;point[2,0,2,0]=.20;point[2,0,3,0]=.03
    data=dict(done=torch.tensor([[False],[False],[True]]),goal_index=torch.zeros(T,1,dtype=torch.long),
        s1=state,contact_stable=stable,contact_loss_point=point,
        contact_loss_position=torch.zeros(T,1,6,3),contact_loss_rotation=torch.eye(3).expand(T,1,6,3,3))
    return clock,data


def test_completed_part_survives_later_bad_contact_until_other_part_arrives():
    clock,data=fixture();r=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert r['success_rate']==1
    assert abs(r['best_contact_error_cm']-2.5)<1e-5
    assert abs(r['worst_part_error_cm']-3)<1e-5
    assert sum(r['failure_reasons'].values())==0
    assert r['segments'][0]['case_id']==116
    ev,_,_=_replay_events(data,clock,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,stable_contact_frames=2)
    assert ev[:,0,2].tolist()==[True,False,False]
    assert ev[:,0,3].tolist()==[False,False,True]


def test_unqualified_geometry_never_latches_and_missing_error_has_denominator():
    clock,data=fixture();data['contact_stable'][:,:,3]=False
    r=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert r['success_rate']==0 and r['best_contact_error_cm'] is None
    assert r['contact_error_samples']==0
    assert r['failure_reasons']['missing_stable_contact']==1
    assert r['segments'][0]['part_missing_attempts'][3]==1


def test_incomplete_part_uses_latest_stable_and_never_observes_after_done():
    stable=torch.ones(4,1,1,dtype=torch.bool);live=torch.tensor([[True],[True],[True],[False]])
    errors=torch.tensor([.08,.07,.09,.01]).square().reshape(4,1,1)
    selected=completed_stable_samples(stable,live,errors,torch.tensor(.05))
    assert selected[:,0,0].tolist()==[False,False,True,False]


def test_completed_contact_does_not_override_root_failure():
    clock,data=fixture();data['s1']=data['s1'].clone();data['s1'][-1,0,0]=.5
    r=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert r['success_rate']==0 and r['failure_reasons']['root_position']==1


def test_interrupted_contact_keeps_guidance_but_not_false_body_endpoint():
    clock,data=fixture();data['contact_stable'].zero_()
    data['body_endpoint_valid']=torch.tensor([[True],[True],[False]])
    ev,terminal,schedule=_replay_events(data,clock,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,stable_contact_frames=2)
    assert ev[-1,0,2:4].all() and int(ev.sum())==2
    assert not terminal.any() and len(schedule[2]['root'])==0


def test_collision_interruption_suppresses_body_even_if_targets_were_reached():
    clock,data=fixture();data['s1']=data['s1'].clone();data['s1'][-1,0,0]=.5
    data['body_endpoint_valid']=torch.ones_like(data['done'])
    data['contact_interrupted']=data['done'].clone()
    data['collision_mask']=torch.zeros(3,1,6,dtype=torch.bool);data['collision_mask'][-1,0,2]=True
    data['collision_in_region']=torch.zeros_like(data['collision_mask'])
    _,terminal,schedule=_replay_events(data,clock,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,stable_contact_frames=2)
    assert not terminal.any() and all(len(row['root'])==0 for row in schedule.values())
    r=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert r['success_rate']==0 and r['failure_reasons']['root_position']==1


def test_stage_handoff_requires_its_own_completion_witness():
    clock,data=fixture()
    clock.event_ends=torch.tensor([1,2]);clock.target_masks=clock.target_masks.expand(2,-1);clock.keep_masks=clock.keep_masks.expand(2,-1)
    clock.material_local=clock.material_local.expand(2,-1,-1,-1);clock.material_targets=clock.material_targets.expand(2,-1,-1,-1)
    clock.material_contact_tolerances=torch.tensor([.05,.05]);clock.data['point']=clock.data['point'].expand(2,-1,-1)
    data['goal_index'][2]=1
    plan={'length':2,'enabled':torch.ones(2,1,dtype=torch.bool),'ends':torch.tensor([[2],[3]]),'truncated':torch.zeros(2,1,dtype=torch.bool)}
    ev,_,_=_replay_events(data,clock,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',plan,stable_contact_frames=2)
    assert ev[:,0,2].tolist()==[True,False,True]


def test_collision_interruption_keeps_goal_at_actual_stop_state():
    clock,data=fixture()
    data['contact_interrupted']=data['done'].clone()
    data['collision_mask']=torch.zeros(3,1,6,dtype=torch.bool);data['collision_mask'][-1,0,2]=True
    data['collision_in_region']=torch.zeros_like(data['collision_mask'])
    events={}
    ev,terminal,schedule=_replay_events(data,clock,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,native_events=events,stable_contact_frames=2)
    assert int(ev.sum())==2 and not terminal.any()
    assert ev[:,0,2].tolist()==[True,False,False]
    assert ev[:,0,3].tolist()==[False,False,True]
    assert events['mask'][2,0,3]
    assert events['required_terms'][0,2:4].eq(1).all()


def test_first_step_collision_uses_collected_stop_without_exempting_contact():
    clock,data=fixture();data['contact_stable'].fill_(True)
    data['contact_interrupted']=data['done'].clone()
    data['collision_mask']=torch.zeros(3,1,6,dtype=torch.bool);data['collision_mask'][0,0,2]=True
    data['collision_in_region']=torch.zeros_like(data['collision_mask'])
    events={}
    ev,terminal,_=_replay_events(data,clock,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,native_events=events,stable_contact_frames=2)
    assert ev[0,0,2:4].all() and int(ev.sum())==2
    assert 'pre_action_guidance' not in events
    assert events['mask'][0,0,2:4].all()
    assert not terminal.any()


def test_normal_timeout_still_has_target_guidance():
    clock,data=fixture();data['contact_stable'].zero_()
    data['contact_interrupted']=torch.zeros_like(data['done'])
    events={}
    ev,_,_=_replay_events(data,clock,torch.tensor([0]),torch.tensor([3]),torch.tensor([False]),'accumulated_target',None,native_events=events,stable_contact_frames=2)
    assert ev[-1,0,2:4].all() and int(events['required_terms'].sum())==2
