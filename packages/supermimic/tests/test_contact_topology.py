from copy import deepcopy
import pytest
import torch
from supermimic.contact_topology import support_topology,classify_transition,physical_intervals,ADMISSION
from supermimic.contact_stability import CONTRACT
from supermimic.contact_phase import VERSION
from supermimic.contact_library import load_slice_clock,history_indices


def data(T=7):
    s=torch.zeros(T,71);s[:,2]=.8;s[:,3]=1
    return dict(surface_local=torch.zeros(T,6,dtype=torch.long),native_part_position=torch.zeros(T,6,3),native_part_rotation=torch.eye(3).expand(T,6,3,3).clone(),source_supervision_valid=torch.ones(T,dtype=torch.bool),s=s.clone(),s1=s.clone(),a=torch.zeros(T,29),a_prev=torch.zeros(T,29),xi=torch.zeros(T,52),next_xi=torch.zeros(T,52),f=torch.arange(T),ep_step=torch.arange(T),term=torch.zeros(T,dtype=torch.bool),active=torch.zeros(T,6,dtype=torch.bool),known=torch.ones(T,6,dtype=torch.bool),contact_stable=torch.zeros(T,6,dtype=torch.bool),contact_load=torch.zeros(T,6),contact_material_speed=torch.zeros(T,6),contact_row_stable=torch.ones(T,6,dtype=torch.bool),point=torch.zeros(T,6,3))


def test_tracking_terminal_is_not_source_rejection():
    d=data();d['term'][-1]=True
    spans,valid,_,_=physical_intervals(d)
    assert spans==[(0,7)] and valid.all()


def test_physical_failure_and_state_discontinuity_are_separate_intervals():
    d=data(5);d['s1'][2,2]=.1;d['s'][3]=d['s1'][2]
    spans,valid,_,_=physical_intervals(d)
    assert spans==[(0,2),(4,5)] and valid.tolist()==[True,True,False,False,True]
    d=data(5);d['f'][3:]+=10
    assert physical_intervals(d)[0]==[(0,3),(3,5)]






def test_active_scraping_is_not_an_established_support_and_unknown_is_explicit():
    d=data(3);d['active'][:,0]=True;d['contact_load'][:,0]=400;d['contact_material_speed'][:,0]=2.5
    states=support_topology(d,[[[10],[],[],[],[],[]] for _ in range(3)])
    assert all(s['mask']==0 and s['raw_mask']==1 for s in states)
    d['known'][1]=False;states=support_topology(d,[[[],[],[],[],[],[]] for _ in range(3)])
    assert classify_transition(states,d,0,3,torch.zeros(6,dtype=torch.bool))['family']=='unknown_evidence'


def fixture(tmp_path):
    d=data(8);d['contact_stable'][3,0]=True;d['contact_stable'][7,0]=True
    target=torch.zeros(2,6,dtype=torch.bool);target[:,0]=True
    meta=dict(task_manifest_sha256='task',source_checkpoint_sha256='checkpoint',selection_manifest_sha256='selection',stable_support_contract=deepcopy(CONTRACT),contact_phase_version=VERSION,source_evidence_admission=deepcopy(ADMISSION),recording_spans=[dict(motion=0,begin=0,end=4),dict(motion=0,begin=4,end=8)])
    x=dict(schema='native_evidence_contact_slices_v2',metadata=deepcopy(meta),starts=torch.tensor([0,4]),stops=torch.tensor([4,8]),goals=torch.tensor([3,7]),motion_ids=torch.tensor([0,0]),trace_ids=torch.tensor([0,1]),case_ids=torch.tensor([1,2]),target_masks=target,keep_masks=torch.zeros_like(target),support_masks=torch.zeros_like(target),source_valid=torch.ones(2,dtype=torch.bool),contact_goal_valid=torch.ones(2,dtype=torch.bool),restart_diagnostic=torch.zeros(2,dtype=torch.int8))
    return dict(data=d,metadata=meta),x,tmp_path/'cases.pt'


def test_restart_failures_are_accepted_and_history_stays_in_own_trace(tmp_path):
    rec,x,path=fixture(tmp_path);rec['data']['term'][-1]=True;torch.save(x,path)
    clock=load_slice_clock(rec,path)
    assert clock.source_evidence_contract==ADMISSION
    assert history_indices(clock,torch.tensor([4]),5).tolist()==[[4,4,4,4,4]]
    assert len(clock.event_ends)==2


def test_diagnostic_goal_flag_does_not_reject_valid_endpoint(tmp_path):
    rec,x,path=fixture(tmp_path);x['contact_goal_valid'][0]=False;torch.save(x,path)
    assert len(load_slice_clock(rec,path).event_ends)==2
    x['contact_goal_valid'][0]=True;x['source_valid'][1]=False;torch.save(x,path)
    with pytest.raises(ValueError,match='source-invalid'):load_slice_clock(rec,path)
    x['source_valid'][1]=True;x['trace_ids'][1]=-1;torch.save(x,path)
    with pytest.raises(ValueError,match='trace identity'):load_slice_clock(rec,path)




def test_unqualified_weak_contact_surfaces_are_preserved_in_native_topology():
    from supermimic.contact_topology import native_contact_phases
    d=data(3);d['active'][:,0]=True;d['contact_load'][:,0]=5
    surfaces=[[[10],[],[],[],[],[]],[[11],[],[],[],[],[]],[[11],[],[],[],[],[]]]
    raw=native_contact_phases(d,surfaces);support=support_topology(d,surfaces)
    assert len(raw)==2 and raw[0]['attachments'][0]==[10] and raw[1]['attachments'][0]==[11]
    assert all(x['mask']==1 for x in raw) and all(x['mask']==0 for x in support)


def test_new_long_source_transfers_cannot_be_silently_overwritten():
    from types import SimpleNamespace
    from supermimic.contact_library import require_source_replay_capacity
    clock=SimpleNamespace(segment_starts=torch.tensor([0,300]),segment_stops=torch.tensor([234,320]))
    with pytest.raises(ValueError,match='244'):require_source_replay_capacity(clock,160,10)
    assert require_source_replay_capacity(clock,244,10)==244
    assert require_source_replay_capacity(clock,160,10,segment=2)==30


def test_recorded_history_context_is_retained_without_supervised_goals(tmp_path):
    rec,x,path=fixture(tmp_path);x['starts'][0]=2;rec['data']['source_supervision_valid'][:2]=False
    torch.save(x,path);clock=load_slice_clock(rec,path)
    assert history_indices(clock,torch.tensor([2]),3).tolist()==[[0,0,1]]
    x['starts'][0]=0;torch.save(x,path)
    with pytest.raises(ValueError,match='History-only'):load_slice_clock(rec,path)


def test_trimmed_source_history_never_reads_excluded_prefix(tmp_path):
    from supermimic.contact_source_motion import SOURCE_MOTION_CONTRACT
    rec,x,path=fixture(tmp_path)
    rec['metadata']['source_motion_contract']=deepcopy(SOURCE_MOTION_CONTRACT)
    x['metadata']['source_motion_contract']=deepcopy(SOURCE_MOTION_CONTRACT)
    x['schema']='native_motion_contact_slices_v3'
    for meta in (rec['metadata'],x['metadata']):
        meta['recording_spans'][0]['history_begin']=2
        meta['recording_spans'][1]['history_begin']=4
    x['starts'][0]=2;rec['data']['source_supervision_valid'][:2]=False
    torch.save(x,path);clock=load_slice_clock(rec,path)
    assert history_indices(clock,torch.tensor([2,3]),3).tolist()==[[2,2,2],[2,2,2]]
    x['starts'][0]=0;torch.save(x,path)
    with pytest.raises(ValueError,match='History-only'):load_slice_clock(rec,path)


def test_offline_segmentation_settings_do_not_gate_endpoint_loading(tmp_path):
    rec,x,path=fixture(tmp_path)
    x['metadata']['source_evidence_admission']={'version':'offline-audit-only'}
    torch.save(x,path)
    clock=load_slice_clock(rec,path)
    assert clock.source_evidence_contract=={'version':'offline-audit-only'}


def test_actual_unqualified_target_is_still_rejected(tmp_path):
    rec,x,path=fixture(tmp_path)
    rec['data']['contact_stable'][3,0]=False
    torch.save(x,path)
    with pytest.raises(ValueError,match='unqualified required part'):
        load_slice_clock(rec,path)
