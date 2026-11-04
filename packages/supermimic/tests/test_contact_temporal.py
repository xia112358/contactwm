"""End-to-end contracts for the single offline support/transfer timeline."""
import pytest
import torch
from supermimic.contact_temporal import segmentation_contact_state
from supermimic.contact_topology import support_topology,classify_limb_roles
from supermimic.stable_contact_slices import load_transfer_slices


def data(T=24):
    return dict(active=torch.zeros(T,6,dtype=torch.bool),known=torch.ones(T,6,dtype=torch.bool),
        contact_load=torch.zeros(T,6),contact_material_speed=torch.zeros(T,6),contact_stable=torch.zeros(T,6,dtype=torch.bool),
        contact_row_stable=torch.zeros(T,6,dtype=torch.bool),surface_local=torch.zeros(T,6,dtype=torch.long),
        native_part_position=torch.zeros(T,6,3),native_part_rotation=torch.eye(3).expand(T,6,3,3).clone(),s1=torch.cat((torch.zeros(T,3),torch.ones(T,1),torch.zeros(T,67)),dim=1),point=torch.zeros(T,6,3))


def bearing(d,p,start,stop):
    d['active'][start:stop,p]=True;d['contact_load'][start:stop,p]=100


def land(d,p,start):
    bearing(d,p,start,len(d['active']));d['native_part_position'][start:,p,0]=.1


def roles(d,start,stop):
    timeline=segmentation_contact_state(d)
    states=support_topology(d,[[[] for _ in range(6)] for _ in range(len(d['active']))],timeline)
    return classify_limb_roles(states,d,start,stop,torch.zeros(6,dtype=torch.bool),timeline)


def test_sustained_support_retains_onset_and_separate_confirmation():
    d=data(10);land(d,0,2);t=segmentation_contact_state(d)
    assert t['episodes']==[dict(part=0,start=2,confirmed=6,surface=0,stop=10)]
    cuts,hits=load_transfer_slices(d,t)
    assert hits[:,0].nonzero().flatten().tolist()==[2]
    assert [(c['start'],c['stop']) for c in cuts]==[(0,10)]
    assert t['transfers'][0]['boundary_eligible']


@pytest.mark.parametrize('duration',[1,2,3,4])
def test_short_touch_is_raw_evidence_not_support_or_boundary(duration):
    d=data();bearing(d,0,5,5+duration);d['native_part_position'][:,0,0]=torch.arange(24)*.01
    t=segmentation_contact_state(d);cuts,hits=load_transfer_slices(d,t)
    assert not t['support'].any() and not hits.any()
    assert d['active'][:,0].sum()==duration
    assert [(c['start'],c['stop']) for c in cuts]==[(0,24)]


def test_short_touch_inside_flight_does_not_split_its_identity():
    d=data();bearing(d,0,0,6);bearing(d,0,9,11);bearing(d,0,17,24)
    d['native_part_position'][6:18,0,0]=torch.linspace(0,.2,12);d['native_part_position'][18:,0,0]=.2
    t=segmentation_contact_state(d)
    assert len(t['transfers'])==1
    assert (t['transfers'][0]['start'],t['transfers'][0]['arrival'])==(6,17)
    assert roles(d,9,11)['limb_roles']['LL']=='T'


def test_short_stationary_same_face_gap_only_bridges_derived_state():
    d=data();bearing(d,0,0,24);d['active'][8:11,0]=False;d['contact_load'][8:11,0]=0
    before={k:v.clone() for k,v in d.items()};t=segmentation_contact_state(d)
    assert t['bridged'][8:11,0].all() and t['support'][:,0].all()
    assert not t['transfers'] and all(torch.equal(d[k],v) for k,v in before.items())


@pytest.mark.parametrize('mode',['unknown','surface','motion'])
def test_invalid_gap_evidence_cannot_bridge(mode):
    d=data();bearing(d,0,0,24);d['active'][8:11,0]=False;d['contact_load'][8:11,0]=0
    if mode=='unknown':d['known'][9,0]=False
    if mode=='surface':d['surface_local'][11:,0]=1
    if mode=='motion':d['native_part_position'][9,0,0]=.1
    assert not segmentation_contact_state(d)['bridged'].any()


def test_load_hysteresis_and_exit_do_not_wait_for_row_identity():
    d=data(10);bearing(d,0,0,10);d['contact_load'][5:,0]=torch.tensor([29,20,31,14,14])
    t=segmentation_contact_state(d)
    assert t['support'][:,0].tolist()==[True]*8+[False]*2
    assert not d['contact_stable'].any()


def test_sustained_sliding_preserves_bearing_without_inventing_transfer():
    d=data(12);bearing(d,0,0,12);d['contact_material_speed'][6:,0]=2
    t=segmentation_contact_state(d)
    assert t['support'][:,0].all()
    assert d['active'][:,0].all() and not t['transfers']


def test_unknown_interval_does_not_connect_transfer_evidence():
    d=data();bearing(d,0,0,6);land(d,0,16);d['known'][10,0]=False
    t=segmentation_contact_state(d)
    assert all(not (e['start']<10<e['stop']) for e in t['transfers'])


def test_nearby_arrivals_merge_and_root_motion_is_not_support_motion():
    d=data();land(d,0,3);land(d,1,5);d['s1'][:,0]=torch.arange(24)*.1
    cuts,_=load_transfer_slices(d)
    assert cuts[0]['arrivals']==[(3,0),(5,1)] and cuts[0]['stop']==24
    assert roles(d,10,24)['role_key']=='SSNN'


def test_merge_window_does_not_chain_unboundedly():
    d=data(60)
    for p in (0,1,2,3):bearing(d,p,0,6)
    land(d,0,22);land(d,2,27);land(d,1,32);land(d,3,37)
    d['active'][30:42,0]=False;d['contact_load'][30:42,0]=0
    d['native_part_position'][30:43,0,0]=torch.linspace(.1,.3,13)
    d['native_part_position'][43:,0,0]=.3
    cuts,_=load_transfer_slices(d)
    assert [c['stop'] for c in cuts]==[28,60]
    assert [f for c in cuts for f,p in c['arrivals'] if p==0]==[22,42]
    assert all(len([p for _,p in c['arrivals']])==len({p for _,p in c['arrivals']}) for c in cuts)


def test_surface_change_blocks_arrival_merge():
    d=data(40);bearing(d,0,0,6);bearing(d,1,0,6)
    land(d,0,16);land(d,1,21);d['surface_local'][21:,0]=1
    cuts,_=load_transfer_slices(d)
    assert [c['stop'] for c in cuts]==[17,40]


def test_foot_to_knee_form_changes_only_at_observed_contact():
    d=data();bearing(d,0,0,6);land(d,4,14)
    t=segmentation_contact_state(d);cuts,_=load_transfer_slices(d,t)
    assert t['form_changes']==[dict(frame=14,limb=0,source_part=0,target_part=4)]
    assert [(c['start'],c['stop']) for c in cuts]==[(0,14),(14,24)]
    assert roles(d,0,6)['contact_forms']['LL']=='foot'
    assert (t['forms'][6:14,0]==-1).all()
    assert roles(d,6,14)['contact_forms']['LL']=='none'
    assert roles(d,6,14)['limb_roles']['LL']=='T'
    assert roles(d,14,19)['contact_forms']['LL']=='knee'
    assert roles(d,6,19)['limb_roles']['LL']=='T'
    with pytest.raises(ValueError,match='multiple main contact forms'):roles(d,0,19)


def test_form_and_nearby_arrival_arbitration_retains_every_event():
    d=data(30);bearing(d,0,0,10);land(d,4,18);land(d,2,5)
    t=segmentation_contact_state(d);cuts,_=load_transfer_slices(d,t)
    assert 18 in [c['stop'] for c in cuts]
    assert len({b for c in cuts for b in (c['start'],c['stop']) if 17<=b<=19})==1
    assert sorted((f,p) for c in cuts for f,p in c['arrivals'])==sorted((e['arrival'],e['part']) for e in t['transfers'] if e['arrival'] is not None)
    coverage=torch.zeros(30,dtype=torch.int)
    for c in cuts:coverage[c['start']:c['stop']]+=1
    assert (coverage==1).all()


def test_simultaneous_sustained_foot_and_knee_is_reported():
    d=data();bearing(d,0,0,24);bearing(d,4,0,24)
    with pytest.raises(ValueError,match='Simultaneous sustained'):segmentation_contact_state(d)


def test_two_frame_knee_touch_does_not_change_foot_form():
    d=data();bearing(d,0,0,24);bearing(d,4,8,10)
    t=segmentation_contact_state(d)
    assert not t['form_changes'] and (t['forms'][:,0]==0).all()
    assert roles(d,0,24)['contact_forms']['LL']=='foot'


def test_arrival_after_segment_boundary_keeps_transfer_role():
    d=data();land(d,0,10);d['native_part_position'][5:10,0,0]=torch.linspace(0,.08,5)
    c=roles(d,10,15)
    assert c['limb_roles']['LL']=='T' and c['role_evidence'][0]['transfer_started_before_segment']
    assert roles(d,15,24)['limb_roles']['LL']=='S'


def test_single_frame_in_flight_reads_same_transfer():
    d=data();bearing(d,0,0,6);land(d,0,17)
    d['native_part_position'][6:17,0,0]=torch.linspace(0,.1,11)
    assert roles(d,12,13)['limb_roles']['LL']=='T'


def test_terminal_handoff_is_support_until_next_segment():
    d=data(30);bearing(d,0,0,8);land(d,0,20)
    d['native_part_position'][8:20,0,0]=torch.linspace(0,.1,12)
    assert roles(d,0,10)['limb_roles']['LL']=='S'
    assert roles(d,10,25)['limb_roles']['LL']=='T'


def test_unfinished_trace_tail_is_transfer_but_stationary_airborne_is_not():
    d=data();bearing(d,0,0,6)
    assert roles(d,10,24)['limb_roles']['LL']=='N'
    d['native_part_position'][6:,0,0]=torch.arange(18)*.01
    assert roles(d,10,24)['limb_roles']['LL']=='T'


def test_row_quality_never_changes_partition_forms_or_roles():
    d=data();bearing(d,1,0,24);land(d,0,10)
    before=roles(d,0,15);a,_=load_transfer_slices(d)
    d['contact_stable'][:]=True;d['contact_row_stable'][:]=True
    b,_=load_transfer_slices(d)
    assert before==roles(d,0,15)
    assert [(c['start'],c['stop']) for c in a]==[(c['start'],c['stop']) for c in b]


def test_repeated_transfers_have_distinct_event_identity():
    d=data(40);bearing(d,0,0,6);bearing(d,0,14,21);bearing(d,0,30,40)
    d['native_part_position'][6:15,0,0]=torch.linspace(0,.1,9);d['native_part_position'][15:21,0,0]=.1
    d['native_part_position'][21:31,0,0]=torch.linspace(.1,.2,10);d['native_part_position'][31:,0,0]=.2
    t=segmentation_contact_state(d);cuts,hits=load_transfer_slices(d,t)
    assert hits[:,0].nonzero().flatten().tolist()==[14,30]
    assert len({e['id'] for e in t['transfers']})==2
    assert [c['stop'] for c in cuts]==[15,40]


def test_moving_unsupported_part_with_no_source_evidence_is_inactive():
    d=data();d['native_part_position'][:,0,0]=torch.arange(24)*.1
    assert roles(d,0,24)['limb_roles']['LL']=='N'


def test_mirror_swaps_whole_role_vector_and_form_pairs():
    d=data();bearing(d,0,0,24);land(d,1,10);bearing(d,2,0,24)
    order=[1,0,3,2,5,4]
    mirrored={k:v[:,order] if k!='s1' else v for k,v in d.items()}
    a=roles(d,0,15);b=roles(mirrored,0,15)
    assert a['role_key']=='STSN' and b['role_key']=='TSNS'
    assert a['mirror_role_key']==b['mirror_role_key']


def test_weak_raw_contacts_are_not_promoted_by_long_duration():
    d=data();bearing(d,0,0,24);d['contact_load'][:,0]=5
    assert not segmentation_contact_state(d)['support'].any()
    assert roles(d,0,24)['limb_roles']['LL']=='N'


def test_cross_boundary_arrival_is_valid_role_but_not_legacy_release():
    d=data();land(d,0,5);land(d,1,12)
    cuts,_=load_transfer_slices(d)
    # Boundaries are never expanded to recreate a release for an old trainer.
    assert all(c['start']==(cuts[i-1]['stop'] if i else 0) for i,c in enumerate(cuts))
    assert roles(d,12,17)['limb_roles']['RL']=='T'


def test_surface_audit_reads_same_transfer_not_point_quality():
    from supermimic.contact_topology import phase_ranges,annotate_surface_transitions,classify_transition
    d=data();bearing(d,0,0,6);land(d,0,16);d['surface_local'][16:,0]=1
    t=segmentation_contact_state(d)
    attachments=[[[10 if f<16 else 11],[],[],[],[],[]] for f in range(24)]
    states=support_topology(d,attachments,t);new=torch.tensor([1,0,0,0,0,0],dtype=torch.bool)
    c=dict(trace_id=0,start=10,goal=20,new=new.tolist(),topology_floor=0,transfer_ids=[0],topology=classify_transition(states,d,10,21,new,t))
    phases=[dict(trace_id=0,**x) for x in phase_ranges(states)]
    annotate_surface_transitions(d,[dict(begin=0,transfer_events=t['transfers'],model_fingerprint='native')],[c],phases)
    relation=c['topology']['landing_surfaces'][0]
    assert relation['relation']=='cross_surface' and relation['release_frame']==6
    assert relation['source_shapes']==[10] and relation['target_shapes']==[11]
    assert not d['contact_stable'].any()


def test_terminal_flight_does_not_relabel_support_handoff_as_transfer():
    d=data();bearing(d,0,0,10);land(d,1,8)
    d['native_part_position'][10:,0,0]=torch.arange(14)*.03
    c=roles(d,6,13)
    assert c['role_key']=='STNN'
    assert roles(d,13,24)['limb_roles']['LL']=='T'


def test_loaded_scrape_with_moving_body_pose_is_not_a_settled_arrival():
    d=data(30);bearing(d,0,0,6);bearing(d,0,10,18);land(d,0,23)
    d['native_part_position'][6:24,0,0]=torch.linspace(0,.3,18)
    d['native_part_position'][24:,0,0]=.3
    t=segmentation_contact_state(d)
    assert not any(e['start']<=10<e['stop'] for e in t['episodes'])
    assert len(t['transfers'])==1 and t['transfers'][0]['arrival']==23
    assert d['active'][10:18,0].all()


def test_rolling_loaded_departure_waits_for_native_release():
    d=data(30);bearing(d,0,0,12);land(d,0,22)
    d['native_part_position'][8:23,0,0]=torch.linspace(0,.3,15);d['native_part_position'][23:,0,0]=.3
    t=segmentation_contact_state(d)
    assert t['transfers'][0]['start']==12
    assert not d['active'][t['transfers'][0]['start'],0]


def test_missing_material_motion_is_not_stationary_evidence():
    d=data();bearing(d,0,3,24)
    t=segmentation_contact_state(d)
    assert not t['material_motion_known'][3,0] and t['material_motion_known'][4:,0].all()
    assert t['episodes'][0]['confirmed']>=7


def test_body_carried_hand_does_not_retroactively_transfer_for_whole_gap():
    d=data(35);d['s1'][:,0]=torch.arange(35)*.01
    d['native_part_position'][:,2,0]=d['s1'][:,0]
    # A long unsupported carry followed by an independent late placement.
    d['s1'][25:,0]=.25;d['native_part_position'][25:,2,0]=.25
    bearing(d,2,25,35)
    assert roles(d,0,15)['limb_roles']['LA']=='N'


def test_near_form_cut_cannot_merge_two_arrivals_of_same_part():
    d=data(40);bearing(d,0,0,10);land(d,4,25)
    land(d,1,9);d['active'][17:27,1]=False;d['contact_load'][17:27,1]=0
    d['native_part_position'][17:28,1,0]=torch.linspace(.1,.25,11);d['native_part_position'][28:,1,0]=.25
    t=segmentation_contact_state(d);cuts,_=load_transfer_slices(d,t)
    for c in cuts:
        parts=[p for _,p in c['arrivals']]
        assert len(parts)==len(set(parts))


def test_continuous_loaded_rolling_does_not_create_transfer():
    d=data(30);bearing(d,0,0,30)
    d['native_part_position'][8:20,0,0]=torch.linspace(0,.08,12)
    d['native_part_position'][20:,0,0]=.08
    t=segmentation_contact_state(d)
    assert not t['transfers']
    assert t['support'][:,0].all()
    assert roles(d,0,30)['limb_roles']['LL']=='S'


def test_continuous_sliding_and_resettling_is_not_relocation():
    d=data(30);bearing(d,0,0,30)
    d['contact_material_speed'][8:18,0]=.8
    d['native_part_position'][8:19,0,0]=torch.linspace(0,.1,11)
    d['native_part_position'][19:,0,0]=.1
    t=segmentation_contact_state(d)
    assert not t['transfers']
    cuts,hits=load_transfer_slices(d,t)
    assert not hits.any() and len(cuts)==1


def test_sliding_does_not_start_transfer_before_later_release():
    d=data(35);bearing(d,0,0,15);land(d,0,25)
    d['contact_material_speed'][7:15,0]=.8
    d['native_part_position'][7:26,0,0]=torch.linspace(0,.3,19)
    d['native_part_position'][26:,0,0]=.3
    t=segmentation_contact_state(d)
    assert len(t['transfers'])==1
    assert t['transfers'][0]['start']==15


def test_initial_contact_is_retained_without_creating_a_prefix_fragment():
    d=data(35);bearing(d,0,0,12);land(d,1,3);land(d,0,25)
    t=segmentation_contact_state(d);cuts,hits=load_transfer_slices(d,t)
    assert hits[3,1]
    assert cuts[0]['start']==0 and cuts[0]['stop']==35
    assert (3,1) in cuts[0]['arrivals'] and (25,0) in cuts[0]['arrivals']


def test_hand_reversal_does_not_cut_a_continuous_engagement():
    d=data(60);land(d,2,50)
    d['native_part_position'][:26,2,0]=torch.linspace(0,-.4,26)
    d['native_part_position'][26:51,2,0]=torch.linspace(-.4,.1,25)
    t=segmentation_contact_state(d);e=t['transfers'][0]
    assert e['activity_ranges']==[[0,51]]
    assert roles(d,0,20)['limb_roles']['LA']=='T'
    assert roles(d,35,51)['limb_roles']['LA']=='T'
    # A moving first engagement has an observed completion despite unknown history.
    assert e['boundary_eligible']


def test_initial_and_terminal_airborne_frames_have_no_assigned_form():
    d=data(30);bearing(d,0,8,18)
    d['native_part_position'][:,0,0]=torch.arange(30)*.01
    d['native_part_position'][8:18,0,0]=.08
    t=segmentation_contact_state(d)
    assert (t['forms'][:8,0]==-1).all()
    assert (t['forms'][18:,0]==-1).all()
    assert roles(d,20,30)['limb_roles']['LL']=='T'
    assert roles(d,20,30)['contact_forms']['LL']=='none'


def test_bridged_contact_gap_does_not_invent_observed_form():
    d=data();bearing(d,0,0,24);d['active'][8:10,0]=False;d['contact_load'][8:10,0]=0
    t=segmentation_contact_state(d)
    assert t['support'][8:10,0].all()
    assert (t['forms'][8:10,0]==-1).all()


def test_idle_hand_can_engage_without_an_old_contact():
    d=data(45);land(d,2,30)
    d['native_part_position'][15:31,2,0]=torch.linspace(0,.1,16)
    e=segmentation_contact_state(d)['transfers'][0]
    assert e['source_part'] is None and e['participation_kind']=='engage'
    assert roles(d,0,15)['limb_roles']['LA']=='N'
    assert roles(d,18,28)['limb_roles']['LA']=='T'
    assert roles(d,31,45)['limb_roles']['LA']=='S'


def test_stationary_existing_contact_can_become_support_by_loading():
    d=data(30);bearing(d,2,0,30);d['contact_load'][:15,2]=5
    t=segmentation_contact_state(d);e=t['transfers'][0]
    assert e['arrival']==15 and e['activity_ranges']==[[15,16]]
    assert d['active'][:,2].all() and e['motion_m']==0
    assert roles(d,10,20)['limb_roles']['LA']=='T'


def test_unloading_without_native_release_is_disengagement():
    d=data(30);bearing(d,2,0,30);d['contact_load'][15:,2]=5
    e=segmentation_contact_state(d)['transfers'][0]
    assert e['participation_kind']=='disengage' and e['activity_ranges']==[[15,16]]
    assert roles(d,15,16)['limb_roles']['LA']=='T'
    assert roles(d,16,30)['limb_roles']['LA']=='N'


def test_idle_between_withdrawal_and_reach_is_not_continuous_transfer():
    d=data(70);bearing(d,2,0,10);bearing(d,2,55,70)
    d['native_part_position'][10:20,2,0]=torch.linspace(0,-.2,10)
    d['native_part_position'][20:40,2,0]=-.2
    d['native_part_position'][40:56,2,0]=torch.linspace(-.2,.1,16)
    d['native_part_position'][56:,2,0]=.1
    assert roles(d,12,18)['limb_roles']['LA']=='T'
    assert roles(d,25,35)['limb_roles']['LA']=='N'
    assert roles(d,45,56)['limb_roles']['LA']=='T'


def test_withdrawal_ends_at_idle_without_needing_a_new_contact():
    d=data(40);bearing(d,2,0,10)
    d['native_part_position'][10:21,2,0]=torch.linspace(0,-.2,11)
    d['native_part_position'][21:,2,0]=-.2
    t=segmentation_contact_state(d);e=t['transfers'][0]
    assert e['arrival'] is None and e['participation_kind']=='disengage'
    assert roles(d,12,20)['limb_roles']['LA']=='T'
    assert roles(d,25,40)['limb_roles']['LA']=='N'


def test_resting_hand_jitter_does_not_extend_withdrawal_to_trace_end():
    d=data(60);bearing(d,2,0,10)
    d['native_part_position'][10:21,2,0]=torch.linspace(0,-.2,11)
    d['native_part_position'][21:,2,0]=-.2+torch.arange(39).remainder(2)*.004
    e=segmentation_contact_state(d)['transfers'][0]
    assert e['stop']<=22
    assert roles(d,30,60)['limb_roles']['LA']=='N'


def test_load_reallocation_does_not_require_airborne_motion():
    d=data(40);bearing(d,0,0,40);d['contact_load'][10:20,0]=5
    t=segmentation_contact_state(d)
    assert d['active'][:,0].all()
    assert len(t['transfers'])==1 and t['transfers'][0]['arrival']==20
    assert roles(d,20,25)['limb_roles']['LL']=='T'
    assert roles(d,14,18)['limb_roles']['LL']=='N'


def test_initial_moving_engagement_is_eligible_but_settled_tail_is_absorbed():
    d=data(45);land(d,0,20)
    d['native_part_position'][:21,0,0]=torch.linspace(0,.3,21)
    d['native_part_position'][21:,0,0]=.3
    t=segmentation_contact_state(d);cuts,hits=load_transfer_slices(d,t)
    assert hits[20,0] and t['transfers'][0]['boundary_eligible']
    assert [(c['start'],c['stop']) for c in cuts]==[(0,45)]


def test_observed_initial_departure_survives_short_unconfirmed_support():
    d=data(40);bearing(d,0,0,3);land(d,0,20)
    d['native_part_position'][3:21,0,0]=torch.linspace(0,.3,18)
    d['native_part_position'][21:,0,0]=.3
    t=segmentation_contact_state(d);cuts,hits=load_transfer_slices(d,t)
    assert not t['support'][:3,0].any() and hits[20,0]
    assert t['transfers'][0]['boundary_eligible']
    assert [(c['start'],c['stop']) for c in cuts]==[(0,40)]
    assert cuts[0]['arrivals']==[(20,0)]


@pytest.mark.parametrize('offset',[0,7])
def test_adjacent_form_and_arrival_cuts_do_not_create_one_frame_fragment(offset):
    d=data(50+offset);bearing(d,4,0,20+offset);bearing(d,5,0,10+offset)
    land(d,1,19+offset);land(d,0,20+offset);land(d,3,18+offset)
    motion=dict(start=0,stop=len(d['active']),posture_phases=[dict(start=0,stop=15+offset,kind='support_posture_motion')])
    cuts,hits=load_transfer_slices(d,source_motion=motion)
    assert [(c['start'],c['stop']) for c in cuts]==[(0,15+offset),(15+offset,20+offset),(20+offset,50+offset)]
    assert cuts[1]['arrivals']==[(18+offset,3),(19+offset,1)]
    t=segmentation_contact_state(d)
    for c in cuts:
        for limb in range(4):assert len({p for p in t['forms'][c['start']:c['stop'],limb].tolist() if p>=0})<=1


@pytest.mark.parametrize('unload_at',[22,26])
def test_adjacent_event_groups_merge_once_without_swallowing_next_same_part_event(unload_at):
    d=data(45)
    for p in (0,1,2):bearing(d,p,0,6)
    land(d,0,14);land(d,1,20);land(d,2,22)
    d['active'][unload_at:32,0]=False;d['contact_load'][unload_at:32,0]=0
    d['native_part_position'][26:33,0,0]=torch.linspace(.1,.3,7)
    d['native_part_position'][33:,0,0]=.3
    cuts,_=load_transfer_slices(d)
    assert [c['stop'] for c in cuts]==[23,45]
    assert cuts[0]['arrivals']==[(14,0),(20,1),(22,2)]
    assert cuts[1]['arrivals']==[(32,0)]


def test_eventless_tail_is_absorbed_without_deleting_source_frames():
    d=data(40);bearing(d,0,0,6);land(d,0,20)
    cuts,_=load_transfer_slices(d)
    assert len(cuts)==1 and (cuts[0]['start'],cuts[0]['stop'])==(0,40)
    assert cuts[0]['arrivals']==[(20,0)]
    assert any(x['reason']=='terminal_settling_without_independent_event' for x in cuts[0]['merged_arrival_boundaries'])


def test_new_unfinished_departure_preserves_preceding_completion():
    d=data(40);bearing(d,0,0,6);land(d,0,20);bearing(d,1,0,25)
    d['native_part_position'][25:,1,0]=torch.arange(15)*.03
    cuts,_=load_transfer_slices(d)
    assert [c['stop'] for c in cuts]==[21,40]
    assert cuts[-1]['phase_kind']=='incomplete_transfer'
    assert roles(d,21,40)['limb_roles']['RL']=='T'
    assert not segmentation_contact_state(d)['transfers'][-1]['completed']


def test_independent_posture_is_retained_but_its_eventless_tail_is_absorbed():
    d=data(50);bearing(d,0,0,50);land(d,1,15)
    motion=dict(start=0,stop=50,posture_phases=[dict(start=20,stop=40,kind='rise')])
    cuts,_=load_transfer_slices(d,source_motion=motion)
    assert [(c['start'],c['stop']) for c in cuts]==[(0,20),(20,50)]
    assert cuts[-1]['phase_kind']=='rise'


@pytest.mark.parametrize('part',[1,2,3])
def test_cross_boundary_disengagement_continuation_is_not_an_independent_tail(part):
    d=data(45);bearing(d,0,0,6);land(d,0,22);bearing(d,part,0,18)
    d['native_part_position'][18:,part,0]=torch.arange(27)*.03
    t=segmentation_contact_state(d);cuts,_=load_transfer_slices(d,t)
    assert [(c['start'],c['stop']) for c in cuts]==[(0,45)]
    event=next(e for e in t['transfers'] if e['part']==part)
    assert event['start']==18 and event['arrival'] is None
    assert event['id'] in cuts[0]['transfer_ids']
    assert any(x['reason']=='participation_without_support_establishment' for x in cuts[0]['merged_arrival_boundaries'])


def test_new_completed_departure_is_not_prior_event_settling():
    d=data(40);bearing(d,0,0,6);land(d,0,20);bearing(d,1,0,21)
    d['native_part_position'][21:31,1,0]=torch.arange(10)*.03
    d['native_part_position'][31:,1,0]=.27
    cuts,_=load_transfer_slices(d)
    assert [c['stop'] for c in cuts]==[21,40]


def test_new_activity_after_idle_is_not_a_continuation_of_old_withdrawal():
    d=data(65);bearing(d,0,0,6);land(d,0,30);bearing(d,2,0,10);land(d,2,50)
    d['native_part_position'][10:20,2,0]=torch.linspace(0,-.2,10)
    d['native_part_position'][20:40,2,0]=-.2
    d['native_part_position'][40:51,2,0]=torch.linspace(-.2,.1,11)
    t=segmentation_contact_state(d)
    # A separate posture boundary closes the interval before the arrival.
    motion=dict(start=0,stop=65,posture_phases=[dict(start=48,stop=65,kind='rise')])
    cuts,_=load_transfer_slices(d,t,motion)
    assert [(c['start'],c['stop']) for c in cuts][:2]==[(0,31),(31,48)]
    assert any(c['start']==48 for c in cuts)  # independent posture remains
    e=next(e for e in t['transfers'] if e['part']==2)
    assert e['start']<31 and any(31<=a<48 for a,b in e['activity_ranges'])


def settling_contact(material):
    d=data(15);bearing(d,1,0,10)
    d['native_part_position'][:6,1,0]=torch.arange(6)*.04
    d['native_part_position'][6:,1,0]=.2
    d['contact_material_speed'][5:10,1]=torch.tensor(material)
    return d


def test_measured_impact_decay_can_establish_support_before_release():
    d=settling_contact([.26,.19,.005,.025,.04])
    t=segmentation_contact_state(d)
    assert t['episodes']==[dict(part=1,start=5,confirmed=9,surface=0,stop=10)]
    assert t['transfers'][0]['arrival']==5


@pytest.mark.parametrize('material',[[.04,.19,.005,.025,.04],[.26,.3,.005,.025,.04],[.26,.05,.19,.025,.04],[.3,.3,.3,.3,.3]])
def test_scrape_or_late_velocity_burst_is_not_impact_settling(material):
    assert not segmentation_contact_state(settling_contact(material))['episodes']


def test_impact_decay_without_stopped_part_pose_is_not_support():
    d=settling_contact([.26,.19,.005,.025,.04])
    d['native_part_position'][5:,1,0]=.2+torch.arange(10)*.02
    assert not segmentation_contact_state(d)['episodes']


def test_missing_initial_motion_cannot_prove_impact_decay():
    d=settling_contact([.26,.19,.005,.025,.04])
    d['active'][4,1]=False;d['contact_load'][4,1]=0
    assert not segmentation_contact_state(d)['episodes']


def test_new_withdrawal_preserves_completion_without_cutting_at_withdrawal():
    d=data(55);bearing(d,0,0,6);land(d,0,20);bearing(d,2,0,28)
    d['native_part_position'][28:39,2,0]=torch.linspace(0,-.2,11)
    d['native_part_position'][39:,2,0]=-.2
    t=segmentation_contact_state(d);cuts,_=load_transfer_slices(d,t)
    event=next(e for e in t['transfers'] if e['part']==2)
    assert event['completed'] and event['arrival'] is None
    assert [c['stop'] for c in cuts]==[21,55]


def with_substep_evidence(d):
    from supermimic.contact_substep_support import SubstepSupport
    monitor=SubstepSupport(.002);T=len(d['active']);samples=[]
    for t in range(T*10):
        frame=t//10
        observation={k:d[k][frame:frame+1] for k in ('active','known','surface_local')}
        features={k:d[k][frame:frame+1] for k in ('contact_load','contact_material_speed')}
        poses=(d['native_part_position'][frame:frame+1],d['native_part_rotation'][frame:frame+1])
        samples.append(monitor.update(observation,features,poses,torch.tensor([t])))
    d.update({'substep_support_'+key:torch.stack([s[key][0] for s in samples]).reshape(T,10,6) for key in samples[0]})
    d.update(ep_step=torch.arange(T),substep_contact_active=d['active'][:,None].expand(-1,10,-1),substep_contact_known=d['known'][:,None].expand(-1,10,-1))
    return d


def test_substep_establishment_does_not_wait_for_five_control_samples():
    d=data(6);bearing(d,0,1,3);with_substep_evidence(d)
    timeline=segmentation_contact_state(d)
    assert timeline['episodes']==[dict(part=0,start=1,stop=3,confirmed=1,surface=0)]
    assert timeline['transfers'][0]['arrival']==1


def test_substep_projection_preserves_stationary_gap_debounce():
    d=data();bearing(d,0,0,24);d['active'][8:11,0]=False;d['contact_load'][8:11,0]=0
    with_substep_evidence(d);timeline=segmentation_contact_state(d)
    assert timeline['support'][:,0].all()
    assert not timeline['transfers']


def test_posture_continuation_does_not_keep_confirmation_latency_fragment():
    d=data(30);bearing(d,4,0,10);bearing(d,0,10,30)
    temporal=segmentation_contact_state(d)
    event=next(e for e in temporal['transfers'] if e['part']==0)
    # A native substep window straddles the control-frame boundary.
    event['confirmed']=11
    phases=dict(start=0,stop=30,posture_phases=[dict(start=11,stop=30,kind='rise')])
    cuts,_=load_transfer_slices(d,temporal,phases)
    assert [(c['start'],c['stop']) for c in cuts]==[(0,10),(10,30)]
    assert cuts[-1]['phase_kind']=='rise'
    assert phases['posture_phases'][0]['start']==11


def test_independent_posture_onset_outside_confirmation_window_is_preserved():
    d=data(35);bearing(d,4,0,10);bearing(d,0,10,35)
    temporal=segmentation_contact_state(d)
    next(e for e in temporal['transfers'] if e['part']==0)['confirmed']=11
    phases=dict(start=0,stop=35,posture_phases=[dict(start=18,stop=35,kind='rise')])
    cuts,_=load_transfer_slices(d,temporal,phases)
    assert any(c['start']==18 for c in cuts)


def test_placement_continuity_does_not_fabricate_support_or_current_form():
    d=data();bearing(d,0,0,24);bearing(d,1,0,24)
    d['active'][8:11,0]=False;d['contact_load'][8:11,0]=0
    with_substep_evidence(d);d['support_placement_resume']=torch.zeros(24,6,dtype=torch.bool);d['support_placement_resume'][11,0]=True
    before={k:v.clone() for k,v in d.items()};timeline=segmentation_contact_state(d)
    assert not timeline['support'][8:11,0].any()
    assert timeline['placement_support'][:,0].all()
    assert (timeline['forms'][8:11,0]==-1).all()
    assert not timeline['transfers']
    assert all(torch.equal(d[k],v) for k,v in before.items())
    assert roles(d,8,11)['limb_roles']['LL']=='S'


def test_first_moving_engagement_is_separate_from_next_step():
    d=data(65);bearing(d,1,0,25);land(d,0,20);land(d,1,45)
    d['native_part_position'][:21,0,0]=torch.linspace(0,.3,21)
    d['native_part_position'][21:,0,0]=.3
    cuts,_=load_transfer_slices(d)
    assert [c['stop'] for c in cuts]==[21,65]
    assert cuts[0]['arrivals']==[(20,0)]
    assert cuts[1]['arrivals']==[(45,1)]


def test_stationary_initial_confirmation_does_not_become_new_engagement():
    d=data(30);bearing(d,0,3,30)
    t=segmentation_contact_state(d)
    assert not t['transfers'][0]['boundary_eligible']
    cuts,_=load_transfer_slices(d,t)
    assert [c['stop'] for c in cuts]==[30]


def test_confirmation_inside_same_posture_does_not_split_its_first_frame():
    d=data(50);bearing(d,4,0,20);land(d,0,20);bearing(d,1,0,35)
    d['native_part_position'][35:,1,0]=torch.arange(15)*.03
    motion=dict(start=0,stop=50,posture_phases=[dict(start=20,stop=50,kind='rise')])
    cuts,_=load_transfer_slices(d,source_motion=motion)
    assert [(c['start'],c['stop']) for c in cuts]==[(0,20),(20,50)]
