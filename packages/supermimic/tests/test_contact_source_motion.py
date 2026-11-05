import torch
from supermimic.contact_temporal import segmentation_contact_state
from supermimic.contact_source_motion import source_motion_analysis
from supermimic.stable_contact_slices import load_transfer_slices


def trace(T=60):
    s=torch.zeros(T,71);s[:,2]=.8;s[:,3]=1
    a=torch.zeros(T,6,dtype=torch.bool);a[:,:2]=True
    return dict(s1=s,active=a,known=torch.ones_like(a),contact_load=a.float()*150,
        contact_material_speed=torch.zeros(T,6),surface_local=torch.zeros(T,6,dtype=torch.long),
        native_part_position=torch.zeros(T,6,3),native_part_rotation=torch.eye(3).expand(T,6,3,3).clone())


def analyze(d):
    t=segmentation_contact_state(d)
    return t,source_motion_analysis(d,t)


def test_static_edge_waits_are_excluded_but_source_is_unchanged():
    d=trace();d['s1'][20:40,0]=torch.linspace(0,.2,20);d['s1'][40:,0]=.2
    before={k:v.clone() for k,v in d.items()};t,a=analyze(d)
    assert {x['reason'] for x in a['exclusions']}=={'static_prefix','static_suffix'}
    assert a['start']>=15 and a['stop']<=45
    cuts,_=load_transfer_slices(d,t,a);coverage=torch.zeros(60,dtype=torch.int)
    for c in cuts:coverage[c['start']:c['stop']]+=1
    for x in a['exclusions']:coverage[x['start']:x['stop']]+=1
    assert (coverage==1).all() and all(torch.equal(d[k],v) for k,v in before.items())


def test_rise_is_a_posture_phase_not_static_or_contact_transfer():
    d=trace(35);d['s1'][:,2]=torch.linspace(.8,1.05,35);d['s1'][:,16]=torch.linspace(1.5,.3,35)
    t,a=analyze(d)
    assert not a['exclusions'] and not t['transfers']
    assert [x['kind'] for x in a['posture_phases']]==['rise']


def test_mid_trace_standing_is_not_deleted_or_joined_away():
    d=trace();d['s1'][:15,0]=torch.linspace(-.15,0,15);d['s1'][45:,0]=torch.linspace(0,.15,15)
    _,a=analyze(d)
    assert (a['start'],a['stop'])==(0,60) and not a['exclusions']


def test_normal_partial_initial_and_terminal_steps_are_not_anomalies():
    d=trace(30);d['active'][:5,0]=False;d['contact_load'][:5,0]=0
    d['active'][22:,1]=False;d['contact_load'][22:,1]=0
    d['native_part_position'][:6,0,0]=torch.linspace(-.1,0,6)
    d['native_part_position'][22:,1,0]=torch.linspace(0,.15,8)
    d['s1'][:,0]=torch.linspace(0,.3,30)
    _,a=analyze(d)
    assert (a['start'],a['stop'])==(0,30) and not a['exclusions']


def test_vertical_initialization_drop_requires_settling_evidence():
    d=trace(40);d['active'][:8]=False;d['contact_load'][:8]=0
    d['s1'][:9,2]=torch.linspace(1.0,.8,9);d['s1'][15:,0]=torch.linspace(0,.2,25)
    _,a=analyze(d)
    assert any(x['reason']=='initialization_drop' for x in a['exclusions'])
    assert a['start']>=9


def test_upright_descent_is_not_terminal_fall():
    d=trace(30);d['s1'][:,2]=torch.linspace(1.2,.8,30);d['active'][20:]=False;d['contact_load'][20:]=0
    _,a=analyze(d)
    assert not any(x['reason']=='terminal_fall' for x in a['exclusions'])


def test_terminal_fall_needs_both_inversion_trend_and_downward_motion():
    d=trace(40);d['s1'][20:,2]=torch.linspace(.8,.2,20)
    angle=torch.linspace(0,1.8,20);d['s1'][20:,3]=torch.cos(angle/2);d['s1'][20:,4]=torch.sin(angle/2)
    _,a=analyze(d)
    tail=next(x for x in a['exclusions'] if x['reason']=='terminal_fall')
    assert tail['stop']==40 and 20<=tail['start']<35


def test_all_static_source_has_no_fabricated_training_segment():
    d=trace();t,a=analyze(d);cuts,_=load_transfer_slices(d,t,a)
    assert not cuts and a['start']==a['stop']


def test_partial_edges_are_removed_as_whole_segments_not_retained_as_tasks():
    d=trace(60);d['s1'][:,0]=torch.linspace(0,.6,60)
    d['active'][:6,0]=False;d['contact_load'][:6,0]=0
    d['native_part_position'][:7,0,0]=torch.linspace(-.2,0,7)
    d['active'][12:25,0]=False;d['contact_load'][12:25,0]=0
    d['native_part_position'][12:26,0,0]=torch.linspace(0,.2,14);d['native_part_position'][26:,0,0]=.2
    d['active'][42:,1]=False;d['contact_load'][42:,1]=0
    d['native_part_position'][42:,1,0]=torch.linspace(0,.3,18)
    raw={k:v.clone() for k,v in d.items()};t,a=analyze(d);cuts,_=load_transfer_slices(d,t,a)
    assert {e['reason'] for e in a['exclusions']}=={'incomplete_initial_action','incomplete_terminal_action'}
    assert cuts and all(not c['incomplete_edge_activity'] for c in cuts)
    assert any((25,0) in c['arrivals'] for c in cuts)
    coverage=torch.zeros(60,dtype=torch.long)
    for c in cuts:coverage[c['start']:c['stop']]+=1
    for e in a['exclusions']:coverage[e['start']:e['stop']]+=1
    assert (coverage==1).all()
    assert all(torch.equal(d[k],v) for k,v in raw.items())


def test_initial_edge_does_not_follow_a_long_concurrent_hand_reach():
    d=trace(80);d['s1'][:,0]=torch.linspace(0,.8,80)
    d['active'][:6,0]=False;d['contact_load'][:6,0]=0
    d['native_part_position'][:7,0,0]=torch.linspace(-.2,0,7)
    d['active'][60:,2]=True;d['contact_load'][60:,2]=100
    d['native_part_position'][:61,2,0]=torch.linspace(0,1,61);d['native_part_position'][61:,2,0]=1
    t,a=analyze(d);cuts,_=load_transfer_slices(d,t,a)
    assert a['start']==7
    assert any((60,2) in c['arrivals'] for c in cuts)


def test_initial_support_confirmation_latency_is_not_a_truncated_action():
    d=trace(50);d['s1'][:,0]=torch.linspace(0,.5,50)
    d['contact_load'][:3,0]=0
    d['native_part_position'][:4,0,0]=torch.linspace(0,.009,4);d['native_part_position'][4:,0,0]=.009
    d['active'][12:25,0]=False;d['contact_load'][12:25,0]=0
    d['native_part_position'][12:26,0,0]=torch.linspace(.009,.2,14);d['native_part_position'][26:,0,0]=.2
    t,a=analyze(d);cuts,_=load_transfer_slices(d,t,a)
    assert a['start']==0
    assert any((25,0) in c['arrivals'] for c in cuts)
    assert not any(e['reason']=='incomplete_initial_action' for e in a['exclusions'])
