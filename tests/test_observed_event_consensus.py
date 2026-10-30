import numpy as np
from somaforge_core.observed_event_consensus import build_consensus, audit_consensus
from somaforge_core.consensus_event_alignment import align_observed_events


def evidence(start, extra=False):
    m = np.zeros((80, 6), bool); s = np.full((80, 6), -1)
    m[:40, 0] = True; s[:40, 0] = 0
    m[start:65, 2] = True; s[start:65, 2] = 1
    if extra: m[70:74, 5] = True; s[70:74, 5] = 2
    return m, s


def test_population_clock_not_representative_and_minority_not_anchor():
    data = [evidence(t, extra=t==20) for t in (20, 25, 30)]
    c = build_consensus(*zip(*data))
    assert len(c['events']) == 2 and len(c['minority_episodes']) == 1
    assert c['events'][1]['interval'][0] == 25
    reports = [audit_consensus(c, m, s) for m,s in data]
    assert all(r['passed'] for r in reports)
    maps, audit = align_observed_events(reports, 80, reference_index=0)
    assert audit['timeline'] == 'population event consensus'
    assert 25 in audit['output_anchors']
    assert abs(maps[0,25]-20) < c['evidence_policy']['stable_frames']
    assert not c['original_reference_used'] and not c['task_success_verified']


def test_overlap_and_order_come_from_new_executions():
    data = [evidence(t) for t in (20,25,30)]
    c = build_consensus(*zip(*data))
    m,s = evidence(45)  # Loses the observed contact overlap.
    report = audit_consensus(c,m,s)
    assert not report['passed']
    assert all(r['passed'] for r in report['checks'])
    assert any(not r['passed'] for r in report['relation_checks'])


def test_population_is_permutation_invariant():
    data = [evidence(t) for t in (20,25,30)]
    a = build_consensus(*zip(*data)); b=build_consensus(*zip(*data[::-1]))
    assert [(e['part'],e['surface'],e['interval']) for e in a['events']] == [(e['part'],e['surface'],e['interval']) for e in b['events']]
    assert a['relations'] == b['relations']


def test_no_surface_switch_bridging_and_no_geometry_contact_fallback():
    m,s=evidence(20); m[25:28,2]=True;s[25:28,2]=2
    c=build_consensus([m]*3,[s]*3)
    assert [e['surface'] for e in c['events'] if e['part']==2] == [1,2,1]
    bad=s.copy();bad[m]=-1
    import pytest
    with pytest.raises(ValueError, match='primary surface'): build_consensus([m]*3,[bad]*3)


def test_distinct_surface_itineraries_are_not_fused():
    from somaforge_core.observed_event_consensus import strategy_groups
    data = [evidence(t) for t in (20,25,30)]
    m,s=evidence(20);s[m & (s==1)]=2
    groups=strategy_groups([d[0] for d in data]+[m]*3, [d[1] for d in data]+[s]*3)
    assert groups == [[0,1,2],[3,4,5]]


def test_majority_contact_guide_does_not_copy_representative():
    from somaforge_core.native_rollout_consensus import ContactGuard
    import torch
    from types import SimpleNamespace
    masks=np.zeros((3,8,6),bool); faces=np.full((3,8,6),-1)
    masks[0,:,0]=True;faces[0,:,0]=0
    masks[1:,:,2]=True;faces[1:,:,2]=1
    class FK:
        def link_poses(self,q,names):
            return torch.zeros(8,len(names),3),torch.eye(3).expand(8,len(names),3,3)
    guard=ContactGuard(SimpleNamespace(),{},[],masks,faces)
    guard.prepare(0,torch.zeros(8,36),FK())
    assert not guard.wanted[:,0].any() and guard.wanted[:,2].all()
    assert (guard.faces[:,2]==1).all()


def test_missing_event_does_not_discard_others_observed_anchor():
    data=[evidence(t) for t in (20,25,30)]
    data[2][0][:,2]=False;data[2][1][:,2]=-1
    c=build_consensus(*zip(*data))
    reports=[audit_consensus(c,m,s) for m,s in data]
    maps,audit=align_observed_events(reports,80,reference_index=0)
    assert 22.5 in audit['output_anchors']  # Two observed votes, not intersection of all three.
    assert audit['clock_diagnostics'][2]['missing_event_keys']
    assert not audit['clock_diagnostics'][0]['missing_event_keys']
    assert np.all(np.diff(maps,axis=1)>0)


def test_transport_does_not_promote_minority_incidental_contacts():
    from somaforge_core.observed_event_consensus import observed_consensus_subset
    data=[evidence(25,extra=i==0) for i in range(3)]
    c=build_consensus(*zip(*data));r=audit_consensus(c,*data[0])
    own=observed_consensus_subset(c,r)
    assert audit_consensus(own,*evidence(25))['passed']
    assert not audit_consensus(own,*evidence(45))['passed']


def test_exclusive_endpoint_is_valid_for_continuity_bounds():
    from somaforge_core.event_continuity import EventContinuityGuard
    from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
    m=np.zeros((20,6),bool);s=np.full((20,6),-1);m[:,0]=True;s[:,0]=0
    c=build_consensus([m]*3,[s]*3);r=audit_consensus(c,m,s)
    q=np.zeros((3,20,36),np.float32);q[:,:,2]=.8;q[:,:,3]=1
    guard=EventContinuityGuard(q,[r]*3,50,CanonicalG1ForwardKinematics())
    # Output releases before the last frame; source episode ends are exclusive n.
    out_m=m.copy();out_m[-3:,0]=False;out_s=s.copy();out_s[-3:,0]=-1
    out=audit_consensus(c,out_m,out_s)
    assert guard.check(q[0],CanonicalG1ForwardKinematics(),out)['passed']


def test_earlier_contact_is_variation_not_error_when_overlap_is_retained():
    data=[evidence(25) for _ in range(3)]
    # Establish the foot later than the output hand, while maintaining overlap.
    for m,s in data: m[:10,0]=False;s[:10,0]=-1
    c=build_consensus(*zip(*data))
    m,s=evidence(5);m[:10,0]=False;s[:10,0]=-1
    r=audit_consensus(c,m,s)
    assert r['passed'] and r['relation_variation_count']
    assert r['missing_event_count']==0


def test_one_missing_event_is_not_multiplied_by_incident_relations():
    data=[evidence(25) for _ in range(3)];c=build_consensus(*zip(*data))
    m,s=evidence(25);m[:,2]=False;s[:,2]=-1
    r=audit_consensus(c,m,s)
    assert r['failed_checks']==r['missing_event_count']==1
    assert any(row['passed'] is None for row in r['relation_checks'])


def test_correspondence_does_not_pair_climb_and_descent_just_to_match_count():
    from somaforge_core.observed_event_consensus import match
    first=[dict(surface=1,interval=[210,270]),dict(surface=1,interval=[280,320]),dict(surface=1,interval=[855,910])]
    second=[dict(surface=1,interval=[215,319]),dict(surface=1,interval=[854,909])]
    pairs=match(first,second)
    assert (2,1) in pairs and (1,1) not in pairs


def test_continuous_contact_can_preserve_two_confirmations_without_shared_release():
    c=dict(schema='test',evidence_policy=dict(stable_frames=3,max_dropout_frames=3),relations=[],
        events=[dict(id=0,part=0,surface=0,interval=[0,40]),
                dict(id=1,part=0,surface=0,interval=[50,80],distinct_from_previous=False)])
    m=np.zeros((80,6),bool);s=np.full((80,6),-1);m[:,0]=True;s[:,0]=0
    assert audit_consensus(c,m,s)['passed']
    c['events'][1]['distinct_from_previous']=True
    assert not audit_consensus(c,m,s)['passed']


def test_clock_rate_smoothing_reduces_bends_preserves_bounds_and_endpoints():
    from somaforge_core.population_event_clock import smooth_clock_rates
    rates=np.r_[np.ones(20),np.full(2,3.),np.full(2,.25),np.ones(20)]
    rates*=len(rates)/rates.sum()
    mapping=np.r_[0,np.cumsum(rates)]
    out=smooth_clock_rates(mapping,3,(.25,4.))
    np.testing.assert_allclose(out[[0,-1]],mapping[[0,-1]])
    assert .25 <= np.diff(out).min() and np.diff(out).max() <= 4.
    assert np.abs(np.diff(out,n=2)).max() < np.abs(np.diff(mapping,n=2)).max()/3
    identity=np.arange(50.)
    np.testing.assert_allclose(smooth_clock_rates(identity,3,(.25,4.)),identity)
