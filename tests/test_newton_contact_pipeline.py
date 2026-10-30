import numpy as np
import pytest
from somaforge_core.newton_contacts import constraint_decision, reduce_snapshot, PARTS, SCHEMA
from somaforge_core.newton_contact_data import touchdown_events
from somaforge_core.newton_contact_query import query_contacts, install_contact_query


def test_strict_kernel_boundary_and_nonfinite():
    a,b=constraint_decision(np.array([.019,.02,.01]),np.full(3,.02),np.array([1,1,2]),np.zeros((3,1),int),3)
    assert a.tolist()==[True,False,False]
    assert np.array_equal(a,b)
    with pytest.raises(ValueError,match='Nonfinite'):
        constraint_decision(np.array([np.nan]),np.array([.02]),np.array([1]),np.array([[0]]),1)


def test_events_do_not_fabricate_boundary_contact_or_modify_labels():
    mask=np.zeros((14,6),bool);mask[2:7,0]=True;mask[8:,1]=True
    before=mask.copy()
    events=touchdown_events(mask,merge_gap=10)
    assert len(events)==2
    assert all(mask[f,parts].all() for f,parts in events)
    np.testing.assert_array_equal(mask,before)


def test_missing_worker_does_not_fall_back(monkeypatch):
    install_contact_query(None)
    monkeypatch.delenv('SOMAFORGE_NEWTON_CONTACT_URL',raising=False)
    with pytest.raises(RuntimeError,match='no geometric fallback'):
        query_contacts(np.zeros((1,36)),[np.zeros((1,3)),np.eye(3)[None],np.ones((1,3)),np.zeros(1)])


def test_observation_is_not_filtered_by_intention():
    import importlib.util
    import torch
    from types import SimpleNamespace
    from contact_solver import contact_validity as mod
    def provider(q,scene):
        return dict(schema=SCHEMA,parts=PARTS,active=[[0,0,1,0,0,0]],unallocated=[[0]*6],
                    position_w=np.zeros((1,6,3)),surface=[[-1,-1,1,-1,-1,-1]],
                    surface_catalog=[dict(surface=1,normal_w=[0,0,1])],
                    pairs=[[dict(part=2,surface=1,allocated=True,dist=0.,
                                 includemargin=.02,position_w=[0,0,0])]])
    install_contact_query(provider)
    try:
        p=SimpleNamespace(qpos=torch.zeros((1,36)),active_contact=torch.zeros((1,6),dtype=torch.bool))
        observed,_,surface=mod.observed_contacts(p,torch.zeros((1,6)),torch.zeros((1,6,3)),
            torch.zeros((1,3)),torch.eye(3)[None],torch.ones((1,3)),torch.zeros(1))
        assert observed[0,2] and surface[0,2]==1
    finally:
        install_contact_query(None)


def test_corrupt_saved_activation_rejected():
    raw=dict(count=1,dist=np.array([.025]),includemargin=np.array([.02]),type=np.array([1]),
        efc_address=np.array([[0]]),active=np.array([True]),constraint_allocated=np.array([True]))
    with pytest.raises(ValueError,match='Inconsistent saved'):
        reduce_snapshot(raw,body_labels=[],body_env={},shape_surface={})


def test_partial_friction_rows_are_not_fully_allocated():
    active,allocated=constraint_decision(np.array([.019]),np.array([.02]),np.array([1]),
        np.array([[0,1,-1,-1]]),1,np.array([4]))
    assert active[0] and not allocated[0]


def test_edge_normal_does_not_erase_actual_surface_contact():
    raw=dict(count=1,dist=np.array([.019]),includemargin=np.array([.02]),type=np.array([1]),
        efc_address=np.array([[0]]),active=np.array([True]),constraint_allocated=np.array([True]),
        worldid=np.array([0]),body0=np.array([-1]),body1=np.array([0]),shape0=np.array([0]),shape1=np.array([1]),
        frame_w=np.array([[[-.707,.707,-.03],[0,0,0],[0,0,0]]]),position_w=np.array([[0,0,.1095]]),
        geometry_point0_w=np.array([[0,0,.1]]),geometry_point1_w=np.array([[0,0,.119]]))
    reduced=reduce_snapshot(raw,body_labels=['left_knee_link'],body_env={0:0},
        shape_surface={0:[dict(surface=1,normal_w=[0,0,1],plane_offset=.1)]})
    assert reduced.active[4] and reduced.surface[4]==1
    np.testing.assert_allclose(reduced.position_w[4],[0,0,.119])


def test_surface_change_can_be_a_touchdown_without_lost_contact():
    mask=np.zeros((12,6),bool);mask[:,0]=True
    surface=np.zeros((12,6),int);surface[6:,0]=1
    events=touchdown_events(mask,surfaces=surface)
    assert len(events)==1 and events[0][0]==6 and events[0][1][0]


def test_inactive_candidates_remain_separate_from_contact_truth():
    raw=dict(count=1,dist=np.array([.021]),includemargin=np.array([.02]),type=np.array([1]),
        efc_address=np.array([[-1]]),active=np.array([False]),constraint_allocated=np.array([False]),
        worldid=np.array([0]),body0=np.array([-1]),body1=np.array([0]),shape0=np.array([0]),shape1=np.array([1]),
        frame_w=np.array([[[0,0,1],[1,0,0],[0,1,0]]]),position_w=np.array([[0,0,.1105]]),
        geometry_point0_w=np.array([[0,0,.1]]),geometry_point1_w=np.array([[0,0,.121]]))
    reduced=reduce_snapshot(raw,body_labels=['left_knee_link'],body_env={0:0},
        shape_surface={0:[dict(surface=1,normal_w=[0,0,1],plane_offset=.1)]},include_candidates=True)
    assert not reduced.active.any() and not reduced.unallocated.any() and not reduced.pairs
    assert len(reduced.candidate_pairs)==1
    assert not reduced.candidate_pairs[0]['constraint_active']
