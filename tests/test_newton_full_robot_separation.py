import numpy as np
import pytest
from somaforge_core.newton_contacts import full_robot_separation


def snapshot():
    return dict(count=3,worldid=np.array([0,0,0]),type=np.ones(3,int),
        body0=np.array([-1,3,3]),body1=np.array([3,4,8]),shape0=np.array([0,7,7]),
        shape1=np.array([7,8,9]),dist=np.array([-.03,-.01,-.5]),
        active=np.ones(3,bool),constraint_allocated=np.ones(3,bool))


def test_full_body_includes_noncontact_body_and_excludes_other_robot():
    r=full_robot_separation(snapshot(),body_env={3:0,4:0,8:1},shape_surface={0:0})
    assert r['terrain_penetration_m']==.03 and r['self_penetration_m']==.01
    assert r['terrain_rows']==1 and r['self_rows']==1


def test_unallocated_full_body_contact_is_error():
    s=snapshot();s['constraint_allocated'][0]=False
    with pytest.raises(ValueError,match='Unallocated'):
        full_robot_separation(s,body_env={3:0,4:0,8:1},shape_surface={0:0})


def test_invalid_shallower_witness_is_reported_even_when_deepest_normal_is_valid():
    from generator.rollout_termination import geometry_failure,TerminationLimits
    s=snapshot();s['dist'][:2]=[-.0008,-.0004]
    s['geometry_point0_w']=np.zeros((3,3));s['geometry_point1_w']=np.zeros((3,3))
    s['frame_w']=np.tile(np.eye(3),(3,1,1));s['frame_w'][1,0]=np.nan
    r=full_robot_separation(s,body_env={3:0,4:0,8:1},shape_surface={0:0})
    assert len(r['invalid_penetrating_witnesses'])==1
    assert geometry_failure(r,TerminationLimits(penetration_m=.001))=='invalid_geometry'
