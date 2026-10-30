import numpy as np
import pytest
from somaforge_core.newton_support import support_motion


def snapshot(force,velocity):
    n=len(force)
    return dict(count=n,active=np.ones(n,bool),constraint_allocated=np.ones(n,bool),
        frame_w=np.tile(np.array([[0.,0,1],[1.,0,0],[0,1.,0]]),(n,1,1)),
        position_w=np.zeros((n,3)),force_on_body1_w=np.array(force,float),
        torque_on_body1_w=np.zeros((n,3)),point_velocity0_w=np.zeros((n,3)),
        point_velocity1_w=np.array(velocity,float),pre_point_velocity0_w=np.zeros((n,3)),
        pre_point_velocity1_w=np.array(velocity,float),
        application_point0_local=np.zeros((n,3)),application_point1_local=np.zeros((n,3)))


def measure(s):
    n=s['count']
    return support_motion(s,eligible=np.ones(n,bool),robot_side=np.ones(n,int),
                          groups=np.zeros(n,int),group_count=2)


def test_unloaded_rotating_points_do_not_masquerade_as_support_motion():
    s=snapshot([[0,0,100],[0,0,0],[0,0,0]],[[0,0,0],[.2,0,0],[-.2,0,0]])
    r=measure(s)
    assert r['rms_tangent_speed_m_s'][0]==0
    assert r['loaded_contact_rows'].tolist()==[0]
    assert not r['load_bearing'][1] and np.isnan(r['rms_tangent_speed_m_s'][1])


def test_loaded_velocities_do_not_cancel_at_virtual_support_center():
    s=snapshot([[0,0,100],[0,0,100]],[[.1,0,0],[-.1,0,0]])
    np.testing.assert_allclose(measure(s)['rms_tangent_speed_m_s'][0],.1)


def test_stationary_loaded_point_does_not_hide_other_loaded_sliding_points():
    s=snapshot([[0,0,10],[0,0,90]],[[0,0,0],[.1,0,0]])
    r=measure(s)
    assert r['minimum_loaded_speed_m_s'][0]==0
    np.testing.assert_allclose(r['rms_tangent_speed_m_s'][0],np.sqrt(.9)*.1)
    assert r['maximum_loaded_speed_m_s'][0]==.1


def test_zero_force_is_known_unloaded_not_static():
    result=measure(snapshot([[0,0,0]],[[0,0,0]]))
    assert not result['load_bearing'].any()
    assert result['load_known'].all()
    assert result['load_state'].tolist()==['unloaded','no_contact']


def test_motion_is_relative_to_actual_environment_contact_body():
    s=snapshot([[0,0,100]],[[.1,0,0]])
    s['point_velocity0_w'][:]=s['point_velocity1_w']
    assert measure(s)['rms_tangent_speed_m_s'][0]==0


def test_force_never_replaces_actual_activation_or_allocation():
    s=snapshot([[0,0,100]],[[0,0,0]])
    s['constraint_allocated'][0]=False
    with pytest.raises(ValueError,match='unallocated'):measure(s)
    s['constraint_allocated'][0]=True;s.pop('force_on_body1_w')
    with pytest.raises(ValueError,match='unknown'):measure(s)


def test_primary_side_contact_is_excluded_without_promoting_candidate_top():
    from somaforge_core.newton_support import select_support_contacts
    s=snapshot([[0,0,100]],[[0,0,0]])
    s.update(dist=np.array([-.001]),includemargin=np.array([.02]),type=np.ones(1,int),
        efc_address=np.zeros((1,1),int),constraint_rows=np.ones(1,int),
        body0=np.array([-1]),body1=np.array([0]),shape0=np.array([0]),shape1=np.array([1]),worldid=np.array([0]))
    s['position_w'][0]=[0,.5,.5];s['frame_w'][0]=[[1,0,0],[0,1,0],[0,0,1]]
    binding=dict(body_env={0:0},shape_surface={0:[
        dict(surface=0,normal_w=[1,0,0],triangles_w=[[[0,0,0],[0,1,0],[0,1,1]]]),
        dict(surface=1,normal_w=[0,0,1],triangles_w=[[[0,0,1],[1,0,1],[0,1,1]]])]},
        surface_catalog=[dict(surface=0,normal_w=[1,0,0]),dict(surface=1,normal_w=[0,0,1])])
    r=select_support_contacts(s,binding,['left_ankle_roll_link'])
    assert not r['eligible'].any() and r['excluded_rows']==[0]
    s['body0'][0]=0
    assert not select_support_contacts(s,binding,['left_ankle_roll_link'])['eligible'].any()
