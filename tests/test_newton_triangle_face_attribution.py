import numpy as np
import pytest
from somaforge_core.newton_contacts import actual_contact_faces,triangle_distance_squared,reduce_snapshot


def faces():
    return [dict(surface=0,normal_w=[0,0,1],plane_offset=0.,attribution_schema='native_triangle_nearest_v1',
        triangles_w=[[[-2,-2,0],[2,-2,0],[2,2,0]],[[-2,-2,0],[2,2,0],[-2,2,0]]]),
        dict(surface=1,normal_w=[0,0,1],plane_offset=1.,attribution_schema='native_triangle_nearest_v1',
        triangles_w=[[[0,0,1],[1,0,1],[1,1,1]],[[0,0,1],[1,1,1],[0,1,1]]])]


def test_off_plane_witness_is_not_an_activation_threshold():
    point=np.array([.6714248657,.1352715492,.0025482597]);before=point.copy()
    assert actual_contact_faces(point,np.array([0,0,1]),faces())[0]['surface']==0
    np.testing.assert_array_equal(point,before)


def test_real_finite_face_not_infinite_plane_or_xy_box():
    binding=[dict(surface=1,normal_w=[0,0,1],triangles_w=[[[0,0,1],[1,0,1],[0,1,1]]]),
        dict(surface=2,normal_w=[1,0,0],triangles_w=[[[1,0,0],[1,1,0],[1,1,2]],[[1,0,0],[1,1,2],[1,0,2]]])]
    # Inside triangle XY bounding box, but outside the actual triangle.
    assert actual_contact_faces(np.array([.9,.9,1.]),np.array([0,0,1]),binding)[0]['surface']==2


@pytest.mark.parametrize('active',[False,True])
def test_constraint_decision_and_points_preserved(active):
    a=np.array([active]);point=np.array([[.67,.13,.00254826]])
    raw=dict(count=1,dist=np.array([-.005 if active else .03]),includemargin=np.array([.02]),type=np.array([1]),
        efc_address=np.array([[0 if active else -1]]),active=a,constraint_allocated=a.copy(),worldid=np.array([0]),
        body0=np.array([-1]),body1=np.array([0]),shape0=np.array([0]),shape1=np.array([1]),
        frame_w=np.array([[[0,0,1],[0,1,0],[-1,0,0]]]),position_w=point.copy(),
        geometry_point0_w=point.copy(),geometry_point1_w=point-np.array([0,0,.005]))
    result=reduce_snapshot(raw,body_labels=['left_knee_link'],body_env={0:0},shape_surface={0:faces()},include_candidates=True)
    assert bool(result.active[4])==active and not result.unallocated.any()
    assert bool(result.candidate_pairs[0]['constraint_active'])==active
    if active:np.testing.assert_array_equal(result.position_w[4],raw['geometry_point1_w'][0].astype(np.float32))


def test_missing_geometry_is_not_silently_guessed():
    with pytest.raises(ValueError,match='face absent'):
        actual_contact_faces(np.array([.1,.1,.0025]),np.array([0,0,1]),[dict(surface=0,normal_w=[0,0,1],plane_offset=0.)])
