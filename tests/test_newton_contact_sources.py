import numpy as np
import pytest
from somaforge_core.newton_contact_sources import decode_mesh_triangle_key, source_face_metadata, source_normal_fan


def key(a,b,triangle,point=0):return (a<<43)|(b<<23)|(((triangle<<1)|1)<<3)|point


def test_source_key_and_wrong_shape_mapping():
    k=key(0,73,12,2)
    assert decode_mesh_triangle_key(k,0,73)['triangle_index']==12
    assert decode_mesh_triangle_key(k,0,73)['manifold_point']==2
    with pytest.raises(ValueError,match='mapping'):decode_mesh_triangle_key(k,0,74)
    with pytest.raises(ValueError,match='did not write'):decode_mesh_triangle_key(-1,0,73)


def test_native_dynamic_source_layout_is_lossless():
    from somaforge_core.newton_contact_sources import canonical_source_keys
    for bits, sub_bits in ((7,23),(7,3),(20,23)):
        sub = 3 if sub_bits == 3 else ((7 << 1 | 1) << 3)
        native = (42 << (sub_bits+bits)) | (73 << sub_bits) | sub
        expected = (42 << 43) | (73 << 23) | sub
        assert canonical_source_keys([native],bits,sub_bits).tolist() == [expected]
    with pytest.raises(ValueError,match='layout'):
        canonical_source_keys([1 << 38],7,23)


def test_native_analytic_sphere_source_has_no_manifold_bits():
    from somaforge_core.newton_contact_sources import expand_native_sphere_triangle_keys
    raw = (1 << 23) | ((18 << 1) | 1)
    convex = key(0,2,18,2)
    out = expand_native_sphere_triangle_keys([raw,convex],[8,3,10])
    assert out.tolist() == [key(0,1,18),convex]
    assert decode_mesh_triangle_key(out[0],0,1)['triangle_index'] == 18


def test_source_triangle_and_adjacent_faces_are_distinct():
    binding=[dict(surface=1,triangle_indices=[7],triangles_w=[[[0,0,1],[1,0,1],[0,1,1]]]),
             dict(surface=9,triangle_indices=[8],triangles_w=[[[0,0,1],[1,0,1],[0,0,0]]])]
    p=source_face_metadata(key(0,73,7),0,0,73,binding)
    assert p['source_surface']==1 and p['edge_adjacent_surfaces']==[9]
    # Being adjacent does not make the side the source of this contact.
    assert p['status']=='mesh_triangle'


def test_missing_source_triangle_is_not_guessed_from_distance():
    with pytest.raises(ValueError,match='unique actual face'):
        source_face_metadata(key(0,73,7),0,0,73,[dict(surface=1,triangle_indices=[8],triangles_w=[])])


def test_edge_from_side_triangle_can_be_top_owned_but_true_side_stays_side():
    side=dict(surface=9,normal_w=[0,-1,0],triangles_w=[[[0,0,1],[1,0,1],[0,0,0]]])
    top=dict(surface=1,normal_w=[0,0,1],triangles_w=[[[0,0,1],[1,0,1],[0,1,1]]])
    floor=dict(surface=0,normal_w=[0,0,1],triangles_w=[[[0,0,0],[1,0,0],[0,1,0]]])
    faces=[side,top,floor]
    r=source_normal_fan(side['triangles_w'][0],side,faces,[0,-.1,.99],[.5,0,1])
    assert r['primary_surface']==1 and 0 not in r['incident_surface_candidates']
    assert source_normal_fan(side['triangles_w'][0],side,faces,[0,-1,0],[.3,0,.4])['primary_surface']==9


def test_normal_fan_cannot_select_unrelated_parallel_plane():
    side=dict(surface=9,normal_w=[0,-1,0],triangles_w=[[[0,0,1],[1,0,1],[0,0,0]]])
    distant=dict(surface=1,normal_w=[0,0,1],triangles_w=[[[0,0,5],[1,0,5],[0,1,5]]])
    assert source_normal_fan(side['triangles_w'][0],side,[side,distant],[0,-.1,.99],[.5,0,1])['primary_surface']==9


def test_inward_side_witness_cannot_acquire_remote_top():
    side=dict(surface=9,normal_w=[0,-1,0],triangle_indices=[7],
              triangles_w=[[[0,0,1],[1,0,1],[0,0,0]]])
    top=dict(surface=1,normal_w=[0,0,1],triangle_indices=[4],
             triangles_w=[[[0,0,1],[1,0,1],[0,1,1]]])
    witness=np.array([.2,0,.3])
    r=source_face_metadata(key(0,73,7),0,0,73,[side,top],[0,1,1e-7],witness)
    assert r['primary_surface']==9
    assert r['incident_surface_candidates']==[9]
    np.testing.assert_array_equal(witness,[.2,0,.3])
    with pytest.raises(ValueError,match='Missing actual'):
        source_face_metadata(key(0,73,7),0,0,73,[side,top],[0,1,1e-7])


def test_reducer_uses_source_and_normal_without_changing_activation_or_witness():
    from somaforge_core.newton_contacts import reduce_snapshot
    n=np.array([0,-.1,.99]);n/=np.linalg.norm(n)
    point=np.array([[.5,0,.999999]])
    raw=dict(count=1,dist=np.array([.017]),includemargin=np.array([.02]),type=np.array([1]),
        efc_address=np.array([[0]]),active=np.array([True]),constraint_allocated=np.array([True]),
        worldid=np.array([0]),body0=np.array([-1]),body1=np.array([0]),shape0=np.array([0]),shape1=np.array([73]),
        frame_w=np.array([[n,[1,0,0],[0,0,1]]]),position_w=point.copy(),
        geometry_point0_w=point.copy(),geometry_point1_w=point+n*.017,
        source_key=np.array([key(0,73,7)]),same_pass_source_mapping_verified=True)
    faces=[dict(surface=9,normal_w=[0,-1,0],triangle_indices=[7],triangles_w=[[[0,0,1],[1,0,1],[0,0,0]]]),
           dict(surface=1,normal_w=[0,0,1],triangle_indices=[4],triangles_w=[[[0,0,1],[1,0,1],[0,1,1]]])]
    r=reduce_snapshot(raw,body_labels=['left_knee_link'],body_env={0:0},shape_surface={0:faces})
    assert r.active[4] and r.surface[4]==1 and not r.unallocated.any()
    assert r.pairs[0]['contact_source']['source_surface']==9
    assert r.pairs[0]['surface_attribution']=='newton_source_triangle_witness_normal_fan_v2'
    assert r.pairs[0]['dist']==.017
    np.testing.assert_array_equal(raw['geometry_point0_w'],point)


def test_single_collision_pass_carries_its_own_source_and_checks_mapping():
    from types import SimpleNamespace
    from somaforge_core.newton_contact_sources import capture_constraint_snapshot
    array=lambda v:SimpleNamespace(numpy=lambda:np.asarray(v))
    contacts=SimpleNamespace(rigid_contact_shape0=array([0]),rigid_contact_shape1=array([73]))
    solver=SimpleNamespace(_contact_tid_to_cid=array([0]))
    calls=[]
    class Capture:
        def collide(self,state,contacts):calls.append('collide');return np.array([key(0,73,7)])
    def snapshot():
        calls.append('snapshot')
        r={k:np.array([0]) for k in ('worldid','body0','body1','type','dist','includemargin',
            'active','constraint_allocated','constraint_rows')}
        r.update(count=1,shape0=np.array([0]),shape1=np.array([73]))
        for k in ('position_w','geometry_point0_w','geometry_point1_w'):r[k]=np.zeros((1,3))
        r['frame_w']=np.eye(3)[None]
        return r
    raw=capture_constraint_snapshot(None,None,contacts,solver,snapshot,Capture())
    assert calls==['collide','snapshot'] and raw['same_pass_source_mapping_verified']
    solver._contact_tid_to_cid=array([-1])
    with pytest.raises(ValueError,match='Missing solver'):
        capture_constraint_snapshot(None,None,contacts,solver,snapshot,Capture())
