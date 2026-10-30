"""Geometry/gradient contracts, independently of Newton contact activation."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from contact_solver.native_contact_position import (
    NativeContactPositionRouter, NativeContactRegionGeometry,
    clip_triangles, segment_triangles_squared, sphere_patch_squared, sphere_patch_batch_squared,
    native_position_statistics,
)
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _rotation_vector_matrix
from somaforge_core.motion_contracts import BODY_NAMES
from somaforge_core.robot_assets import canonical_g1_asset_metadata
from somaforge_core.solid_distance import SOLID_GEOMETRY_SCHEMA
from somaforge_core.contact_source_geometry import SOURCE_NORMAL_FAN_SCHEMA


def sphere_cost(value, *, clipped=False):
    center=value.new_zeros(3)
    position=value[:3]
    rotation=_rotation_vector_matrix(value[3:])
    normals=value.new_tensor([[0.,1.,0.]]) if clipped else value.new_empty((0,3))
    offsets=value.new_tensor([.001]) if clipped else value.new_empty(0)
    a=value.new_tensor([.08,.03,0.]);b=a+value.new_tensor([0.,0.,.02])
    return sphere_patch_squared(center,.035,normals,offsets,a,b,position,rotation)


def test_sphere_rotation_without_geometry_motion_has_zero_gradient():
    pose=torch.tensor([.01,.0,.025,.2,.3,-.4],dtype=torch.float64,requires_grad=True)
    loss=sphere_cost(pose)
    gradient=torch.autograd.grad(loss,pose)[0]
    torch.testing.assert_close(gradient[3:],torch.zeros(3,dtype=pose.dtype),atol=1e-14,rtol=0)
    assert torch.autograd.gradcheck(sphere_cost,(pose,),eps=1e-6,atol=1e-8,rtol=1e-5)


def test_clipped_sphere_moving_regional_boundary_has_true_scalar_derivative():
    pose=torch.tensor([.01,-.01,.037,.2,.3,-.4],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda q:sphere_cost(q,clipped=True),(pose,),eps=1e-6,atol=1e-8,rtol=1e-5)


@pytest.mark.parametrize('cuts',[0,1,2])
def test_batched_sphere_matches_scalar_values_and_pose_derivatives(cuts):
    rng=torch.Generator().manual_seed(712+cuts)
    pose=torch.randn(24,6,generator=rng,dtype=torch.float64)
    pose[:,:3]*=.04;pose.requires_grad_()
    center=pose.new_tensor([.002,-.003,.001]);radius=.035
    normals=pose.new_tensor([[0.,1.,0.],[1.,0.,0.]])[:cuts]
    offsets=pose.new_tensor([.001,.006])[:cuts]
    a=torch.randn(24,3,generator=rng,dtype=pose.dtype)*.07;b=a+pose.new_tensor([0.,0.,.02])
    rotation=_rotation_vector_matrix(pose[:,3:])
    batched=sphere_patch_batch_squared(center,radius,normals,offsets,a,b,pose[:,:3],rotation)
    scalar=torch.stack([sphere_patch_squared(center,radius,normals,offsets,a[i],b[i],pose[i,:3],rotation[i]) for i in range(len(pose))])
    torch.testing.assert_close(batched,scalar,atol=1e-12,rtol=1e-10)
    gb=torch.autograd.grad(batched.sum(),pose,retain_graph=True)[0]
    gs=torch.autograd.grad(scalar.sum(),pose)[0]
    torch.testing.assert_close(gb,gs,atol=1e-11,rtol=1e-9)


def test_batched_sphere_inside_intersection_and_tangent_have_finite_gradients():
    position=torch.tensor([[0.,0.,0.],[.035,0.,0.],[0.,0.,.01]],dtype=torch.float64,requires_grad=True)
    a=torch.zeros_like(position);b=a+position.new_tensor([0.,0.,.02])
    value=sphere_patch_batch_squared(position.new_zeros(3),.035,position.new_empty(0,3),position.new_empty(0),
        a,b,position,torch.eye(3,dtype=position.dtype).expand(3,-1,-1))
    torch.testing.assert_close(value,position.new_tensor([.015**2,0.,.025**2]))
    assert bool(torch.autograd.grad(value.sum(),position)[0].isfinite().all())


def test_sphere_normal_interval_has_no_attraction_inside_and_retains_far_signal():
    def value(z):
        position=torch.stack((z*0,z*0,z))
        return sphere_patch_squared(z.new_zeros(3),.005,z.new_empty((0,3)),z.new_empty(0),
            z.new_tensor([0.,0.,0.]),z.new_tensor([0.,0.,.02]),position,torch.eye(3,dtype=z.dtype))
    for height in (.006,.012,.024):
        z=torch.tensor(height,dtype=torch.float64,requires_grad=True)
        loss=value(z)
        assert loss.item()==0 and torch.autograd.grad(loss,z)[0].item()==0
    z=torch.tensor(.1,dtype=torch.float64,requires_grad=True)
    loss=value(z)
    torch.testing.assert_close(loss,loss.new_tensor(.075**2),atol=1e-14,rtol=0)
    assert torch.autograd.grad(loss,z)[0]>0


def test_mesh_clipping_keeps_surface_without_creating_region_caps():
    tri=np.array([[[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.]]])
    clipped=clip_triangles(tri,np.array([[1.,0.,0.]]),np.array([0.]))
    assert len(clipped)>0
    assert np.all(clipped[...,0]<=0) and np.all(clipped[...,2]==0)
    a=torch.tensor([[0.,0.,-.1]],dtype=torch.float64);b=-a
    assert segment_triangles_squared(a,b,torch.tensor(clipped)[None]).item()==0


def test_triangle_distance_is_continuous_at_edge_and_has_correct_gradient():
    tri=torch.tensor([[[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]],dtype=torch.float64)
    def cost(x):
        moved=tri+x.reshape(1,1,1,3)
        return segment_triangles_squared(x.new_tensor([[.5,.6,.1]]),x.new_tensor([[.5,.6,.2]]),moved)
    offset=torch.tensor([.01,.02,.03],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(cost,(offset,),eps=1e-6,atol=1e-8,rtol=1e-5)


def fixture():
    # Synthetic analytic geometry with the real canonical link hierarchy. It
    # is a geometry fixture, never a simulation or training contact label.
    bodies=('left_ankle_roll_link','right_ankle_roll_link','left_sphere_hand_link',
        'right_sphere_hand_link','left_knee_link','right_knee_link')
    names=tuple(dict.fromkeys((*bodies,*BODY_NAMES[1:7])))
    fk=CanonicalG1ForwardKinematics().double()
    q=torch.zeros(1,36,dtype=torch.float64);q[:,3]=1;q[:,2]=1
    centers,_=fk.link_poses(q,bodies)
    shapes=[dict(shape=i,body=body,kind='sphere',scale=[.15,0.,0.],transform=[0.,0.,0.,0.,0.,0.,1.]) for i,body in enumerate(bodies)]
    faces=[dict(surface=i,normal_w=[0.,0.,1.],plane_offset=float(centers[0,i,2])-.16) for i in range(6)]
    metadata=dict(link_names=names,configured_margin=.02,surface_attribution_schema=SOURCE_NORMAL_FAN_SCHEMA,
        surface_catalog=faces,solid_geometry=dict(schema=SOLID_GEOMETRY_SCHEMA,model_fingerprint='synthetic geometry unit test',
            robot_asset=canonical_g1_asset_metadata(),shapes=shapes,allowed_pairs=[]))
    bounds=SimpleNamespace(low=torch.full((6,3),-.1),high=torch.full((6,3),.1))
    points=centers.detach().clone();points[:,:,2]-=.16;points[:,:,0]+=.3
    active=torch.zeros(1,6,dtype=torch.bool);active[:,0]=True
    surfaces=torch.arange(6)[None]
    return metadata,fk,q,bounds,points,active,surfaces


def test_router_matches_direct_geometry_and_retains_prediction_gradient():
    metadata,fk,q,bounds,points,active,surfaces=fixture();q.requires_grad_()
    router=NativeContactPositionRouter({5:metadata})
    result=router(fk,q,torch.tensor([5]),points,active,surfaces,None,bounds)
    direct=NativeContactRegionGeometry(metadata,fk,bounds.low,bounds.high).distances(q,points,active,surfaces,None)
    torch.testing.assert_close(result,direct)
    gradient=torch.autograd.grad(result.sum(),q)[0]
    assert bool(gradient.isfinite().all()) and gradient[0,0]<0
    assert torch.count_nonzero(result[~active])==0
    with pytest.raises(ValueError,match='Missing initialized'):
        router(fk,q,torch.tensor([9]),points,active,surfaces,None,bounds)


def test_regional_target_interval_keeps_xyz_and_matches_existing_minimum():
    metadata,fk,q,bounds,points,active,surfaces=fixture()
    q[:,2]+=.2;points[:,:,1]+=.22;q.requires_grad_()
    router=NativeContactPositionRouter({5:metadata})
    args=(fk,q,torch.tensor([5]),points,active,surfaces,None,bounds)
    regional=router(*args,per_region=True)
    assert regional.shape==(1,6,4)
    assert regional[:,1:].count_nonzero()==0
    torch.testing.assert_close(regional[:,0].amin(-1),router(*args)[:,0])
    gradient=torch.autograd.grad(regional.sum(),q)[0]
    assert bool(gradient.isfinite().all())
    assert gradient[0,0]<0 and gradient[0,1]<0 and gradient[0,2]>0


def test_world_chart_preserves_value_and_q_gradient():
    metadata,fk,q,bounds,points,active,surfaces=fixture();q.requires_grad_()
    first=NativeContactRegionGeometry(metadata,fk,bounds.low,bounds.high)
    angle=.73;rotation=torch.tensor(Rotation.from_euler('z',angle).as_matrix(),dtype=q.dtype)[None]
    origin=q.new_tensor([[3.,-2.,.7]])
    shifted=copy.deepcopy(metadata)
    for f in shifted['surface_catalog']:f['plane_offset']+=.7
    second=NativeContactRegionGeometry(shifted,fk,bounds.low,bounds.high)
    world=origin[:,None]+torch.einsum('bij,bpj->bpi',rotation,points)
    a=first.distances(q,points,active,surfaces,None)
    b=second.distances(q,world,active,surfaces,None,world_frame=(origin,rotation))
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(a.sum(),q)[0],torch.autograd.grad(b.sum(),q)[0],atol=1e-11,rtol=1e-11)


def test_scene_batching_preserves_distinct_planes_margins_and_robot_geometry():
    metadata,fk,q,bounds,points,active,surfaces=fixture()
    second=copy.deepcopy(metadata);second['configured_margin']=.035
    for face in second['surface_catalog']:face['plane_offset']+=.06
    third=copy.deepcopy(second);third['solid_geometry']['shapes'][0]['scale'][0]=.12
    metadata={0:metadata,1:second,2:third};routes=torch.tensor([1,0,2,1])
    q=q.repeat(4,1).requires_grad_();points=points.repeat(4,1,1)
    active=active.repeat(4,1);surfaces=surfaces.repeat(4,1)
    router=NativeContactPositionRouter(metadata)
    batched=router(fk,q,routes,points,active,surfaces,None,bounds)
    direct=torch.cat([NativeContactRegionGeometry(metadata[int(key)],fk,bounds.low,bounds.high).distances(
        q[i:i+1],points[i:i+1],active[i:i+1],surfaces[i:i+1],None) for i,key in enumerate(routes)])
    assert router.geometry_keys[0]==router.geometry_keys[1]!=router.geometry_keys[2]
    torch.testing.assert_close(batched,direct,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(batched.sum(),q)[0],torch.autograd.grad(direct.sum(),q)[0],atol=1e-11,rtol=1e-11)


def test_missing_native_configuration_and_old_attribution_are_explicit_errors():
    metadata,fk,_,bounds,*_=fixture()
    for key in ('configured_margin','solid_geometry','surface_attribution_schema'):
        bad=copy.deepcopy(metadata);bad.pop(key)
        with pytest.raises(ValueError):NativeContactRegionGeometry(bad,fk,bounds.low,bounds.high)


def test_http_and_tensor_statistics_share_geometry_and_require_real_contacts():
    metadata,fk,q,bounds,points,active,surfaces=fixture();q.requires_grad_()
    model=SimpleNamespace(fk=fk,region_geometry=bounds)
    router=NativeContactPositionRouter({0:metadata});routes=torch.zeros(1,dtype=torch.long)
    witness=points[0,0].detach().clone();witness[0]-=.3;witness[2]+=.01
    pair=dict(part=0,surface=0,position_w=witness.tolist(),dist=.01,includemargin=.02,allocated=True,constraint_active=True)
    http=dict(pairs=[[pair]],surface_catalog=metadata['surface_catalog'])
    tensor=dict(schema='newton_device_witness_batch_v1',pairs=dict(eligible=torch.tensor([True]),
        sample=torch.tensor([0]),part=torch.tensor([0]),primary_surface=torch.tensor([0]),geometry_point1_w=witness[None]))
    a=native_position_statistics(model,router,q,routes,points,active,surfaces,http)
    b=native_position_statistics(model,router,q,routes,points,active,surfaces,tensor)
    for first,second in zip(a,b):torch.testing.assert_close(first,second)
    assert a[1].item() and a[2].item()==1
    absent=dict(pairs=[[]],surface_catalog=metadata['surface_catalog'])
    error,complete,count=native_position_statistics(model,router,q,routes,points,active,surfaces,absent)
    torch.testing.assert_close(error,a[0])
    assert not complete.item() and count.item()==0
    gradient=torch.autograd.grad(error.sum(),q)[0]
    assert gradient[0,0]<0 and bool(gradient.isfinite().all())


def test_shared_metadata_uses_solver_margins_and_preserves_shape_export():
    from somaforge_core.newton_tensor_transport import native_geometry_metadata
    query=SimpleNamespace(shape_surface={7:[dict(surface=4,normal_w=[0.,0.,1.],plane_offset=.4)]},
        configured_terrain_includemargins=[.017,.023],worlds=1,provenance={'model_fingerprint':'unit'},
        solid_geometry={'schema':SOLID_GEOMETRY_SCHEMA})
    result=native_geometry_metadata(query,('left_ankle_roll_link',))
    assert result['configured_margin']==.017
    assert result['surface_attribution_schema']==SOURCE_NORMAL_FAN_SCHEMA
    assert result['solid_geometry'] is query.solid_geometry
    assert result['surface_catalog'][0]['surface']==4


@pytest.mark.parametrize('duplicate_faces', [False, True])
def test_selected_triangle_backward_matches_all_faces(duplicate_faces):
    from contact_solver.native_contact_position import _segment_triangle_costs
    generator = torch.Generator().manual_seed(831)
    a = torch.randn(8, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    b = torch.randn(8, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    triangles = torch.randn(1, 7, 3, 3, generator=generator, dtype=torch.float64)
    if duplicate_faces:
        triangles = triangles.repeat(1, 2, 1, 1)
    triangles.requires_grad_()
    expected = _segment_triangle_costs(a, b, triangles).amin(-1)
    actual = segment_triangles_squared(a, b, triangles)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    expected_grad = torch.autograd.grad(expected.sum(), (a, b, triangles))
    actual_grad = torch.autograd.grad(actual.sum(), (a, b, triangles))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, atol=1e-12, rtol=1e-11)
    with torch.no_grad():
        torch.testing.assert_close(segment_triangles_squared(a, b, triangles), expected)
