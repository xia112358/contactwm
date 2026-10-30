import math
from types import SimpleNamespace

import numpy as np
import torch

from contact_solver.shape_surface_band import NativeShapeSurfaceBand, rounded_cap_cost, triangle_band_cost, enabled_face_shape_ids, minimum_bounds


def geometry():
    g = NativeShapeSurfaceBand.__new__(NativeShapeSurfaceBand)
    dtype = torch.float64
    polygon = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]], dtype=dtype)
    up = torch.tensor([0., 0., 1.], dtype=dtype)
    edges = polygon.roll(-1,0)-polygon
    outward = torch.linalg.cross(edges, up.expand_as(edges))
    outward = outward/outward.norm(dim=-1,keepdim=True)
    g.reference = SimpleNamespace(polygons={1:polygon}, faces={1:(outward, None, up, 0.)})
    g.margin = .02
    g.low = np.array([[-.1,-.1,-.1]]*6)
    g.high = np.array([[.1,.1,.1]]*6)
    return g


def sphere(part=2):
    return dict(part=part, kind='sphere', shape=1, center=np.zeros(3), radius=.1)


def cube():
    return dict(part=0, kind='mesh', shape=2, vertices=np.array(
        [[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.1,.1)]))


def rotation(angle=0.):
    a = torch.as_tensor(angle, dtype=torch.float64)
    z, o = a*0, a*0+1
    return torch.stack((a.cos(), z, a.sin(), z, o, z, -a.sin(), z, a.cos())).reshape(3,3)


def test_mesh_band_has_one_scalar_for_approach_and_recovery():
    g = geometry()
    for height, expected in ((-.5,.36), (.1,0.), (.11,0.), (.12,0.), (.2,.0064)):
        p = torch.tensor([0.,0.,height], dtype=torch.float64, requires_grad=True)
        values = g.mesh_costs(cube(), p, rotation(), 1)
        torch.testing.assert_close(values, values.new_full((4,),expected), atol=1e-12,rtol=1e-8)
        grad = torch.autograd.grad(values.sum(),p)[0]
        if height < .1:
            assert grad[2] < 0
        elif height > .12:
            assert grad[2] > 0
        else:
            assert grad.abs().max() < 1e-12


def test_mesh_side_to_top_recovery_is_continuous_at_footprint_boundary():
    g = geometry()
    # EPA would use a shallow horizontal exit, while the top-contact cap
    # supplies a continuous upward/diagonal recovery distance.
    for x in (1.099999, 1.1, 1.100001):
        p = torch.tensor([x,.23,-.03], dtype=torch.float64)
        value = g.mesh_costs(cube(),p,rotation(),1).amin()
        assert abs(float(value)-.0169) < 1e-8
    p = torch.tensor([1.105,.23,-.03],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:g.mesh_costs(cube(),x,rotation(),1), (p,))


def test_mesh_edge_clearance_uses_euclidean_native_margin():
    g=geometry()
    point=torch.tensor([1.105,.23,.11],dtype=torch.float64,requires_grad=True)
    # Diagonal clearance to the box-top edge is sqrt(5mm^2+10mm^2),
    # inside the actual 20mm band. A vertical-only prism rejects it.
    value=g.mesh_costs(cube(),point,rotation(),1).amin()
    assert value<1e-12
    assert torch.autograd.grad(value,point)[0].abs().max()<1e-10


def test_disabled_shape_terrain_pairs_do_not_supply_contact_geometry():
    g=geometry()
    scene=dict(shapes=[dict(shape=0,body=None,kind='box',scale=[1.,1.,.5],
        transform=[0.,0.,-.5,0.,0.,0.,1.]),
        dict(shape=7,body='left_ankle_roll_link'),dict(shape=99,body='right_ankle_roll_link')],
        allowed_pairs=[[0,1],[1,2]])
    assert enabled_face_shape_ids(scene,g.reference.polygons,g.reference.faces)=={1:{7}}


def test_rotating_mesh_has_true_derivative_with_fresh_hull_queries():
    g = geometry()
    p = torch.tensor([.13,.27,-.17],dtype=torch.float64)
    angle = torch.tensor(.31,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a:g.mesh_costs(cube(),p,rotation(a),1), (angle,))


def test_near_flat_foot_can_activate_all_regions_without_a_unique_minkowski_witness():
    g = geometry()
    angle = .03
    height = .1*(math.cos(angle)+math.sin(angle))+.003
    p = torch.tensor([.43,-.15,height],dtype=torch.float64,requires_grad=True)
    # A large ground face and nearly flat sole permit multiple simultaneous
    # material regions. Selecting one hull point-pair must not discard them.
    values = g.mesh_costs(cube(),p,rotation(angle),1)
    assert values.max() < 1e-12
    assert torch.autograd.grad(values.sum(),p)[0].abs().max() < 1e-10


def test_sphere_top_and_rounded_edge_are_in_the_same_native_band():
    g = geometry()
    r = rotation()
    for p in ([0.,0.,.1], [0.,0.,.11], [1.06,0.,.08], [1.066,0.,.088]):
        p = torch.tensor(p,dtype=torch.float64)
        assert g.sphere_costs(sphere(),p,r,1).amin() < 1e-12
    p = torch.tensor([1.06,0.,.04],dtype=torch.float64,requires_grad=True)
    value = g.sphere_costs(sphere(),p,r,1).amin()
    expected = (.1-math.hypot(.06,.04))**2
    torch.testing.assert_close(value,value.new_tensor(expected),atol=1e-12,rtol=1e-8)
    grad = torch.autograd.grad(value,p)[0]
    assert grad[0] < 0 and grad[2] < 0
    assert torch.autograd.gradcheck(lambda x:g.sphere_costs(sphere(),x,r,1).amin(),(p,))


def test_sphere_deep_embedding_does_not_take_bottom_exit():
    g = geometry()
    p = torch.tensor([0.,0.,-.5],dtype=torch.float64,requires_grad=True)
    value = g.sphere_costs(sphere(),p,rotation(),1).amin()
    torch.testing.assert_close(value,value.new_tensor(.36))
    assert torch.autograd.grad(value,p)[0][2] < 0


def test_rounded_corner_normals_point_outward_and_rotation_keeps_regions():
    g = geometry()
    p = torch.tensor([1.06,1.06,math.sqrt(.01-.0036-.0036)],dtype=torch.float64)
    value = g.sphere_costs(sphere(),p,rotation(),1)
    assert value.amin() < 1e-12
    # Material at the +x top corner comes from the sphere's negative-x half.
    assert value[0] < 1e-12 and value[1] > 1e-5


def test_triangle_band_does_not_give_an_interior_zero_by_tangent_margin():
    tri = torch.tensor([[[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]]],dtype=torch.float64)
    normal = tri.new_tensor([0.,0.,1.])
    p = tri.new_tensor([0.,0.,-.01]).requires_grad_()
    value = triangle_band_cost(tri,normal,p,.02)
    torch.testing.assert_close(value,value.new_tensor([.0001]))
    assert torch.autograd.grad(value.sum(),p)[0][2] < 0


def test_batched_sphere_features_match_scalar_values_and_gradients():
    g = geometry(); g.shapes={1:sphere()}
    class FK:
        def link_poses(self,q,names):
            return q[:,None,:3].expand(-1,len(names),-1),torch.stack([rotation(a) for a in q[:,3]])[:,None].expand(-1,len(names),-1,-1)
    q = torch.tensor([[1.06,.2,.04,.13],[0.,0.,-.4,.23]],dtype=torch.float64,requires_grad=True)
    surfaces = torch.full((2,6),-1,dtype=torch.long);surfaces[:,2]=1
    batched,_ = g.costs(FK(),q,surfaces)
    expected = torch.stack([g.sphere_costs(sphere(),p[:3],rotation(p[3]),1) for p in q])
    torch.testing.assert_close(batched[:,2,:2],expected)
    selected = batched[:,2,:2].amin(-1).sum()
    direct = expected.amin(-1).sum()
    torch.testing.assert_close(torch.autograd.grad(selected,q,retain_graph=True)[0],torch.autograd.grad(direct,q)[0])


def test_interval_bounds_select_one_geometry_instead_of_two_incompatible_minima():
    values=torch.tensor([[4.,0.],[0.,9.],[3.,2.],[2.,3.]],dtype=torch.float64,requires_grad=True)
    selected=minimum_bounds(values,torch.tensor([0,0,1,1]),2)
    torch.testing.assert_close(selected,values.new_tensor([[4.,0.],[2.5,2.5]]))
    # A four-unit interior violation cannot become zero by borrowing the
    # lower bound of a different nine-unit separated geometry alternative.
    torch.testing.assert_close(torch.autograd.grad(selected.sum(),values)[0],values.new_tensor([[1.,1.],[0.,0.],[.5,.5],[.5,.5]]))


def test_shared_bounds_preserve_total_shape_cost_and_its_true_derivative():
    g=geometry();g.shapes={1:sphere(),2:dict(cube(),part=2)}
    class FK:
        def link_poses(self,q,names):
            return q[:,None,:3].expand(-1,len(names),-1),torch.stack([rotation(a) for a in q[:,3]])[:,None].expand(-1,len(names),-1,-1)
    q=torch.tensor([[1.06,.2,.04,.13],[0.,0.,-.4,.23],[0.,0.,.11,0.]],dtype=torch.float64,requires_grad=True)
    surfaces=torch.full((3,6),-1,dtype=torch.long);surfaces[:,2]=1
    old,_=g.costs(FK(),q,surfaces)
    bounds,_=g.intervals(FK(),q,surfaces)
    torch.testing.assert_close(bounds.cost[:,2,:2],old[:,2,:2],atol=1e-12,rtol=1e-8)
    torch.testing.assert_close(torch.autograd.grad(bounds.cost[:,2,:2].sum(),q,retain_graph=True)[0],
        torch.autograd.grad(old[:,2,:2].sum(),q)[0],atol=1e-10,rtol=1e-6)
    assert bounds.lower[0,2,:2].min()>0
    assert bounds.lower[2,2,:2].min()<1e-12
    assert bounds.upper[2,2,:2].min()<1e-12
