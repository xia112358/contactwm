from types import SimpleNamespace
import torch
from supermimic.contact_collision_objective import ContinuousClearance,terrain_distances


def fixture(boxes=(),target=(0.,0.,0.)):
    nodes=torch.zeros(6,1,3)
    surface=SimpleNamespace(atlas=SimpleNamespace(points_local=nodes))
    for part in range(6):setattr(surface,f'triangles_{part}',torch.empty(0,3,3))
    targets=torch.tensor(target).expand(1,6,4,3).clone()
    clock=SimpleNamespace(material_targets=targets,paired_material_targets=targets[:,:,:1],paired_material_valid=torch.ones(1,6,1,dtype=torch.bool),event_ends=torch.tensor([1]),data={'point':targets[:,:,0]})
    return ContinuousClearance(surface,[{'boxes':list(boxes)}],clock)


def loss_at(objective,xyz,blocked=False):
    p=torch.zeros(1,6,3);p[0,0]=torch.tensor(xyz);p.requires_grad_()
    enabled=torch.zeros(1,6,dtype=torch.bool);enabled[0,0]=True
    c=objective.loss(p,torch.eye(3).expand(1,6,3,3),torch.tensor([0]),torch.tensor([0]),enabled,torch.full_like(enabled,blocked))
    c.sum().backward();return c,p.grad


def test_clearance_is_active_before_native_contact():
    c,g=loss_at(fixture(),(1.,0.,.005))
    torch.testing.assert_close(c[0,0],torch.tensor(.01))
    assert g[0,0,2]<0


def test_actual_goal_contact_is_not_pushed_away():
    c,g=loss_at(fixture(),(0.,0.,0.))
    assert not c.any() and not g.any()


def test_goal_permission_does_not_exempt_the_whole_obstacle():
    box=(0.,0.,.5,1.,0.,0.,0.,2.,2.,1.)
    obj=fixture([box],target=(0.,0.,1.))
    allowed,g=loss_at(obj,(0.,0.,1.));assert not allowed.any() and not g.any()
    outside,g=loss_at(obj,(.8,0.,1.));assert outside[0,0]>0 and g[0,0,2]<0


def test_entry_support_is_exempt_until_native_release():
    obj=fixture()
    c,g=loss_at(obj,(1.,0.,0.),blocked=True)
    assert not c.any() and not g.any()
    c,g=loss_at(obj,(1.,0.,0.),blocked=False)
    assert c[0,0]>0 and g[0,0,2]<0


def test_signed_box_distance_outside_inside_and_corner():
    p=torch.tensor([[[1.005,0.,.5],[.99,0.,.5],[1.003,0.,1.004]]],requires_grad=True)
    d,near=terrain_distances(p,torch.tensor([[0.,0.,.5]]),torch.eye(3)[None],torch.tensor([[1.,1.,.5]]))
    torch.testing.assert_close(d[0,:,0],torch.tensor([.005,-.01,.005]),atol=1e-6,rtol=1e-4)
    torch.testing.assert_close(near[0,:,0],torch.tensor([[1.,0.,.5],[1.,0.,.5],[1.,0.,1.]]))
    d[:,:,0].sum().backward()
    torch.testing.assert_close(p.grad[0,2],torch.tensor([.6,0.,.8]),atol=1e-4,rtol=1e-4)


def test_far_geometry_is_zero_with_finite_gradient():
    c,g=loss_at(fixture(),(1.,0.,1.))
    assert not c.any() and not g.any() and torch.isfinite(g).all()
