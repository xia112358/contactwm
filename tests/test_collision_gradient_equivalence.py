import pytest
import torch
from contact_solver.research.sampled_penetration import penetration
from contact_solver.collision_geometry import full_geometry_box_ground_penetration


@pytest.mark.parametrize('margin,tolerance',[(0.,0.),(.002,.005)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_penetration_values_and_gradients(margin,tolerance,dtype):
    torch.manual_seed(172)
    points=torch.randn(2,5,30,3,dtype=dtype,requires_grad=True)
    with torch.no_grad():
        points[0,0,:7]=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[1,1,1],[-1,0,0],[0,0,1],[0,0,-1]],dtype=dtype)
    scene=dict(box_center=torch.zeros(2,3,dtype=dtype),box_rotation=torch.eye(3,dtype=dtype)[None].expand(2,-1,-1),
               box_half_extents=torch.ones(2,3,dtype=dtype),ground_height=torch.zeros(2,dtype=dtype),
               margin_m=margin,planned_contact_tolerance_m=tolerance)
    parts=torch.arange(30)%7-1
    contact=torch.randint(0,2,(2,5,6))
    old=full_geometry_box_ground_penetration(points,parts,contact,**scene)
    new=penetration(points,parts,contact,**scene)
    for a,b in zip(old,new):torch.testing.assert_close(a,b,rtol=0,atol=0)
    oldgrad=torch.autograd.grad(sum(x.sum() for x in old),points,retain_graph=True)[0]
    newgrad=torch.autograd.grad(sum(x.sum() for x in new),points)[0]
    torch.testing.assert_close(oldgrad,newgrad,rtol=0,atol=0)
