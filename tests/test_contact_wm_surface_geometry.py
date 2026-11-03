import torch
from holosoma.contact_wm.surface_geometry import HandPlaneContact
from somaforge_core import G1_29DOF_JOINT_ORDER


def test_hand_surface_translation_gradient_and_plane_midpoint():
    model=HandPlaneContact(G1_29DOF_JOINT_ORDER,collision_profile="surface_v2").double()
    assert len(model.links)==1
    s=torch.zeros(1,71,dtype=torch.float64);s[:,6]=1;s[:,2]=1
    n=torch.tensor([[0.,0.,1.]],dtype=torch.float64);plane=torch.tensor([[0.,0.,.7]],dtype=torch.float64)
    s.requires_grad_(True);point=model(s,n,plane)
    grad=torch.autograd.grad(point.sum(),s)[0]
    torch.testing.assert_close(grad[0,:3],torch.tensor([1.,1.,.5],dtype=torch.float64),atol=1e-8,rtol=1e-8)
    shift=torch.tensor([[.02,-.03,.04]],dtype=torch.float64)
    moved=s.detach().clone();moved[:,:3]+=shift
    torch.testing.assert_close(model(moved,n,plane+shift),point.detach()+shift,atol=1e-8,rtol=1e-8)
    assert torch.isfinite(grad).all()


def test_hand_surface_orientation_gradient_matches_finite_difference():
    model=HandPlaneContact(G1_29DOF_JOINT_ORDER,collision_profile="surface_v2").double()
    assert len(model.links)==1
    s=torch.zeros(1,71,dtype=torch.float64);s[:,6]=1;s[:,2]=1;s[:,13:42]=.13
    s.requires_grad_(True);n=torch.tensor([[0.,0.,1.]],dtype=torch.float64);p=torch.tensor([[0.,0.,.7]],dtype=torch.float64)
    loss=model(s,n,p)[0,0];g=torch.autograd.grad(loss,s)[0]
    j=13+list(G1_29DOF_JOINT_ORDER).index('left_wrist_pitch_joint');eps=1e-6
    a=s.detach().clone();b=a.clone();a[0,j]+=eps;b[0,j]-=eps
    finite=(model(a,n,p)[0,0]-model(b,n,p)[0,0])/(2*eps)
    torch.testing.assert_close(g[0,j],finite,atol=1e-6,rtol=1e-4)
