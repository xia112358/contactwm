import torch
from contact_solver.research.step14_qp_teacher import coordinates,pose


def fixture():
    r=torch.zeros(36,dtype=torch.float64)
    r[3:7]=torch.tensor([.8,.1,-.2,.3],dtype=r.dtype)
    r[3:7]/=r[3:7].norm()
    s=r.new_tensor([.02]*3+[.1]*32)
    return r,s


def test_chart_roundtrip_and_quaternion_sign():
    r,s=fixture();z=torch.linspace(-1,1,35,dtype=r.dtype)
    q=pose(z,r,s)
    assert torch.allclose(coordinates(q,r,s),z,atol=1e-12)
    q[3:7]*=-1
    assert torch.allclose(coordinates(q,r,s),z,atol=1e-12)


def test_zero_and_nonzero_chart_gradients():
    r,s=fixture()
    for z in (torch.zeros(35,dtype=r.dtype),torch.linspace(-.3,.3,35,dtype=r.dtype)):
        z.requires_grad_()
        assert torch.autograd.gradcheck(lambda x:coordinates(pose(x,r,s),r,s),(z,))


def test_teacher_descent_equals_target_displacement():
    r,s=fixture();target=torch.linspace(-.7,.7,35,dtype=r.dtype)
    z=torch.zeros_like(target,requires_grad=True)
    loss=.5*(coordinates(pose(z,r,s),r,s)-target).square().sum()
    assert torch.allclose(-torch.autograd.grad(loss,z)[0],target,atol=1e-12)
