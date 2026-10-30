import torch
from contact_solver.research.step14_root_cost import displacement_cost


def reference():
    q=torch.zeros(36,dtype=torch.float64);q[3]=1
    return q


def test_normalized_root_translation_is_1000_times_joint():
    r=reference();translation=r.clone();joint=r.clone()
    translation[0]=.02;joint[7]=.1
    assert displacement_cost(translation,r,1000)==1000*displacement_cost(joint,r,1000)


def test_quaternion_sign_and_initial_gradient():
    r=reference();q=r.clone().requires_grad_()
    assert torch.equal(torch.autograd.grad(displacement_cost(q,r,1000),q)[0],torch.zeros_like(q))
    q=r.clone();q[3:7]*=-1
    assert displacement_cost(q,r,1000)==0


def test_nonzero_rotation_and_translation_gradient():
    r=reference();q=r.clone();q[0]=.003;q[4]=.04;q[7]=.05
    q.requires_grad_()
    assert torch.autograd.gradcheck(lambda x:displacement_cost(x,r,1000),(q,))
