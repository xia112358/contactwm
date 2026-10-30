import torch
from contact_solver.research.recovery_field import box_sdf


def test_box_sdf_inside_outside_and_descent_sign():
    point = torch.tensor([[0., 0., .8], [2., 2., 0.]], dtype=torch.float64, requires_grad=True)
    phi = box_sdf(point, torch.zeros(3), torch.eye(3, dtype=torch.float64), torch.ones(3))
    torch.testing.assert_close(phi, point.new_tensor([-.2, 2**.5]))
    loss = (-phi).relu().square().sum()
    grad = torch.autograd.grad(loss, point)[0]
    assert grad[0, 2] < 0
    assert box_sdf(point-.1*grad, torch.zeros(3), torch.eye(3, dtype=torch.float64), torch.ones(3))[0] > phi[0]


def test_rotated_box_gradient_matches_finite_difference():
    theta = torch.tensor(.43, dtype=torch.float64)
    c, s = theta.cos(), theta.sin()
    basis = torch.stack((c, -s, c*0, s, c, c*0, c*0, c*0, c*0+1)).reshape(3, 3)
    center = torch.tensor([.2, -.5, .3], dtype=torch.float64)
    point = (torch.tensor([[.7, .1, .2]], dtype=torch.float64) @ basis.T + center).requires_grad_()
    assert torch.autograd.gradcheck(lambda p: box_sdf(p, center, basis, torch.ones(3)), (point,))
