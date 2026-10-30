import torch
from generator.trajectory_budget import densify_qpos, derivative_metrics, smooth_budget_loss


def trajectory(n=19):
    q = torch.zeros(2,n,36)
    q[...,3] = 1
    q[...,0] = torch.linspace(0,1,n).square()
    return q


def test_densify_endpoints_and_quaternion_sign():
    q = trajectory()
    q[:,1::2,3:7] *= -1
    dense = densify_qpos(q,10)
    assert dense.shape == (2,181,36)
    torch.testing.assert_close(dense[:,::10],q)
    torch.testing.assert_close(dense[...,3:7].norm(dim=-1),torch.ones(2,181))


def test_duration_scaling_and_sign_invariance():
    q = trajectory()
    a = derivative_metrics(q,torch.ones(2))
    q[:,1::2,3:7] *= -1
    b = derivative_metrics(q,torch.ones(2)*2)
    for k in a:
        order = int(k.split('_d')[1][0])
        torch.testing.assert_close(a[k],b[k]*2**order)


def test_finite_stationary_gradient_and_reference_detached():
    q = trajectory()*0
    q[...,3] = 1
    q.requires_grad_()
    reference = q.detach().clone().requires_grad_()
    loss, ratio = smooth_budget_loss(q,torch.ones(2),reference,torch.ones(2))
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert reference.grad is None
    assert loss == 0


def test_collision_resampling_gradient():
    q = trajectory().requires_grad_()
    densify_qpos(q,10).square().mean().backward()
    assert torch.isfinite(q.grad).all()


def test_root_budget_has_gradient_and_duration_changes_cost():
    q = trajectory().requires_grad_()
    reference = q.detach()
    loss, ratios = smooth_budget_loss(q,torch.full((2,),0.36),reference,torch.full((2,),0.36))
    assert loss > 0 and bool((ratios > 1).all())
    loss.backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    slow, _ = smooth_budget_loss(q.detach(),torch.full((2,),0.72),reference,torch.full((2,),0.72))
    assert slow < loss


def test_resampling_preserves_physical_quadratic_acceleration():
    a = derivative_metrics(trajectory(19),torch.full((2,),0.36))
    b = derivative_metrics(trajectory(64),torch.full((2,),0.36))
    torch.testing.assert_close(a['root_d2_peak'],b['root_d2_peak'],rtol=3e-4,atol=1e-6)
