import pytest
import torch

from contact_solver.shape_face_interval import interval_cost, triangle_prism_support, sphere_prism_support


def prism(dtype=torch.float64):
    normals = torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]]], dtype=dtype)
    return normals, torch.ones(1, 4, dtype=dtype), torch.tensor([[0., 0., 1.]], dtype=dtype)


def test_one_interval_has_zero_scalar_and_gradient_inside_actual_margin():
    gap = torch.tensor([-.1, 0., .01, .02, .1], dtype=torch.float64, requires_grad=True)
    cost = interval_cost(gap, gap.new_tensor(.02))
    torch.testing.assert_close(cost, gap.new_tensor([.01, 0., 0., 0., .0064]))
    grad = torch.autograd.grad(cost.sum(), gap)[0]
    torch.testing.assert_close(grad, gap.new_tensor([-.2, 0., 0., 0., .16]))
    with pytest.raises(ValueError, match='finite gaps'):
        interval_cost(gap.new_tensor([torch.inf]), gap.new_tensor(.02))


def test_triangle_support_clips_overhang_instead_of_penalizing_infinite_plane():
    n, b, d = prism()
    # The outside vertex is 10cm below the plane, but the true in-footprint
    # skin stays above it. The minimum is an edge/side intersection.
    tri = torch.tensor([[[[2., 0., -.1], [0., -.5, .2], [0., .5, .2]]]], dtype=torch.float64)
    torch.testing.assert_close(triangle_prism_support(tri, n, b, d), d.new_tensor([.05]))
    # Break the support-edge tie before comparing a unique derivative. At a
    # tied minimum a central finite difference is not a unique subgradient.
    tri[0, 0, 2, 2] += .013
    tri.requires_grad_()
    assert torch.autograd.gradcheck(lambda x: triangle_prism_support(x, n, b, d), (tri,))


def test_prism_corner_inside_a_large_triangle_is_not_missed():
    n, b, d = prism()
    # All three vertices and all original triangle edges are outside. The
    # support is created by two clipping planes at the footprint corner.
    tri = torch.tensor([[[[-4., -4., -4.], [4., -4., 4.], [0., 5., 0.]]]], dtype=torch.float64)
    torch.testing.assert_close(triangle_prism_support(tri, n, b, d), d.new_tensor([-1.]))


def test_disjoint_skin_is_explicit_not_a_zero_contact_residual():
    n, b, d = prism()
    tri = torch.tensor([[[[2., 0., 0.], [3., 0., 0.], [2., 1., 0.]]]], dtype=torch.float64)
    assert torch.isinf(triangle_prism_support(tri, n, b, d)).all()
    assert torch.isinf(sphere_prism_support(d.new_tensor([[3., 0., 0.]]), .1, n, b, d)).all()


def test_sphere_overhang_has_analytical_clipped_support_and_correct_derivative():
    n, b, d = prism()
    center = d.new_tensor([[1.1, 0., .2]]).requires_grad_()
    expected = .2-(.2**2-.1**2)**.5
    torch.testing.assert_close(sphere_prism_support(center, .2, n, b, d), d.new_tensor([expected]))
    assert torch.autograd.gradcheck(lambda x: sphere_prism_support(x, .2, n, b, d), (center,))


def test_sphere_footprint_corner_and_anatomical_cut_are_both_respected():
    n, b, d = prism()
    center = d.new_tensor([[1.1, 1.1, .2]])
    value = sphere_prism_support(center, .2, n, b, d)
    torch.testing.assert_close(value, d.new_tensor([.2-(.04-.02)**.5]))
    # A regional upper hemisphere must not borrow the lower hemisphere's skin.
    n = torch.cat((n, d.new_tensor([[[0., 0., -1.]]])), 1)
    b = torch.cat((b, d.new_tensor([[-.2]])), 1)
    torch.testing.assert_close(sphere_prism_support(center, .2, n, b, d), d.new_tensor([.2]))


def test_support_is_invariant_to_duplicate_triangles_and_rigid_chart_change():
    n, b, d = prism()
    tri = torch.tensor([[[[2., 0., -.1], [0., -.5, .2], [0., .5, .2]]]], dtype=torch.float64)
    value = triangle_prism_support(tri, n, b, d)
    torch.testing.assert_close(triangle_prism_support(tri.repeat(1, 5, 1, 1), n, b, d), value)
    angle = .7
    r = torch.eye(3, dtype=torch.float64)
    shift = d.new_tensor([3., -.7, .4])
    # Use a true rigid matrix, without float32 trigonometry rounding.
    r[:2, :2] = torch.tensor([[angle]], dtype=torch.float64).cos()*torch.eye(2, dtype=torch.float64) + torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)*torch.tensor(angle, dtype=torch.float64).sin()
    transformed = triangle_prism_support(tri@r.T+shift, n@r.T, b+(n@r.T*shift).sum(-1), d@r.T)
    torch.testing.assert_close(transformed-(d@r.T*shift).sum(-1), value)
