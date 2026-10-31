"""Sparse surface guidance must preserve the dense loss and actor gradients."""
from types import SimpleNamespace
import pytest
import torch
from somaforge_core.contact_system.atlas import SurfaceAtlas
from somaforge_core.contact_system.objectives import smooth_region_loss
from supermimic.contact_surface import SurfaceRegionObjective


@pytest.mark.parametrize("parts", [[4], [5, 1], list(range(6))])
@pytest.mark.parametrize("radius", [0., .015])
def test_sparse_loss_and_pose_gradients_match_dense(parts, radius):
    generator = torch.Generator().manual_seed(17)
    nodes = torch.randn(6, 32, 3, generator=generator, dtype=torch.float64) * .03
    normals = torch.zeros_like(nodes); normals[..., 2] = 1
    areas = torch.rand(6, 32, generator=generator, dtype=torch.float64) + .1
    atlas = SurfaceAtlas(nodes, normals, areas, parts=tuple(map(str, range(6))))
    obj = SurfaceRegionObjective.__new__(SurfaceRegionObjective)
    torch.nn.Module.__init__(obj)
    obj.atlas = atlas
    obj.geometry = SimpleNamespace(poses=lambda state: state)
    obj.temperature, obj.radius, obj.target_normal = .02, radius, [0., 0., 1.]
    p = (torch.randn(3, 6, 3, generator=generator, dtype=torch.float64)*.01).requires_grad_()
    R = torch.eye(3, dtype=torch.float64).expand(3, 6, 3, 3).clone().requires_grad_()
    target = (torch.randn(3, 6, 3, generator=generator, dtype=torch.float64)*.01).requires_grad_()
    n = torch.zeros_like(target); n[..., 2] = 1
    actual = obj.region_loss((p, R), target, parts, normals=n, normal_weight=2., tangent_weight=.7)
    reference = smooth_region_loss(atlas.world(p, R)[:, parts], target[:, parts], n[:, parts],
        areas[parts], temperature=.02, halfwidth=radius, normal_weight=2., tangent_weight=.7)
    for key, value in actual.items():
        expected = torch.zeros_like(value).index_copy(1, torch.tensor(parts), reference[key])
        torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(actual['loss'].sum(), (p, R, target), retain_graph=True)
    b = torch.autograd.grad(reference['loss'].sum(), (p, R, target))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
    assert atlas.world(p, R, parts=parts).shape == (3, len(parts), 32, 3)
