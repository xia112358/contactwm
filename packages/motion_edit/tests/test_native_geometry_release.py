"""Optimizer geometry tests; these do not certify native contact labels."""
import unittest
import numpy as np
from somaforge_core.native_geometry_release import GeometryRelease, FrameGeometryConstraint, FastFrameGeometryConstraint


class GeometryTest(unittest.TestCase):
    def geometry(self):
        g = GeometryRelease.__new__(GeometryRelease)
        g.axes = np.array([[[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]])
        g.low = np.array([[0., -1., -1.]])
        g.high = np.array([[0., 1., 1.]])
        g.radii = np.array([.1])
        g.margin = np.array([[.02]])
        g.guard = 1e-6
        return g

    def test_complete_support_not_centroid(self):
        g = self.geometry()
        # One vertex is close even when the center is far away.
        value = np.asarray(g.gaps(np.array([[[0., 0., .11], [0., 0., .5]]])))
        self.assertLess(value.item(), 0)

    def test_finite_face_allows_side_release(self):
        g = self.geometry()
        self.assertGreater(float(g.gaps(np.array([[[1.3, 0., .05]]]))[0, 0]), 0)
        self.assertLess(float(g.gaps(np.array([[[0., 0., .05]]]))[0, 0]), 0)

    def test_derivative(self):
        import torch
        g = self.geometry()
        z = torch.tensor(.2, requires_grad=True)
        points = torch.stack((z*0, z*0, z)).reshape(1, 1, 3)
        derivative, = torch.autograd.grad(g.gaps(points)[0, 0], z)
        self.assertAlmostEqual(float(derivative), 1.)

    def test_triangle_diagonal_is_not_xy_bounding_box(self):
        g = self.geometry()
        triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        edges = np.roll(triangle, -1, axis=0)-triangle
        axes = np.cross(edges, [0., 0., 1.])
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        g.axes = np.concatenate(([[0., 0., 1.]], axes))[None]
        projection = triangle @ g.axes[0].T
        g.low = projection.min(0)[None]; g.high = projection.max(0)[None]
        # Both positions lie inside the triangle's XY bounding rectangle.
        self.assertGreater(float(g.gaps(np.array([[[.8, .8, .01]]]))[0,0]), 0)
        self.assertLess(float(g.gaps(np.array([[[.3, .3, .01]]]))[0,0]), 0)

    def test_only_source_absent_allowed_pairs(self):
        g = self.geometry()
        g.parts = np.array([0])
        g.allowed = np.array([[True]])
        c = FrameGeometryConstraint(g, np.ones(6, bool), lambda x:(np.array([[-1.]]),np.ones((1,1,len(x)))))
        self.assertTrue(c.feasible(np.zeros(2)))
        self.assertEqual(c.fun(np.zeros(2)).size, 0)

    def test_roundoff_cannot_consume_clearance_guard(self):
        g = self.geometry()
        g.parts = np.array([0]); g.allowed = np.array([[True]])
        def constraint(value):
            return FrameGeometryConstraint(g, np.zeros(6, bool), lambda x:(np.array([[value]]),np.ones((1,1,len(x)))))
        self.assertTrue(constraint(-1e-16).feasible(np.zeros(2)))
        self.assertFalse(constraint(-g.guard).feasible(np.zeros(2)))

    def fast(self, mask=None):
        g = self.geometry()
        g.parts = np.array([0]); g.allowed = np.array([[True]])
        g.indices = np.array([0])
        g.vertices = np.array([[[0., 0., 0.], [.2, .1, .1]]])
        forward = lambda x, shapes: g.vertices[shapes]+x[:3]
        gradient = lambda x, indices, local, normals: normals
        return g, FastFrameGeometryConstraint(g, np.zeros(6, bool) if mask is None else mask, 3, forward, gradient)

    def test_fast_support_values_and_derivatives(self):
        g, c = self.fast()
        rng = np.random.default_rng(123)
        for x in rng.uniform(-2, 2, (20, 4)):
            reference = np.asarray(g.gaps(g.vertices+x[:3])).ravel()
            np.testing.assert_allclose(c.fun(x), reference, atol=2e-7)
            grad = c.jac(x)
            numerical = []
            for j in range(4):
                dx = np.eye(4)[j]*1e-6
                numerical.append((c.fun(x+dx)-c.fun(x-dx))/2e-6)
            np.testing.assert_allclose(grad, np.stack(numerical, axis=-1), atol=1e-8)

    def test_fast_gradient_lazy_and_face_parameters_ignored(self):
        g, c = self.fast()
        x = np.array([0., 0., .3, 4.])
        c.fun(x)
        self.assertEqual(c.jacobian_calls, 0)
        x[3] = -8.
        c.fun(x)
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.jac(x)[0, 3], 0)
        c.jac(x)
        self.assertEqual(c.jacobian_calls, 1)

    def test_fast_no_active_release_does_not_call_fk(self):
        g, c = self.fast(np.ones(6, bool))
        c.forward_vertices = lambda *args: self.fail('No release shapes should be evaluated')
        self.assertTrue(c.feasible(np.zeros(4)))
        self.assertEqual(c.jac(np.zeros(4)).shape, (0, 4))


if __name__ == '__main__':
    unittest.main()
