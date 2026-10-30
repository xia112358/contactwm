import unittest
import numpy as np
from somaforge_core.contact_aggregation import first_surface_acquisitions, release_supervision, align_event_anchors, phase_support_penalties


class ContactAggregationTests(unittest.TestCase):
    def events(self, active, surfaces=None):
        active = np.asarray(active, dtype=bool)[:, None]
        surfaces = np.zeros_like(active, dtype=int) if surfaces is None else np.asarray(surfaces)[:, None]
        return first_surface_acquisitions(active, surfaces, window=3, occupancy=2/3)

    def test_dropouts_do_not_create_actions_even_when_synchronous(self):
        for length in (1, 4, 12):
            sequence = [1]*5 + [0]*length + [1]*5
            self.assertEqual(self.events(sequence), [])
            self.assertFalse(release_supervision(np.tile(sequence, (15, 1))).any())

    def test_first_acquisition_and_surface_transition(self):
        rows = self.events([0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual([(r['frame'], r['surface']) for r in rows], [(2, 0), (5, 1)])

    def test_isolated_candidate_and_incomplete_tail_are_not_anchors(self):
        self.assertEqual(self.events([0, 1, 0, 0, 0, 0, 1]), [])

    def test_certification_does_not_override_actual_contact(self):
        active = np.array([[True, False], [True, False], [False, False]])
        np.testing.assert_array_equal(release_supervision(active, np.ones_like(active)), [False, True])

    def test_observations_are_not_filled(self):
        active = np.array([[True], [False], [True], [True]])
        original = active.copy()
        first_surface_acquisitions(active, np.zeros_like(active), window=3, occupancy=2/3)
        np.testing.assert_array_equal(active, original)

    def test_repeated_event_identity_and_order_are_preserved(self):
        q = np.zeros((30, 36)); q[:, 3] = 1.; q[:, 0] = np.arange(30)*.01
        source = q.copy(); source[2:, 0] = q[:-2, 0]
        mapping, times = align_event_anchors(q, source, [0, 7, 15, 22, 29], band=3)
        np.testing.assert_array_equal(times, [0, 9, 17, 24, 29])
        self.assertTrue((np.diff(mapping) >= 0).all())
        self.assertTrue((abs(mapping-np.arange(30)) <= 3).all())

    def test_invalid_event_order_is_rejected(self):
        q = np.zeros((10, 36)); q[:, 3] = 1.
        with self.assertRaises(ValueError):
            align_event_anchors(q, q, [0, 5, 4, 9])

    def test_static_motion_does_not_arbitrarily_shift_events(self):
        q = np.zeros((20, 36)); q[:, 3] = 1.
        mapping, _ = align_event_anchors(q, q, [0, 7, 12, 19])
        np.testing.assert_array_equal(mapping, np.arange(20))

    def test_support_allows_rotation_about_a_fixed_contact(self):
        import torch
        theta = torch.linspace(0, .3, 12, dtype=torch.float64)
        r = torch.eye(3, dtype=torch.float64).repeat(12, 1, 1)
        r[:, 0, 0] = theta.cos(); r[:, 1, 1] = theta.cos()
        r[:, 0, 1] = -theta.sin(); r[:, 1, 0] = theta.sin()
        cloud = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
        p = -(r @ cloud[0])[:, None, :]
        p.requires_grad_()
        tasks = [(0, 11, 0, cloud, torch.tensor([0., 0., 1.], dtype=torch.float64), 0.)]
        normal, tangent = phase_support_penalties(p, r[:, None], tasks, scale=.001)
        self.assertGreater(float((p[-1]-p[0]).norm()), .1)
        self.assertLess(float(normal+tangent), 1e-15)
        shifted = p + torch.stack((theta, theta*0, theta*0), -1)[:, None]
        _, moving = phase_support_penalties(shifted, r[:, None], tasks, scale=.001)
        self.assertGreater(float(moving), 1.)
        moving.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        again = phase_support_penalties(p, r[:, None], tasks, scale=.001)
        self.assertLess(float(sum(again)), 1e-15)

    def test_upper_envelope_does_not_lift_an_existing_contact(self):
        import torch
        p = torch.zeros((3, 1, 3), requires_grad=True)
        r = torch.eye(3).repeat(3, 1, 1, 1)
        tasks = [(0, 2, 0, torch.zeros((1, 3)), torch.tensor([0., 0., 1.]), .001)]
        normal, _ = phase_support_penalties(p, r, tasks, scale=.001)
        normal.backward()
        self.assertEqual(float(normal.detach()), 0.)
        self.assertEqual(float(p.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
