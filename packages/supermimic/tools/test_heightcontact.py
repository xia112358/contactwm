"""CPU regressions for height raster sampling and observation contents."""
import sys
from pathlib import Path
import unittest

import torch
from supermimic.lowstep_observations import HeightContactContext, HeightForceContext


class HeightContactTests(unittest.TestCase):
    def setUp(self):
        box = [0, 0, 0.175, 1, 0, 0, 0, 1, 1, 0.35]
        self.context = HeightContactContext([{"boxes": [box]}, {"boxes": [box]}])
        self.position = torch.tensor([[-0.1, 0, 0.8]])
        self.rotation = torch.eye(3).unsqueeze(0)
        self.xi = torch.zeros(1, 36)

    def test_known_box_top_and_floor(self):
        features = self.context.features((self.position, self.rotation), self.xi)
        self.assertEqual(features.shape, (1, 123))
        self.assertAlmostEqual(float(features[0, 5 * 11 + 3]), -0.45, places=5)
        self.assertAlmostEqual(float(features[0, 10]), -0.8, places=5)

    def test_contacts_only_change_contact_channels(self):
        before = self.context.features((self.position, self.rotation), self.xi)
        self.xi[:, 34:36] = torch.tensor([1., 0.])
        after = self.context.features((self.position, self.rotation), self.xi)
        torch.testing.assert_close(before[:, :121], after[:, :121])
        torch.testing.assert_close(after[:, 121:], torch.tensor([[1., 0.]]))

    def test_task_identifier_is_not_a_network_feature(self):
        before = self.context.features((self.position, self.rotation), self.xi)
        self.xi[:, 33] = 1
        after = self.context.features((self.position, self.rotation), self.xi)
        torch.testing.assert_close(before, after)

    def test_bilinear_height_gradient(self):
        position = torch.tensor([[0.43, 0, 0.8]], requires_grad=True)
        self.context.features((position, self.rotation), self.xi).sum().backward()
        self.assertTrue(torch.isfinite(position.grad).all())
        self.assertGreater(float(position.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(position.grad[:, 2].abs().sum()), 0)

    def test_force_frame_roundtrip_and_magnitude(self):
        # A +90 degree heading maps world +x to local -y.
        rotation = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        world = torch.zeros(1, 18)
        world[0, :3] = torch.tensor([1., 0., 2.])
        local = HeightForceContext.local_forces((self.position, rotation), world)
        torch.testing.assert_close(local[0, :3], torch.tensor([0., -1., 2.]))
        torch.testing.assert_close(HeightForceContext.world_forces((self.position, rotation), local), world)
        context = HeightForceContext([{"boxes": [[0., 0., .175, 1., 0., 0., 0., 1., 1., .35]]}])
        xi = torch.zeros(1, 52)
        xi[:, 34:] = world
        features = context.features((self.position, rotation), xi)
        self.assertEqual(features.shape[-1], 139)
        torch.testing.assert_close(features[:, 121:], local)


if __name__ == "__main__":
    unittest.main()
