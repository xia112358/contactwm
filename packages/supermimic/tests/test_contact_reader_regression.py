"""Unknown/reset worlds and inactive native capacity cannot become measurements."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import numpy as np
import torch

from supermimic.contact_native import NativeContacts


class NativeReaderRegressionTest(unittest.TestCase):
    def reader(self):
        r = NativeContacts.__new__(NativeContacts)
        r.direct = False
        r.env = SimpleNamespace(N=3, device='cpu', ep_step=torch.tensor([0, 8, 8]))
        r.binding = {'body_env': {10: 0, 20: 1, 30: 2}}
        return r

    def test_rescope_invalidates_both_binding_caches(self):
        r = self.reader()
        r.device_selector = object()
        r.evidence_binding = object()
        r.set_observed_worlds([0, 2])
        self.assertFalse(hasattr(r, 'device_selector'))
        self.assertFalse(hasattr(r, 'evidence_binding'))
        self.assertEqual(r.binding['body_env'], {10: 0, 30: 2})
        self.assertEqual(r._known_mask()[:, 0].tolist(), [False, False, True])

    def test_unknown_sentinels_leave_raw_and_known_values_unchanged(self):
        r = self.reader()
        r.set_observed_worlds([0, 2])
        raw = torch.arange(54, dtype=torch.float32).reshape(3, 6, 3)
        raw_copy = raw.clone()
        result = r._mask_unknown(dict(active=torch.ones(3, 6, dtype=torch.bool),
            point=raw, force=raw, paired_force_w=raw,
            surface=torch.full((3, 6), 4), surface_local=torch.full((3, 6), 2),
            load=torch.ones(3, 6), slip=torch.ones(3, 6, dtype=torch.float64),
            contact_count=torch.ones(3, 6, dtype=torch.int64),
            body_contact_count=torch.ones(3, 4, dtype=torch.int64),
            body_contact_point=torch.ones(3, 4, 3),
            body_contact_known=torch.ones(3, 4, dtype=torch.bool)))
        self.assertFalse(result['active'][:2].any())
        self.assertTrue(result['active'][2].all())
        self.assertTrue(torch.isnan(result['slip'][:2]).all())
        self.assertTrue((result['surface_local'][:2] == -1).all())
        self.assertTrue((result['point'][:2] == 0).all())
        self.assertTrue((result['force'][:2] == 0).all())
        self.assertTrue((result['body_contact_count'][:2] == 0).all())
        self.assertFalse(result['body_contact_known'][:2].any())
        torch.testing.assert_close(result['point'][2], raw[2])
        torch.testing.assert_close(raw, raw_copy)

    def test_unscoped_reset_is_unknown_until_first_step(self):
        r = self.reader()
        self.assertEqual(r._known_mask()[:, 0].tolist(), [False, True, True])
        r.env.ep_step[0] = 1
        self.assertTrue(r._known_mask().all())

    def test_invalid_active_normal_is_rejected_without_part_filter(self):
        snapshot = dict(count=1, dist=np.array([-.01, -.01]),
            includemargin=np.zeros(2), type=np.ones(2, dtype=int),
            frame_w=np.zeros((2, 3, 3)))
        self.assertTrue(NativeContacts._normal_error(snapshot))
        snapshot['frame_w'][0] = np.eye(3)
        # Unused capacity may contain a zero normal; it is not a native row.
        self.assertFalse(NativeContacts._normal_error(snapshot))
        snapshot['count'] = 2
        self.assertTrue(NativeContacts._normal_error(snapshot))
        # A real but inactive row does not have an active constraint normal.
        snapshot['type'][1] = 0
        self.assertFalse(NativeContacts._normal_error(snapshot))
        snapshot['frame_w'][0, 0, 0] = float('nan')
        self.assertTrue(NativeContacts._normal_error(snapshot))

    def test_normal_check_accepts_native_numpy_and_device_tensor_counts(self):
        # snapshot_solver_contacts exports count as a zero-dimensional ndarray;
        # read_device supplies a scalar torch tensor. Unused capacity stays out.
        snapshot = dict(dist=np.array([-.01, -.01], dtype=np.float32),
            includemargin=np.zeros(2, np.float32), type=np.ones(2, dtype=int),
            frame_w=np.array([np.eye(3), np.zeros((3, 3))], dtype=np.float32))
        for count in (1, np.int32(1), np.asarray(1, dtype=np.int32),
                      np.array([1], dtype=np.int32), torch.tensor(1)):
            with self.subTest(count_type=type(count), shape=getattr(count, 'shape', None)):
                self.assertFalse(NativeContacts._normal_error(dict(snapshot, count=count)))
        self.assertTrue(NativeContacts._normal_error(dict(snapshot, count=np.asarray(2, dtype=np.int32))))
        with self.assertRaises(ValueError):
            NativeContacts._normal_error(dict(snapshot, count=np.array([1, 2])))

    def test_export_serializes_live_world0_friction_without_mutating_template(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="box" size="1 1 1" friction=".2 .005 .0001"/></worldbody></mujoco>')
        before = model.geom_friction.copy()
        fields = {}
        for key in ('geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap', 'geom_friction'):
            values = np.stack([getattr(model, key).copy() for _ in range(2)])
            if key == 'geom_friction':
                values[0, :, 0] = .9
                values[1, :, 0] = .4
            fields[key] = SimpleNamespace(numpy=lambda a=values: a)
        r = self.reader()
        r.collision_profile = 'legacy_v1'
        r.solver = SimpleNamespace(mj_model=model, mjw_model=SimpleNamespace(**fields))
        r.env.sim = SimpleNamespace(backend_manager=SimpleNamespace(get_solver_dt=lambda: .005))
        modules = {
            'somaforge_core.robot_assets': SimpleNamespace(canonical_g1_asset_metadata=lambda **kw: kw),
            'somaforge_core.collision_profiles': SimpleNamespace(audit_surface_mujoco=lambda model: None),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules):
            path = Path(directory) / 'solver.mjb'
            r.export_collision_model(path)
            saved = mujoco.MjModel.from_binary_path(str(path))
            np.testing.assert_allclose(saved.geom_friction[:, 0], .9)
            np.testing.assert_array_equal(model.geom_friction, before)
            self.assertEqual(saved.opt.timestep, .005)
            self.assertNotEqual(model.opt.timestep, saved.opt.timestep)
            meta = json.loads(path.with_suffix('.mjb.json').read_text())
            self.assertEqual(meta['exported_world'], 0)
            self.assertEqual(meta['per_world_parameter_differences'], ['geom_friction'])
            with self.assertRaises(FileExistsError):
                r.export_collision_model(path)


if __name__ == '__main__':
    unittest.main()
