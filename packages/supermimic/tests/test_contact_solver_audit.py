"""CPU execution of the same Warp kernels captured by the native GPU audit."""
import unittest
import json
from types import SimpleNamespace

import numpy as np
import warp as wp

from supermimic.contact_solver_audit import ContactSolverAudit, COUNTERS


class ContactSolverAuditTest(unittest.TestCase):
    def fixture(self, *, cone=0):
        arr = lambda x, dtype=None: wp.array(np.asarray(x), dtype=dtype, device='cpu')
        frames = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        contact = SimpleNamespace(
            dist=arr([-.001, -.002, 0.1], wp.float32),
            includemargin=arr([0., 0., 0.], wp.float32),
            worldid=arr([0, 1, 0], int), type=arr([1, 1, 1], int),
            pos=arr(np.zeros((3, 3), np.float32), wp.vec3),
            frame=arr(frames, wp.mat33), dim=arr([3, 1, 3], int),
            efc_address=arr([[0, 1, 2, 3], [0, -1, -1, -1], [-1, -1, -1, -1]], int))
        z = lambda: arr(np.zeros((2, 4), np.float32), float)
        data = SimpleNamespace(contact=contact, nacon=arr([3], int),
            nefc=arr([4, 1], int), overflow=arr([0, 0], int),
            qpos=z(), qvel=z(), qacc=z(),
            efc=SimpleNamespace(force=z(), D=z(), pos=z(), aref=z()))
        audit = ContactSolverAudit.__new__(ContactSolverAudit)
        audit.data = data
        audit.device = 'cpu'
        audit.cone = cone
        audit.frame_tolerance = 1e-4
        audit.first = wp.zeros(6, dtype=int, device='cpu')
        audit.detail = wp.zeros(14, device='cpu')
        audit.counters = wp.zeros(len(COUNTERS), dtype=wp.int64, device='cpu')
        return audit

    def test_valid_active_and_inactive_rows(self):
        a = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'has not executed'):
            a.check()
        a.launch()
        a.check()
        r = a.report()
        self.assertEqual(r['active_contact_rows'], 2)
        self.assertEqual(r['contact_rows'], 3)
        self.assertEqual(r['live_constraint_rows'], 5)
        self.assertIsNone(r['first_failure'])

    def test_zero_normal_is_sticky_and_keeps_exact_row(self):
        a = self.fixture()
        frame = a.data.contact.frame.numpy()
        frame[1] = 0
        a.data.contact.frame.assign(frame)
        a.launch()
        frame[1] = np.eye(3)
        a.data.contact.frame.assign(frame)
        a.launch()
        with self.assertRaisesRegex(RuntimeError, 'nonorthonormal'):
            a.check()
        f = a.report()['first_failure']
        self.assertEqual((f['world'], f['row'], f['native_solve']), (1, 1, 1))
        self.assertEqual(f['detail'][5:], [0.] * 9)

    def test_checks_every_allocated_pyramid_row(self):
        a = self.fixture()
        addresses = a.data.contact.efc_address.numpy()
        addresses[0, 3] = -1
        a.data.contact.efc_address.assign(addresses)
        a.launch()
        with self.assertRaisesRegex(RuntimeError, 'not_allocated'):
            a.check()
        f = a.report()['first_failure']
        self.assertEqual((f['row'], f['column'], f['value']), (0, 3, -1))

    def test_overflow_and_nonfinite_constraint_fail(self):
        for field in ('overflow', 'force'):
            with self.subTest(field=field):
                a = self.fixture()
                if field == 'overflow':
                    a.data.overflow.assign(np.array([0, 2], np.int32))
                else:
                    force = a.data.efc.force.numpy()
                    force[1, 0] = np.nan
                    a.data.efc.force.assign(force)
                a.launch()
                with self.assertRaises(RuntimeError):
                    a.check()
                self.assertEqual(a.report()['first_failure']['world'], 1)

    def test_contact_capacity_exceeded_is_detected_without_out_of_bounds(self):
        a = self.fixture()
        a.data.nacon.assign(np.array([4], np.int32))
        a.launch()
        with self.assertRaisesRegex(RuntimeError, 'contact_count_out_of_capacity'):
            a.check()

    def test_nonfinite_failure_report_is_strict_json(self):
        for value, encoded in [(np.nan, 'NaN'), (np.inf, '+Infinity'), (-np.inf, '-Infinity')]:
            with self.subTest(value=encoded):
                a = self.fixture()
                force = a.data.efc.force.numpy()
                force[1, 0] = value
                a.data.efc.force.assign(force)
                a.launch()
                report = json.loads(json.dumps(a.report(), allow_nan=False))
                self.assertEqual(report['first_failure']['detail'][0], encoded)
                self.assertEqual(report['first_failure']['reason'], 'nonfinite_live_constraint')


if __name__ == '__main__':
    unittest.main()
