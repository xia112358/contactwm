"""Native CCD repair regressions, without GPU simulation or training."""
import ast
import copy
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import warp as wp
from mujoco_warp._src import collision_convex, warp_util
from mujoco_warp._src.collision_gjk import _witness_on_face
from mujoco_warp._src.math import make_frame

from supermimic import contact_manifold_normals as repair


@wp.kernel
def _native_clipped_intersection(result: wp.array[wp.mat33]):
    # Clipping can put the first manifold vertex exactly on the reference face.
    # The EPA pair still encodes penetration along +Z.
    w1, w2, dist = _witness_on_face(
        wp.vec3(1.0, 1.0, 0.0), wp.vec3(0.0),
        wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, -1.0))
    result[0] = make_frame(w1 - w2)
    result[1] = make_frame(wp.vec3(0.0, 0.0, 0.002))


class ManifoldNormalRepairTest(unittest.TestCase):
    def setUp(self):
        self.factory = collision_convex.ccd_kernel_builder
        self.markers = {
            name: getattr(collision_convex, name)
            for name in ('_supermimic_normal_patch', '_supermimic_normal_patch_factory')
            if hasattr(collision_convex, name)
        }
        self.cache = dict(warp_util._KERNEL_CACHE)
        warp_util._KERNEL_CACHE.clear()
        for name in self.markers:
            delattr(collision_convex, name)

    def tearDown(self):
        collision_convex.ccd_kernel_builder = self.factory
        for name in ('_supermimic_normal_patch', '_supermimic_normal_patch_factory'):
            if hasattr(collision_convex, name):
                delattr(collision_convex, name)
        for name, value in self.markers.items():
            setattr(collision_convex, name, value)
        warp_util._KERNEL_CACHE.clear()
        warp_util._KERNEL_CACHE.update(self.cache)

    def test_late_install_rejects_all_cached_parameterizations(self):
        factory_key = hash(inspect.unwrap(self.factory).__name__)
        for parameters in ((6, 7, 64, 64, True, 0, 128, 0),
                           (5, 6, 32, 16, False, 2, 64, 1)):
            with self.subTest(parameters=parameters):
                key = tuple(hash(value) for value in parameters) + (factory_key,)
                warp_util._KERNEL_CACHE[key] = object()
                with self.assertRaisesRegex(RuntimeError, 'before creating any solver'):
                    repair.install()
                self.assertIs(collision_convex.ccd_kernel_builder, self.factory)
                self.assertFalse(hasattr(collision_convex, '_supermimic_normal_patch'))
                warp_util._KERNEL_CACHE.clear()

    def test_unrelated_cached_factory_is_preserved(self):
        key = (123, hash('unrelated_kernel_builder'))
        sentinel = object()
        warp_util._KERNEL_CACHE[key] = sentinel
        repair.install()
        self.assertIs(warp_util._KERNEL_CACHE[key], sentinel)

    def test_install_is_idempotent_after_patched_kernels_are_cached(self):
        repair.install()
        patched = collision_convex.ccd_kernel_builder
        warp_util._KERNEL_CACHE[(123, hash('ccd_kernel_builder'))] = object()
        repair.install()
        self.assertIs(collision_convex.ccd_kernel_builder, patched)
        repair.validate_runtime('manifold_v2')

    def test_upstream_and_patched_contracts_cannot_be_mixed(self):
        repair.validate_runtime('legacy')
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            repair.validate_runtime('manifold_v2')
        repair.install()
        for profile in ('legacy', 'firm_v2', 'manifold_v1'):
            with self.subTest(profile=profile), self.assertRaisesRegex(RuntimeError, 'disagrees'):
                repair.validate_runtime(profile)
        for profile in ('manifold_v2', 'manifold_soft_v1', 'manifold_balanced_v1'):
            repair.validate_runtime(profile)

    def test_legacy_apply_checks_actual_solver_and_does_not_relabel_failures(self):
        import mujoco
        from supermimic.contact_response import apply
        disabled = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        def env_with(dt, flags):
            manager = SimpleNamespace(
                get_solver_dt=lambda: dt, get_model=lambda: None,
                _solver=SimpleNamespace(mj_model=SimpleNamespace(
                    opt=SimpleNamespace(disableflags=flags))))
            return SimpleNamespace(sim=SimpleNamespace(backend_manager=manager),
                                   contact_response='not_yet_validated')
        valid = env_with(.005, disabled)
        apply(valid, 'legacy')
        self.assertEqual(valid.contact_response, 'legacy')
        for dt, flags, error in ((.002, disabled, 'step disagree'),
                                 (.005, 0, 'flag disagree')):
            env = env_with(dt, flags)
            with self.assertRaisesRegex(ValueError, error):
                apply(env, 'legacy')
            self.assertEqual(env.contact_response, 'not_yet_validated')
        repair.install()
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            apply(valid, 'legacy')

    def test_marker_alone_cannot_claim_the_patch(self):
        repair.install()
        collision_convex.ccd_kernel_builder = self.factory
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            repair.install()
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            repair.validate_runtime('manifold_v2')

    def test_changed_upstream_semantics_fail_closed(self):
        source = inspect.getsource(inspect.unwrap(self.factory))
        # Preserve both old frame statements while changing the sign convention.
        changed = source.replace('if dist <= margin:', 'if dist >= margin:')
        self.assertNotEqual(source, changed)
        with mock.patch.object(repair.inspect, 'getsource', return_value=changed):
            with self.assertRaisesRegex(RuntimeError, 'Unsupported native CCD factory'):
                repair.install()

    def test_patch_only_changes_normal_source_and_preserves_orientation(self):
        original = ast.parse(inspect.getsource(inspect.unwrap(self.factory)))
        repair.install()
        patched = ast.parse(inspect.getsource(inspect.unwrap(collision_convex.ccd_kernel_builder)))
        # Verify the native points, distance, activation and contact material
        # pipeline are identical, not merely statistically similar in a rollout.
        expected = copy.deepcopy(original)
        class ReplaceWitness(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'make_frame':
                    for name in ('left', 'right'):
                        value = getattr(node.args[0], name)
                        if isinstance(value, ast.Subscript):
                            setattr(node.args[0], name, ast.Name(
                                id={'witness1': 'w1', 'witness2': 'w2'}[value.value.id],
                                ctx=ast.Load()))
                return self.generic_visit(node)
        expected = ReplaceWitness().visit(expected)
        self.assertEqual(ast.dump(patched), ast.dump(expected))
        # Execute the actual native branch with coincident clipped witnesses.
        branches = [node for node in ast.walk(patched) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and ast.unparse(node.test) == 'dist <= margin']
        self.assertEqual(len(branches), 1)
        program = compile(ast.fix_missing_locations(ast.Module(body=[branches[0]], type_ignores=[])),
                          '<native-normal-orientation>', 'exec')
        for dist, margin, w1, w2 in (
                (-.002, 0., [0., 0., .002], [0., 0., 0.]),
                (.003, .001, [0., 0., 0.], [0., 0., .002]),
                (.001, .001, [0., 0., .002], [0., 0., 0.])):
            scope = dict(dist=dist, margin=margin, w1=np.array(w1), w2=np.array(w2),
                         witness1=np.zeros((4, 3)), witness2=np.zeros((4, 3)),
                         make_frame=lambda normal: normal / np.linalg.norm(normal))
            exec(program, scope)
            np.testing.assert_array_equal(scope['frame'], [0., 0., 1.])

    def test_native_clipping_reproduces_zero_old_normal(self):
        result = wp.zeros(2, dtype=wp.mat33, device='cpu')
        wp.launch(_native_clipped_intersection, dim=1, inputs=[result], device='cpu')
        frames = result.numpy()
        np.testing.assert_array_equal(frames[0, 0], np.zeros(3))
        np.testing.assert_allclose(frames[1] @ frames[1].T, np.eye(3), atol=1e-6)
        np.testing.assert_array_equal(frames[1, 0], [0., 0., 1.])


if __name__ == '__main__':
    unittest.main()
