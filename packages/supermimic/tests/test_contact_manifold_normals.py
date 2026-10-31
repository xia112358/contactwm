"""Native CCD repair regressions, without GPU simulation or training."""
import ast
import copy
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import warp as wp
from mujoco_warp._src import collision_convex, collision_gjk, warp_util
from mujoco_warp._src.collision_gjk import _witness_on_face, _attach_face, _epa_witness, Polytope, Geom
from mujoco_warp._src.types import GeomType
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


@wp.kernel
def _native_shallow_epa_witness(
        vert: wp.array[wp.vec3], vert_index: wp.array[int], face: wp.array[int],
        projection: wp.array[wp.vec3], norm2: wp.array[float],
        witnesses: wp.array[wp.vec3], depth: wp.array[float], frames: wp.array[wp.mat33]):
    pt = Polytope()
    pt.vert = vert
    pt.vert_index = vert_index
    pt.face = face
    pt.face_pr = projection
    pt.face_norm2 = norm2
    _attach_face(pt, 0, 0, 1, 2)
    x1, x2, dist = _epa_witness(pt, Geom(), Geom(), GeomType.BOX, GeomType.BOX, 0)
    witnesses[0] = x1
    witnesses[1] = x2
    depth[0] = dist
    frames[0] = make_frame(x1 - x2)
    frames[1] = make_frame(pt.face_pr[0])


class ManifoldNormalRepairTest(unittest.TestCase):
    _markers = ('_supermimic_normal_patch', '_supermimic_normal_patch_factory',
                '_supermimic_normal_patch_epa_helper', repair._EPA_HELPER)

    def setUp(self):
        self.factory = collision_convex.ccd_kernel_builder
        self.markers = {
            name: getattr(collision_convex, name)
            for name in self._markers
            if hasattr(collision_convex, name)
        }
        self.cache = dict(warp_util._KERNEL_CACHE)
        warp_util._KERNEL_CACHE.clear()
        for name in self.markers:
            delattr(collision_convex, name)

    def tearDown(self):
        collision_convex.ccd_kernel_builder = self.factory
        for name in self._markers:
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
        repair.validate_runtime('manifold_v3')

    def test_upstream_and_patched_contracts_cannot_be_mixed(self):
        repair.validate_runtime('legacy')
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            repair.validate_runtime('manifold_v2')
        repair.install()
        for profile in ('legacy', 'firm_v2', 'manifold_v1', 'manifold_v2',
                        'manifold_soft_v1', 'manifold_balanced_v1'):
            with self.subTest(profile=profile), self.assertRaisesRegex(RuntimeError, 'disagrees'):
                repair.validate_runtime(profile)
        for profile in ('manifold_v3',):
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
            repair.validate_runtime('manifold_v3')

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
        repair.install(repair.WITNESS_PATCH_ID)
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

    def test_native_epa_reconstruction_can_lose_nonzero_penetration(self):
        # Real float32 support vertices: Minkowski closest-face depth survives,
        # but separately reconstructing the two witnesses rounds both to 0.5.
        eps = 2.0 ** -25
        vertices = np.array([
            [.15, .05, .25 + eps], [.25, .25, .25],
            [.35, .05, .25 + eps], [.25, .25, .25],
            [.25, .45, .75], [.25, .25, .75],
        ], dtype=np.float32)
        arrays = [wp.array(vertices, dtype=wp.vec3, device='cpu'),
                  wp.zeros(6, dtype=int, device='cpu'), wp.zeros(1, dtype=int, device='cpu'),
                  wp.zeros(1, dtype=wp.vec3, device='cpu'), wp.zeros(1, dtype=float, device='cpu'),
                  wp.zeros(2, dtype=wp.vec3, device='cpu'), wp.zeros(1, dtype=float, device='cpu'),
                  wp.zeros(2, dtype=wp.mat33, device='cpu')]
        wp.launch(_native_shallow_epa_witness, dim=1, inputs=arrays, device='cpu')
        witnesses, depth, frames = arrays[5].numpy(), arrays[6].numpy(), arrays[7].numpy()
        self.assertLess(float(depth[0]), 0.0)
        self.assertGreater(float(depth[0]), -1e-7)
        np.testing.assert_array_equal(witnesses[0], witnesses[1])
        np.testing.assert_array_equal(frames[0, 0], np.zeros(3))
        np.testing.assert_allclose(frames[1] @ frames[1].T, np.eye(3), atol=1e-6)
        self.assertGreater(float(frames[1, 0, 2]), .999999)

    def test_historical_patch_remains_distinct_and_reproducible(self):
        repair.install(repair.WITNESS_PATCH_ID)
        repair.validate_runtime('manifold_v2')
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            repair.validate_runtime('manifold_v3')
        with self.assertRaisesRegex(RuntimeError, 'different native CCD normal patch'):
            repair.install()

    def test_projection_preserved_before_margin_or_shape_eligibility(self):
        repair.install()
        helper = getattr(collision_convex, repair._EPA_HELPER)
        patched = ast.parse(inspect.getsource(helper.func)).body[0]
        original = ast.parse(inspect.getsource(collision_gjk.epa_phase.func)).body[0]
        # Removing only the extra output and its saved projection reconstructs
        # the complete original phase, including margin/shape checks and returns.
        stripped = copy.deepcopy(patched)
        stripped.name = original.name
        stripped.returns = copy.deepcopy(original.returns)
        stripped.body = [node for node in stripped.body if not (
            isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'contact_direction')]
        for node in ast.walk(stripped):
            if isinstance(node, ast.Return):
                node.value.elts.pop()
        self.assertEqual(ast.dump(stripped), ast.dump(original))
        source = inspect.getsource(helper.func)
        self.assertLess(source.index('contact_direction = pt.face_pr[idx]'),
                        source.index('# multicontact not supported for margin'))
        self.assertIn('return dist, 1, x1, x2, idx, contact_direction', source)

    def test_projection_branch_handles_sign_and_world_translation(self):
        repair.install()
        tree = ast.parse(inspect.getsource(inspect.unwrap(collision_convex.ccd_kernel_builder)))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                     and node.name == 'eval_ccd_write_contact')
        selected = [node for node in block.body if (
            isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'contact_direction') or (
            isinstance(node, ast.If) and ast.unparse(node.test) in ('needs_epa', 'dist <= margin')
            and any(isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name)
                    and child.targets[0].id in ('contact_direction', 'frame') for child in node.body))]
        self.assertEqual(len(selected), 3)
        program = compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                          '<native-projection-normal>', 'exec')
        for offset in (np.zeros(3), np.array([1.7, 2.6, .7]), np.array([120., -80., 4.])):
            for needs_epa, dist, margin, direction in (
                    (True, -2.98e-8, 0., [0., 0., 2.98e-8]),
                    (True, .001 - 2.98e-8, .001, [0., 0., 2.98e-8]),
                    (False, .003, .001, [0., 0., -.002]),
                    (False, -.002, 0., [0., 0., .002])):
                scope = dict(needs_epa=needs_epa, dist=dist, margin=margin,
                             w1=np.array(offset, dtype=np.float32),
                             w2=np.array(offset, dtype=np.float32),
                             epa_direction=np.array(direction, dtype=np.float32),
                             make_frame=lambda normal: normal / np.linalg.norm(normal))
                if not needs_epa:
                    scope['w1'] += np.array(direction, dtype=np.float32)
                exec(program, scope)
                np.testing.assert_array_equal(scope['frame'], [0., 0., 1.])


if __name__ == '__main__':
    unittest.main()
