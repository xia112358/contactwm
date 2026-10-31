"""Preserve the EPA separating direction through manifold reconstruction.

A clipped witness may lie exactly on the plane intersection, so its point-pair
separation is zero even though the parent EPA pair is penetrating. The native
kernel previously used the first clipped pair to orient ALL manifold contacts.
The original EPA witnesses also lose their separation to float32 rounding near
touching. The current repair uses the EPA closest-face projection before either
witness reconstruction or clipping, preserving native distances/points/activation.
The historical witness repair remains selectable for reproducible diagnostics.
The installed package is never edited. Verify the complete upstream factory and
install before any native CCD kernel has been cached.
"""
import hashlib
import inspect
import linecache

WITNESS_PATCH_ID = 'epa_witness_manifold_normal_v1'
PATCH_ID = 'epa_projection_manifold_normal_v2'
_PATCH_IDS = (WITNESS_PATCH_ID, PATCH_ID)
# Audit the complete factory, not just two matching statements: their surrounding
# distance/orientation semantics and the decorator are part of this repair.
_UPSTREAM_FACTORY_SHA256 = 'eed87a3ebce717c0a1ed1af6a71c86272f90b81bb41dcbd52f03c9ee4d7b0dd5'
_UPSTREAM_EPA_SHA256 = 'dac0f83800ee56f9712160a9ad859967cbc1c92bcfc7420694c0486ac270c6b3'
_EPA_HELPER = '_supermimic_epa_phase_projection_v2'


def _installed(module):
    marker = getattr(module, '_supermimic_normal_patch', None)
    factory = getattr(module, '_supermimic_normal_patch_factory', None)
    if marker is None and factory is None:
        return 'upstream'
    if marker not in _PATCH_IDS or factory is not module.ccd_kernel_builder:
        raise RuntimeError('Native CCD normal patch identity changed; start a fresh process')
    if marker == PATCH_ID:
        helper = getattr(module, _EPA_HELPER, None)
        if helper is None or helper is not getattr(module, '_supermimic_normal_patch_epa_helper', None):
            raise RuntimeError('Native EPA projection helper identity changed; start a fresh process')
    return marker


def validate_runtime(profile):
    """A process-global kernel patch cannot be labelled as upstream physics."""
    from mujoco_warp._src import collision_convex as module
    from .contact_response import contract
    expected = contract(profile)['manifold_normal']
    if _installed(module) != expected:
        raise RuntimeError(f'Contact response {profile} disagrees with the process CCD kernel; '
                           'start a fresh process with the requested contact response')


def install(patch_id=PATCH_ID):
    from mujoco_warp._src import collision_convex as module
    from mujoco_warp._src import warp_util
    if patch_id not in _PATCH_IDS:
        raise ValueError(f'Unknown native CCD normal patch: {patch_id}')
    installed = _installed(module)
    if installed != 'upstream':
        if installed != patch_id:
            raise RuntimeError('A different native CCD normal patch is already installed; '
                               'start a fresh process for the requested physics version')
        return
    factory = inspect.unwrap(module.ccd_kernel_builder)
    # cache_kernel keys by factory NAME, not function identity. Redefining a
    # warmed factory would return old kernels while claiming the patch is active;
    # removing its entries would still leave existing solvers/graphs untouched.
    factory_key = hash(factory.__name__)
    if any(key[-1] == factory_key for key in warp_util._KERNEL_CACHE):
        raise RuntimeError('Native CCD kernels already exist; install the normal fix before '
                           'creating any solver, in a fresh process')
    source = inspect.getsource(factory)
    if hashlib.sha256(source.encode()).hexdigest() != _UPSTREAM_FACTORY_SHA256:
        raise RuntimeError('Unsupported native CCD factory source; re-audit the manifold-normal patch')
    if patch_id == WITNESS_PATCH_ID:
        replacements = {
            'frame = make_frame(witness1[0] - witness2[0])': 'frame = make_frame(w1 - w2)',
            'frame = make_frame(witness2[0] - witness1[0])': 'frame = make_frame(w2 - w1)',
        }
    else:
        from mujoco_warp._src import collision_gjk
        epa_source = inspect.getsource(collision_gjk.epa_phase.func)
        if hashlib.sha256(epa_source.encode()).hexdigest() != _UPSTREAM_EPA_SHA256:
            raise RuntimeError('Unsupported native EPA phase source; re-audit the manifold-normal patch')
        # Export the true closest-face projection independently of multicontact
        # eligibility. Native epa_phase discards idx for margins and non-polyhedral
        # shape pairs; those still need the same stable EPA contact normal.
        epa_replacements = {
            'def epa_phase(': f'def {_EPA_HELPER}(',
            'Tuple[float, int, wp.vec3, wp.vec3, int]': 'Tuple[float, int, wp.vec3, wp.vec3, int, wp.vec3]',
            'return result.dist, 1, result.x1, result.x2, -1':
                'return result.dist, 1, result.x1, result.x2, -1, result.x1 - result.x2',
            'return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1':
                'return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1, wp.vec3()',
            '  # multicontact not supported for margin':
                '  contact_direction = pt.face_pr[idx]\n\n  # multicontact not supported for margin',
            'return dist, 1, x1, x2, idx': 'return dist, 1, x1, x2, idx, contact_direction',
        }
        for old, new in epa_replacements.items():
            if epa_source.count(old) != 1:
                raise RuntimeError('Unsupported native EPA phase; re-audit the manifold-normal patch')
            epa_source = epa_source.replace(old, new)
        epa_filename = '<supermimic_' + patch_id + '_epa>'
        linecache.cache[epa_filename] = (len(epa_source), None, epa_source.splitlines(True), epa_filename)
        namespace = dict(collision_gjk.__dict__)
        exec(compile(epa_source, epa_filename, 'exec'), namespace)
        helper = namespace[_EPA_HELPER]
        # face_pr is the outward closest-face projection in geom1 - geom2
        # Minkowski space, the same quantity used by EPA for penetration depth.
        # Its separate return remains valid even when multiccd_idx is discarded.
        # If no EPA face exists, preserve the original GJK separation direction.
        insertion = '''    contact_direction = w1 - w2
    if needs_epa:
      contact_direction = epa_direction

'''
        anchor = '    if dist >= gap and not is_collision_sensor:\n'
        replacements = {
            '    ccdid = int(-1)': '    epa_direction = wp.vec3()\n    ccdid = int(-1)',
            'dist, ncollision, w1, w2, multiccd_idx = epa_phase(':
                f'dist, ncollision, w1, w2, multiccd_idx, epa_direction = {_EPA_HELPER}(',
            anchor: insertion + anchor,
            'frame = make_frame(witness1[0] - witness2[0])': 'frame = make_frame(contact_direction)',
            'frame = make_frame(witness2[0] - witness1[0])': 'frame = make_frame(-contact_direction)',
        }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Unsupported native CCD kernel; re-audit the manifold-normal patch')
        source = source.replace(old, new)
    filename = '<supermimic_' + patch_id + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    if patch_id == PATCH_ID:
        setattr(module, _EPA_HELPER, helper)
    exec(compile(source, filename, 'exec'), module.__dict__)
    module._supermimic_normal_patch = patch_id
    module._supermimic_normal_patch_factory = module.ccd_kernel_builder
    if patch_id == PATCH_ID:
        module._supermimic_normal_patch_epa_helper = helper
