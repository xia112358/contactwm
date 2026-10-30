"""Keep the EPA separating normal when clipping a multi-point contact manifold.

A clipped witness may lie exactly on the plane intersection, so its point-pair
separation is zero even though the parent EPA pair is penetrating. The native
kernel previously used the first clipped pair to orient ALL manifold contacts.
Use the original EPA witnesses, preserving native distances/points/activation.
The installed package is never edited. Verify the complete upstream factory and
install before any native CCD kernel has been cached.
"""
import hashlib
import inspect
import linecache

PATCH_ID = 'epa_witness_manifold_normal_v1'
# Audit the complete factory, not just two matching statements: their surrounding
# distance/orientation semantics and the decorator are part of this repair.
_UPSTREAM_FACTORY_SHA256 = 'eed87a3ebce717c0a1ed1af6a71c86272f90b81bb41dcbd52f03c9ee4d7b0dd5'


def _installed(module):
    marker = getattr(module, '_supermimic_normal_patch', None)
    factory = getattr(module, '_supermimic_normal_patch_factory', None)
    if marker is None and factory is None:
        return False
    if marker != PATCH_ID or factory is not module.ccd_kernel_builder:
        raise RuntimeError('Native CCD normal patch identity changed; start a fresh process')
    return True


def validate_runtime(profile):
    """A process-global kernel patch cannot be labelled as upstream physics."""
    from mujoco_warp._src import collision_convex as module
    from .contact_response import contract
    expected = contract(profile)['manifold_normal'] != 'upstream'
    if _installed(module) != expected:
        raise RuntimeError(f'Contact response {profile} disagrees with the process CCD kernel; '
                           'start a fresh process with the requested contact response')


def install():
    from mujoco_warp._src import collision_convex as module
    from mujoco_warp._src import warp_util
    if _installed(module):
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
    replacements = {
        'frame = make_frame(witness1[0] - witness2[0])': 'frame = make_frame(w1 - w2)',
        'frame = make_frame(witness2[0] - witness1[0])': 'frame = make_frame(w2 - w1)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Unsupported native CCD kernel; re-audit the manifold-normal patch')
        source = source.replace(old, new)
    filename = '<supermimic_' + PATCH_ID + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), module.__dict__)
    module._supermimic_normal_patch = PATCH_ID
    module._supermimic_normal_patch_factory = module.ccd_kernel_builder
