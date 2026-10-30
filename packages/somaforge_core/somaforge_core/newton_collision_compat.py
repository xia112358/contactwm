"""Install shared numeric collision repair before any Newton pipeline is built.

No package files, geometry, margin, gap, sorting, or contact activation changes.
The hash guard intentionally rejects unreviewed upstream collision code changes.
"""
import hashlib
from importlib.metadata import version
from pathlib import Path

LEGACY_SCHEMA='newton_convex_conditional_retry_sat_v2'
NATIVE_SCHEMA='newton_native_1_7_0_7e75abe_v1'
NATIVE_VERSION='1.7.0'
SCHEMA=NATIVE_SCHEMA if version('newton') == NATIVE_VERSION else LEGACY_SCHEMA
UPSTREAM_SHA256='d9b836df5759caff9e92107c429a41d34367d22b7fc00590b8e94b4d28b38b41'
NATIVE_HASHES={
    'contact_reduction_global': 'd64ef5487b4f351edb30e067000c929cd1a0cac719dff516e725d6defc956590',
    'collision_convex': 'b32bb4567b0c8fead3850f737fc873dfc10ba4a58ea848339c750d402159decb',
    'collision_core': 'daccb7ad4fe6a2acd0ef5fae83bbd6ba3e655d9e16226677b003e7fc1870de1a',
    'mpr': '92dc1a988df254855593f2f171a83823f44bc4b167e6d105aeeab26f10c16b0e',
    'support_function': '2d2931860de1a47a7b8905e4de66dc09b5a3126e596be6172398cfa716890f6f',
    'narrow_phase': '2cee9b6f4b41eb36e7ca11d8ad64859ae927de6700e9d46f2cc84d4b1b4478c0',
}


def install():
    from newton._src.geometry import collision_convex,collision_core
    if SCHEMA == NATIVE_SCHEMA:
        from importlib import import_module
        for name, expected in NATIVE_HASHES.items():
            module=import_module(f'newton._src.geometry.{name}')
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Newton main {name} changed; review pinned native collision compatibility')
        # Do not replace upstream factories with the legacy retry/SAT copies.
        for name in ('create_solve_convex_multi_contact','create_solve_convex_single_contact'):
            if not getattr(collision_convex,name).__module__.startswith('newton.'):
                raise RuntimeError('Newton main collision factory was replaced; review external patches')
        return SCHEMA
    from . import newton_convex_retry as repaired
    if hashlib.sha256(Path(collision_convex.__file__).read_bytes()).hexdigest()!=UPSTREAM_SHA256:
        raise RuntimeError('Newton convex implementation changed; review conditional retry compatibility')
    for name in ('create_solve_convex_multi_contact','create_solve_convex_single_contact'):
        function=getattr(repaired,name)
        setattr(collision_convex,name,function)
        setattr(collision_core,name,function)
    return SCHEMA


def check_device(device):
    """Warp may only log a device trap; explicitly propagate it to Python."""
    import warp as wp
    from warp._src import context
    device=wp.get_device(device)
    if device.is_cuda:
        error=context.runtime.core.wp_cuda_context_check(device.context)
        if error:
            raise RuntimeError(f'Newton collision/device failure ({error}); check SOMAFORGE_UNRESOLVED_CONVEX_CONTACT diagnostics')


def install_simulation_guard():
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from isaaclab.physics import PhysicsManager
    if getattr(NewtonManager,'_somaforge_collision_guard',False):return
    original=NewtonManager.step.__func__
    def step(cls):
        result=original(cls)
        check_device(PhysicsManager._device)
        return result
    NewtonManager.step=classmethod(step)
    NewtonManager._somaforge_collision_guard=True


def install_mesh_query_launch_tuning():
    """Use 64-lane mesh queries in WBT before Newton captures its CUDA graph.

    Sparse robot/terrain queries do not benefit from the native 256-lane
    default. Only the launch width changes; CPU and heightfield paths retain
    their native settings. Reapply after contact-pipeline initialization so
    later scene rebuilds also receive the setting before graph capture.
    """
    if SCHEMA != NATIVE_SCHEMA:
        return

    from functools import wraps
    import logging

    from isaaclab_newton.physics.newton_manager import NewtonManager

    if getattr(NewtonManager, '_somaforge_mesh_query_launch_tuning', False):
        return
    original = NewtonManager._initialize_contacts.__func__

    @wraps(original)
    def initialize_contacts(cls):
        result = original(cls)
        pipeline = cls._collision_pipeline
        if pipeline is not None:
            narrow_phase = pipeline.narrow_phase
            if pipeline.device.is_cuda and narrow_phase.has_meshes and not narrow_phase.has_heightfields:
                narrow_phase.tile_size_mesh_convex = 64
                logging.getLogger(__name__).info('WBT Newton mesh query block width: 64 (before CUDA graph capture).')
        return result

    NewtonManager._initialize_contacts = classmethod(initialize_contacts)
    NewtonManager._somaforge_mesh_query_launch_tuning = True
