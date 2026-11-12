"""API-only bridge for local Isaac Lab 3 and the loaded Newton build.

No collision algorithm, asset, margin or contact activation is changed here.
"""
from __future__ import annotations

from pathlib import Path
from functools import wraps
from importlib.metadata import version

from .newton_collision_compat import SCHEMA, NATIVE_SCHEMA


def _module_fingerprint(module):
    """Identity of the loaded code, including local collision-writer changes."""
    import hashlib
    package = Path(module).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_provenance() -> dict[str, str]:
    """Describe the actual solver stack; version strings are not an allowlist."""
    import hashlib
    import json
    import newton
    module = Path(newton.__file__).resolve()
    dependencies = {name: version(name) for name in ('warp-lang', 'mujoco', 'mujoco-warp')}
    source = _module_fingerprint(module)
    identity = dict(newton=newton.__version__, source_sha256=source, **dependencies)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(runtime_id=f"newton-{newton.__version__}+sha256:{digest}",
                version=newton.__version__, module=str(module), **identity)


NEWTON_RUNTIME_ID = runtime_provenance()['runtime_id']


def install():
    runtime_provenance()
    if SCHEMA != NATIVE_SCHEMA:
        return
    import newton
    import newton.solvers
    # GA snapshots Newton 1.6's deformable ranges during model finalization.
    # Newton 1.7 renamed these identities to surface/volume. Expose the real
    # lists through read-only aliases; never fabricate empty ranges.
    for old, new in (
        ("_cloth_label", "surface_label"),
        ("_cloth_particle_start", "_surface_particle_start"),
        ("_cloth_particle_end", "_surface_particle_end"),
        ("_soft_label", "volume_label"),
        ("_soft_particle_start", "_volume_particle_start"),
        ("_soft_particle_end", "_volume_particle_end"),
    ):
        if not hasattr(newton.ModelBuilder, old):
            setattr(newton.ModelBuilder, old, property(lambda self, name=new: getattr(self, name)))
    # Upstream's supported compatibility flag keeps the existing actuator
    # coordinate layout; do not silently reinterpret trained policy targets.
    newton.use_coord_layout_targets = False
    # Renamed upstream enum, same bits and semantics for the local Lab API.
    if not hasattr(newton.solvers, 'SolverNotifyFlags'):
        newton.solvers.SolverNotifyFlags = newton.ModelFlags
    from newton.selection import ArticulationView
    if not getattr(ArticulationView, '_somaforge_target_names', False):
        original = ArticulationView.get_attribute
        def get_attribute(self, name, source):
            name = {'joint_target_pos': 'joint_target_q',
                    'joint_target_vel': 'joint_target_qd'}.get(name, name)
            return original(self, name, source)
        ArticulationView.get_attribute = get_attribute
        ArticulationView._somaforge_target_names = True
    # Isaac Lab 0.13 still calls the deprecated sensing_obj API. Translate
    # names only; Newton owns sensor selection, force storage, and updates.
    from newton.sensors import SensorContact
    if not getattr(SensorContact, '_somaforge_sensor_names', False):
        original_sensor_init = SensorContact.__init__

        @wraps(original_sensor_init)
        def sensor_init(self, *args, **kwargs):
            for old, new in (('sensing_obj_bodies', 'sensing_bodies'), ('sensing_obj_shapes', 'sensing_shapes')):
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"Specify only one of {old} and {new}")
                    kwargs[new] = kwargs.pop(old)
            original_sensor_init(self, *args, **kwargs)

        SensorContact.__init__ = sensor_init
        for old, new in (('sensing_obj_idx', 'sensing_indices'), ('sensing_obj_type', 'sensing_type'),
                         ('sensing_obj_transforms', 'sensing_transforms')):
            if not hasattr(SensorContact, old):
                setattr(SensorContact, old, property(
                    lambda self, name=new: getattr(self, name),
                    lambda self, value, name=new: setattr(self, name, value),
                ))
        SensorContact._somaforge_sensor_names = True
