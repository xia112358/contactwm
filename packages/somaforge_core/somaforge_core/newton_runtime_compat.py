"""API-only bridge for local Isaac Lab 3 and the sole supported Newton build.

No collision algorithm, asset, margin or contact activation is changed here.
"""
from __future__ import annotations

from pathlib import Path
from functools import wraps
from importlib.metadata import version
import os

from .newton_collision_compat import SCHEMA, NATIVE_SCHEMA


NEWTON_VERSION = "1.7.0"
NEWTON_SOURCE_COMMIT = "7e75abe28b8d69522bc0b34837aaed2c303dff5a"
NEWTON_RUNTIME_ID = f"newton-{NEWTON_VERSION}+{NEWTON_SOURCE_COMMIT}"
RUNTIME_VERSIONS = {"newton": NEWTON_VERSION, "warp-lang": "1.18.0", "mujoco": "3.15.0", "mujoco-warp": "3.15.0"}


def runtime_provenance() -> dict[str, str]:
    """Validate and describe the only Newton runtime accepted by SomaForge."""

    import newton

    module = Path(newton.__file__).resolve()
    expected_text = os.environ.get("SOMAFORGE_NEWTON_ENV")
    if newton.__version__ != NEWTON_VERSION:
        raise RuntimeError(
            f"Unsupported Newton {newton.__version__} at {module}; "
            f"expected {NEWTON_RUNTIME_ID}"
        )
    for distribution, expected_version in RUNTIME_VERSIONS.items():
        actual = version(distribution)
        if actual != expected_version:
            raise RuntimeError(f"Unsupported {distribution} {actual}; expected {expected_version} for {NEWTON_RUNTIME_ID}")
    if expected_text is not None:
        expected = Path(expected_text).resolve()
        if not module.is_relative_to(expected):
            raise RuntimeError(
                f"Newton module {module} is outside required runtime {expected}"
            )
    return {
        "runtime_id": NEWTON_RUNTIME_ID,
        "version": NEWTON_VERSION,
        "source_commit": NEWTON_SOURCE_COMMIT,
        "module": str(module),
        **{name: version(name) for name in RUNTIME_VERSIONS},
    }


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
