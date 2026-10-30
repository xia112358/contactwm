import pytest


def test_shared_factories_and_idempotent_install():
    from somaforge_core import newton_collision_compat as compat
    from newton._src.geometry import collision_convex,collision_core
    if compat.SCHEMA == compat.NATIVE_SCHEMA:
        before=[getattr(module,name) for module in (collision_convex,collision_core)
                for name in ('create_solve_convex_multi_contact','create_solve_convex_single_contact')]
        assert compat.install()==compat.NATIVE_SCHEMA
        assert compat.install()==compat.NATIVE_SCHEMA
        after=[getattr(module,name) for module in (collision_convex,collision_core)
               for name in ('create_solve_convex_multi_contact','create_solve_convex_single_contact')]
        assert before == after
        return
    from somaforge_core import newton_convex_retry as repaired
    assert compat.install()==repaired.SCHEMA
    assert compat.install()==repaired.SCHEMA
    for name in ('create_solve_convex_multi_contact','create_solve_convex_single_contact'):
        assert getattr(collision_core,name) is getattr(repaired,name)
        assert getattr(collision_convex,name) is getattr(repaired,name)


def test_unreviewed_upstream_fails_closed(monkeypatch):
    from somaforge_core import newton_collision_compat as compat
    monkeypatch.setattr(compat,'UPSTREAM_SHA256','unreviewed')
    monkeypatch.setattr(compat,'NATIVE_HASHES',{'mpr':'unreviewed'})
    with pytest.raises(RuntimeError,match='review'):
        compat.install()


def test_runtime_bridge_native_target_names(monkeypatch):
    from somaforge_core import newton_runtime_compat as runtime
    from somaforge_core import newton_collision_compat as compat
    if compat.SCHEMA != compat.NATIVE_SCHEMA:
        pytest.skip('Native main bridge only')
    import newton
    from newton.selection import ArticulationView
    runtime.install()
    first = ArticulationView.get_attribute
    runtime.install()
    assert ArticulationView.get_attribute is first
    assert newton.solvers.SolverNotifyFlags is newton.ModelFlags
    assert newton.use_coord_layout_targets is False
    class FakeView:
        def _get_attribute_values(self, name, source):
            return name, source
    marker = object()
    assert first(FakeView(), 'joint_target_pos', marker) == ('joint_target_q', marker)
    assert first(FakeView(), 'joint_target_vel', marker) == ('joint_target_qd', marker)
    assert first(FakeView(), 'joint_f', marker) == ('joint_f', marker)


def test_release_bridge_translates_sensor_names_without_changing_selection(monkeypatch):
    import newton.sensors
    from somaforge_core import newton_runtime_compat as runtime

    class Sensor:
        def __init__(self, model, *, sensing_bodies=None, sensing_shapes=None, **kwargs):
            self.received = model, sensing_bodies, sensing_shapes, kwargs
            self.sensing_indices = [3, 8]
            self.sensing_type = 'body'
            self.sensing_transforms = object()

    monkeypatch.setattr(newton.sensors, 'SensorContact', Sensor)
    runtime.install()
    first = Sensor.__init__
    runtime.install()
    assert Sensor.__init__ is first
    marker = object()
    sensor = Sensor(marker, sensing_obj_bodies='Robot/*', sensing_obj_shapes=None, measure_total=True)
    assert sensor.received == (marker, 'Robot/*', None, {'measure_total': True})
    assert sensor.sensing_obj_idx is sensor.sensing_indices
    assert sensor.sensing_obj_type == 'body'
    assert sensor.sensing_obj_transforms is sensor.sensing_transforms
    sensor.sensing_obj_idx = [1]
    assert sensor.sensing_indices == [1]
    with pytest.raises(TypeError, match='only one'):
        Sensor(marker, sensing_obj_shapes='old', sensing_shapes='new')


def test_runtime_guard_rejects_mixed_dependency_stack(monkeypatch):
    from somaforge_core import newton_runtime_compat as runtime
    original = runtime.version
    monkeypatch.setattr(runtime, 'version', lambda name: '1.17.0' if name == 'warp-lang' else original(name))
    with pytest.raises(RuntimeError, match='warp-lang'):
        runtime.runtime_provenance()


def test_mesh_query_tuning_applies_after_initialization_and_scene_rebuild(monkeypatch):
    import sys
    from types import SimpleNamespace

    from somaforge_core import newton_collision_compat as compat

    monkeypatch.setattr(compat, 'SCHEMA', compat.NATIVE_SCHEMA)
    calls = []
    marker = object()

    class Manager:
        _collision_pipeline = None

        @classmethod
        def _initialize_contacts(cls):
            calls.append(cls)
            cls._collision_pipeline = SimpleNamespace(
                device=SimpleNamespace(is_cuda=True),
                narrow_phase=SimpleNamespace(has_meshes=True, has_heightfields=False, tile_size_mesh_convex=256),
            )
            return marker

    monkeypatch.setitem(sys.modules, 'isaaclab_newton.physics.newton_manager', SimpleNamespace(NewtonManager=Manager))
    compat.install_mesh_query_launch_tuning()
    wrapped = Manager._initialize_contacts.__func__
    compat.install_mesh_query_launch_tuning()
    assert Manager._initialize_contacts.__func__ is wrapped

    class SolverManager(Manager):
        pass

    assert SolverManager._initialize_contacts() is marker
    first_pipeline = SolverManager._collision_pipeline
    assert first_pipeline.narrow_phase.tile_size_mesh_convex == 64
    assert SolverManager._initialize_contacts() is marker
    assert SolverManager._collision_pipeline is not first_pipeline
    assert SolverManager._collision_pipeline.narrow_phase.tile_size_mesh_convex == 64
    assert calls == [SolverManager, SolverManager]


@pytest.mark.parametrize('case', ['cpu', 'heightfield', 'no_mesh', 'no_pipeline', 'legacy'])
def test_mesh_query_tuning_preserves_other_native_paths(monkeypatch, case):
    import sys
    from types import SimpleNamespace

    from somaforge_core import newton_collision_compat as compat

    monkeypatch.setattr(compat, 'SCHEMA', compat.LEGACY_SCHEMA if case == 'legacy' else compat.NATIVE_SCHEMA)
    width = 1 if case == 'cpu' else 128 if case == 'heightfield' else 256
    narrow_phase = SimpleNamespace(
        has_meshes=case != 'no_mesh', has_heightfields=case == 'heightfield', tile_size_mesh_convex=width,
    )
    pipeline = SimpleNamespace(device=SimpleNamespace(is_cuda=case != 'cpu'), narrow_phase=narrow_phase)
    marker = object()

    class Manager:
        _collision_pipeline = None

        @classmethod
        def _initialize_contacts(cls):
            cls._collision_pipeline = None if case == 'no_pipeline' else pipeline
            return marker

    original = Manager._initialize_contacts.__func__
    monkeypatch.setitem(sys.modules, 'isaaclab_newton.physics.newton_manager', SimpleNamespace(NewtonManager=Manager))
    compat.install_mesh_query_launch_tuning()
    assert Manager._initialize_contacts() is marker
    assert narrow_phase.tile_size_mesh_convex == width
    if case == 'legacy':
        assert Manager._initialize_contacts.__func__ is original
