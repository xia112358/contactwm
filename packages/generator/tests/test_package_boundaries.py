"""Public compatibility and dependency-direction regression checks."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def legacy_import_path(monkeypatch):
    """Historical aliases are an explicit dependency of compatibility tests."""
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / 'packages/climb00_pipeline'))


@pytest.mark.usefixtures('legacy_import_path')
def test_legacy_module_identity_and_monkeypatch(monkeypatch):
    for legacy, canonical in (
        ('full1000_position_predictor', 'generator.full1000_position_predictor'),
        ('contact_q_generator', 'generator.contact_q_generator'),
        ('newton_witness_loss', 'contact_solver.newton_witness_loss'),
        ('contracts', 'somaforge_core.motion_contracts'),
    ):
        old = importlib.import_module('climb00_pipeline.' + legacy)
        new = importlib.import_module(canonical)
        assert old is new
        monkeypatch.setattr(old, '_compatibility_probe', object(), raising=False)
        assert new._compatibility_probe is old._compatibility_probe


def test_solver_imports_without_loading_generator():
    subprocess.run([sys.executable, '-c', '''
import importlib, pkgutil, sys
import contact_solver
for module in pkgutil.iter_modules(contact_solver.__path__):
    if module.name != 'research':
        importlib.import_module('contact_solver.' + module.name)
assert not any(n == 'generator' or n.startswith('generator.') for n in sys.modules)
assert not any(n == 'climb00_pipeline' or n.startswith('climb00_pipeline.') for n in sys.modules)
assert not any(n == 'motion_edit' or n.startswith('motion_edit.') for n in sys.modules)
'''], check=True)


def test_shared_newton_import_does_not_initialize_editing_or_prediction():
    subprocess.run([sys.executable, '-c', '''
import sys
from somaforge_core.newton_collision import DirectNewtonCollisionScene
from somaforge_core.newton_collision_filter import apply_self_collision_filters_to_builder
assert DirectNewtonCollisionScene.__module__ == 'somaforge_core.newton_collision'
assert callable(apply_self_collision_filters_to_builder)
assert not any(n.split('.')[0] in ('motion_edit', 'contact_solver', 'generator') for n in sys.modules)
'''], check=True)


def test_maintained_package_dependencies_have_no_cycle():
    import tomllib
    root = Path(__file__).resolve().parents[3] / 'packages'
    names = ('somaforge_core', 'contact_solver', 'motion_edit', 'generator')
    projects = {name: tomllib.loads((root / name / 'pyproject.toml').read_text())['project'] for name in names}
    graph = {project['name']: {dep.split('[')[0] for dep in project.get('dependencies', [])}
             for project in projects.values()}

    def visit(name, ancestors):
        assert name not in ancestors, f'Package dependency cycle: {ancestors + (name,)}'
        for dependency in graph.get(name, ()):
            if dependency in graph:
                visit(dependency, ancestors + (name,))

    for name in graph:
        visit(name, ())


@pytest.mark.usefixtures('legacy_import_path')
def test_shared_types_keep_identity():
    from climb00_pipeline import TrajectoryPrediction as old_output
    from somaforge_core.prediction_contracts import TrajectoryPrediction as shared_output
    from climb00_pipeline.unified_interaction import project_interaction_q_trajectory as old_project
    from contact_solver.trajectory_projection import project_interaction_q_trajectory
    assert old_output is shared_output
    assert old_project is project_interaction_q_trajectory
