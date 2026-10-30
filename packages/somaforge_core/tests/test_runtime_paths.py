"""Native workers use maintained packages without inheriting legacy search paths."""
from pathlib import Path
import os
import json

import pytest

from somaforge_core import runtime_paths


@pytest.fixture
def worker_environment(monkeypatch):
    monkeypatch.delenv('SOMAFORGE_ENV_PROFILE', raising=False)
    monkeypatch.delenv('SOMAFORGE_ENV_PREFIX', raising=False)
    root = Path('/workspace')
    lab = Path('/isaaclab')
    monkeypatch.setattr(runtime_paths, 'project_root', lambda: root)
    monkeypatch.setattr(runtime_paths, 'isaaclab_path', lambda: lab)
    monkeypatch.setattr(runtime_paths, 'default_ik_conda_environment', lambda: str(root / 'runtime/environments/conda/somaforge'))

    def directories(path):
        # Enumerating workspace packages would reintroduce arbitrary local code.
        assert path == lab / 'source'
        return iter([path / 'isaaclab_tasks', path / 'README.md', path / 'isaaclab'])

    monkeypatch.setattr(Path, 'iterdir', directories)
    monkeypatch.setattr(Path, 'is_dir', lambda path: path.suffix != '.md')
    return root, lab


def test_worker_paths_include_only_maintained_packages_and_sorted_lab_modules(worker_environment):
    root, lab = worker_environment
    env = runtime_paths.newton_subprocess_environment()
    paths = env['PYTHONPATH'].split(os.pathsep)
    assert paths == list(map(str, [
        root, root / 'src/holosoma',
        root / 'packages/somaforge_core', root / 'packages/contact_solver',
        root / 'packages/generator', root / 'packages/motion_edit', root / 'packages/supermimic',
        lab / 'source/isaaclab', lab / 'source/isaaclab_tasks',
    ]))
    assert env['SOMAFORGE_NEWTON_ENV'] == str(root / 'runtime/environments/conda/somaforge')
    assert env['SOMAFORGE_ROOT'] == str(root)
    assert env['ISAACLAB_PATH'] == str(lab)


def test_worker_environment_drops_ambient_paths_without_mutating_parent(worker_environment, monkeypatch):
    inherited = {
        'PYTHONPATH': '/old/packages/climb00_pipeline:/unrelated',
        'PYTHONHOME': '/other/python',
        'HOLOSOMA_ISAACLAB3_NEWTON_HEADLESS_EXPERIENCE': '/old/headless.kit',
        'HOLOSOMA_ISAACLAB3_NEWTON_KIT_EXPERIENCE': '/old/window.kit',
        'CUDA_VISIBLE_DEVICES': '2',
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    before = dict(os.environ)
    env = runtime_paths.newton_subprocess_environment()
    assert 'climb00_pipeline' not in env['PYTHONPATH'] and '/unrelated' not in env['PYTHONPATH']
    for key in inherited.keys() - {'PYTHONPATH', 'CUDA_VISIBLE_DEVICES'}:
        assert key not in env
    assert env['CUDA_VISIBLE_DEVICES'] == '2' and env['PYTHONNOUSERSITE'] == '1'
    assert dict(os.environ) == before


def test_explicit_main_profile_routes_native_worker_to_new_prefix(worker_environment, monkeypatch):
    root, _ = worker_environment
    prefix = root / 'runtime/environments/conda/somaforge'
    monkeypatch.setenv('SOMAFORGE_ENV_PROFILE', 'main')
    monkeypatch.setenv('SOMAFORGE_ENV_PREFIX', str(prefix))
    monkeypatch.setenv('CONDA_PREFIX', '/old/ik')
    monkeypatch.setenv('VIRTUAL_ENV', '/old/venv')
    monkeypatch.setenv('PYTHONUSERBASE', '/old/user')
    monkeypatch.setenv('PATH', '/old/ik/bin:/usr/bin')
    before = dict(os.environ)
    env = runtime_paths.newton_subprocess_environment()
    assert runtime_paths.newton_environment() == prefix
    assert env['SOMAFORGE_NEWTON_ENV'] == str(prefix)
    assert env['CONDA_PREFIX'] == str(prefix)
    assert env['PATH'].split(os.pathsep)[0] == str(prefix / 'bin')
    assert '/old' not in env['PATH']
    assert 'VIRTUAL_ENV' not in env and 'PYTHONUSERBASE' not in env
    assert dict(os.environ) == before


@pytest.mark.parametrize('value', ['', 'relative/path', '/outside/env', '/workspace/runtime/environments/conda'])
def test_explicit_main_profile_rejects_missing_or_unmanaged_prefix(worker_environment, monkeypatch, value):
    monkeypatch.setenv('SOMAFORGE_ENV_PROFILE', 'main')
    monkeypatch.setenv('SOMAFORGE_ENV_PREFIX', value)
    with pytest.raises(RuntimeError, match='Managed'):
        runtime_paths.newton_environment()


def test_retired_profile_cannot_select_a_second_runtime(worker_environment, monkeypatch):
    root, _ = worker_environment
    monkeypatch.setenv('SOMAFORGE_ENV_PROFILE', 'ik')
    monkeypatch.setenv('SOMAFORGE_ENV_PREFIX', str(root / 'runtime/environments/conda/somaforge-ik'))
    assert runtime_paths.newton_environment() == root / 'runtime/environments/conda/somaforge'


def test_unmanaged_shell_ik_default_uses_main_catalog(monkeypatch):
    monkeypatch.delenv('SOMAFORGE_ENV_PROFILE', raising=False)
    root = runtime_paths.project_root()
    monkeypatch.delenv('SOMAFORGE_ENV_CATALOG', raising=False)
    assert runtime_paths.default_ik_conda_environment() == str(root / 'runtime/environments/conda/somaforge')


@pytest.mark.parametrize('profile', ['main', 'ik'])
def test_managed_ik_default_uses_catalog_prefix(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(runtime_paths, 'project_root', lambda: tmp_path)
    monkeypatch.setenv('SOMAFORGE_ENV_PROFILE', profile)
    monkeypatch.delenv('SOMAFORGE_ENV_CATALOG', raising=False)
    config = tmp_path / 'configs/environments/profiles.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'schema': 'somaforge_environment_profiles_v1',
        'profiles': {'main': {'prefix': 'runtime/environments/conda/custom-main'}}}))
    assert runtime_paths.default_ik_conda_environment() == str(tmp_path / 'runtime/environments/conda/custom-main')


def test_managed_ik_default_rejects_external_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, 'project_root', lambda: tmp_path)
    monkeypatch.setenv('SOMAFORGE_ENV_PROFILE', 'main')
    monkeypatch.delenv('SOMAFORGE_ENV_CATALOG', raising=False)
    config = tmp_path / 'configs/environments/profiles.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'schema': 'somaforge_environment_profiles_v1',
        'profiles': {'main': {'prefix': '../external'}}}))
    with pytest.raises(RuntimeError, match='Managed IK environment'):
        runtime_paths.default_ik_conda_environment()


def test_managed_ik_default_uses_explicit_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, 'project_root', lambda: tmp_path)
    monkeypatch.setenv('SOMAFORGE_ENV_PROFILE', 'main')
    config = tmp_path / 'custom-profiles.json'
    monkeypatch.setenv('SOMAFORGE_ENV_CATALOG', str(config))
    config.write_text(json.dumps({'schema': 'somaforge_environment_profiles_v1',
        'profiles': {'main': {'prefix': 'runtime/environments/conda/explicit-main'}}}))
    assert runtime_paths.default_ik_conda_environment() == str(tmp_path / 'runtime/environments/conda/explicit-main')
