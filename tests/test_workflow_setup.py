"""Fresh clones must not depend on workstation state or change data during planning."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import setup_workflow_environment as setup
import run_workflow as workflow


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'renamed-checkout'
    (root / 'configs').mkdir(parents=True)
    registry = root / 'configs/workflow_resources.json'
    registry.write_bytes((ROOT / 'configs/workflow_resources.json').read_bytes())
    return root


def options(*values):
    import argparse
    parser = argparse.ArgumentParser()
    for name in ('resources', 'isaaclab', 'base-python', 'prefix'):
        parser.add_argument('--' + name, type=Path)
    return parser.parse_args(values)


def test_default_install_has_no_base_environment_and_uses_exact_upstream_lock(checkout):
    plan = setup.make_plan(options(), root=checkout, environ={})
    assert plan['base_python'] is None
    assert plan['runtime'] == str(checkout.parent / 'contactwm-data/supermimic')
    sync = next(row for row in plan['commands'] if row[:2] == ['uv', 'sync'])
    assert '--frozen' in sync and '--python' in sync
    assert '3.12' in sync and 'importers' in sync
    assert any(row[-1] == setup.LAB_COMMIT for row in plan['commands'])
    assert not (checkout / 'runtime').exists()
    assert not (checkout.parent / 'contactwm-data').exists()
    assert not any('/perception/' in arg or '/home/zhouxia/' in arg.replace(str(checkout), '')
                   for command in plan['commands'] for arg in command)


def test_installer_and_launcher_use_same_local_registry(checkout, tmp_path):
    local = checkout / 'configs/workflow_resources.local.json'
    document = json.loads((checkout / 'configs/workflow_resources.json').read_text())
    document['supermimic']['runtime_root'] = str(tmp_path / 'custom-storage')
    local.write_text(json.dumps(document))
    plan = setup.make_plan(options(), root=checkout, environ={})
    args, tail = workflow.parse_args(['tracking', '--dry-run'])
    launch = workflow.build_launch(args, tail, root=checkout, environ={})
    assert plan['resources'] == str(local) == str(launch.resources)
    assert plan['runtime'] == str(launch.runtime_root)
    assert plan['python'] == launch.argv[0]
    assert 'CUDA_VISIBLE_DEVICES' not in launch.environment
    assert 'MUJOCO_EGL_DEVICE_ID' not in launch.environment


def test_explicit_registry_wins_over_local_file(checkout, tmp_path):
    local = checkout / 'configs/workflow_resources.local.json'
    local.write_text('not valid json')
    registry = checkout / 'configs/workflow_resources.json'
    env = {'SOMAFORGE_RESOURCES': str(registry), 'SOMAFORGE_SUPERMIMIC_RUNTIME': str(tmp_path / 'override')}
    plan = setup.make_plan(options(), root=checkout, environ=env)
    args, tail = workflow.parse_args(['tracking'])
    launch = workflow.build_launch(args, tail, root=checkout, environ=env)
    assert str(launch.resources) == plan['resources'] == str(registry)
    assert str(launch.runtime_root) == plan['runtime'] == str(tmp_path / 'override')


def test_configured_runtime_inside_checkout_is_rejected(checkout):
    with pytest.raises(ValueError, match='outside the checkout'):
        setup.make_plan(options(), root=checkout, environ={'SOMAFORGE_SUPERMIMIC_RUNTIME': str(checkout / 'data')})


def test_existing_interpreter_symlink_is_preserved_for_explicit_overlay(checkout, tmp_path):
    base = tmp_path / 'base/bin/python'
    base.parent.mkdir(parents=True)
    base.symlink_to(sys.executable)
    plan = setup.make_plan(options('--base-python', str(base)), root=checkout, environ={})
    assert plan['base_python'] == str(base)
    assert not any(command[:2] == ['uv', 'sync'] for command in plan['commands'])
    assert any('--system-site-packages' in command for command in plan['commands'])


def test_existing_unowned_environment_is_not_overwritten(checkout):
    plan = setup.make_plan(options(), root=checkout, environ={})
    prefix = Path(plan['prefix'])
    prefix.mkdir(parents=True)
    sentinel = prefix / 'keep'
    sentinel.write_text('untouched')
    with pytest.raises(ValueError, match='unowned'):
        setup.install(plan, root=checkout)
    assert sentinel.read_text() == 'untouched'
    assert not (checkout.parent / 'contactwm-data').exists()


def test_direct_package_runtime_uses_local_configuration(checkout, monkeypatch):
    spec = importlib.util.spec_from_file_location('_test_supermimic_paths', ROOT / 'packages/supermimic/supermimic/paths.py')
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    monkeypatch.setattr(paths, 'workspace_root', lambda: checkout)
    monkeypatch.delenv('SOMAFORGE_RESOURCES', raising=False)
    monkeypatch.delenv('SOMAFORGE_SUPERMIMIC_RUNTIME', raising=False)
    document = json.loads((checkout / 'configs/workflow_resources.json').read_text())
    document['supermimic']['runtime_root'] = '../custom-data'
    (checkout / 'configs/workflow_resources.local.json').write_text(json.dumps(document))
    assert paths.runtime_root() == checkout.parent / 'custom-data'
