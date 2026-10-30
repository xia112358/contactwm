"""Workflow routing preserves backend recipes and keeps dry runs non-mutating."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_workflow', ROOT / 'scripts/run_workflow.py')
workflow = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = workflow
spec.loader.exec_module(workflow)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / 'source'
    (root / 'configs').mkdir(parents=True)
    registry = root / 'configs/workflow_resources.json'
    runtime = tmp_path / 'runtime'
    registry.write_text(json.dumps({
        'supermimic': {'python': sys.executable, 'runtime_root': str(runtime / 'supermimic')},
        'ppo': {'python': sys.executable, 'runtime_root': str(runtime / 'ppo')},
    }))
    return root, registry, runtime


def launch(workspace, values, forwarded=(), env=None):
    root, _, _ = workspace
    options, tail = workflow.parse_args([*values, '--', *forwarded])
    return workflow.build_launch(options, tail, root=root, environ=env or {})


@pytest.mark.parametrize('backend,entrypoint', [
    ('supermimic', ('-m', 'supermimic.train')),
    ('ppo', ('scripts/train_php_wbt.py',)),
])
def test_tracking_uses_existing_implementation_and_preserves_native_arguments(workspace, backend, entrypoint):
    root, _, runtime = workspace
    native = ['--visualizer', 'kit', '--headless', '--checkpoint', 'tracker.pt', '--output-dir', 'run with spaces']
    plan = launch(workspace, ['tracking', '--backend', backend], native)
    expected = entrypoint if backend == 'supermimic' else (str(root / entrypoint[0]),)
    assert plan.argv[1:1 + len(expected)] == expected
    assert list(plan.argv[1 + len(expected):]) == native
    assert plan.profile == backend
    assert plan.runtime_root == runtime / backend
    # GPU/algorithm options are passed to the existing parser, never rewritten.
    assert native[-1] in plan.argv


@pytest.mark.parametrize('stage,module,profile', [
    ('collect', 'supermimic.contact_collect', 'supermimic'),
    ('initialize', 'supermimic.contact_initialize', 'supermimic'),
    ('contact', 'supermimic.contact_train', 'supermimic'),
    ('predictor', 'generator.train_full1000_position', 'ppo'),
])
def test_remaining_stages_use_current_entrypoints(workspace, stage, module, profile):
    plan = launch(workspace, [stage], ['--help'])
    assert plan.argv[1:] == ('-m', module, '--help')
    assert plan.profile == profile


@pytest.mark.parametrize('stage', ['collect', 'initialize', 'contact', 'predictor'])
def test_ppo_selection_is_only_the_original_tracking_stage(workspace, stage):
    with pytest.raises(ValueError, match='applies only to tracking'):
        launch(workspace, [stage, '--backend', 'ppo'])


def test_default_main_flow_preserves_supermimic_teacher_and_dagger_artifact_arguments(workspace):
    tracking = launch(workspace, ['tracking'])
    assert tracking.backend == 'supermimic'
    assert tracking.argv[1:] == ('-m', 'supermimic.train')
    artifacts = ['--checkpoint', 'tracking.pt', '--recording', 'rollout.pt', '--slices', 'cases_validated.pt']
    for stage in ('initialize', 'contact'):
        native = artifacts + (['--resume', 'accepted.pt', '--prior_checkpoint', 'accepted.pt'] if stage == 'contact' else [])
        plan = launch(workspace, [stage], native)
        assert plan.profile == 'supermimic'
        assert list(plan.argv[3:]) == native
        assert '--tracking_backend' not in plan.argv


@pytest.mark.parametrize('dry_run', [True, False])
def test_launch_plan_is_printed_only_for_dry_run(workspace, monkeypatch, capsys, dry_run):
    root, registry, runtime = workspace
    monkeypatch.setattr(workflow, 'ROOT', root)
    # Main's build default is bound at definition; provide the fixture root explicitly.
    original = workflow.build_launch
    monkeypatch.setattr(workflow, 'build_launch', lambda options, tail: original(options, tail, root=root, environ={}))
    executed = []
    monkeypatch.setattr(workflow, 'execute', executed.append)
    before = set(sys.modules)
    args = ['tracking', '--resources', str(registry)] + (['--dry-run'] if dry_run else [])
    assert workflow.main([*args, '--', '--help']) == 0
    output = capsys.readouterr().out
    if dry_run:
        assert json.loads(output)['argv'][-1] == '--help'
        assert not executed
    else:
        assert len(executed) == 1
        assert output == ''
    assert not runtime.exists()
    assert not any(name == 'isaaclab' or name.startswith('isaaclab.') for name in set(sys.modules) - before)


def test_python_precedence_and_missing_python_is_actionable_without_mkdir(workspace):
    root, registry, runtime = workspace
    nonexistent = root.parent / 'missing-python'
    plan = launch(workspace, ['tracking', '--python', str(nonexistent)], env={'SOMAFORGE_SUPERMIMIC_PYTHON': sys.executable})
    assert plan.argv[0] == str(nonexistent)
    assert not plan.as_dict()['interpreter_available']
    with pytest.raises(ValueError, match='not executable'):
        workflow.execute(plan, environ={})
    assert not runtime.exists()
    assert launch(workspace, ['tracking'], env={'SOMAFORGE_SUPERMIMIC_PYTHON': sys.executable}).argv[0] == sys.executable


def test_registry_and_runtime_overrides_use_canonical_source_only(workspace):
    root, registry, runtime = workspace
    other = root.parent / 'other-runtime'
    plan = launch(workspace, ['tracking'], env={
        'SOMAFORGE_RESOURCES': str(registry), 'SOMAFORGE_SUPERMIMIC_RUNTIME': str(other),
        'PYTHONPATH': '/old/sibling/source', 'CUDA_VISIBLE_DEVICES': '3,2',
    })
    assert plan.runtime_root == other
    assert plan.environment['PYTHONPATH'].split(os.pathsep) == [str(root / rel) for rel in workflow.LOCAL_PACKAGES] + [str(root)]
    assert '/old/sibling/source' not in plan.environment['PYTHONPATH']
    assert all(Path(plan.environment[name]).is_relative_to(other) for name in workflow.CACHE_DIRS)
    assert plan.environment['CUDA_VISIBLE_DEVICES'] == '3,2'
    assert plan.environment['MUJOCO_EGL_DEVICE_ID'] == '3'


def test_explicit_external_isaaclab_source_is_configured_without_import(workspace):
    root, registry, _ = workspace
    lab = root.parent / 'isaaclab'
    package = lab / 'source/isaaclab'
    package.mkdir(parents=True)
    (package / 'pyproject.toml').write_text('')
    payload = json.loads(registry.read_text())
    payload['supermimic']['isaaclab_root'] = str(lab)
    registry.write_text(json.dumps(payload))
    plan = launch(workspace, ['tracking'])
    assert plan.environment['ISAACLAB_PATH'] == str(lab)
    assert str(package) in plan.environment['PYTHONPATH'].split(os.pathsep)


def test_data_cannot_default_to_real_source_directory(workspace):
    root, registry, _ = workspace
    payload = json.loads(registry.read_text())
    payload['supermimic']['runtime_root'] = str(root / 'runtime-data')
    registry.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='outside the source checkout'):
        launch(workspace, ['tracking'])


def test_ppo_runtime_symlink_preserves_existing_external_layout(workspace):
    root, registry, runtime = workspace
    payload = json.loads(registry.read_text())
    payload['ppo']['runtime_root'] = 'runtime/current'
    registry.write_text(json.dumps(payload))
    (root / 'runtime').mkdir()
    runtime.mkdir()
    (root / 'runtime/current').symlink_to(runtime, target_is_directory=True)
    assert launch(workspace, ['tracking', '--backend', 'ppo']).runtime_root == runtime


def test_backend_arguments_need_separator_and_are_never_silently_consumed():
    with pytest.raises(SystemExit):
        workflow.parse_args(['tracking', '--task_manifest', 'manifest.json'])


def test_execute_keeps_argv_boundaries_and_creates_only_external_caches(workspace, monkeypatch):
    root, _, runtime = workspace
    plan = launch(workspace, ['tracking'], ['--log_dir', 'a directory with spaces'])
    calls = []
    monkeypatch.setattr(os, 'chdir', lambda path: calls.append(('cwd', path)))
    monkeypatch.setattr(os, 'execve', lambda executable, argv, env: calls.append(('exec', executable, argv, env)))
    workflow.execute(plan, environ={'PYTHONPATH': '/old', 'PYTHONHOME': '/old', 'API_TOKEN': 'private'})
    assert calls[0] == ('cwd', root)
    _, executable, argv, env = calls[1]
    assert executable == sys.executable
    assert argv[-1] == 'a directory with spaces'
    assert 'PYTHONHOME' not in env
    assert env['API_TOKEN'] == 'private'
    assert 'API_TOKEN' not in plan.as_dict()['environment']
    assert all(Path(plan.environment[name]).is_dir() for name in workflow.CACHE_DIRS)


def test_native_help_does_not_create_runtime_directories(workspace, monkeypatch):
    _, _, runtime = workspace
    plan = launch(workspace, ['predictor'], ['--help'])
    calls = []
    monkeypatch.setattr(os, 'chdir', lambda path: None)
    monkeypatch.setattr(os, 'execve', lambda *args: calls.append(args))
    workflow.execute(plan, environ={})
    assert len(calls) == 1
    assert not runtime.exists()
