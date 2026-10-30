import json
from pathlib import Path
import pytest
from supermimic import paths


def test_explicit_runtime_is_independent_of_checkout(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    monkeypatch.setenv('SOMAFORGE_SUPERMIMIC_RUNTIME', str(runtime))
    assert paths.runtime_path('logs', 'trial') == runtime / 'logs/trial'
    assert not runtime.exists()


def test_resource_file_selects_runtime_without_sibling_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv('SOMAFORGE_SUPERMIMIC_RUNTIME', raising=False)
    monkeypatch.setattr(paths, 'workspace_root', lambda: tmp_path)
    config = tmp_path / 'resources.json'
    config.write_text(json.dumps({'supermimic': {'runtime_root': 'external/runtime'}}))
    monkeypatch.setenv('SOMAFORGE_RESOURCES', 'resources.json')
    assert paths.runtime_root() == tmp_path / 'external/runtime'


def test_missing_runtime_configuration_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.delenv('SOMAFORGE_SUPERMIMIC_RUNTIME', raising=False)
    monkeypatch.delenv('SOMAFORGE_RESOURCES', raising=False)
    monkeypatch.setattr(paths, 'workspace_root', lambda: tmp_path)
    with pytest.raises(RuntimeError, match='Configure supermimic.runtime_root'):
        paths.runtime_root()


def test_relative_environment_override_is_rejected(monkeypatch):
    monkeypatch.setenv('SOMAFORGE_SUPERMIMIC_RUNTIME', 'runtime')
    with pytest.raises(ValueError, match='absolute'):
        paths.runtime_root()


def test_packaged_contact_profile_retains_existing_contract():
    profile = paths.package_path('experiments/contact_completion.json')
    payload = json.loads(profile.read_text())
    assert payload['default_tolerance_m'] == .05
    assert payload['overrides'][0]['segment_bounds'] == [478, 513]
