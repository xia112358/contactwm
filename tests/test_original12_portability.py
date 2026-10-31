"""Immutable task data must relocate without a simulator or old checkout."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'packages/supermimic'), str(ROOT / 'packages/somaforge_core')]
from supermimic.task_manifest import load_task_manifest, verify_task_manifest


def fixture_task(tmp_path, monkeypatch):
    root = tmp_path / 'new-checkout'
    runtime = root / 'runtime/current'
    data = runtime / 'workflow/original12'
    data.mkdir(parents=True)
    fields = ('csv', 'source_motion', 'source_terrain', 'terrain_usd')
    task = {'name': 'fixture', 'original_frame_bounds': [700, 830], 'sampling_weight': .5}
    for field in fields:
        p = data / (field + '.fixture')
        p.write_bytes(('immutable ' + field).encode())
        task[field] = '/historical-machine/selected/' + p.name
        task[field + '_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = data / 'task_max76_with_walk.json'
    manifest.write_text(json.dumps(dict(schema='supermimic_somaforge_lowstep_v1', tasks=[task],
                                       motion_dir='/historical-machine/selected')))
    mapping = runtime / 'manifests/published_data_paths.json'
    mapping.parent.mkdir()
    mapping.write_text(json.dumps(dict(schema='somaforge_published_data_paths_v1', files=[], prefixes=[
        dict(source='/historical-machine/selected', target=str(data))])))
    monkeypatch.setenv('SOMAFORGE_ROOT', str(root))
    monkeypatch.delenv('SOMAFORGE_PUBLISHED_DATA_PATHS', raising=False)
    return manifest, data


def test_task_relocates_without_rewriting_identity_or_semantics(tmp_path, monkeypatch):
    manifest, data = fixture_task(tmp_path, monkeypatch)
    original = manifest.read_bytes()
    spec = load_task_manifest(manifest)
    assert Path(spec['tasks'][0]['csv']) == data / 'csv.fixture'
    assert spec['tasks'][0]['original_frame_bounds'] == [700, 830]
    assert spec['tasks'][0]['sampling_weight'] == .5
    report = verify_task_manifest(manifest)
    assert report['tasks'] == 1
    assert report['manifest_sha256'] == hashlib.sha256(original).hexdigest()
    assert manifest.read_bytes() == original


def test_modified_input_is_rejected_after_relocation(tmp_path, monkeypatch):
    manifest, data = fixture_task(tmp_path, monkeypatch)
    (data / 'source_motion.fixture').write_text('changed')
    with pytest.raises(ValueError, match='Task input changed'):
        verify_task_manifest(manifest)


def test_relative_paths_are_manifest_relative_even_from_other_cwd(tmp_path, monkeypatch):
    manifest, data = fixture_task(tmp_path, monkeypatch)
    document = json.loads(manifest.read_text())
    document['motion_dir'] = '.'
    for key in ('csv', 'source_motion', 'source_terrain', 'terrain_usd'):
        document['tasks'][0][key] = key + '.fixture'
    manifest.write_text(json.dumps(document))
    monkeypatch.chdir(tmp_path)
    assert load_task_manifest(manifest)['motion_dir'] == str(data)
    assert verify_task_manifest(manifest)['tasks'] == 1


def load_downloader(monkeypatch):
    spec = importlib.util.spec_from_file_location('_test_original12_download', ROOT / 'scripts/download_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_release_uses_gh_without_exposing_tokens_and_checks_hash(tmp_path, monkeypatch):
    module = load_downloader(monkeypatch)
    payload = b'private release fixture'
    record = dict(name='input.tar.gz', url='https://github.com/owner/repo/releases/download/v1/input.tar.gz',
                  size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    def unauthorized(*args, **kwargs):
        raise urllib.error.HTTPError(record['url'], 404, 'Not Found', {}, None)
    monkeypatch.setattr(module.urllib.request, 'urlopen', unauthorized)
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/usr/bin/gh')
    commands = []
    def download(argv, **kwargs):
        commands.append(argv)
        Path(argv[-1]).write_bytes(payload)
    monkeypatch.setattr(module.subprocess, 'run', download)
    cache = tmp_path / 'cache'
    assert module.acquire(record, cache).read_bytes() == payload
    assert commands[0][1:4] == ['release', 'download', 'v1']
    assert '--repo' in commands[0] and 'owner/repo' in commands[0]
    record['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Conflicting cached asset'):
        module.acquire(record, cache)


def test_authenticated_fallback_rejects_non_github_and_wildcard_assets(tmp_path, monkeypatch):
    module = load_downloader(monkeypatch)
    for url, name in [('https://example.test/owner/repo/releases/download/v1/input.tar.gz', 'input.tar.gz'),
                      ('https://github.com/owner/repo/releases/download/v1/*.tar.gz', '*.tar.gz')]:
        with pytest.raises(ValueError, match='exact github.com'):
            module.authenticated_github_asset(dict(url=url, name=name), tmp_path / 'partial')
