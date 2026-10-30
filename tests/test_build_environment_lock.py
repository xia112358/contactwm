"""The pip lock preserves the selected upstream graph and immutable artifacts."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'build_environment_lock', Path(__file__).resolve().parents[1] / 'scripts/build_environment_lock.py')
lock_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock_builder)


def package(name, version='1', **kwargs):
    return dict(name=name, version=version, source={'registry': 'https://pypi.org/simple'}, **kwargs)


def test_official_graph_activates_requested_extras_and_platform_only():
    graph = {'package': [
        package('isaaclab-dev', dependencies=[{'name': 'common'},
            {'name': 'windows', 'marker': "sys_platform == 'win32'"}],
            **{'optional-dependencies': {'test': [{'name': 'pytest'}],
                'isaacsim': [{'name': 'sim', 'extra': ['gpu']}], 'unused': [{'name': 'unused'}]}}),
        package('common'), package('windows'), package('pytest'), package('unused'),
        package('sim', **{'optional-dependencies': {'gpu': [{'name': 'driver'}]}}), package('driver'),
    ]}
    assert set(lock_builder.official_packages(graph)) == {'isaaclab-dev', 'common', 'pytest', 'sim', 'driver'}


def test_official_graph_rejects_ambiguous_active_versions():
    graph = {'package': [package('isaaclab-dev', dependencies=[{'name': 'common'}]),
                         package('common', '1'), package('common', '2')]}
    with pytest.raises(ValueError, match='Ambiguous'):
        lock_builder.official_packages(graph)


def test_seed_rejects_a_different_lab_commit_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(lock_builder.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='wrong\n'))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='pinned Isaac Lab commit'):
        lock_builder.seed(tmp_path, output)
    assert not output.exists()


def test_seed_rejects_modified_official_lock_without_creating_output(tmp_path, monkeypatch):
    (tmp_path / 'uv.lock').write_text('modified')
    def git_run(argv, **kwargs):
        return SimpleNamespace(stdout=lock_builder.LAB_COMMIT if 'rev-parse' in argv else b'committed')
    monkeypatch.setattr(lock_builder.subprocess, 'run', git_run)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='differs from the pinned'):
        lock_builder.seed(tmp_path, output)
    assert not output.exists()


def test_wheel_selection_uses_compatible_artifact_and_preserves_digest():
    selected = package('example', wheels=[
        {'url': 'https://example.org/example-1-py3-none-win_amd64.whl', 'hash': 'sha256:' + 'a' * 64},
        {'url': 'https://example.org/example-1-py3-none-any.whl', 'hash': 'sha256:' + 'b' * 64},
    ])
    result = lock_builder.wheel_record(selected)
    assert result['url'].endswith('none-any.whl') and result['sha256'] == 'b' * 64


def finish_input(tmp_path, name='example', url='https://example.org/example.whl', digest='b' * 64):
    seed = tmp_path / 'seed.json'
    report = tmp_path / 'report.json'
    output = tmp_path / 'result.lock.txt'
    seed.write_text(json.dumps(dict(isaaclab_commit='c' * 40, upstream_lock_sha256='d' * 64,
        records={'retained': dict(version='1', url='https://example.org/retained.whl', sha256='a' * 64)})))
    report.write_text(json.dumps({'install': [dict(metadata=dict(name=name, version='2'),
        download_info=dict(url=url, archive_info=dict(hashes=dict(sha256=digest))))]}))
    return seed, report, output


def test_finish_merges_closures_and_records_provenance(tmp_path):
    seed, report, output = finish_input(tmp_path)
    lock_builder.finish(seed, report, output)
    text = output.read_text()
    assert 'retained @ https://example.org/retained.whl#sha256=' + 'a' * 64 in text
    assert 'example @ https://example.org/example.whl#sha256=' + 'b' * 64 in text
    provenance = json.loads(output.with_suffix('.provenance.json').read_text())
    assert provenance['package_count'] == 2
    assert set(provenance['records']) == {'example', 'retained'}


def test_finish_rejects_changes_to_retained_upstream_versions(tmp_path):
    seed, report, output = finish_input(tmp_path, name='retained')
    with pytest.raises(ValueError, match='retained upstream pin'):
        lock_builder.finish(seed, report, output)
    assert not output.exists()


@pytest.mark.parametrize('kwargs', [dict(name='onnxruntime'), dict(name='opencv-python'),
                                   dict(url='file:///old/wheel.whl'), dict(digest='bad')])
def test_finish_rejects_conflicting_provider_or_mutable_artifact(tmp_path, kwargs):
    with pytest.raises(ValueError):
        lock_builder.finish(*finish_input(tmp_path, **kwargs))
