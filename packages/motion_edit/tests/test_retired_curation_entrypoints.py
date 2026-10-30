"""Retired user entrypoints cannot silently route into a legacy workflow."""
import argparse
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from motion_edit import cli


RETIRED = (
    'workbench', 'workbench-action', 'import-manual-cuts', 'export-cutter-segments',
    'accept', 'reject', 'list-layer', 'migrate-layer-to-canonical',
    'import-lte-catalog', 'force-retarget',
)


@pytest.mark.parametrize('command', RETIRED)
def test_retired_command_is_absent_not_merely_hidden(command):
    parser = cli.build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert command not in commands.choices
    with pytest.raises(SystemExit):
        parser.parse_args([command])


@pytest.mark.parametrize('command', ['export-manifest', 'export-split-npz'])
def test_export_requires_version_and_rejects_old_layer_even_with_version(command):
    suffix = ['--output', 'out.json'] if command == 'export-manifest' else []
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, *suffix])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, '--motion-version-id', 'v1',
                                      '--source', 'accepted/old', *suffix])


def test_retired_handlers_and_modules_are_absent():
    for name in ('_cmd_workbench', '_cmd_workbench_action', '_cmd_curate', '_cmd_list_layer',
                 '_cmd_import_manual_cuts', '_cmd_export_cutter_segments',
                 '_cmd_import_lte_catalog', '_cmd_force_retarget'):
        assert not hasattr(cli, name)
    for module in ('motion_edit.workbench.server', 'motion_edit.adapters.lte'):
        assert importlib.util.find_spec(module) is None
    import motion_edit.workbench as workbench
    assert not hasattr(workbench, 'make_workbench_server')


@pytest.mark.parametrize('handler', [cli._cmd_export_manifest, cli._cmd_export_split_npz])
def test_direct_legacy_calls_fail_before_reading_or_writing(handler):
    with patch.object(cli, 'read_motion_version') as read_version:
        with pytest.raises(ValueError, match='retired'):
            handler(SimpleNamespace(source='accepted/old', motion_version_id='v1'))
    read_version.assert_not_called()


def test_canonical_manifest_export_uses_version_and_its_segments(tmp_path):
    args = cli.build_parser().parse_args(['export-manifest', '--motion-version-id', 'v1',
                                         '--output', str(tmp_path/'manifest.json')])
    version, segments = object(), [object()]
    with patch.object(cli, 'read_motion_version', return_value=version) as read_version, \
         patch.object(cli, 'read_canonical_segments', return_value=segments) as read_segments, \
         patch.object(cli, 'export_motion_version_manifest') as export:
        args.func(args)
    read_version.assert_called_once_with('v1')
    read_segments.assert_called_once_with('v1')
    export.assert_called_once_with(args.output, version=version, segments=segments)


def test_canonical_split_export_keeps_version_identity(tmp_path):
    from motion_edit.schema import SegmentRecord
    args = cli.build_parser().parse_args(['export-split-npz', '--motion-version-id', 'v1',
                                         '--output-dir', str(tmp_path)])
    version = SimpleNamespace(motion_path='source.npz', token_catalog_path=None)
    segment = SegmentRecord(segment_id='s1', motion_id='m1', start_frame=0, end_frame=4, source='canonical')
    with patch.object(cli, 'read_motion_version', return_value=version), \
         patch.object(cli, 'read_canonical_segments', return_value=[segment]), \
         patch.object(cli, 'export_split_npz', return_value=[]) as export:
        args.func(args)
    exported = export.call_args.args[1][0]
    assert exported.motion_path == 'source.npz'
    assert exported.metadata['motion_version_id'] == 'v1'
