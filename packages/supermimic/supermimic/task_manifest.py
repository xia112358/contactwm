"""Resolve task inputs on this host while retaining immutable manifest identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TOP_LEVEL_PATHS = ('motion_dir', 'visualization_mjcf', 'source_catalogue', 'source_manifest')
TASK_PATHS = ('csv', 'terrain_usd', 'source_motion', 'source_terrain')


def load_task_manifest(path):
    from somaforge_core.published_data import relocate_published_values, resolve_published_path
    path = Path(path).expanduser().resolve()
    spec = relocate_published_values(json.loads(path.read_text()))
    # Relative paths in new manifests are relative to that manifest, never cwd.
    for key in TOP_LEVEL_PATHS:
        if spec.get(key):
            spec[key] = str(resolve_published_path(spec[key], path.parent))
    for key in ('urdf_path', 'usd_path'):
        if spec.get('robot_asset', {}).get(key):
            spec['robot_asset'][key] = str(resolve_published_path(spec['robot_asset'][key], path.parent))
    for task in spec.get('tasks', []):
        for key in TASK_PATHS:
            if task.get(key):
                task[key] = str(resolve_published_path(task[key], path.parent))
    return spec


def verify_task_manifest(path):
    """Verify the complete execution inputs without importing a simulator."""
    spec = load_task_manifest(path)
    if spec.get('schema') != 'supermimic_somaforge_lowstep_v1' or not spec.get('tasks'):
        raise ValueError('Unsupported or empty task manifest')
    checked = set()
    for row, fields in [(spec, ('visualization_mjcf', 'source_catalogue', 'source_manifest')),
                        *[(task, TASK_PATHS) for task in spec['tasks']]]:
        for key in fields:
            if not row.get(key):
                continue
            source = Path(row[key])
            if not source.is_file():
                raise FileNotFoundError(f'Missing {key}: {source}; install the task data component first')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != row[key + '_sha256']:
                raise ValueError(f'Task input changed: {source}')
            checked.add(str(source))
    directory = Path(spec['motion_dir'])
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing motion_dir: {directory}')
    for task in spec['tasks']:
        # The published original12 directory is a distinct copy of its selected CSVs.
        selected = directory / Path(task['csv']).name
        if not selected.is_file() or hashlib.sha256(selected.read_bytes()).hexdigest() != task['csv_sha256']:
            raise ValueError(f'Selected motion directory differs from task CSV: {selected}')
    return dict(manifest=str(Path(path).resolve()), manifest_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                tasks=len(spec['tasks']), checked_files=len(checked),
                scope='File integrity and path relocation only; no physics or policy execution')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_task_manifest(args.manifest), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
