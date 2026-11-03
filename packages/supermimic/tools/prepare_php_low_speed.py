"""Select complete PHP walk-command clips; preserve original motion/terrain pairs."""
from __future__ import annotations
from supermimic.paths import runtime_path

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=runtime_path('data/php296_surface_v2/task.json'))
    parser.add_argument('--output', type=Path, default=runtime_path('data/php_low_speed_surface_v2'))
    args = parser.parse_args()
    source, output = args.source.absolute(), args.output.absolute()
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    spec = json.loads(source.read_text())
    if spec['source_component'] != 'php_all_five_groups_296' or len(spec['tasks']) != 296:
        raise ValueError('Expected the canonical full PHP296 manifest')
    if spec['collision_profile'] != 'surface_v2':
        raise ValueError('Expected surface_v2 physics')
    entries = {}
    for manifest in spec['source_manifests']:
        if digest(manifest['path']) != manifest['sha256']:
            raise ValueError(f'Source manifest changed: {manifest["path"]}')
        for entry in json.loads(Path(manifest['path']).read_text())['motion_files']:
            key = (manifest['group'], entry['motion_id'])
            if key in entries:
                raise ValueError(f'Duplicate source identity: {key}')
            entries[key] = entry
    selected, decisions = [], []
    for index, original in enumerate(spec['tasks']):
        task = copy.deepcopy(original)
        entry = entries[(task['group'], task['source_motion_id'])]
        if entry['motion_sha256'] != task['source_motion_sha256']:
            raise ValueError(f'Source identity mismatch: {task["name"]}')
        name = Path(entry['source_file']).name
        if task['group'] == 'locomotion':
            # Reject unknown commands rather than silently interpreting them as walking.
            if not re.fullmatch(r'cfg\d+_(?:(?:stand|(?:run_)?(?:forward|left_45|left_90|right_45|right_90))_\d+(?:\.\d+)?s_)+motion\.npz', name):
                raise ValueError(f'Unknown locomotion command sequence: {name}')
        elif not ('walk' in name or name.startswith('run_')):
            raise ValueError(f'Unknown obstacle gait: {name}')
        keep = re.search(r'(?:^|_)run_', name) is None
        decisions.append(dict(name=task['name'], source_task_index=index,
                              source_motion_id=task['source_motion_id'], source_file=entry['source_file'],
                              selected=keep, reason='walk commands only' if keep else 'contains run commands'))
        if not keep:
            continue
        for field in ('csv', 'terrain_usd', 'source_motion', 'source_terrain'):
            if digest(task[field]) != task[field + '_sha256']:
                raise ValueError(f'Changed asset: {task[field]}')
        task.update(source_task_index=index, source_file=entry['source_file'])
        selected.append(task)
    counts = Counter(task['group'] for task in selected)
    if dict(counts) != dict(locomotion=13, lowstep=56, lowclimb=40):
        raise ValueError(f'Unexpected walk-only membership: {dict(counts)}')
    (output / 'motions').mkdir(parents=True)
    for task in selected:
        csv = output / 'motions' / Path(task['csv']).name
        csv.symlink_to(task['csv'])
        task['csv'] = str(csv)
        task['sampling_weight'] = 1. / counts[task['group']]
    result = copy.deepcopy(spec)
    result.update(source_component='php_walk_only_109', motion_dir=str(output / 'motions'),
                  tasks=selected, group_counts=dict(counts),
                  selection_source_manifest=str(source), selection_source_manifest_sha256=digest(source),
                  selection_contract='Whole clips with walking commands only; any run command excludes the entire clip. No duration or numeric speed threshold. Original terrain layout and source identities preserved.')
    report = dict(selected=len(selected), excluded=len(decisions)-len(selected), groups=dict(counts),
                  duration_seconds=sum(t['frames']/t['fps'] for t in selected), decisions=decisions)
    (output / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'task.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'decisions'}, indent=2))


if __name__ == '__main__':
    main()
