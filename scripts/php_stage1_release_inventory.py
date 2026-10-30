#!/usr/bin/env python3
"""Inventory the explicitly selected PHP Stage 1 checkpoint and native references.

Original files and embedded hashes are immutable. The release installer supplies
path aliases; publication never regenerates contacts or promotes readiness flags.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def build(root, php_checkout):
    root = root.resolve()
    runtime = (root / 'runtime/current').resolve()
    destination_root = Path('runtime/current/motions/php_stage1_20261025_v1')
    aliases = {str(root): '.', str(runtime): 'runtime/current'}
    components = {}

    def component(name, purpose, dependencies=()):
        result = dict(id=name, purpose=purpose, status='existing_experimental_stage1',
                      dependencies=list(dependencies), files=[], missing_inputs=[],
                      provenance_notes=['Original bytes retained; fixed-q native contact intentions are not dynamic support certification.'])
        components[name] = result
        return result

    def bind(source, target):
        source = source.absolute()
        for value in (str(source), str(source.resolve())):
            if value in aliases and aliases[value] != str(target):
                raise ValueError(f'Conflicting source alias: {value}')
            aliases[value] = str(target)

    destinations = {}
    def add(comp, source, target=None):
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if target is None:
            choices = sorted(aliases.items(), key=lambda row: len(Path(row[0]).parts), reverse=True)
            for old, new in choices:
                if source.is_relative_to(old):
                    target = Path(new) / source.relative_to(old)
                    break
        if target is None or not str(target).startswith('runtime/current/'):
            raise ValueError(f'No portable data destination for {source}')
        prior = destinations.setdefault(str(source), str(target))
        if prior != str(target):
            raise ValueError(f'Conflicting destination for {source}')
        if not any(r['source'] == str(source) for r in comp['files']):
            comp['files'].append(dict(source=str(source), destination=str(target), size=source.stat().st_size))

    def tree(comp, source):
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.lock', '.log'):
                add(comp, path)

    deps = component('php-stage1-runtime-inputs', 'Original low-climb teacher 03600, its 40-task manifest inputs, and upstream attribution')
    refs = component('php-stage1-native-references', '40 canonical native references, native point-pair labels, 775 action segments and provenance', [deps['id']])
    student = component('php-stage1-checkpoint', 'Unmodified student 07000 checkpoint, native configuration and 40-task evaluation reports', [refs['id']])
    prepared = root / 'tmp/php_action_segments_20261024/segments_all40_v1'
    bind(prepared, destination_root / 'prepared')
    tree(refs, prepared)
    manifest = json.loads((prepared / 'train.json').read_text())
    if len(manifest['motion_files']) != 40 or manifest['reference_sampling'] != 'fixed_q_forward_no_integration':
        raise ValueError('Unexpected Stage 1 reference identity')
    teacher = Path(manifest['teacher']['path'])
    add(deps, teacher)
    add(deps, teacher.parent / 'holosoma_config.yaml')
    original = Path(manifest['teacher']['training_manifest'])
    tree(deps, original.parent)
    original_data = json.loads(original.read_text())
    for row in original_data['motion_files'] + original_data['terrains']:
        if 'source_file' in row:
            source = Path(row['source_file'])
            add(deps, source if source.is_absolute() else original.parent / source)
    # Use the same upstream attribution source as the existing PHP data release.
    php = php_checkout.expanduser().resolve()
    for name in ('LICENSE', 'NOTICE', 'README.md', 'release-assets.json'):
        add(deps, php / name, destination_root / 'upstream' / name)

    source_manifest = Path(manifest['source_manifest'])
    bind(source_manifest.parent, destination_root / 'extraction')
    add(refs, source_manifest)
    for row in manifest['motion_files']:
        mid = int(row['motion_id'])
        motion = Path(row['motion_file'])
        bind(motion.parent, destination_root / 'native' / f'motion_{mid:06d}')
        add(refs, motion)
        with np.load(row['contact_file'], allow_pickle=False) as z:
            meta = json.loads(z['metadata_json'].item())
        for key in ('source_labels', 'events', 'evidence'):
            add(refs, Path(meta[key]))
        add(refs, Path(meta['alignment']['path']))

    model = root / 'runtime/current/models/php_continuous/low_climb_20261024_112654/PHPSegmentDistill/20261024_112701-low_climb_continuous_20261024_112654-locomotion'
    add(student, model / 'model_07000.pt')
    add(student, model / 'holosoma_config.yaml')
    evaluation = root / 'tmp/php_continuous_20261024'
    bind(evaluation, destination_root / 'evaluation')
    add(student, evaluation / 'stage1_7000_acceptance_20261025.json')
    for name in ('eval7000_20261025_v1', 'eval7000_20261025_contacts_v1'):
        add(student, evaluation / name / 'evaluation.json')
        add(student, evaluation / name / 'binding.json')
    return dict(schema='somaforge_data_release_inventory_v1', repo_root=str(root),
                source_roots_aliases=[dict(source=k, target=v) for k, v in sorted(aliases.items())],
                components=list(components.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--php-checkout', type=Path, default=Path.home() / 'php_parkour')
    args = parser.parse_args()
    inventory = build(args.root, args.php_checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(inventory, stream, indent=2)
        stream.write('\n')
    for c in inventory['components']:
        print(c['id'], len(c['files']), sum(f['size'] for f in c['files']))
