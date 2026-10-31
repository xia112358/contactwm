#!/usr/bin/env python3
"""Package explicitly supplied original12 inputs without rewriting evidence bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from build_data_release import build, verify_release

DESTINATION = 'runtime/current/workflow/original12'
TASK_FIELDS = ('csv', 'source_motion', 'terrain_usd', 'source_terrain')


def inventory(manifest, license_file, notice_file):
    manifest = manifest.expanduser().absolute()
    spec = json.loads(manifest.read_text())
    if (spec.get('source_unit') != 'original_database_clip' or spec.get('composed_trajectories') is not False
            or len(spec.get('tasks', [])) != 12):
        raise ValueError('Expected the original12 non-composed original-clip manifest')
    rows = {}
    def add(source, relative, expected=None):
        source = Path(source).expanduser().absolute()
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected and actual != expected:
            raise ValueError(f'Source checksum mismatch: {source}')
        destination = DESTINATION + '/' + relative
        row = dict(source=str(source), destination=destination, size=source.stat().st_size)
        if destination in rows and rows[destination] != row:
            raise ValueError(f'Duplicate destination with different source: {destination}')
        rows[destination] = row
    add(manifest, manifest.name)
    for task in spec['tasks']:
        for field in TASK_FIELDS:
            source = Path(task[field])
            if source.is_relative_to(manifest.parent):
                relative = source.relative_to(manifest.parent).as_posix()
            elif field == 'source_terrain':
                relative = 'upstream/terrain/' + source.parent.name + '/' + source.name
            else:
                raise ValueError(f'Unexpected external task input: {source}')
            add(source, relative, task[field + '_sha256'])
        selected = Path(spec['motion_dir']) / Path(task['csv']).name
        add(selected, selected.relative_to(manifest.parent).as_posix(), task['csv_sha256'])
    add(spec['source_catalogue'], 'provenance/catalogue.json', spec['source_catalogue_sha256'])
    add(spec['visualization_mjcf'], 'visualization/canonical_visual.xml', spec['visualization_mjcf_sha256'])
    add(license_file, 'upstream/LICENSE')
    add(notice_file, 'upstream/NOTICE')
    # Asset paths resolve through the destination checkout, whose own bootstrap
    # verifies the canonical robot. No robot, motion or terrain is substituted.
    suffix = '/src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf'
    old_checkout = spec['robot_asset']['urdf_path'].removesuffix(suffix)
    if old_checkout == spec['robot_asset']['urdf_path']:
        raise ValueError('Unknown canonical robot source path')
    aliases = [dict(source=str(manifest.parent), target=DESTINATION),
               dict(source=str(manifest.parent.resolve()), target=DESTINATION),
               dict(source=old_checkout, target='.')]
    if spec.get('somaforge_root') and spec['somaforge_root'] != old_checkout:
        aliases.append(dict(source=spec['somaforge_root'], target='.'))
    return dict(source_roots_aliases=aliases, components=[dict(id='original12',
        purpose='Twelve original step/climb<=76cm and walking references with paired terrain, canonical poses and provenance',
        status='kinematic_references_fresh_native_contact_collection_required', dependencies=[],
        provenance_notes=[
            'Original JSON/CSV/NPZ/USD/OBJ/XML bytes retained; manifest and input hashes unchanged.',
            'Catalogue is retained as provenance inventory; only the twelve selected task execution inputs are distributed.',
            'No policy, world-model checkpoint, contact recording, inferred labels or convergence claim is included.',
            'Source PHP clips and terrain retain their upstream Apache-2.0 license and notice.'],
        files=list(rows.values()))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--license', type=Path, required=True)
    parser.add_argument('--notice', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh release output directory')
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1]/'configs/original12_release.json')
    parser.add_argument('--repository', default='xia112358/contactwm')
    parser.add_argument('--tag', default='original12-20261031-v1')
    args = parser.parse_args()
    data = inventory(args.manifest, args.license, args.notice)
    # The build inventory is derived publication metadata, never source evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory_file = args.output.parent/(args.output.name + '.inventory.json')
    if inventory_file.exists():
        raise FileExistsError(f'Choose a fresh output name: {inventory_file}')
    inventory_file.write_text(json.dumps(data, indent=2)+'\n')
    catalog = build([inventory_file], args.output, args.repository, args.tag, args.catalog)
    verify_release(catalog, args.output)


if __name__ == '__main__':
    main()
