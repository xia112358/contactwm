"""Download verified pre-composition PHP databases and index original clips.

This prepares kinematic source data only. It never trains or treats source
contact hints as physical surface-v2 contacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import struct
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import numpy as np

from supermimic.paths import runtime_root, package_path

ROOT = runtime_root()
SOURCE_FILES = [
    'LICENSE',
    'motion_matching/motion_matching/scenarios/skill_registry.py',
    'motion_matching/motion_matching/scenarios/builtin.py',
    'motion_matching/motion_matching/core/database.py',
    'motion_matching/motion_matching/core/features.py',
    'motion_matching/motion_matching/metadata/constants.py',
    'motion_matching/motion_matching/utils/math_utils.py',
    'motion_matching/motion_matching/generate_database.py',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def download(url, path, size=None, digest=None):
    if path.exists():
        if size is not None and path.stat().st_size != size:
            raise ValueError(f'Existing file has wrong size; preserved: {path}')
        if digest is not None and sha(path) != digest:
            raise ValueError(f'Existing file has wrong hash; preserved: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    with urlopen(url, timeout=60) as response, temporary.open('xb') as stream:
        total = 0
        while chunk := response.read(1024*1024):
            total += len(chunk)
            if size is not None and total > size:
                raise ValueError('Download larger than declared release')
            stream.write(chunk)
    if (size is not None and total != size) or (digest is not None and sha(temporary) != digest):
        raise ValueError(f'Download validation failed; preserved: {temporary}')
    temporary.rename(path)


def relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or str(p) != value:
        raise ValueError(f'Unsafe release path: {value}')
    return Path(*p.parts)


def fetch_assets(pin, output):
    archive = output/'archives'/Path(pin['archive']['url']).name
    download(pin['archive']['url'], archive, pin['archive']['size'], pin['archive']['sha256'])
    expected = {v['path']: v for v in pin['files']}
    raw = output/'databases'
    with tarfile.open(archive, 'r:gz') as bundle:
        seen = set()
        for member in bundle:
            if member.isdir():
                continue
            path = relative(member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f'Unexpected release member: {member.name}')
            entry = expected[member.name]
            if member.size != entry['size']:
                raise ValueError(f'Release member size mismatch: {member.name}')
            target = raw/path
            if target.is_symlink():
                raise ValueError(f'Refusing symlinked raw asset: {target}')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source, dest)
            if target.stat().st_size != entry['size'] or sha(target) != entry['sha256']:
                raise ValueError(f'Raw asset checksum failed: {target}')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Missing declared release members')
    print(f'Verified {len(seen)} original database/terrain assets', flush=True)
    sources = []
    for name in SOURCE_FILES:
        target = output/'upstream'/name
        download(f'https://raw.githubusercontent.com/amazon-far/php_parkour/{pin["commit"]}/{name}', target)
        sources.append(dict(path=name, sha256=sha(target)))
    return raw, sources


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_database(path):
    """Exact published binary layout, with EOF/dimension/range validation."""
    with path.open('rb') as f:
        def integer():
            return struct.unpack('<I', f.read(4))[0]
        def array(dtype, components=None):
            shape = (integer(),) if components is None else (integer(), integer(), components)
            size = int(np.prod(shape))*np.dtype(dtype).itemsize
            if size > path.stat().st_size:
                raise ValueError('Invalid database dimensions')
            raw = f.read(size)
            if len(raw) != size:
                raise ValueError('Truncated database')
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        result = {key: array('<f4', dim) for key, dim in (
            ('bone_positions', 3), ('bone_velocities', 3),
            ('bone_rotations', 4), ('bone_angular_velocities', 3))}
        for key in ('bone_parents', 'range_starts', 'range_stops'):
            result[key] = array('<i4')
        result['source_contact_hints'] = array('u1', 1)[..., 0]
        if f.read(1):
            raise ValueError('Unexpected trailing database bytes')
    n, bones = result['bone_positions'].shape[:2]
    if bones != 31 or result['bone_parents'].shape != (bones,):
        raise ValueError('Expected original G1 database skeleton')
    for name in ('bone_positions', 'bone_velocities', 'bone_rotations', 'bone_angular_velocities'):
        if result[name].shape[:2] != (n, bones) or not np.isfinite(result[name]).all():
            raise ValueError(f'Invalid original bone array: {name}')
    starts, stops = result['range_starts'], result['range_stops']
    if len(starts) != len(stops) or not len(starts) or starts[0] != 0 or stops[-1] != n:
        raise ValueError('Original clip bounds do not cover database')
    if np.any(starts >= stops) or not np.array_equal(starts[1:], stops[:-1]):
        raise ValueError('Original clips overlap or leave gaps')
    if np.any(np.linalg.norm(result['bone_rotations'], axis=-1) < 1e-8):
        raise ValueError('Zero original quaternion')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data/php_originals_v1')
    p.add_argument('--download-only', action='store_true')
    a = p.parse_args()
    pin_path = package_path('experiments/php_original_release.json')
    pin = json.loads(pin_path.read_text())
    output = a.output.absolute(); output.mkdir(parents=True, exist_ok=True)
    raw, sources = fetch_assets(pin, output)
    write_json(output/'download-verification.json', dict(schema='php_original_download_v1',
        repository=pin['repository'], commit=pin['commit'], release_manifest_sha256=sha(pin_path),
        archive=pin['archive'], verified_files=len(pin['files']), upstream_sources=sources))
    if a.download_only:
        return
    # Export implementation is intentionally separate from the verified download.
    export_originals(output, pin)


def export_originals(output, pin):
    import re
    upstream = output/'upstream/motion_matching/motion_matching'
    registry = load_module(upstream/'scenarios/skill_registry.py', 'php_original_registry')
    # Joint names are declared alongside the official axes, avoiding a second
    # independently guessed robot ordering.
    axes_source = (upstream/'metadata/constants.py').read_text()
    names = re.findall(r'# \d+: (\w+_joint)', axes_source)
    if len(names) != 29:
        raise ValueError('Unsupported upstream joint ordering')
    bones = ['simulation_root', 'base_link', *names]
    mirror_names = [n.replace('left_', 'TMP_').replace('right_', 'left_').replace('TMP_', 'right_') for n in bones]
    mirror_order = np.array([bones.index(n) for n in mirror_names])
    exports = output/'original_clips'
    catalogue_path = output/'catalogue.json'
    if catalogue_path.exists() or exports.exists():
        raise FileExistsError('Original exports already exist; use a fresh output to rebuild')
    exports.mkdir()
    records, databases, windows, excluded = [], [], [], []
    by_database = {}
    for filename in sorted((output/'databases').glob('*.bin')):
        data = read_database(filename)
        starts, stops = data['range_starts'], data['range_stops']
        family = filename.stem.removeprefix('database_g1_')
        family_dir = exports/family; family_dir.mkdir()
        database_hash = sha(filename)
        unit_error = abs(np.linalg.norm(data['bone_rotations'], axis=-1)-1)
        database = dict(name=family, path=str(filename), sha256=database_hash,
                        frames=len(data['bone_positions']), fps=60.,
                        original_range_count=len(starts),
                        source_nonunit_quaternion_count=int((unit_error > 2e-4).sum()),
                        source_max_quaternion_norm_error=float(unit_error.max()))
        databases.append(database)
        family_records = []
        for index, (start, stop) in enumerate(zip(starts.tolist(), stops.tolist())):
            clip_id = f'{family}/range_{index:03d}'
            mirror_of = None; mirror_error = None
            if index % 2 == 1 and stop-start == stops[index-1]-starts[index-1]:
                prior = slice(int(starts[index-1]), int(stops[index-1]))
                mirrored_p = data['bone_positions'][prior][:, mirror_order]*[1,-1,1]
                mirrored_q = data['bone_rotations'][prior][:, mirror_order]*[1,-1,1,-1]
                actual_q = data['bone_rotations'][start:stop]
                pos_error = float(abs(mirrored_p-data['bone_positions'][start:stop]).max())
                rot_error = float(np.minimum(np.linalg.norm(mirrored_q-actual_q, axis=-1),
                                             np.linalg.norm(mirrored_q+actual_q, axis=-1)).max())
                mirror_error = dict(position_m=pos_error, quaternion_component_norm=rot_error)
                if pos_error < 2e-4 and rot_error < 2e-4:
                    mirror_of = family_records[-1]['clip_id']
            path = family_dir/f'range_{index:03d}.npz'
            metadata = dict(schema='php_original_database_clip_v1', source_unit='original_database_clip',
                clip_id=clip_id, source_database=str(filename), source_database_sha256=database_hash,
                original_frame_bounds=[start, stop], end_exclusive=True, fps=60.,
                mirrored_from=mirror_of, root_quaternion='wxyz',
                joint_names=names, bone_names=bones,
                contacts='source two-foot hints only; not native contact truth or force data',
                pose_format='unaltered original local bone hierarchy; not canonical training state',
                source_commit=pin['commit'], physical_contacts_collected=False,
                requires_native_canonicalization=True)
            values = {key: data[key][start:stop] for key in (
                'bone_positions', 'bone_velocities', 'bone_rotations', 'bone_angular_velocities', 'source_contact_hints')}
            np.savez_compressed(path, **values, bone_parents=data['bone_parents'],
                                original_frame_index=np.arange(start,stop), fps=np.asarray(60.),
                                metadata_json=np.asarray(json.dumps(metadata)))
            # Verify serialization reproduces every source array exactly.
            with np.load(path, allow_pickle=False) as saved:
                for key, value in values.items():
                    np.testing.assert_array_equal(saved[key], value)
                np.testing.assert_array_equal(saved['original_frame_index'], np.arange(start,stop))
            record = dict(clip_id=clip_id, database=family, database_sha256=database_hash,
                original_range_index=index, original_frame_bounds=[start,stop], frames=stop-start,
                fps=60., duration_s=(stop-start)/60., path=str(path), sha256=sha(path),
                mirrored_from=mirror_of, mirror_check=mirror_error,
                source_nonunit_quaternion_count=int((unit_error[start:stop]>2e-4).sum()))
            records.append(record); family_records.append(record)
        by_database[filename.name] = family_records
        print(f'Exported {family}: {len(family_records)} original ranges', flush=True)
    # Skill names are aliases/windows into original clips, not extra copies.
    # Invalid upstream ranges are exposed and excluded rather than clamped.
    for skill, cfg in registry.SKILL_REGISTRY.items():
        if skill == registry.LOCOMOTION_KEY:
            continue
        bin_name, _ = registry.resolve_paths(skill,cfg)
        ranges = by_database[bin_name]
        terrains = cfg.get('terrains', [])
        for variant, (start, search_end, end) in enumerate(cfg['search_ranges']):
            matches = [r for r in ranges if r['original_frame_bounds'][0] <= start <= search_end < end <= r['original_frame_bounds'][1]]
            if len(matches) != 1:
                excluded.append(dict(skill=skill, variant=variant, bounds=[start,search_end,end],
                                     reason='upstream search range outside a single original clip; not corrected silently'))
                continue
            r = matches[0]; i = r['original_range_index']
            terrain = output/'databases'/terrains[i] if i < len(terrains) else None
            if terrain is None or not terrain.is_file():
                raise ValueError(f'Missing original terrain for {skill}/{variant}')
            windows.append(dict(skill=skill, variant=variant, clip_id=r['clip_id'],
                original_frame_bounds=[start,end], entry_search_end=search_end,
                bounds_convention='registry end marker retained verbatim, not a new contact segment',
                terrain=str(terrain), terrain_sha256=sha(terrain),
                terrain_metadata=str(terrain.with_suffix('.json')), mirrored_from=r['mirrored_from']))
    catalogue = dict(schema='php_original_source_catalogue_v1', source_unit='original_database_clip',
        source_commit=pin['commit'], composed_trajectories=False, fps=60.,
        databases=databases, clips=records, skill_windows=windows, excluded_upstream_windows=excluded,
        original_clip_count=len(records), verified_mirror_count=sum(r['mirrored_from'] is not None for r in records),
        physical_contacts_collected=False,
        next_step='Canonicalize selected original clips with matching terrain; native contact collection before segmentation. Do not reuse composed rollout boundaries.')
    write_json(catalogue_path,catalogue)
    write_selection(output, catalogue)
    print(json.dumps(dict(databases=len(databases), original_ranges=len(records),
        verified_mirrors=catalogue['verified_mirror_count'],
        quarantined_windows=len(excluded), catalogue=str(catalogue_path)),indent=2),flush=True)


def write_selection(output, catalogue):
    records, windows = catalogue['clips'], catalogue['skill_windows']
    catalogue_path = output/'catalogue.json'
    # Preserve the user's current low-speed skill scope; all other originals are
    # downloaded for later use, without silently enabling high-speed training.
    selected = [r['clip_id'] for r in records if r['database'] in ('step','climb_76') and r['mirrored_from'] is None]
    locomotion = [r['clip_id'] for r in records if r['database'] == 'locomotion' and r['mirrored_from'] is None]
    mirrors = [r['clip_id'] for r in records if r['mirrored_from'] in selected]
    write_json(output/'low_speed_sources.json', dict(schema='php_original_source_selection_v1',
        source_unit='original_database_clip', catalogue=str(catalogue_path), catalogue_sha256=sha(catalogue_path),
        primary_clip_ids=selected, optional_mirror_clip_ids=mirrors,
        locomotion_context_clip_ids=locomotion,
        selected_skill_windows=[w for w in windows if w['skill'] in ('step','climb_76') and w['clip_id'] in selected],
        composed_trajectories=False, physical_contacts_collected=False,
        selection_note='Use registered original low-speed step/climb windows once. Locomotion is retained as a separate context library, not automatically admitted as low-speed contact tasks. Mirrors are explicit augmentation.'))


if __name__ == '__main__':
    main()
