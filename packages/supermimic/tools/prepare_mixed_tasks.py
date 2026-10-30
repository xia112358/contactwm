"""Combine all prepared step/climb motions with canonical, obstacle-free walking."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import copy
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
pass
from prepare_lowstep import digest, write_terrain


def main():
    output = runtime_path('data/mixed_terrain_walk')
    if (output / 'task.json').exists():
        raise FileExistsError(output / 'task.json')
    sources = [('lowstep', 'lowstep_boxes_bounds_fixed'), ('lowclimb', 'lowclimb_boxes'),
               ('highclimb', 'highclimb_boxes')]
    inputs = [(group, runtime_path('data') / folder / 'task.json') for group, folder in sources]
    specs = [(group, json.loads(path.read_text())) for group, path in inputs]
    base = copy.deepcopy(specs[0][1])
    soma=workspace_root()
    pass
    from somaforge_core.published_data import resolve_published_path
    from somaforge_core.motion_schema import G1_29DOF_JOINT_ORDER, decode_kinematics_provenance
    from somaforge_core.robot_assets import decode_robot_asset_json
    source_manifest = soma / 'runtime/current/motions/php_wbt_v1_20261016/motions-locomotion/manifest.json'
    locomotion = json.loads(source_manifest.read_text())
    assert locomotion['robot_asset_sha256'] == base['robot_asset']['urdf_sha256']
    terrains = {t['terrain_id']: t for t in locomotion['terrains']}
    (output / 'motions').mkdir(parents=True, exist_ok=True)
    (output / 'terrains').mkdir(exist_ok=True)
    tasks = []
    for group, spec in specs:
        assert spec['robot_asset'] == base['robot_asset']
        for task in spec['tasks']:
            task = copy.deepcopy(task)
            task['group'] = group
            # Existing canonical CSVs remain byte-for-byte unchanged.
            tasks.append(task)
    selected = []
    for entry in locomotion['motion_files']:
        terrain = terrains[entry['terrain_id']]
        raw = resolve_published_path(terrain['source_file'])
        assert digest(raw) == terrain['source_sha256']
        boxes = np.load(raw, allow_pickle=False)
        # Published filename encodes behavior; exclude running and all obstacles.
        if len(boxes) or 'run_' in Path(entry['source_file']).name:
            continue
        source = source_manifest.parent / entry['motion_file']
        ground = source_manifest.parent / terrain['terrain_file']
        assert digest(source) == entry['motion_sha256']
        assert digest(ground) == terrain['terrain_sha256']
        with np.load(source, allow_pickle=False) as data:
            decode_robot_asset_json(data['robot_asset_json'], context=str(source))
            decode_kinematics_provenance(data['kinematics_provenance_json'], context=str(source))
            assert float(data['fps']) == 50
            q = data['joint_pos']
            names = data['joint_names'].tolist()
            joints = q[:, 7:][:, [names.index(n) for n in G1_29DOF_JOINT_ORDER]]
            values = np.concatenate((q[:, :3], q[:, [4, 5, 6, 3]], joints), axis=1)
            assert np.isfinite(values).all()
        name = 'flatwalk_' + entry['motion_id'].split('_')[-1]
        csv_path = output / 'motions' / (name + '.csv')
        np.savetxt(csv_path, values, delimiter=',', fmt='%.9g')
        tasks.append(dict(name=name, group='flatwalk', source_motion_id=entry['motion_id'],
                          source_motion=str(source), source_motion_sha256=digest(source),
                          csv=str(csv_path), csv_sha256=digest(csv_path),
                          source_terrain=str(ground), source_terrain_sha256=digest(ground),
                          boxes=[], frames=len(values), fps=50.0))
        selected.append(entry['motion_id'])
    assert len(selected) == 9, selected
    counts = Counter(t['group'] for t in tasks)
    shape_count = 1 + max(len(t['boxes']) for t in tasks)
    for i, task in enumerate(tasks):
        for field in ('source_motion', 'source_terrain', 'csv'):
            assert digest(task[field]) == task[field + '_sha256']
        terrain_path = output / 'terrains' / f'terrain_{i:04d}.usda'
        if terrain_path.exists():
            terrain_path.unlink()
        count = write_terrain(task['source_terrain'], terrain_path, shape_count, np.asarray(task['boxes']))
        assert count == 1 + len(task['boxes'])
        task.update(terrain_usd=str(terrain_path), terrain_usd_sha256=digest(terrain_path),
                    layout_padding_colliders=shape_count-count,
                    sampling_weight=1.0 / counts[task['group']])
    base.update(source_component='mixed_lowstep_lowclimb_highclimb_flatwalk',
                motion_dir=str(output / 'motions'), tasks=tasks,
                terrain_observation='local_heightmap_11x11_force6x3_heading_v1',
                contact_labels='runtime measured six-part total contact forces (normal + friction)',
                source_manifests=[dict(path=str(p), sha256=digest(p)) for _,p in inputs] +
                                 [dict(path=str(source_manifest), sha256=digest(source_manifest))],
                group_counts=dict(counts),
                sampling_contract='Equal group mass; uniform clip allocation within group; return-based reset bins within clip',
                flatwalk_selection=dict(source=str(source_manifest), obstacle_count=0,
                                        excluded_behavior='run_', motion_ids=selected))
    base.pop('source_manifest', None)
    base.pop('source_manifest_sha256', None)
    base['terrain_layout']['shapes_per_world'] = shape_count
    (output / 'task.json').write_text(json.dumps(base, indent=2)+'\n')
    print(json.dumps({'manifest':str(output/'task.json'), 'groups':dict(counts),
                      'frames':sum(t['frames'] for t in tasks), 'shapes_per_world':shape_count},indent=2))


if __name__ == '__main__':
    main()
