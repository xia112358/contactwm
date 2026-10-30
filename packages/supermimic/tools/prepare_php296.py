"""Prepare all five canonical PHP groups, preserving source motions and terrains."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
pass
from prepare_lowstep import digest,write_terrain

COMPONENTS={'locomotion':('motions-locomotion',100),'lowstep':('motions-low-step',56),
            'highstep':('motions-high-step',60),'lowclimb':('motions-low-climb-76',40),
            'highclimb':('motions-high-climb-76',40)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=runtime_path('data/php296_surface_v2'))
    args=parser.parse_args();output=args.output.absolute()
    if output.exists():raise FileExistsError(f'Choose a fresh output directory: {output}')
    base=copy.deepcopy(json.loads((runtime_path('data/lowclimb_boxes/task.json')).read_text()))
    soma=workspace_root();pass
    from somaforge_core.robot_assets import decode_robot_asset_json,validate_g1_asset_metadata
    from somaforge_core.motion_schema import G1_29DOF_JOINT_ORDER,decode_kinematics_provenance
    from somaforge_core.published_data import resolve_published_path
    tasks=[];sources=[]
    # Verify every source before writing generated data.
    for group,(component,expected) in COMPONENTS.items():
        manifest=soma/'runtime/current/motions/php_wbt_v1_20261016'/component/'manifest.json'
        spec=json.loads(manifest.read_text());entries=spec['motion_files']
        if len(entries)!=expected:raise ValueError(f'Unexpected {component} count')
        terrains={t['terrain_id']:t for t in spec['terrains']}
        sources.append(dict(group=group,path=str(manifest),sha256=digest(manifest)))
        for index,entry in enumerate(entries):
            terrain=terrains[entry['terrain_id']]
            motion=manifest.parent/entry['motion_file'];ground=manifest.parent/terrain['terrain_file']
            raw=resolve_published_path(terrain['source_file'])
            for path,sha in [(motion,entry['motion_sha256']),(ground,terrain['terrain_sha256']),(raw,terrain['source_sha256'])]:
                if digest(path)!=sha:raise ValueError(f'Source fingerprint mismatch: {path}')
            boxes=np.load(raw,allow_pickle=False)[:,terrain['php_variant']]
            if boxes.shape!=(len(boxes),10) or not np.isfinite(boxes).all():raise ValueError(f'Invalid boxes: {raw}')
            tasks.append(dict(name=f'{group}_{index:04d}',group=group,source_motion_id=entry['motion_id'],
                source_motion=str(motion),source_motion_sha256=digest(motion),
                source_terrain=str(ground),source_terrain_sha256=digest(ground),boxes=boxes.tolist()))
    (output/'motions').mkdir(parents=True);(output/'terrains').mkdir()
    slots=1+max(len(task['boxes']) for task in tasks)
    counts=Counter(task['group'] for task in tasks)
    for index,task in enumerate(tasks):
        source=Path(task['source_motion'])
        with np.load(source,allow_pickle=False) as data:
            asset=decode_robot_asset_json(data['robot_asset_json'],context=str(source))
            validate_g1_asset_metadata(asset,context=str(source))
            decode_kinematics_provenance(data['kinematics_provenance_json'],context=str(source))
            if float(data['fps'])!=50.:raise ValueError(f'Expected 50 Hz: {source}')
            names=data['joint_names'].tolist();q=data['joint_pos']
            joints=q[:,7:][:,[names.index(name) for name in G1_29DOF_JOINT_ORDER]]
            values=np.concatenate((q[:,:3],q[:,[4,5,6,3]],joints),axis=1)
            if values.shape[1]!=36 or not np.isfinite(values).all():raise ValueError(f'Invalid motion: {source}')
        csv=output/'motions'/(task['name']+'.csv');np.savetxt(csv,values,delimiter=',',fmt='%.9g')
        terrain=output/'terrains'/f'terrain_{index:04d}.usda'
        actual=write_terrain(task['source_terrain'],terrain,slots,np.asarray(task['boxes']))
        if actual!=1+len(task['boxes']):raise ValueError(f'Terrain components disagree: {terrain}')
        task.update(csv=str(csv),csv_sha256=digest(csv),terrain_usd=str(terrain),terrain_usd_sha256=digest(terrain),
                    layout_padding_colliders=slots-actual,frames=len(values),fps=50.,sampling_weight=1./counts[task['group']])
        if index%50==0:print(f'Prepared {index+1}/{len(tasks)}',flush=True)
    base.pop('source_manifest',None);base.pop('source_manifest_sha256',None)
    base.update(source_component='php_all_five_groups_296',motion_dir=str(output/'motions'),tasks=tasks,
        source_manifests=sources,group_counts=dict(counts),collision_profile='surface_v2',
        terrain_observation='local_heightmap_11x11_force6x3_heading_v1',
        contact_labels='runtime force6; source reference contacts are not reused as surface-v2 physical truth',
        sampling_contract='Equal group mass, uniform clip allocation within group; recent30 all-from-start failure retry at runtime',
        excluded_datasets=['LAFAN1'])
    base['terrain_layout']['shapes_per_world']=slots
    path=output/'task.json';path.write_text(json.dumps(base,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(path),groups=dict(counts),clips=len(tasks),frames=sum(t['frames'] for t in tasks),
        shapes_per_world=slots,excluded_datasets=base['excluded_datasets']),indent=2))


if __name__=='__main__':main()
