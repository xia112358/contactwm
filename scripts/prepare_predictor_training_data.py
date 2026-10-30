"""Build complete predictor caches from self-observed demonstration events."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from prepare_predictor_event_paths import prepare
from generator.support_path_acceptance import file_evidence, validate_support_paths
from generator.training_data import split_name, box_obbs
from generator.newton_query_supervision import validate_native_box_scene
from generator.fullbody_dataset import _resample_qpos
from somaforge_core.demonstration_events import SCHEMA as EVENT_SCHEMA, PARTS
from somaforge_core.contact_dataset import contact_dataset_fingerprint
from somaforge_core.motion_contracts import BODY_NAMES
from somaforge_core.robot_assets import decode_robot_asset_json, encode_robot_asset_json
from somaforge_core.newton_contact_data import load_contact_labels, require_current_newton_manifest
from somaforge_core.contact_face_selection import ground_top_catalog


def build_cache(manifest_path, output, phase_frames=64):
    manifest = json.loads(manifest_path.read_text())
    require_current_newton_manifest(manifest, context='soft joined predictor preprocessing')
    data = {name:[] for name in ('start_contacts','current_contacts','current_surfaces',
        'end_contact','target_contacts','target_surfaces','persistent_contact',
        'interaction_touchdown','box_origin','box_basis','box_edge_start','box_height')}
    samples = []; qs = []; geometry_errors = []
    for motion_id, entry in enumerate(manifest['motion_files']):
        contact = load_contact_labels(entry['motion_file'], entry['newton_contact_file'])
        plan = json.loads(Path(entry['edit_plan_file']).read_text())
        surfaces = [json.loads(x) for x in Path(plan['metadata']['target_surface_catalog']).read_text().splitlines() if x.strip()]
        selected = ground_top_catalog(surfaces)
        ground, top = selected[0], selected[1]
        if abs(float(ground['metadata']['ground_z'])) > 1.e-7:
            raise ValueError('Predictor box scene contract requires ground at z=0')
        polygon = np.asarray(top['metadata']['polygon_world'], dtype=np.float64)
        edge_u, edge_v = polygon[1]-polygon[0], polygon[2]-polygon[1]
        normal = np.asarray(top['normal'], dtype=np.float64)
        height = float(top['origin'][2])-float(ground['metadata']['ground_z'])
        world_rotation = np.stack((edge_u/np.linalg.norm(edge_u), edge_v/np.linalg.norm(edge_v), normal), -1)
        with np.load(entry['newton_contact_file'], allow_pickle=False) as labels:
            native_scene = json.loads(labels['contact_semantics_json'].item())['scene']
        geometry_errors.append(validate_native_box_scene(
            torch.tensor((polygon.mean(0)-.5*height*normal)[None]),
            torch.tensor(world_rotation[None]),
            torch.tensor([[.5*np.linalg.norm(edge_u), .5*np.linalg.norm(edge_v), .5*height]]),
            torch.tensor([float(ground['metadata']['ground_z'])]), native_scene))
        with np.load(entry['motion_file'],allow_pickle=False) as z:
            decode_robot_asset_json(z['robot_asset_json'],context=str(entry['motion_file']))
            q=z['joint_pos'].copy(); body_names=z['body_names'].astype(str).tolist()
            torso=body_names.index(BODY_NAMES[0]); pos=z['body_pos_w'][:,torso].copy()
            quat=z['body_quat_w'][:,torso].copy(); fps=float(z['fps'])
        events=[json.loads(x) for x in Path(entry['event_segments_file']).read_text().splitlines() if x.strip()]
        for event_index,e in enumerate(events):
            start,end=int(e['start_frame']),int(e['end_frame'])
            if not 0 <= start < end < len(q): raise ValueError('Invalid demonstration action interval')
            origin=pos[start].astype(np.float64)
            rotation=Rotation.from_quat(quat[start],scalar_first=True).as_matrix()
            yaw=np.arctan2(rotation[1,0],rotation[0,0]); basis=Rotation.from_euler('z',-yaw).as_matrix()
            local=q[start:end+1].astype(np.float64).copy()
            local[:,:3]=(local[:,:3]-origin)@basis.T
            local[:,3:7]=Rotation.from_matrix(basis@Rotation.from_quat(local[:,3:7],scalar_first=True).as_matrix()).as_quat(scalar_first=True)
            qs.append(_resample_qpos(local.astype(np.float32),phase_frames))
            active_start=contact['contact_part_mask'][start]; active_end=contact['contact_part_mask'][end]
            keep=np.zeros(6,bool); establish=np.zeros(6,bool)
            for name in e['persistent_parts']: keep[PARTS.index(name)]=True
            for touchdown in e.get('touchdown_events',[]): establish[int(touchdown['part_index'])]=True
            if np.any((keep|establish)&~active_end):
                raise ValueError('Observed event roles disagree with their own endpoint contact')
            if np.any(keep&establish):raise ValueError('Conflicting event roles')
            for name,value in (
                ('start_contacts',active_start),('end_contact',active_end),
                ('current_surfaces',contact['contact_surface'][start]),
                ('target_surfaces',contact['contact_surface'][end]),
                ('persistent_contact',keep),('interaction_touchdown',establish)):
                data[name].append(value.copy())
            for key, frame, active in (('current_contacts',start,active_start),('target_contacts',end,active_end)):
                points=(contact['contact_position_w'][frame]-origin)@basis.T
                points[~active]=0; data[key].append(points.astype(np.float32))
            world_basis=np.stack([top['tangent_u'],top['tangent_v'],top['normal']],axis=-1)
            data['box_origin'].append((basis@(np.asarray(top['origin'])-origin)).astype(np.float32))
            data['box_basis'].append((basis@world_basis).astype(np.float32))
            data['box_edge_start'].append(np.asarray([[v['u'],v['v']] for v in top['metadata']['polygon_surface_coordinates']],np.float32))
            data['box_height'].append(float(top['origin'][2])-float(ground['metadata']['ground_z']))
            samples.append(dict(motion_id=motion_id,current_frame=start,target_frame=end,
                event_index=event_index,segment_id=e['segment_id'],height=float(plan['metadata']['height_scale']),
                duration_s=(end-start)/fps))
    for key in data: data[key]=np.asarray(data[key])
    if not samples:raise ValueError('No demonstration action samples')
    box_obbs(data)
    certificate=validate_support_paths(manifest_path,samples)
    fingerprint=contact_dataset_fingerprint(manifest_path)
    summary=dict(schema='explicit_event_predictor_cache_v1',samples=len(samples),
        splits={name:sum(split_name(s['height']).startswith(name) for s in samples)
                for name in ('train','validation','test')},demonstration_event_evidence=certificate,
        contact_dataset_fingerprint=fingerprint,source_manifest=file_evidence(manifest_path),
        scene_geometry_validation=dict(schema='predictor_native_box_geometry_v1',
            verified_motions=len(geometry_errors), max_error_m=max(geometry_errors),
            face_selection='primary_horizontal_upward_faces_v1'))
    torch.save(dict(data=data,samples=samples,summary=summary,robot_asset_json=encode_robot_asset_json()),output/'data.pt')
    np.savez_compressed(output/'q_cache_64.npz',target_q=np.asarray(qs,np.float32),
        motion_ids=np.asarray([s['motion_id'] for s in samples],np.int64),
        current_frames=np.asarray([s['current_frame'] for s in samples],np.int64),
        target_frames=np.asarray([s['target_frame'] for s in samples],np.int64),
        robot_asset_json=encode_robot_asset_json(),contact_dataset_fingerprint=fingerprint)
    (output/'cache_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); collection=args.collection.absolute(); out=args.output.absolute()
    out.mkdir(parents=True,exist_ok=args.resume)
    index=json.loads((collection/'index.json').read_text())
    entries=index['motion_files']
    def certify(item):
        motion_id,entry=item
        if not entry['join_passed']:raise ValueError('Collection has an unverified soft connection; rebuild it before extracting labels')
        folder=out/'certificates'/entry['motion_id']
        if (folder/'manifest.json').exists():
            single=json.loads((folder/'manifest.json').read_text())
            validate_support_paths(folder/'manifest.json',[])
            row=single['motion_files'][0]
            for key,field in (('cropped_motion','motion_file'),('cropped_contacts','newton_contact_file')):
                if file_evidence(entry[key])['sha256'] != file_evidence(row[field])['sha256']:
                    raise ValueError('Resume evidence belongs to a different demonstration; use a new output directory')
            clock=[(int(e['start_frame']),int(e['end_frame'])) for line in
                Path(entry['cropped_events']).read_text().splitlines() if line.strip()
                for e in [json.loads(line)]]
            previous=[(int(e['start_frame']),int(e['end_frame'])) for line in
                Path(row['event_segments_file']).read_text().splitlines() if line.strip()
                for e in [json.loads(line)]]
            if clock != previous:
                raise ValueError('Resume action clock changed; use a new output directory')
            return motion_id,entry,single
        prepare(Path(entry['cropped_motion']),Path(entry['cropped_contacts']),
                Path(entry['cropped_events']),folder)
        return motion_id,entry,json.loads((folder/'manifest.json').read_text())
    certified={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(certify,x) for x in enumerate(entries)]):
            motion_id,entry,single=future.result();certified[motion_id]=(entry,single)
            print(json.dumps(dict(certified=len(certified),total=len(entries))),flush=True)
    declarations=[]; motions=[]; terrains={}; runtime=None
    for motion_id in range(len(entries)):
        entry,single=certified[motion_id]
        row=single['motion_files'][0]
        plan=json.loads(Path(entry['plan']).read_text())
        terrain_file=Path(plan['metadata']['target_terrain_mesh'])
        terrain_id=file_evidence(terrain_file)['sha256']
        terrains[terrain_id]=dict(terrain_id=terrain_id,terrain_file=str(terrain_file))
        row.update(motion_id=entry['motion_id'],terrain_id=terrain_id,edit_plan_file=entry['plan'])
        motions.append(row)
        declarations.append(dict(single['demonstration_event_evidence'][0],motion_id=motion_id))
        if runtime is not None and runtime!=single['newton_runtime']:raise ValueError('Mixed Newton runtimes')
        runtime=single['newton_runtime']
    manifest=dict(schema='observed_demonstration_predictor_training_v1',motion_files=motions,
        terrains=list(terrains.values()),demonstration_event_evidence=declarations,newton_runtime=runtime,
        event_contract=dict(schema=EVENT_SCHEMA,
            action_clock='demonstration_intervals_only',contact_targets='own_actual_Newton_observations',
            old_contact_requirements_used=False,quality_filtering=False,boundary_shifts=False),robot_asset_json=encode_robot_asset_json(),training_ready=False,
        experiment_scope='self_observed_demonstration_supervision; actual edited execution loads unknown',
        source_collection=str(collection),soft_joined=True,post_demo_rollouts=False)
    path=out/'manifest.json';path.write_text(json.dumps(manifest,indent=2))
    summary=build_cache(path,out)
    print(json.dumps(dict(samples=summary['samples'],splits=summary['splits'])),flush=True)


if __name__=='__main__': main()
