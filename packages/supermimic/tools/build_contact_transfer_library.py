"""Build a catalogue of complete contact-transfer episodes from fresh native runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-motions',type=int,default=4);p.add_argument('--min-transfers',type=int,default=64)
    a=p.parse_args()
    from supermimic.contact_source import require_original_contact_source
    capture_config=json.loads((a.capture/'config.json').read_text())
    require_original_contact_source(json.loads(Path(capture_config['task_manifest']).read_text()))
    from somaforge_core.robot_assets import validate_g1_asset_metadata
    from somaforge_core.collision_profiles import require_profile
    from supermimic.contact_control import ContactClock,PARTS
    from supermimic.contact_library import build_contact_clock,initial_contact_indices
    accepted=[];excluded=[];common=None;spans=[];offset=0
    for path in sorted(a.capture.glob('motion_*/rollout.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=False);meta=saved['metadata'];data=saved['data']
        asset=json.loads(meta['robot_asset_json']);validate_g1_asset_metadata(asset,context=str(path));require_profile(asset,'surface_v2',context=str(path))
        if meta.get('initialization_transfer') is not None:raise ValueError('Baseline library requires a freshly trained surface-v2 teacher')
        if not meta.get('successful_rollout') or data['term'].any() or not torch.isfinite(data['s1']).all():
            excluded.append(dict(motion=meta['motion'],reason='incomplete or failed native source motion'));continue
        physical={key:meta[key] for key in ('task_manifest_sha256','task_schema','robot_asset_json','observation_schema',
            'robot_collision_schema','force_contract','collision_profile','collision_profile_sha256',
            'terrain_bounding_volume_schema','world_model_schema','perception_contract','contact_timing','dt','state_quaternion') if key in meta}
        if common is not None and physical!=common:raise ValueError('Mixed physical/observation contracts in native library')
        common=physical
        spans.append(dict(motion=int(meta['motion']),begin=offset,end=offset+len(data['s']),source=str(path.resolve()),
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        accepted.append(data);offset+=len(data['s'])
    if not accepted:raise ValueError('No complete surface-native source motion is available')
    keys=set(accepted[0])
    if any(set(d)!=keys for d in accepted):raise ValueError('Incompatible native trajectory fields')
    data={key:torch.cat([d[key] for d in accepted]) for key in keys}
    metadata=dict(**common,schema='supermimic_native_contact_library_v1',motion=None,recording_spans=spans,
        source_checkpoint=json.loads((a.capture/'config.json').read_text())['checkpoint'],
        unit='one whole contact transfer episode, not one simulator timestep')
    library=dict(data=data,metadata=metadata);clock=build_contact_clock(library,hold=1,role_aware=True)
    cases=[]
    for i,(begin,end) in enumerate(zip(clock.bounds,clock.bounds[1:])):
        new=clock.target_masks[end-1]&~clock.keep_masks[end-1];keep=clock.keep_masks[end-1]
        initial=int(initial_contact_indices(clock,torch.tensor([begin]))[0])
        start=data['active'][initial]&data['known'][initial]
        delta=data['point'][end-1]-data['point'][initial]
        cases.append(dict(case=i+1,motion=int(clock.segment_motion[i]),source_frames=list(clock.segment_local_bounds[i]),
            pooled_frames=[begin,end],duration_s=(end-begin)*.02,
            kind='contact_transfer' if bool(new.any()) else 'support_motion',
            initial_support=[PARTS[j] for j in start.nonzero().flatten().tolist()],
            new_contacts=[PARTS[j] for j in new.nonzero().flatten().tolist()],
            maintained_contacts=[PARTS[j] for j in keep.nonzero().flatten().tolist()],
            target_points_w=data['point'][end-1,new].tolist(),
            target_height_m=data['point'][end-1,new,2].tolist(),
            same_part_displacement_m={PARTS[j]:delta[j].tolist() for j in (new&start).nonzero().flatten().tolist()},
            root_displacement_m=(data['s1'][end-1,:3]-data['s'][begin,:3]).tolist()))
    transfers=[c for c in cases if c['kind']=='contact_transfer']
    if len(spans)<a.min_motions or len(transfers)<a.min_transfers:
        raise ValueError(f'Whole-transfer coverage gate failed: {len(spans)} motions / {len(transfers)} transfers; expected at least {a.min_motions}/{a.min_transfers}')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(library,a.output/'rollout.pt')
    catalogue=dict(schema='native_whole_contact_transfer_catalogue_v1',motions=len(spans),whole_transfer_count=len(transfers),
        role_pattern_count=len({(tuple(c['initial_support']),tuple(c['new_contacts']),tuple(c['maintained_contacts'])) for c in transfers}),
        support_motion_count=len(cases)-len(transfers),case_count=len(cases),excluded=excluded,
        unit=metadata['unit'],recording_spans=spans,cases=cases)
    (a.output/'catalogue.json').write_text(json.dumps(catalogue,indent=2)+'\n')
    for name in ('solver-collision.mjb','solver-collision.mjb.json'):shutil.copyfile(a.capture/name,a.output/name)
    print(json.dumps({k:catalogue[k] for k in ('motions','whole_transfer_count','support_motion_count','case_count')},indent=2))


if __name__=='__main__':main()
