"""Partition freshly queried synthesized poses using aligned source support events.

This is a kinematic reference product. Source support qualification is retained
as provenance; no source force, action or dynamic-success flag is transferred.
"""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
import torch
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics
from somaforge_core.rollout_consensus import PART_LINKS
from somaforge_core.observed_event_consensus import episodes
from supermimic.contact_temporal import describe_support_episodes,TEMPORAL_CONTRACT
from supermimic.contact_source_motion import source_motion_analysis
from supermimic.stable_contact_slices import load_transfer_slices


def encode(value):
    if isinstance(value,torch.Tensor):return value.tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def pack(value,dtype=np.float32):
    return base64.b64encode(np.asarray(value,dtype=dtype).tobytes()).decode()


def partition(row,robot,fk):
    job=json.loads(Path(row['jobs']).read_text());case=job['cases'][0]
    directory=Path(job['output_dir'])/'aggregated'/case['key']
    result=json.loads((directory/'result.json').read_text())
    with np.load(result['motion']) as z:q=torch.tensor(z['joint_pos'].copy(),dtype=torch.float32)
    from somaforge_core.newton_contact_data import load_contact_labels
    labels=load_contact_labels(result['motion'],result['contacts'])
    active=torch.tensor(labels['contact_part_mask'],dtype=torch.bool)
    surface=torch.tensor(labels['contact_surface'],dtype=torch.long)
    points=labels['contact_position_w']
    T=len(q);known=torch.ones_like(active)
    pos,rot=fk.link_poses(q,PART_LINKS)
    # A pose-state view for the existing geometric participation/edge analysis.
    # Slots 7:13 are not velocities and are never exported as simulator state.
    state=torch.cat((q[:,:7],torch.zeros(T,6),q[:,7:]),-1)
    data=dict(s1=state,active=active,known=known,surface_local=surface,
              native_part_position=pos,native_part_rotation=rot)
    events=[];bindings=[];consensus=case['support_consensus'];by_id={e['id']:e for e in consensus['events']}
    for check in result['task_acceptance']['checks']:
        if not check['passed']:continue
        episode=check['episode'];a,b=map(int,episode['interval']);p=check['part']
        events.append(dict(part=p,start=a,stop=b,confirmed=a,surface=check['surface']))
        source=by_id[check['action']]
        bindings.append(dict(event_id=check['action'],part=p,start=a,stop=b,
            source_interval=source['interval'],source_votes=source['votes'],source_count=consensus['source_count'],
            qualification='measured source substep support; synthesized pose has actual native contact',
            source_observations=source['observations']))
    # Multiple qualified events may refer to the same still-present contact.
    events=list({(e['part'],e['start'],e['stop'],e['surface']):e for e in events}.values())
    speed=torch.zeros(T,6);angular=torch.zeros_like(speed)
    speed[1:]=(pos[1:]-pos[:-1]).norm(dim=-1)/TEMPORAL_CONTRACT['dt']
    relative=rot[1:]@rot[:-1].transpose(-1,-2)
    angular[1:]=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))/TEMPORAL_CONTRACT['dt']
    moving=(speed>TEMPORAL_CONTRACT['departure_speed_m_s'])|(angular>TEMPORAL_CONTRACT['departure_angular_speed_rad_s'])
    temporal=describe_support_episodes(data,events,active,known,torch.zeros_like(active),moving)
    analysis=source_motion_analysis(data,temporal,initial_bearing=temporal['support'])
    cuts,_=load_transfer_slices(data,temporal,analysis,initial_bearing=temporal['support'])
    for left,right in zip(cuts,cuts[1:]):
        if left['stop']!=right['start']:raise ValueError('Non-contiguous synthesized partition')
    covered=[]
    for cut in cuts:
        if cut['stop']<=cut['start']:raise ValueError('Empty synthesized segment')
        arrivals=[e for e in temporal['transfers'] if e['arrival'] is not None and cut['start']<=e['arrival']<cut['stop']]
        covered.extend(e['id'] for e in arrivals)
        cut['event_bindings']=[e for e in bindings if cut['start']<=e['start']<cut['stop']]
    expected=[e['id'] for e in temporal['transfers'] if e['arrival'] is not None and analysis['start']<=e['arrival']<analysis['stop']]
    if sorted(covered)!=sorted(expected):raise ValueError('Support establishment lost or duplicated by partition')
    assigned={(e['part'],e['surface'],e['start'],e['stop']) for e in events}
    extra=[e for sequence in episodes(active.numpy(),surface.numpy(),stable_frames=1,max_dropout_frames=0) for e in sequence
           if (e['part'],e['surface'],*e['interval']) not in assigned]
    sources=json.loads((Path(job['source_collection'])/'source-executions.json').read_text())
    old_index=next((i for i,e in enumerate(sources) if Path(e['source']).parent.name=='capture-seed17'),0)
    old=torch.load(sources[old_index]['source'],map_location='cpu',weights_only=False)['data']
    oldq=torch.cat((old['s1'][:,:7],old['s1'][:,13:42]),-1)
    with np.load(directory/'alignment.npz') as z:
        index=list(z['source_env_ids']).index(old_index);frame_map=z['source_frame_maps'][index].copy()
    def visual(pose,mask,contact):
        p,r=fk.link_poses(pose,robot['links'])
        return dict(T=len(pose),pos=pack(p),rot=pack(r),active=pack(mask,np.uint8),point=pack(contact))
    full=dict(motion=row['motion'],name=row['name'],frames=T,source_count=len(sources),
        synthesized=visual(q,active,points),source=visual(oldq,old['active'],old['point']),
        source_frame_map=frame_map.tolist(),source_path=sources[old_index]['source'],
        source_support=pack(old['substep_support_event_ready'].any(1),np.uint8),
        support=pack(temporal['support'],np.uint8),bindings=bindings,segments=cuts,
        edges=analysis['exclusions'],effective=[analysis['start'],analysis['stop']],
        unmatched_native_episodes=extra,result=str(directory/'result.json'),selected_stage=result['optimization']['selected_stage'],
        task_acceptance=result['task_acceptance'],dynamic_replay_verified=False,
        boxes=json.loads(Path(row['manifest']).read_text())['tasks'][0]['boxes'])
    receipt=dict(motion=row['motion'],name=row['name'],frames=T,segments=cuts,bindings=bindings,
        exclusions=analysis['exclusions'],effective=full['effective'],unmatched_native_episodes=extra,
        result=full['result'],result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        missing_events=result['task_acceptance']['missing_event_count'],partition_contiguous=True,
        event_coverage=len(covered),expected_events=len(expected),selected_stage=full['selected_stage'])
    return full,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--population',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--robot',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.population/'plan.json').read_text());robot=json.loads(a.robot.read_text());fk=BatchedG1ForwardKinematics()
    rows=[];next_id=1
    for row in plan['jobs']:
        full,receipt=partition(row,robot,fk)
        for cut in full['segments']:cut['case_id']=next_id;next_id+=1
        (a.output/f"motion-{row['motion']:03d}.json").write_text(json.dumps(full,default=encode,separators=(',',':')))
        rows.append(receipt);print(json.dumps(dict(motion=row['motion'],segments=len(full['segments']),missing=receipt['missing_events'],extra=len(full['unmatched_native_episodes']))),flush=True)
    report=dict(schema='synthesized_native_event_partition_v1',motions=rows,cases=next_id-1,policy_training_started=False,
        qualification='source measured substep support events aligned to freshly queried synthesized contacts',
        dynamic_forces_transferred=False,dynamic_replay_verified=False)
    (a.output/'partition.json').write_text(json.dumps(report,default=encode,indent=2))
    (a.output/'index.json').write_text(json.dumps([dict(motion=r['motion'],name=r['name'],segments=len(r['segments']),frames=r['frames']) for r in rows]))

if __name__=='__main__':main()
