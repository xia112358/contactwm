"""Locate force-bearing contacts and measure support motion from fresh rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from somaforge_core.newton_support import SUPPORT_SCHEMA, SUPPORT_FIELDS, select_support_contacts, support_motion
from somaforge_core.newton_contacts import PARTS
from somaforge_core.robot_assets import validate_g1_asset_metadata
from somaforge_core.newton_contact_data import require_current_newton_manifest
from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA


def analyze(path):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(z['_metadata_json'].item())
        validate_g1_asset_metadata(meta.get('robot_asset'),context=str(path))
        require_current_newton_manifest(meta,context=str(path))
        if meta.get('solver_support_semantics',{}).get('schema')!=SUPPORT_SCHEMA:
            raise ValueError('Missing same-solve support evidence; old net body force does not locate support points')
        if 'newton_scene_binding' not in meta:raise ValueError('Actual primary-surface binding unknown')
        if not all('solver_contact_'+field in z for field in SUPPORT_FIELDS):
            raise ValueError('Missing same-solve contact support channels')
        binding=meta['newton_scene_binding'];labels=meta['newton_body_labels'];records=[]
        counts=z['solver_contact_count'];motion_frames=z['motion_time_step']
        prefix_lengths=[]
        for env in range(meta['num_envs']):
            interruptions=np.flatnonzero(motion_frames[:,env]!=np.arange(len(motion_frames)))
            prefix_lengths.append(int(interruptions[0]) if len(interruptions) else len(motion_frames))
        full=max(prefix_lengths)
        complete=[length==full and not bool(z['terminated'][:length,env].any())
                  for env,length in enumerate(prefix_lengths)]
        channels={k[len('solver_contact_'):]:z[k] for k in z.files
                  if k.startswith('solver_contact_') and k!='solver_contact_count'}
        for t in range(len(counts)):
            n=int(counts[t]);s={key:value[t,:n] for key,value in channels.items()}
            s['count']=n
            selected=select_support_contacts(s,binding,labels)
            result=support_motion(s,eligible=selected['eligible'],robot_side=selected['robot_side'],
                                 groups=selected['groups'],group_count=meta['num_envs']*6)
            incoming=dict(s,point_velocity0_w=s['pre_point_velocity0_w'],point_velocity1_w=s['pre_point_velocity1_w'])
            before=support_motion(incoming,eligible=selected['eligible'],robot_side=selected['robot_side'],
                                  groups=selected['groups'],group_count=meta['num_envs']*6)
            speeds={int(i):float(v) for i,v in zip(result['loaded_contact_rows'],result['point_tangent_speed_m_s'])}
            for env in range(meta['num_envs']):
                for part,name in enumerate(PARTS):
                    g=env*6+part;rows=result['loaded_contact_rows'][selected['groups'][result['loaded_contact_rows']]==g]
                    records.append(dict(record=t,motion_frame=int(motion_frames[t,env]),env=env,part=name,
                        contact_activated=bool(result['contact_activated'][g]),
                        load_known=bool(result['load_known'][g]),load_state=str(result['load_state'][g]),
                        loaded_motion_known=bool(result['loaded_motion_known'][g]),slip_state=str(result['slip_state'][g]),
                        load_bearing=bool(result['load_bearing'][g]),normal_force_n=float(result['normal_force_n'][g]),
                        rms_tangent_speed_m_s=float(result['rms_tangent_speed_m_s'][g]) if result['load_bearing'][g] else None,
                        minimum_loaded_speed_m_s=float(result['minimum_loaded_speed_m_s'][g]) if result['load_bearing'][g] else None,
                        maximum_loaded_speed_m_s=float(result['maximum_loaded_speed_m_s'][g]) if result['load_bearing'][g] else None,
                        incoming_rms_tangent_speed_m_s=float(before['rms_tangent_speed_m_s'][g]) if before['load_bearing'][g] else None,
                        loaded_points=[dict(row=int(i),body=labels[int(s[f'body{selected["robot_side"][i]}'][i])],
                            surface=int(selected['surfaces'][i]),
                            position_w=s['position_w'][i].tolist(),tangent_speed_m_s=speeds[int(i)],
                            normal_force_n=float(np.dot(s['force_on_body1_w'][i],s['frame_w'][i,0]))) for i in rows]))
    summaries=[]
    for env in range(meta['num_envs']):
        for part in PARTS:
            known=[r for r in records if r['env']==env and r['part']==part and r['load_bearing']
                   and r['record']<prefix_lengths[env]]
            summaries.append(dict(env=env,part=part,loaded_records=len(known),
                source_episode_frames=prefix_lengths[env],complete_execution=complete[env],
                speed_median_cm_s=float(np.median([r['rms_tangent_speed_m_s'] for r in known])*100) if known else None,
                speed_p95_cm_s=float(np.quantile([r['rms_tangent_speed_m_s'] for r in known],.95)*100) if known else None,
                speed_p99_cm_s=float(np.quantile([r['rms_tangent_speed_m_s'] for r in known],.99)*100) if known else None))
    return dict(schema=SUPPORT_SCHEMA,support_assessment_schema=SUPPORT_ASSESSMENT_SCHEMA,
        recording=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sampling=meta['solver_support_semantics'],scene_fingerprint=binding['model_fingerprint'],
        summaries=summaries,records=records,training_ready=False,
        tolerance_status='source_speed_distribution_only_not_automatic_acceptance_threshold',
        recording_dt_s=meta['dt'],physics_dt_s=meta['sim_dt'],
        velocity_sampling='latest physics solve per recorded control interval; no full-substep slip integral',
        summary_scope='initial uninterrupted episode per environment; resets excluded, incomplete executions flagged',
        interpretation='Actual loaded solver application points; positive force localizes support within native eligible contacts. Resolved qvel at constraint-evaluation geometry, not paired with saved post-integration pose.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=analyze(args.recording)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['summaries'],indent=2))
