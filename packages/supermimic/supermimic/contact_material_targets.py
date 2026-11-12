"""Choose native foot poses at the slice's event-aligned target phase.

A representative contact row is not a representative whole-foot pose. Each
successful execution contributes one stable same-event pose at the target
phase; the existing whole-motion representative supplies the coherent rigid pose.
"""
import hashlib,json
from pathlib import Path
import torch
from .contact_material import CONTRACT,mesh_anchors,world_points
from .contact_placement import solver_part_meshes
from .motion import CONTROL_DT


def aligned_pose_index(contributor, target_frame):
    """Nearest qualified native tick on the existing event-aligned clock.

    Rows sharing a tick are one pose, not extra votes. Alignment transports
    timestamps only: never interpolate poses or contact coordinates.
    """
    return min(range(len(contributor['ticks'])),key=lambda k:(abs(contributor['aligned_frames'][k]-target_frame),contributor['ticks'][k],contributor['rows'][k]))


def pose_medoid(points):
    """Equal execution votes; choose a measured rigid pose, not averaged points."""
    distance=(points[:,None]-points[None,:]).square().sum(-1).mean(-1).sqrt()
    return int(distance.sum(-1).argmin()),distance


def representative_pose_index(candidates, carrier_trace):
    """Keep the population-selected whole trajectory, never splice its poses.

    The upstream motion medoid already selects the carrier using all successful
    executions. Per-event absolute-pose medoids would mix different footfalls.
    """
    return next(i for i,c in enumerate(candidates)
                if str(Path(c['source']).parent.name+'/'+Path(c['source']).name)==carrier_trace+'.pt')


def add_material_targets(slices,proof):
    case_index={int(c):i for i,c in enumerate(slices['case_ids'])}
    spans={s['trace_id']:s for s in slices['metadata']['recording_spans']}
    wanted=slices['target_masks']&~slices['keep_masks']
    targets=torch.zeros(len(case_index),6,4,3);seen=torch.zeros_like(wanted)
    sources={};captures={};audit=[];local=None;case_local=torch.zeros(len(case_index),6,4,3)
    for e in proof:
        i=case_index[e['case']];p=e['part']
        if p>=2:
            from .contact_material_consensus import associated_material_target
            anchor,target,provenance=associated_material_target(e)
            case_local[i,p]=anchor.expand(4,3);targets[i,p]=target.expand(4,3);seen[i,p]=True
            audit.append(dict(case=e['case'],part=p,event_id=e['event_id'],objective='material_matched_stable_contact',**provenance))
            continue
        # s1[goal] is the state after this control interval, on the carrier's clock.
        span=spans[int(slices['trace_ids'][i])]
        target_frame=int(slices['goals'][i])+1-span['begin']
        contributors=list(e['contributors'])
        carrier=span['trace']+'.pt'
        if not any(str(Path(c['source']).parent.name+'/'+Path(c['source']).name)==carrier for c in contributors):
            # Carrier can have a stable row before the event latch confirms it.
            capture=Path(e['contributors'][0]['source']).parent.parent/Path(span['trace']).parent
            source=capture/(Path(span['trace']).name+'.pt')
            raw=torch.load(source,map_location='cpu',weights_only=False)
            contract=json.loads((capture/'substeps/contract.json').read_text())
            world=contract['world_task_ids'].index(int(raw['data']['xi'][0,33]))
            S=round(CONTROL_DT/contract['dt']);ticks=[];rows=[];ready=False
            for frame in range(e['start'],e['stop']+1):
                substeps=torch.load(capture/'substeps'/f'{frame:06d}.pt',map_location='cpu',weights_only=False)['substeps']
                for sub,r in enumerate(substeps):
                    ready |= bool(r['support']['event_ready'][world,p])
                    eligible=(r['eligible']&r['active']&r['allocated']&(r['groups']==world*6+p)).nonzero().flatten()
                    if frame<e['stop'] and bool(r['features']['contact_stable'][world,p]) and len(eligible):
                        ticks.append(frame*S+sub);rows.append(int(eligible[0]))
            if not ticks or not ready:raise ValueError(f'Missing carrier stable evidence for case {e["case"]} part {p}')
            contributors.append(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),start=e['start'],stop=e['stop'],ticks=ticks,rows=rows,aligned_frames=[t/S for t in ticks],event_confirmed_later=True))
        if len(contributors)<2 or len({v['source'] for v in contributors})!=len(contributors):
            raise ValueError('Material pose consensus requires multiple distinct successful executions')
        candidates=[];points=[]
        for contributor in contributors:
            k=aligned_pose_index(contributor,target_frame);tick=contributor['ticks'][k];row=contributor['rows'][k]
            source=Path(contributor['source']);capture=source.parent
            if source not in sources:
                if hashlib.sha256(source.read_bytes()).hexdigest()!=contributor['source_sha256']:
                    raise ValueError('Native pose source hash mismatch')
                raw=torch.load(source,map_location='cpu',weights_only=False)
                if not raw['report']['successful_rollout'] or bool(raw['data']['term'].any()):
                    raise ValueError('Material pose source must be a successful whole execution')
                sources[source]=int(raw['data']['xi'][0,33])
            if capture not in captures:
                contract=json.loads((capture/'substeps/contract.json').read_text())
                anchors=torch.zeros(6,4,3)
                anchors[:2]=torch.stack([mesh_anchors(v) for v in solver_part_meshes(capture)[:2]])
                if local is None:local=anchors
                elif not torch.equal(local,anchors):raise ValueError('Mixed collision geometry in material targets')
                captures[capture]=contract
            contract=captures[capture];world=contract['world_task_ids'].index(sources[source])
            S=round(CONTROL_DT/contract['dt'])
            if not contributor['start']*S<=tick<contributor['stop']*S:
                raise ValueError('Material pose tick lies outside its aligned support event')
            r=torch.load(capture/'substeps'/f'{tick//S:06d}.pt',map_location='cpu',weights_only=False)['substeps'][tick%S]
            if not (r['features']['contact_stable'][world,p] and (r['support']['event_ready'][world,p] or contributor.get('event_confirmed_later',False))):
                raise ValueError('Material pose is not a qualified native support sample')
            if not (r['eligible'][row] and r['active'][row] and r['allocated'][row] and r['groups'][row]==world*6+p):
                raise ValueError('Material pose native-row identity mismatch')
            origin=torch.tensor(contract['world_origins'][world]);position=r['part_position'][world,p]-origin;rotation=r['part_rotation'][world,p]
            native_point=r['rows']['position_w'][row]-origin
            local_witness=rotation.T@(native_point-position)
            anchors_for_pose=local[p] if p<2 else local_witness.expand(4,3)
            points.append(world_points(position,rotation,anchors_for_pose))
            candidates.append(dict(source=str(source),source_sha256=contributor['source_sha256'],solver_tick=tick,native_row=row,
                aligned_frame=contributor['aligned_frames'][k],position=position.tolist(),rotation=rotation.tolist(),
                native_contact_point=native_point.tolist(),local_witness=local_witness.tolist(),event_confirmed_later=contributor.get('event_confirmed_later',False)))
        medoid,distances=pose_medoid(torch.stack(points))
        chosen=representative_pose_index(candidates,span['trace'])
        targets[i,p]=points[chosen];case_local[i,p]=local[p] if p<2 else torch.tensor(candidates[chosen]['local_witness']).expand(4,3);seen[i,p]=True
        audit.append(dict(case=e['case'],part=p,event_id=e['event_id'],target_aligned_frame=target_frame,
            objective='fixed_local_material_anchors_at_native_stable_witness',selection='population_whole_motion_representative_at_aligned_target_phase',
            carrier_trace=span['trace'],population_pose_medoid=medoid,
            contributors=candidates,selected=chosen,mean_pose_dispersion_m=float(distances[chosen].mean()),**candidates[chosen]))
    if not torch.equal(seen,wanted):raise ValueError('Missing or unexpected material target witnesses')
    goals=slices['fixed_goal_points'].clone()
    goals[:,2:]=torch.where(wanted[:,2:,None],targets[:,2:,0],goals[:,2:])
    return dict(slices,fixed_goal_points=goals,fixed_material_local=case_local,fixed_material_targets=targets,
                metadata=dict(slices['metadata'],material_goal_contract=CONTRACT,material_phase_status='invalid',
                    material_phase_reason='Reference segment boundary and same-material target consistency requires review')),audit
