"""Partition the previous source trajectories; policy success is diagnostic only."""
import argparse,csv,hashlib,json,collections
from pathlib import Path
from types import SimpleNamespace
import torch
from supermimic.contact_stability import NativeContactStability,CONTRACT
from supermimic.contact_placement import load_placement_evidence,associate_support_placements,PLACEMENT_CONTRACT
from supermimic.contact_phase import VERSION
from supermimic.contact_temporal import TEMPORAL_CONTRACT,segmentation_contact_state
from supermimic.stable_contact_slices import load_transfer_slices
from supermimic.contact_source_motion import source_motion_analysis,SOURCE_MOTION_CONTRACT
from supermimic.contact_topology import (ADMISSION,SCHEMA,TOPOLOGY_CONTRACT,ROLE_CONTRACT,PARTS,LIMBS,
    physical_intervals,support_topology,phase_ranges,classify_transition,mask_value,mask_name,annotate_surface_transitions,NAMES,native_contact_phases)
from somaforge_core.contact_system.evidence import ContactEvidence,Sampling


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recompute(capture):
    raw=torch.load(capture/'native-rows.pt',map_location='cpu',weights_only=False)
    metadata=json.loads(str(raw[0]['evidence']['evidence_metadata_json']));dt=metadata['sampling']['control_dt']
    support=NativeContactStability(dt);features=[];native_positions=[];native_rotations=[];active=[];known=[];attachments=[];witnesses=[];native_surfaces=[]
    for item in raw:
        saved=item['evidence'];meta=json.loads(str(saved['evidence_metadata_json']))
        e=ContactEvidence(saved,sampling=Sampling(**meta['sampling']),known=saved['known'],control_step=saved['control_step'],provenance=meta['provenance'])
        idx=item['view_rows'];groups=item['groups'];r=e.rows
        if not bool((e.active[idx]&e.allocated[idx]&item['eligible'][idx]).all()):raise ValueError('Stored primary view contains an invalid native row')
        if not torch.equal(r['worldid'][idx],groups//6):raise ValueError('Stored native world/part groups disagree')
        b0,b1=r['body0'][idx],r['body1'][idx]
        if not bool(((b0<0)^(b1<0)).all()):raise ValueError('This dataset requires its captured static-terrain contract')
        side=(b0<0).long();force=r['force_on_body1_w'][idx]*torch.where(side==1,1.,-1.)[:,None]
        view=SimpleNamespace(evidence=e,groups=groups,rows=idx,robot_side=side,force_w=force)
        native_positions.append(item['part_position']);native_rotations.append(item['part_rotation']);features.append(support.update(view,(item['part_position'],item['part_rotation']),torch.as_tensor(saved['control_step'])))
        active.append((torch.bincount(groups,minlength=e.known.numel())>0).reshape(e.known.shape)&e.known);known.append(e.known)
        terrain=torch.where(side==0,r['shape1'][idx],r['shape0'][idx]);sets=[set() for _ in range(e.known.numel())]
        for g,s in zip(groups[force[:,2]>0].tolist(),terrain[force[:,2]>0].tolist()):sets[g].add(s)
        allsets=[set() for _ in range(e.known.numel())]
        for g,shape in zip(groups.tolist(),terrain.tolist()):allsets[g].add(shape)
        native_surfaces.append([[tuple(sorted(allsets[w*6+p])) for p in range(6)] for w in range(len(e.known))])
        attachments.append([[tuple(sorted(sets[w*6+p])) for p in range(6)] for w in range(len(e.known))]);witnesses.append([[r['position_w'][idx[groups==w*6+p]] for p in range(6)] for w in range(len(e.known))])
    return {**{k:torch.stack([x[k] for x in features]) for k in features[0]},'native_part_position':torch.stack(native_positions),'native_part_rotation':torch.stack(native_rotations)},torch.stack(active),torch.stack(known),attachments,witnesses,native_surfaces,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--capture_root',type=Path,help='Explicit fresh capture directory; include every recorded motion, without transferring old restart outcomes');p.add_argument('--edge_exclusions',type=Path,help='Reviewed source-edge exclusions bound to source SHA256');a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);library=a.output/'library';library.mkdir()
    old=torch.load(a.previous/'library/rollout.pt',map_location='cpu',weights_only=False)
    reviewed_edges=json.loads(a.edge_exclusions.read_text()) if a.edge_exclusions is not None else old['metadata'].get('reviewed_edge_exclusions',[])
    applied_edges=[]
    capture_root=a.previous if a.capture_root is None else a.capture_root
    old_lookup={}
    if a.capture_root is None:
        selected_sources={str(Path(s['source']).resolve()) for s in old['metadata']['recording_spans']}
        old_catalogue=json.loads((a.previous/'library/catalogue.json').read_text());old_validation=json.loads((a.previous/'validation/report.json').read_text())['cases']
        for c,v in zip(old_catalogue['cases'],old_validation):
            span=next(s for s in old['metadata']['recording_spans'] if s['motion']==c['motion'])
            old_lookup[(str(Path(span['source']).resolve()),c['source_goal'],c['kind'])]=(c['case'],bool(v['passed']))
    else:
        selected_sources={str(p.resolve()) for p in capture_root.glob('capture-seed*/motion-*.pt')}
    chunks=[];spans=[];cases=[];history=[];previous=[];review=[];failures=[];phases=[];native_phases=[];traces=[];goal_proofs=[];offset=0;covered_frames=0;trimmed_frames=0
    checkpoint=torch.load(old['metadata']['source_checkpoint'],map_location='cpu',weights_only=False)
    hist=int(checkpoint['args']['hist']);expected_contract=old['metadata']['stable_support_contract']
    if expected_contract!=CONTRACT:raise ValueError('Rebuild requires the current support contract')
    for capture in sorted(capture_root.glob('capture-seed*')):
        if not capture.is_dir() or not any(Path(p).parent==capture.resolve() for p in selected_sources):continue
        report=json.loads((capture/'report.json').read_text())
        if report['stable_support_contract']!=CONTRACT or report['checkpoint_sha256']!=old['metadata']['source_checkpoint_sha256'] or report['task_manifest_sha256']!=old['metadata']['task_manifest_sha256']:raise ValueError('Mixed capture support/checkpoint contracts')
        q,truth,known,attachments,witnesses,native_surfaces,rawmeta=recompute(capture)
        placement_evidence=load_placement_evidence(capture) if 'substep_support_contract' in report else None
        for item in report['motions']:
            motion=int(item['motion']);path=capture/f'motion-{motion:03d}.pt'
            if str(path.resolve()) not in selected_sources:continue
            saved=torch.load(path,map_location='cpu',weights_only=False);data=saved['data'];T=len(data['s'])
            if not torch.equal(data['active'],truth[:T,motion]) or not torch.equal(data['known'],known[:T,motion]):raise ValueError('Captured summary disagrees with allocated primary native rows')
            cache_mismatch=int((data['contact_stable']!=q['contact_stable'][:T,motion]).sum())
            data={**data,**{k:v[:T,motion] for k,v in q.items()}}
            placement_audit=[]
            if placement_evidence is not None:
                placement_data,placement_audit=associate_support_placements(data,placement_evidence,motion);data.update(placement_data)
            intervals,valid,finite,physical=physical_intervals(data)
            trace_name=f'{capture.name}/motion-{motion:03d}';trace_info=dict(trace=trace_name,motion=motion,name=item['name'],source=str(path),source_sha256=sha(path),raw_rows=str(capture/'native-rows.pt'),collector_success=bool(item['successful_rollout']),collector_terminal=bool(data['term'][-1]),total_frames=T,valid_frames=int(valid.sum()),physical_invalid_frames=int((~physical).sum()),nonfinite_frames=int((~finite).sum()),valid_intervals=intervals,qualification_cache_mismatches=cache_mismatch,model_fingerprint=rawmeta['provenance']['model_fingerprint']);traces.append(trace_info)
            trace_info['placement_associations']=placement_audit
            if 'substep_support_contract' in report:trace_info['support_sampling_contract']=report['substep_support_contract']
            for t in (~valid).nonzero().flatten().tolist():review.append(dict(trace=trace_name,source_frame=t,reason='nonfinite_source' if not finite[t] else 'physical_fall_or_inversion',source=str(path)))
            bad=(~valid).nonzero().flatten().tolist();bad_runs=[]
            for frame in bad:
                if bad_runs and bad_runs[-1][1]==frame:bad_runs[-1][1]=frame+1
                else:bad_runs.append([frame,frame+1])
            topology_before_failure=support_topology(data,[x[motion] for x in attachments[:T]]) if bad_runs else None
            for lo_bad,hi_bad in bad_runs:
                entry=None if lo_bad==0 else topology_before_failure[lo_bad-1]['mask']
                failures.append(dict(trace=trace_name,motion=motion,source=str(path),source_start=lo_bad,source_stop=hi_bad,
                    kind='numerical_failure' if not bool(finite[lo_bad:hi_bad].all()) else 'physical_failure',
                    topology=dict(entry=mask_name(entry),exit='物理或数值失败',raw_contact_path=[mask_value(a&k) for a,k in zip(data['active'][lo_bad:hi_bad],data['known'][lo_bad:hi_bad])]),
                    context_start=max(0,lo_bad-hist),contact_supervision=False,
                    data={k:v[max(0,lo_bad-hist):hi_bad] for k,v in data.items()}))
            for lo,hi in intervals[1:]:
                if bool(valid[lo-1]):review.append(dict(trace=trace_name,source_frame=lo,reason='source_discontinuity_new_interval',source=str(path)))
            for run,(lo,hi) in enumerate(intervals):
                context_begin=max(0,lo-hist);prefix=lo-context_begin
                if not bool(finite[context_begin:lo].all()):raise ValueError('Recorded source history contains nonfinite context; it cannot be fabricated')
                d={k:v[lo:hi] for k,v in data.items()};surfaces=[x[motion] for x in attachments[lo:hi]];temporal=segmentation_contact_state(d);states=support_topology(d,surfaces,temporal);motion_analysis=source_motion_analysis(d,temporal);cuts,hits=load_transfer_slices(d,temporal,motion_analysis)
                for edge in reviewed_edges:
                    if edge['trace']!=trace_name or not (lo<=edge['start']<edge['stop']<=hi):continue
                    if edge['source_sha256']!=trace_info['source_sha256']:raise ValueError('Reviewed exclusion source has changed')
                    begin,end=edge['start']-lo,edge['stop']-lo
                    if begin==motion_analysis['start'] and end<motion_analysis['stop']:motion_analysis['start']=end
                    elif end==motion_analysis['stop'] and begin>motion_analysis['start']:motion_analysis['stop']=begin
                    else:raise ValueError('Reviewed exclusion must remove only an edge and retain a nonempty interval')
                    if not any(c['start']==begin and c['stop']==end for c in cuts):raise ValueError('Reviewed exclusion no longer matches a complete edge segment')
                    cuts=[c for c in cuts if c['stop']<=begin or c['start']>=end]
                    motion_analysis['exclusions'].append(dict(start=begin,stop=end,reason=edge['reason']))
                    applied_edges.append(edge)
                trimmed_frames+=len(d['s'])-(motion_analysis['stop']-motion_analysis['start'])
                trace_id=len(spans);span=dict(history_begin=offset+prefix+motion_analysis['start'],effective_source_frames=[lo+motion_analysis['start'],lo+motion_analysis['stop']],edge_exclusions=[dict(x,start=lo+x['start'],stop=lo+x['stop']) for x in motion_analysis['exclusions']],posture_phases=motion_analysis['posture_phases'],interior_flags=motion_analysis['interior_flags'],contact_episodes=temporal['episodes'],placement_episodes=temporal['placements'],placement_resumptions=temporal['resumptions'],transfer_events=temporal['transfers'],form_changes=temporal['form_changes'],trace_id=trace_id,trace=trace_name,run=run,motion=motion,name=item['name'],begin=offset,end=offset+hi-context_begin,source=str(path),source_frames=[context_begin,hi],supervised_source_frames=[lo+motion_analysis['start'],lo+motion_analysis['stop']],history_prefix_frames=prefix,source_sha256=sha(path),raw_rows=str(capture/'native-rows.pt'),model_fingerprint=trace_info['model_fingerprint']);spans.append(span)
                if 'substep_support_contract' in report:span['support_sampling_contract']=report['substep_support_contract']
                for phase in native_contact_phases(d,[x[motion] for x in native_surfaces[lo:hi]]):native_phases.append(dict(trace_id=trace_id,trace=trace_name,source_start=lo+phase['start'],source_stop=lo+phase['stop'],**dict(phase,start=prefix+phase['start'],stop=prefix+phase['stop'])))
                for phase in phase_ranges(states):phases.append(dict(trace_id=trace_id,trace=trace_name,source_start=lo+phase['start'],source_stop=lo+phase['stop'],**dict(phase,start=prefix+phase['start'],stop=prefix+phase['stop'])))
                logical=motion_analysis['start']
                for cut in cuts:
                    start,stop,goal=cut['start'],cut['stop'],cut['goal'];new=cut['new'];topology=classify_transition(states,d,logical,stop,new,temporal)
                    source_valid=bool(d['known'][start:stop].all());goal_valid=bool(cut['contact_goal_complete'] and new.any() and (d['contact_stable'][goal]|~new).all())
                    kind=cut['phase_kind'] if cut['phase_kind']!='contact_motion' else 'sustained_contact_transfer' if cut['arrivals'] else 'transfer_motion' if any(v=='T' for v in topology['limb_roles'].values()) else 'support_motion'
                    key=(str(path.resolve()),lo+goal,cut['kind']);diagnostic=old_lookup.get(key)
                    if start!=logical:raise ValueError('Trajectory partition must be contiguous without overlap or backtracking')
                    frame_supervised=bool((d['known'][start:stop]&d['active'][start:stop]&d['contact_stable'][start:stop]).any())
                    case=dict(merged_arrival_boundaries=cut['merged_arrival_boundaries'],transfer_ids=cut['transfer_ids'],boundary_reasons=cut['boundary_reasons'],arrival_frames=[lo+t for t,p in cut['arrivals']],arrival_parts=[p for t,p in cut['arrivals']],frame_point_supervision=frame_supervised,contact_goal_complete=cut['contact_goal_complete'],release_frames=[lo+t if t>=0 else -1 for t in cut['releases']],conversion_class=topology['mirror_role_key'],case=len(cases)+1,trace_id=trace_id,trace=trace_name,motion=motion,name=item['name'],start=offset+prefix+start,stop=offset+prefix+stop,goal=offset+prefix+goal,topology_floor=offset+prefix,source_start=lo+start,source_stop=lo+stop,source_goal=lo+goal,logical_start=lo+logical,context_overlap_frames=max(0,logical-start),kind=kind,base_kind=cut['kind'],new=new.tolist(),keep=cut['keep'].tolist(),support=cut['support'].tolist(),source_valid=source_valid,contact_goal_valid=goal_valid,contact_supervision=source_valid and goal_valid,training_role='contact_goal' if source_valid and goal_valid else 'frame_point_supervision' if frame_supervised else 'trajectory_only',collector_success=trace_info['collector_success'],collector_terminal=trace_info['collector_terminal'],restart_diagnostic=-1 if diagnostic is None else int(diagnostic[1]),previous_case=None if diagnostic is None else diagnostic[0],topology=topology)
                    cases.append(case);history.append(torch.arange(lo+start-hist,lo+start).clamp_min(lo+motion_analysis['start'])-context_begin+offset);previous.append(offset+prefix+max(0,start-1))
                    for part in (new&d['contact_stable'][goal]).nonzero().flatten().tolist():
                        distance=(witnesses[lo+goal][motion][part]-d['point'][goal,part]).norm(dim=-1).min()
                        if distance>1e-5:raise ValueError('Goal witness is not an actual native primary contact row')
                        goal_proofs.append(dict(case=case['case'],trace=trace_name,source_frame=lo+goal,part=PARTS[part],row_stable=bool(d['contact_row_stable'][goal,part]),load_stable=bool(d['contact_load_stable'][goal,part]),material_speed_m_s=float(d['contact_material_speed'][goal,part]),load_n=float(d['contact_load'][goal,part]),qualified=bool(d['contact_stable'][goal,part])))
                    covered_frames+=stop-logical;logical=stop
                if logical!=motion_analysis['stop']:raise ValueError('Logical cuts do not cover their valid source interval')
                chunk={k:v[context_begin:hi] for k,v in data.items()}
                chunk['source_context_only']=torch.arange(hi-context_begin)<prefix
                chunk['source_supervision_valid']=(torch.arange(hi-context_begin)>=prefix+motion_analysis['start'])&(torch.arange(hi-context_begin)<prefix+motion_analysis['stop'])
                chunk['source_physical_valid']=valid[context_begin:hi]
                chunks.append(chunk);offset+=hi-context_begin
        print(json.dumps(dict(stage='recomputed',capture=capture.name,traces=len(traces),cases=len(cases))),flush=True)
    if len(applied_edges)!=len(reviewed_edges):raise ValueError('Not all reviewed edge exclusions matched the source')
    if covered_frames+trimmed_frames!=sum(t['valid_frames'] for t in traces):raise ValueError('Valid motion frames were lost or duplicated in logical cuts')
    retired={'native_restart_report','native_restart_validation','source_library','source_slices_sha256','native_rows','raw_rows','boundary_policy','slice_contract','recording_spans'}
    metadata={k:v for k,v in old['metadata'].items() if k not in retired}
    metadata.update(reviewed_edge_exclusions=reviewed_edges,support_placement_contract=PLACEMENT_CONTRACT,source_motion_contract=SOURCE_MOTION_CONTRACT,recording_spans=spans,source_evidence_admission=ADMISSION,transition_topology=TOPOLOGY_CONTRACT,segmentation_temporal_contract=TEMPORAL_CONTRACT,part_role_contract=ROLE_CONTRACT,stable_support_contract=CONTRACT,contact_phase_version=VERSION,source_capture_policy='same recorded source traces as previous dataset; no new capture selection or tracking-success filter' if a.capture_root is None else 'all explicitly supplied fresh native capture traces; no tracking-success filter; previous restart outcomes do not apply',native_rows=[str(p/'native-rows.pt') for p in sorted(capture_root.glob('capture-seed*')) if p.is_dir()],boundary_policy='support_completion_partition_v10',slice_contract='native source-evidence admission; per-trace history; no tracking policy gate',previous_dataset=dict(path=str(a.previous),summary_sha256=sha(a.previous/'summary.json'),retired_filters=['whole_capture_success','tracking_restart_success']))
    data={k:torch.cat([x[k] for x in chunks]) for k in chunks[0]}
    data['point_loss_valid']=data['known']&data['active']&data['contact_stable']
    if not cases:raise ValueError('No trainable source intervals remain after explicit edge trimming')
    annotate_surface_transitions(data,spans,cases,phases);torch.save(dict(data=data,metadata=metadata),library/'rollout.pt')
    packed=dict(schema=SCHEMA,metadata=metadata,starts=torch.tensor([c['start'] for c in cases]),stops=torch.tensor([c['stop'] for c in cases]),goals=torch.tensor([c['goal'] for c in cases]),motion_ids=torch.tensor([c['motion'] for c in cases]),trace_ids=torch.tensor([c['trace_id'] for c in cases]),case_ids=torch.arange(1,len(cases)+1),history_indices=torch.stack(history),previous_contact_indices=torch.tensor(previous),target_masks=torch.tensor([c['new'] for c in cases])|torch.tensor([c['keep'] for c in cases]),keep_masks=torch.tensor([c['keep'] for c in cases]),support_masks=torch.tensor([c['support'] for c in cases]),source_valid=torch.tensor([c['source_valid'] for c in cases]),contact_goal_valid=torch.tensor([c['contact_goal_valid'] for c in cases]),restart_diagnostic=torch.tensor([c['restart_diagnostic'] for c in cases],dtype=torch.int8))
    torch.save(dict(schema='native_frame_point_supervision_v1',metadata=metadata,case_starts=packed['starts'],case_stops=packed['stops'],valid=data['point_loss_valid'],points=data['point']),library/'frame_point_supervision.pt')
    torch.save(packed,library/'cases.pt');admit=packed['source_valid']&packed['contact_goal_valid']
    frame_admit=torch.tensor([c['frame_point_supervision'] for c in cases])
    for name,mask in [('train',admit),('frame_supervised',frame_admit),('trajectory_only',~frame_admit),('no_endpoint_target',~admit)]:torch.save({k:v[mask] if isinstance(v,torch.Tensor) and v.shape[:1]==(len(cases),) else v for k,v in packed.items()},library/f'cases_{name}.pt')
    classes=collections.defaultdict(list);families=collections.Counter();graph=collections.Counter()
    for c in cases:
        t=c['topology'];classes[c['conversion_class']].append(c['case']);families[t['family']]+=1;graph[(t['entry'],t['exit'],c['conversion_class'])]+=1
    grouped=[dict(role_class=key,limb_roles=dict(zip(LIMBS,key)),count=len(ids),cases=ids) for key,ids in sorted(classes.items(),key=lambda x:-len(x[1]))]
    (library/'catalogue.json').write_text(json.dumps(dict(schema=SCHEMA,admission=ADMISSION,topology_contract=TOPOLOGY_CONTRACT,part_role_contract=ROLE_CONTRACT,cases=cases,traces=traces),ensure_ascii=False,indent=2));(a.output/'topology_classes.json').write_text(json.dumps(grouped,ensure_ascii=False,indent=2));(a.output/'topology_graph.json').write_text(json.dumps(dict(parts=dict(zip(PARTS,NAMES)),scope='six-part established support graph; detailed terrain attachments are per trace/model fingerprint',edges=[dict(source=k[0],target=k[1],family=k[2],count=n) for k,n in graph.items()],failure_edges=[dict(source=x['topology']['entry'],target=x['topology']['exit'],family=x['kind'],trace=x['trace'],source_frames=[x['source_start'],x['source_stop']]) for x in failures],phases=phases),ensure_ascii=False,indent=2));(a.output/'source_review.json').write_text(json.dumps(review,indent=2));(a.output/'goal_evidence.json').write_text(json.dumps(goal_proofs,indent=2))
    flat=[dict(case=c['case'],trace=c['trace'],motion=c['name'],kind=c['kind'],conversion_class=c['conversion_class'],role=c['training_role'],**c['topology']['limb_roles'],**{k+'_form':v for k,v in c['topology']['contact_forms'].items()},role_evidence_complete=c['topology']['role_evidence_complete'],family=c['topology']['family'],topology=c['topology']['topology_key'],mirror_class=c['topology']['mirror_topology_class'],support_modality=c['topology']['support_modality'],landing_surfaces=c['topology']['surface_pattern'],maintained_parts='+'.join(c['topology']['maintained_parts']),recontact_parts='+'.join(c['topology']['recontact_parts']),source_start=c['source_start'],source_stop=c['source_stop'],source_goal=c['source_goal'],logical_start=c['logical_start'],overlap_context_frames=c['context_overlap_frames'],goal_parts='+'.join(c['topology']['goal_parts']),surface_changed_parts='+'.join(c['topology']['surface_changed_parts']),tags=';'.join(c['topology']['tags']),collector_success=c['collector_success'],contact_goal_complete=c['contact_goal_complete'],restart_diagnostic=c['restart_diagnostic'],previous_case=c['previous_case']) for c in cases]
    with (a.output/'topology_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    (a.output/'native_contact_phases.json').write_text(json.dumps(dict(scope='exact native activation and terrain-shape attachments; includes weak and scraping contacts; no support-force gate',parts=dict(zip(PARTS,NAMES)),phases=native_phases),ensure_ascii=False,indent=2))
    recovered=[dict(previous_case=c['previous_case'],new_case=c['case'],role=c['training_role'],collector_success=c['collector_success']) for c in cases if c['restart_diagnostic']==0]
    torch.save(dict(schema='native_source_failure_intervals_v1',intervals=failures),a.output/'failure_intervals.pt')
    (a.output/'previous_restart_failures_retained.json').write_text(json.dumps(recovered,indent=2))
    summary=dict(trimmed_frames=trimmed_frames,edge_exclusions=[dict(trace_id=sp['trace_id'],name=sp['name'],**x) for sp in spans for x in sp['edge_exclusions']],effective_point_loss_valid_part_frames=int((data['point_loss_valid']&data['source_supervision_valid'][:,None]).sum()),source_motion_contract=SOURCE_MOTION_CONTRACT,point_loss_valid_part_frames=int(data['point_loss_valid'].sum()),segments_with_point_loss_frames=sum(bool(data['point_loss_valid'][c['start']:c['stop']].any()) for c in cases),point_loss_scope='per-frame validity retained across the full partition; endpoint-compatible cases_train is a separate legacy supervision view, not source admission',status='complete',training_started=False,motions=len({t['motion'] for t in traces}),source_traces=len(traces),collector_successful_traces=sum(t['collector_success'] for t in traces),collector_failed_traces=sum(not t['collector_success'] for t in traces),valid_frames=covered_frames,recording_frames=offset,recorded_history_prefix_frames=sum(x['history_prefix_frames'] for x in spans),logical_covered_frames=covered_frames,physical_or_numeric_invalid_frames=sum(x['source_stop']-x['source_start'] for x in failures),failure_intervals=len(failures),failure_classes=dict(collections.Counter(x['kind'] for x in failures)),cases=len(cases),contact_goal_cases=int(admit.sum()),trajectory_only_cases=int((~frame_admit).sum()),frame_supervised_cases=int(frame_admit.sum()),legacy_endpoint_ineligible_cases=int((~admit).sum()),conversion_classes=dict(collections.Counter(c['conversion_class'] for c in cases)),surface_relations=dict(collections.Counter(x['relation'] for c in cases for x in c['topology']['landing_surfaces'])),support_modalities=dict(collections.Counter(c['topology']['support_modality'] for c in cases)),structural_family_counts=dict(families),topology_classes=len(grouped),qualified_goal_parts=len(goal_proofs),native_contact_phases=len(native_phases),all_goal_parts_qualified=all(x['qualified'] for x in goal_proofs),source_intervals_from_collector_failures=sum(not c['collector_success'] for c in cases),contact_goals_from_collector_failures=sum(not c['collector_success'] and c['contact_supervision'] for c in cases),previous_restart_failures_retained=len(recovered),restart_failures_admitted_as_contact_goals=sum(c['restart_diagnostic']==0 and c['contact_supervision'] for c in cases),admission=ADMISSION,topology_contract=TOPOLOGY_CONTRACT,artifacts={p.name:dict(path=str(p),sha256=sha(p)) for p in library.glob('*.pt')})
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
