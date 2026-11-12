"""Adapt successful substep captures to the shared population contact consensus."""
import hashlib,json
from pathlib import Path
import torch


def carrier_aligned_frames(source_frames, source_clock, carrier_clock):
    """Map source physical times through consensus onto the carrier's frames."""
    import numpy as np
    # Clocks end at the start of the last control frame, while valid solver
    # samples extend through its substeps. Extend one physical frame so those
    # samples are not silently collapsed to the previous frame boundary.
    source=np.r_[source_clock,source_clock[-1]+1.]
    carrier=np.r_[carrier_clock,carrier_clock[-1]+1.]
    return np.interp(source_frames,source,carrier)


def successful_source_pool(root):
    expected=None;contract=None;chosen={};attempts=[];seen=set()
    for capture in sorted(p for p in Path(root).glob('capture-seed*') if p.is_dir()):
        report=json.loads((capture/'report.json').read_text())
        identities={x['motion']:x['name'] for x in report['motions']}
        current={k:report[k] for k in ('checkpoint_sha256','task_manifest_sha256','stable_support_contract','substep_support_contract')}
        current['reference_provenance']=report.get('reference_provenance')
        if expected is None:expected=identities;contract=current
        if identities!=expected or current!=contract:raise ValueError('Mixed capture task/physics/support contracts')
        for item in report['motions']:
            path=capture/f"motion-{item['motion']:03d}.pt";saved=torch.load(path,map_location='cpu',weights_only=False);data=saved['data']
            complete=bool(item['successful_rollout']) and bool(saved['report']['successful_rollout'])
            finite=all(bool(torch.isfinite(data[k]).all()) for k in ('s','s1','a','a_prev','xi','next_xi'))
            continuous=bool((data['f'][1:]==data['f'][:-1]+1).all() and (data['ep_step'][1:]==data['ep_step'][:-1]+1).all())
            success=complete and finite and continuous and not bool(data['term'].any())
            entry=dict(motion=item['motion'],name=item['name'],source=str(path.resolve()),frames=len(data['s']),success=success,selected=success,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=dict(complete=complete,finite=finite,continuous=continuous,no_termination=not bool(data['term'].any())))
            if entry['source'] in seen:raise ValueError('Duplicate execution cannot receive multiple consensus votes')
            seen.add(entry['source'])
            attempts.append(entry)
            if entry['selected']:chosen.setdefault(item['motion'],[]).append(entry)
    if expected is None:raise ValueError('No native capture reports')
    missing=[expected[i] for i in expected if i not in chosen]
    if missing:raise ValueError(f'Missing successful complete source motions: {missing}')
    return dict(version='complete_successful_source_pool_v2',contract=contract,selected=[entry for rows in chosen.values() for entry in rows],attempts=attempts,
        policy='all verified successful whole trajectories contribute; never splice states across trials')


def aggregate_successful_event_evidence(pool, output):
    """Format adapter to existing somaforge_core event/point-cloud aggregation.

    Dynamics stay on an observed whole trajectory. All successful executions
    vote on its task events and contribute equally to matched target clouds.
    """
    import numpy as np
    from somaforge_core.observed_event_consensus import consensus_from_sequences
    from somaforge_core.contact_aggregation import equal_demo_patch_cloud
    from somaforge_core.consensus_event_alignment import align_observed_events
    from somaforge_core.staged_consensus import motion_medoid
    from .contact_placement import load_placement_evidence,associate_support_placements
    from .contact_temporal import segmentation_contact_state
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    groups={};capture_contract=None
    for entry in pool['selected']:groups.setdefault(entry['motion'],[]).append(entry)
    insufficient={rows[0]['name']:len(rows) for rows in groups.values() if len(rows)<3}
    if insufficient:raise ValueError(f'Need three complete executions per motion: {insufficient}')
    loaded={}
    for capture in sorted({str(Path(e['source']).parent) for e in pool['selected']}):
        capture=Path(capture);contract=json.loads((capture/'substeps/contract.json').read_text())
        identity={k:contract[k] for k in ('robot_asset','part_order','world_task_ids','terrain_faces','selection','dt','world_origins','body_labels')}
        if capture_contract is None:capture_contract=identity
        if identity!=capture_contract:raise ValueError('Cross-rollout contact face/asset/sampling identity mismatch')
        evidence=load_placement_evidence(capture);S=evidence['substeps']
        raw=torch.load(capture/'native-rows.pt',map_location='cpu',weights_only=False)
        positions=torch.stack([r['part_position'] for r in raw]);rotations=torch.stack([r['part_rotation'] for r in raw]);del raw
        for entry in (e for e in pool['selected'] if Path(e['source']).parent==capture):
            data=torch.load(entry['source'],map_location='cpu',weights_only=False)['data'];T=len(data['s']);world=evidence['world_task_ids'].index(entry['motion'])
            data.update(native_part_position=positions[:T,world],native_part_rotation=rotations[:T,world])
            placement,_=associate_support_placements(data,evidence,world);data.update(placement)
            temporal=segmentation_contact_state(data);sequences=[[] for _ in range(6)];clouds={}
            for ep in temporal['placements']:
                p,a,b=ep['part'],ep['start']*S,ep['stop']*S
                faces=torch.as_tensor(evidence['surfaces'][a:b,world,p]);actual=faces[faces>=0]
                if not len(actual):raise ValueError('Measured support placement lacks an unambiguous native terrain identity')
                face=int(actual.mode().values)
                row=dict(part=p,surface=face,kind='establish',interval=[ep['start'],ep['stop']],stable_evidence=[ep['confirmed'],ep['confirmed']+1])
                sequences[p].append(row)
                ticks=(evidence['qualified_points'][a:b,world,p]&(faces==face)).nonzero().flatten()+a
                points=[];point_ticks=[];point_rows=[]
                for tick in ticks.tolist():
                    sample=evidence['native_samples'][tick]
                    use=sample['eligible']&sample['active']&sample['allocated']&(sample['groups']==world*6+p)&(sample['surfaces']==face)
                    rows=use.nonzero().flatten()
                    points.extend((sample['rows']['position_w'][rows]-evidence['world_origins'][world]).tolist())
                    point_ticks.extend([tick]*len(rows));point_rows.extend(rows.tolist())
                clouds[(p,ep['start'],ep['stop'])]=dict(points=np.asarray(points,dtype=np.float32).reshape(-1,3),ticks=np.asarray(point_ticks),rows=point_rows)
            loaded[entry['source']]=dict(sequences=sequences,clouds=clouds,mask=data['substep_contact_active'].any(1).numpy(),
                q=torch.cat((data['s1'][:,:7],data['s1'][:,13:42]),-1).numpy(),substeps=S)
        print(json.dumps(dict(stage='consensus_evidence',capture=capture.name,sources=len(loaded))),flush=True)
        del evidence,positions,rotations
    selected=[];reports=[];all_clouds={}
    for motion,entries in sorted(groups.items()):
        records=[loaded[e['source']] for e in entries]
        consensus=consensus_from_sequences([r['sequences'] for r in records],[r['mask'] for r in records],stable_frames=1,max_dropout_frames=0)
        # The raw execution is a carrier, not the only source of contact evidence.
        _,distances=motion_medoid(np.stack([r['q'] for r in records]))
        missing=[sum(str(i) not in event['observations'] for event in consensus['events']) for i in range(len(entries))]
        representative=min(range(len(entries)),key=lambda i:(missing[i],float(distances[i].sum()),entries[i]['source_sha256']))
        # Preserve the existing event-clock alignment, with each actual source
        # interval as an anchor. Clocks transport evidence timestamps only;
        # poses/contact coordinates are never interpolated or relabelled.
        clock_reports=[dict(consensus_clock={str(e['id']):e['interval'] for e in consensus['events']},
            evidence_policy=consensus['evidence_policy'],checks=[dict(action=e['id'],kind='establish',part=e['part'],
                passed=str(i) in e['observations'],episode=e['observations'].get(str(i))) for e in consensus['events']])
            for i in range(len(entries))]
        clocks,alignment=align_observed_events(clock_reports,len(records[0]['q']),reference_index=representative)
        all_clouds[f'{motion}:alignment']=dict(source_frame_maps=clocks,report=alignment,sources=entries)
        targets=[]
        for event in consensus['events']:
            if str(representative) not in event['observations']:
                continue  # No measured placement on this execution to bind a target to.
            part=event['part'];obs=event['observations'][str(representative)];start,stop=obs['interval']
            contributors=[];clouds=[]
            for key,observed in event['observations'].items():
                i=int(key);a,b=observed['interval'];cloud=records[i]['clouds'][part,a,b]
                if not len(cloud['points']):continue
                contributors.append(dict(source=entries[i]['source'],source_sha256=entries[i]['source_sha256'],source_index=i,start=a,stop=b,ticks=cloud['ticks'].tolist(),rows=cloud['rows'],
                    aligned_frames=carrier_aligned_frames(cloud['ticks']/records[i]['substeps'],clocks[i],clocks[representative]).tolist(),
                    samples=len(cloud['points'])))
                clouds.append(cloud['points'])
            item=dict(part=part,start=start,stop=stop,event_id=event['id'],votes=event['votes'],source_count=len(entries),contributors=contributors)
            if len(clouds)<2:
                item.update(samples=0,reason='stable_target_requires_multiple_successful_executions');targets.append(item);continue
            aggregate=equal_demo_patch_cloud(clouds);chosen=aggregate['representative_index'];cloud=clouds[chosen]
            center=np.median(cloud,axis=0);point_index=int(np.linalg.norm(cloud-center,axis=1).argmin());point=cloud[point_index]
            witness=contributors[chosen];tick=witness['ticks'][point_index];S=records[witness['source_index']]['substeps']
            item.update(samples=sum(len(c) for c in clouds),point=point.tolist(),solver_tick=tick,source_frame=tick//S,substep=tick%S,native_row=witness['rows'][point_index],
                witness_source=witness['source'],witness_source_sha256=witness['source_sha256'],
                representative_cloud_index=chosen,center_dispersion_m=aggregate['center_dispersion_m'])
            all_clouds[f'{motion}:{event["id"]}']=dict(clouds=clouds,point_weights=aggregate['point_weights'],contributors=contributors)
            targets.append(item)
        chosen=dict(entries[representative],placement_target_evidence=targets)
        selected.append(chosen)
        reports.append(dict(motion=motion,name=chosen['name'],sources=entries,representative_index=representative,consensus=consensus,alignment=alignment,
            targets=targets,target_count=len(targets),unresolved_targets=sum(not t['samples'] for t in targets)))
    torch.save(all_clouds,output/'target-clouds.pt')
    (output/'event-consensus.json').write_text(json.dumps(reports,indent=2))
    result=dict(pool,version='successful_substep_population_consensus_v1',selected=selected,
        aggregation=dict(events='somaforge_core.observed_event_consensus.consensus_from_sequences',
            points='somaforge_core.contact_aggregation.equal_demo_patch_cloud',
            alignment='somaforge_core.consensus_event_alignment.align_observed_events; source physical time is mapped through the consensus clock onto carrier physical time, never interpolating contact points or poses',
            event_report=str(output/'event-consensus.json'),point_clouds=str(output/'target-clouds.pt')),
        policy='all successful executions vote equally; representative whole trajectory carries consensus-selected events; raw native contacts unchanged')
    return result
