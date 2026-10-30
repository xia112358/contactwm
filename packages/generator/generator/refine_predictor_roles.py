"""Role-head refinement from demonstrated roles and actual recursive inputs.

The encoder, location/region heads and pose executor remain frozen. The
parent's versioned validity contract applies during training and inference.
Test records never enter feature collection or selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from generator.predictor_runtime import (NativeQueries,Workers,evidence,dump,cpu_tree,
    audit_prediction,configured_acceptance)
from generator.predictor_architecture import load_position_predictor
from generator.full1000_training_variants import event_consistent_roles
from generator.role_validity import role_consistency_loss
from generator.role_validity import mask_role_logits
from generator.depth_observation import predictor_observation
from contact_solver.device_contact_objective import acceptance
from somaforge_core.robot_assets import decode_robot_asset_json
from somaforge_core.contact_face_selection import FACE_POLICY


@dataclass
class Config:
    baseline: Path = Path('baselines/predictor_v1_latest_20261010/baseline.json')
    corpus: Path = Path('runtime/current/motions/infiller_recursive_v2_20261011/corpus.pt')
    output: Path = Path('runtime/current/models/generator/predictor_role_refinement_20261011')
    report: Path = Path('tmp/predictor_role_refinement_20261011')
    steps: int = 400
    evaluation_every: int = 100
    batch_size: int = 128
    learning_rate: float = .0003
    seed: int = 20261011
    smoke: bool = False


def observation(records,query,device,model):
    q=torch.stack([r['q'][0] for r in records]).to(device)
    ids=torch.tensor([int(r['scene_id']) for r in records],device=device)
    actual=query.observed(q,ids)
    scene={k:torch.stack([r['scene'][k] for r in records]).numpy() for k in records[0]['scene']}
    obs=predictor_observation(model,q,actual,scene,ids,query.workers)
    return obs,actual


def role_objective(model,data,indices):
    logits=model.role_head(data['part'][indices])
    contact=data['contact'][indices]; demo=data['demo'][indices]
    validity=getattr(model,'role_validity',False)
    consistency=logits.sum((1,2))*0 if validity else role_consistency_loss(logits,contact)
    classified=mask_role_logits(logits,contact) if validity else logits
    ce=logits.sum((1,2))*0
    if bool(demo.any()):
        ce[demo]=F.cross_entropy(classified[demo].transpose(1,2),data['role'][indices][demo],reduction='none').mean(-1)
    # Retain the old relative preference among semantically possible roles.
    # Unknown recursive targets receive no invented next-event teacher label.
    valid=torch.ones_like(logits,dtype=torch.bool);valid[...,2]=contact
    old=data['old_logits'][indices].masked_fill(~valid,-torch.inf).softmax(-1)
    logp=logits.masked_fill(~valid,-torch.inf).log_softmax(-1).masked_fill(~valid,0.)
    distill=F.kl_div(logp,old,reduction='none').sum(-1).mean(-1)
    per_record=consistency+.1*distill+torch.where(demo,ce,0.)
    groups=[per_record[demo].mean()] if bool(demo.any()) else []
    if bool((~demo).any()): groups.append(per_record[~demo].mean())
    return torch.stack(groups).mean()


@torch.no_grad()
def endpoint_validation(model,records,query,device,limits,report):
    totals=Counter(); passed=Counter(); invalid_keep=Counter()
    for i,r in enumerate(records):
        obs,initial=observation([r],query,device,model)
        prediction=model(**obs)
        actual,audit=audit_prediction(model,query,prediction,obs['current_q'][0],initial,int(r['scene_id']),limits)
        group='demonstration' if r['kind']=='demonstration' else 'generated'
        ok=audit['accepted']
        if group=='demonstration':
            c=r['contract']
            task=acceptance(actual,c['end_contact'][None].to(device),c['end_surface'][None].to(device))
            ok=ok and bool(task['contact_accepted'][0])
        totals[group]+=1; passed[group]+=ok
        invalid_keep[group]+=int('keep_without_current_same_surface_contact' in audit['reasons'])
        if (i+1)%250==0:
            dump(report/'validation_progress.json',dict(evaluated=i+1,total=len(records)))
    return dict(totals=dict(totals),accepted=dict(passed),invalid_keep=dict(invalid_keep))


def run(cfg,device):
    if device=='cpu': raise ValueError('Actual Newton role observations require CUDA')
    if cfg.steps<1 or cfg.evaluation_every<1 or cfg.batch_size<2 or cfg.learning_rate<=0:
        raise ValueError('Invalid role refinement schedule')
    cfg.output.mkdir(parents=True,exist_ok=False);cfg.report.mkdir(parents=True,exist_ok=True)
    dump(cfg.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()})
    torch.set_num_threads(2);torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    baseline=json.loads(cfg.baseline.read_text());parent=baseline['checkpoint']
    if evidence(parent['path'])['sha256']!=parent['sha256']: raise ValueError('Parent Predictor changed')
    saved=torch.load(parent['path'],map_location='cpu',weights_only=False)
    model=load_position_predictor(saved,device=device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.role_head.parameters(): parameter.requires_grad_(True)
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('role_head.')}
    corpus=torch.load(cfg.corpus,map_location='cpu',weights_only=False)
    decode_robot_asset_json(corpus['robot_asset_json'],context='role refinement corpus')
    if corpus['predictor']!=evidence(parent['path']) or corpus['summary']['label_policy']!=FACE_POLICY:
        raise ValueError('Role corpus provenance differs')
    if not cfg.smoke and (not corpus.get('extension') or corpus['extension'].get('smoke')):
        raise ValueError('Formal role refinement requires complete unfiltered recursion')
    selected=[r for r in corpus['records'] if r['split'] in ('train','validation')]
    if cfg.smoke:
        selected=[r for split in ('train','validation') for kind in ('demonstration','predictor_recursive')
                  for r in [next(x for x in selected if x['split']==split and x['kind']==kind)]]
    train=torch.tensor([i for i,r in enumerate(selected) if r['split']=='train'],device=device)
    validation=torch.tensor([i for i,r in enumerate(selected) if r['split']=='validation'],device=device)
    deps=baseline['dependencies']
    for ref in deps.values():
        if evidence(ref['path'])['sha256']!=ref['sha256']: raise ValueError('Stale Newton/source dependency')
    payload=torch.load(deps['data_cache']['path'],map_location='cpu',weights_only=False)
    heights=sorted({round(float(r['height']),8) for r in payload['samples']});del payload
    with Workers(cfg.report/f'newton_{time.time_ns()}',heights,device,query_worlds=16,
        tensor_transport=True,solid_geometry=True,checkpoint=deps['query_checkpoint']['path'],
        source_manifest=deps['query_source_manifest']['path'],source_model=deps['query_source_model']['path'],
        depth_scan_profile=model.depth_scan_profile) as workers:
        if corpus['scene_fingerprints']!={str(e['scene_id']):e['fp'] for e in workers.entries.values()}:
            raise ValueError('Newton scenes changed')
        query=NativeQueries(workers,model.fk,128);cache=defaultdict(list)
        raw=cfg.report/'raw_current_contacts';raw.mkdir(exist_ok=False)
        torch.save(cpu_tree(dict(contact_truth=FACE_POLICY,sampling='same-state fixed-q queries',
            scenes={str(i):c.metadata for i,c in workers.tensor_clients.items()})),raw/'semantics.pt')
        with torch.no_grad():
            for first in range(0,len(selected),cfg.batch_size):
                rows=selected[first:first+cfg.batch_size];obs,actual=observation(rows,query,device,model)
                part=model._encode_points(obs['current_q'],obs['current_contact'],obs['current_anchor'],
                    obs.get('heightmap'),depth=obs)[4]
                cache['part'].append(part);cache['old_logits'].append(model.role_head(part))
                cache['contact'].append(obs['current_contact'])
                roles=torch.stack([r['contract']['role'] for r in rows]).long().to(device)
                contact=obs['current_contact']
                if model.role_validity:
                    metadata=obs
                    target_surface=torch.stack([r['contract']['end_surface'] for r in rows]).to(device)
                    contact=(metadata['current_contact_relations'] &
                        (metadata['contact_surface_ids'][:,None]==target_surface[...,None])).any(-1)
                cache['role'].append(event_consistent_roles(roles,contact))
                cache['demo'].append(torch.tensor([r['kind']=='demonstration' for r in rows],device=device))
                torch.save(dict(first=first,q=obs['current_q'].cpu(),observed=cpu_tree(actual)),raw/f'{first:06d}.pt')
        data={k:torch.cat(v).detach() for k,v in cache.items()}
        torch.save(dict(schema='frozen_encoder_role_features_v1',data=cpu_tree(data),
            rows=[dict(split=r['split'],kind=r['kind'],motion_id=r['motion_id'],sample_id=r['sample_id']) for r in selected],
            parent_predictor=evidence(parent['path']),corpus=evidence(cfg.corpus),robot_asset_json=saved['robot_asset_json']),
            cfg.report/'features.pt')
        val_records=[selected[i] for i in validation.tolist()]
        limits=configured_acceptance(saved['config'])
        initial_gate=endpoint_validation(model,val_records,query,device,limits,cfg.report)
        with torch.no_grad(): best=float(role_objective(model,data,validation))
        initial_score=best
        optimizer=torch.optim.Adam(model.role_head.parameters(),lr=cfg.learning_rate)
        metadata=dict(parent_predictor=evidence(parent['path']),corpus=evidence(cfg.corpus),
            training_config={k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
            training_scope=dict(Counter(selected[i]['kind'] for i in train.tolist())),
            validation_scope=dict(Counter(selected[i]['kind'] for i in validation.tolist())),
            trainable='role_head only',inference_mask=model.role_validity,test_used=False,smoke=cfg.smoke,
            selection='validation role loss; each actual endpoint acceptance group must not regress from parent',
            actual_contact_truth=FACE_POLICY,initial_gate=initial_gate,initial_score=initial_score)
        def save(name,step,score,gate):
            for k,v in frozen.items():
                if not torch.equal(v,model.state_dict()[k]): raise ValueError('Non-role parameters changed')
            torch.save(dict(saved,model=cpu_tree(model.state_dict()),step=step,
                refinement=dict(metadata,step=step,validation_score=score,validation_gate=gate),
                optimizer=cpu_tree(optimizer.state_dict())),cfg.output/name)
        save('best.pt',0,best,initial_gate)
        print(json.dumps(dict(step=0,score=best,gate=initial_gate)),flush=True)
        demo=train[data['demo'][train]];generated=train[~data['demo'][train]]
        if not len(demo) or not len(generated): raise ValueError('Missing role training cohort')
        for step in range(1,cfg.steps+1):
            n=cfg.batch_size//2
            ids=torch.cat((demo[torch.randint(len(demo),(n,),device=device)],
                generated[torch.randint(len(generated),(cfg.batch_size-n,),device=device)]))
            loss=role_objective(model,data,ids)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.role_head.parameters(),10,error_if_nonfinite=True);optimizer.step()
            if step%cfg.evaluation_every==0 or step==cfg.steps:
                with torch.no_grad(): score=float(role_objective(model,data,validation))
                gate=endpoint_validation(model,val_records,query,device,limits,cfg.report)
                eligible=all(gate['accepted'].get(k,0)>=v for k,v in initial_gate['accepted'].items())
                save('last.pt',step,score,gate)
                if eligible and score<best: best=score;save('best.pt',step,score,gate)
                row=dict(step=step,score=score,gate=gate,eligible=eligible,best=best)
                with (cfg.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        if evidence(parent['path'])['sha256']!=parent['sha256']: raise ValueError('Parent Predictor changed')
        dump(cfg.output/'progress.json',dict(status='complete',steps=cfg.steps,best_score=best,parent_unchanged=True))


def main():
    from somaforge_core.launcher import parse_isaaclab_launcher_args
    import tyro
    official = parse_isaaclab_launcher_args(__doc__)
    cfg = tyro.cli(Config)
    if (official.visualizer and not official.headless_explicit) or official.enable_cameras: raise ValueError('Role refinement is headless')
    run(cfg,official.device)


if __name__=='__main__':main()
