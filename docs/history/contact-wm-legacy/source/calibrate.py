"""Calibrate inherited SuperMimic force6 WM on actual PHP Newton transitions.

Holds out entire motions. Uses oracle terrain only to isolate dynamics mismatch;
this is not a deployable observation validation or policy optimization run.
"""
from dataclasses import dataclass
from pathlib import Path
from supermimic.paths import runtime_path
import json
import torch
import numpy as np
from holosoma.contact_wm.supermimic import SuperMimicForceWM
from holosoma.contact_wm.core import fit_loss

@dataclass(frozen=True)
class Options:
    recording:Path
    output_dir:Path
    checkpoint:Path=runtime_path('logs/climb_low_force6_resume1000/ckpt_006000.pt')
    rotations:Path|None=None
    steps:int=200
    batch:int=64
    device:str='cpu'
    seed:int=7
    additional_recordings:tuple[Path,...]=()
    resume:Path|None=None
    hard_fraction:float=.25
    refresh_priority_every:int=50
    learning_rate_scale:float=1.
    recent_recordings:tuple[Path,...]=()
    wm_window:int=1
    sequence_batch:int=16
    sequence_weight:float=.5


def main():
    import tyro
    o=tyro.cli(Options);torch.manual_seed(o.seed);torch.set_num_threads(1)
    if not 0<=o.hard_fraction<=.5 or o.refresh_priority_every<1 or o.learning_rate_scale<=0:raise ValueError('Invalid replay settings')
    out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    keys=('state','action','next_state','active','known','force_w','heights121','previous_force_w','previous_known','observation','transition_valid')
    train=[];validation=[];train_ids=[];validation_ids=[];validation_names=[];sequence_train=[];sequence_validation=[]
    recent_roots={p.resolve() for p in o.recent_recordings}
    if o.wm_window<1:raise ValueError('Positive WM window required')
    directories=[directory for root in (o.recording,*o.additional_recordings) for directory in sorted(root.glob('motion_*'))]
    if len({p.resolve() for p in directories})!=len(directories):raise ValueError('Duplicate replay recordings')
    for directory in directories:
        meta=json.loads((directory/'report.json').read_text());z=np.load(directory/'transitions.npz',allow_pickle=False)
        if meta['schema']!='contact_wm_native_transitions_v1' or meta['control_dt']!=.02:raise ValueError('Unexpected recording semantics')
        row={k:torch.from_numpy(z[k].copy()) for k in keys}
        row['switch']=(row['active']!=row['observation'][:,:6].bool()).any(-1)
        row['transition_valid'] &= row['previous_known'].all(-1)
        row['recent']=torch.full_like(row['transition_valid'],directory.parent.resolve() in recent_roots)
        if o.wm_window>1 and len(row['state'])>=o.wm_window:
            indices=torch.arange(len(row['state'])-o.wm_window+1)[:,None]+torch.arange(o.wm_window)[None]
            indices=indices[row['transition_valid'][indices].all(-1)]
            if len(indices):
                windows={k:v[indices] for k,v in row.items() if k!='observation'}
                (sequence_validation if meta['motion_index'] in (0,7) else sequence_train).append(windows)
        row={k:v[row['transition_valid']] for k,v in row.items()}
        if meta['motion_index'] in (0,7):
            validation.append(row);validation_ids.append(meta['motion_index']);validation_names.append(str(directory.resolve()))
        else:train.append(row);train_ids.append(meta['motion_index'])
    if not train or not validation:raise ValueError('Need separate training and held-out motions')
    merge=lambda rows:{k:torch.cat([r[k] for r in rows]).to(o.device) for k in rows[0]}
    validation_sets={name:{k:v.to(o.device) for k,v in row.items()} for name,row in zip(validation_names,validation)}
    train=merge(train);validation=merge(validation)
    seq_train=merge(sequence_train) if sequence_train else None
    seq_validation=merge(sequence_validation) if sequence_validation else None
    if o.wm_window>1 and (seq_train is None or seq_validation is None):raise ValueError('No valid multi-step training/validation windows')
    m=SuperMimicForceWM(o.rotations);source=m.load_source(o.checkpoint);m.to(o.device)
    optimizer=torch.optim.Adam([{'params':m.net.parameters(),'lr':1e-5},
        {'params':m.contact_head.parameters(),'lr':1e-4},{'params':m.activation_head.parameters(),'lr':1e-4}])
    iteration=0
    if o.resume is not None:
        previous=torch.load(o.resume,map_location=o.device,weights_only=False)
        if previous['schema']!='contact_wm_supermimic_calibration_v1':raise ValueError('Unsupported resume checkpoint')
        m.load_state_dict(previous['wm'],strict=True);optimizer.load_state_dict(previous['optimizer'])
        iteration=int(previous['iteration']);torch.set_rng_state(previous['torch_rng_state'].cpu())
        if o.device.startswith('cuda') and previous['cuda_rng_state']:
            torch.cuda.set_rng_state_all([x.cpu() for x in previous['cuda_rng_state']])
    for group in optimizer.param_groups:group['lr']*=o.learning_rate_scale
    def predict(b):
        physics=b['state'].new_zeros(len(b['state']),33);physics[:,0]=(1.-.75)/.45
        return m.source_step(b['state'],b['action'],physics,b['heights121'],b['previous_force_w'])
    def sequence_prediction(b):
        state=b['state'][:,0];force=b['previous_force_w'][:,0];loss=state.sum()*0
        physics=state.new_zeros(len(state),33);physics[:,0]=(1.-.75)/.45
        for t in range(o.wm_window):
            prediction=m.source_step(state,b['action'][:,t],physics,b['heights121'][:,t],force)
            loss=loss+fit_loss(prediction,b['next_state'][:,t],b['active'][:,t],b['force_w'][:,t],b['known'][:,t],b['transition_valid'][:,t],force_scale=200.)
            state=prediction['state'];force=prediction['force_w']
        return loss/o.wm_window,state,force
    @torch.no_grad()
    def evaluate_sequence():
        if seq_validation is None:return None
        indices=torch.linspace(0,len(seq_validation['state'])-1,min(128,len(seq_validation['state'])),device=o.device).long()
        b={k:v[indices] for k,v in seq_validation.items()};loss,state,force=sequence_prediction(b)
        return dict(samples=len(indices),window=o.wm_window,loss=float(loss),
            final_joint_rmse_rad=float((state[:,13:42]-b['next_state'][:,-1,13:42]).square().mean().sqrt()),
            final_position_error_m=float((state[:,:3]-b['next_state'][:,-1,:3]).norm(dim=-1).mean()))
    @torch.no_grad()
    def evaluate(data):
        collected=[]
        for start in range(0,len(data['state']),64):
            b={k:v[start:start+64] for k,v in data.items()};p=predict(b)
            known=b['known'];observed=(p['contact']>=.5)
            actual=b['active'];force_delta=torch.where(known[...,None],p['force_w']-b['force_w'],0.)
            force_error=force_delta.square().sum((-1,-2))/(3*known.sum(-1)).clamp_min(1)
            collected.append(dict(position=(p['state'][:,:3]-b['next_state'][:,:3]).norm(dim=-1),
                joint=(p['state'][:,13:42]-b['next_state'][:,13:42]).square().mean(-1).sqrt(),force=force_error,
                tp=((observed&actual)&known).sum(-1),fn=((~observed&actual)&known).sum(-1),
                tn=((~observed&~actual)&known).sum(-1),fp=((observed&~actual)&known).sum(-1),switch=b['switch']))
        v={k:torch.cat([r[k] for r in collected]) for k in collected[0]}
        def subset(mask):
            if not mask.any():return None
            r={k:x[mask] for k,x in v.items()};recall=r['tp'].sum()/(r['tp'].sum()+r['fn'].sum()).clamp_min(1)
            specificity=r['tn'].sum()/(r['tn'].sum()+r['fp'].sum()).clamp_min(1)
            return dict(samples=int(mask.sum()),position_error_m=float(r['position'].mean()),joint_rmse_rad=float(r['joint'].mean()),
                force_rmse_N=float(r['force'].mean().sqrt()),activation_recall=float(recall),activation_specificity=float(specificity))
        return dict(all=subset(torch.ones_like(v['switch'])),switch=subset(v['switch']),steady=subset(~v['switch']))
    before=evaluate(validation)
    sequence_before=evaluate_sequence()
    before_by_recording={name:evaluate(row) for name,row in validation_sets.items()}
    switches=train['switch'].nonzero().flatten()
    recent=train['recent'].nonzero().flatten()
    @torch.no_grad()
    def hard_examples():
        scores=[]
        for start in range(0,len(train['state']),64):
            b={k:v[start:start+64] for k,v in train.items()};p=predict(b)
            joint=(p['state'][:,13:42]-b['next_state'][:,13:42]).square().mean(-1)/.01**2
            error=torch.where(b['known'][...,None],p['force_w']-b['force_w'],0.)
            force=error.square().sum((-1,-2))/(3*b['known'].sum(-1)).clamp_min(1)/200.**2
            scores.append(joint+force)
        scores=torch.cat(scores)
        return scores.topk(max(1,len(scores)//5)).indices
    hard=torch.empty(0,dtype=torch.long,device=o.device)
    with (out/'metrics.jsonl').open('x') as log:
        for step in range(o.steps):
            if o.hard_fraction and step%o.refresh_priority_every==0:hard=hard_examples()
            ordinary=torch.randint(len(train['state']),(o.batch,),device=o.device)
            if len(switches):ordinary[:o.batch//2]=switches[torch.randint(len(switches),(o.batch//2,),device=o.device)]
            nhard=int(o.batch*o.hard_fraction)
            if nhard:ordinary[-nhard:]=hard[torch.randint(len(hard),(nhard,),device=o.device)]
            if len(recent):
                # New on-policy samples occupy half the batch; retain random,
                # switch and hard old samples in the other half.
                ordinary[o.batch//2:]=recent[torch.randint(len(recent),(o.batch-o.batch//2,),device=o.device)]
                if nhard:ordinary[o.batch//4:o.batch//4+nhard]=hard[torch.randint(len(hard),(nhard,),device=o.device)]
            b={k:v[ordinary] for k,v in train.items()};optimizer.zero_grad(set_to_none=True)
            p=predict(b);loss=fit_loss(p,b['next_state'],b['active'],b['force_w'],b['known'],b['transition_valid'],force_scale=200.)
            sequence_loss=None
            if seq_train is not None:
                si=torch.randint(len(seq_train['state']),(o.sequence_batch,),device=o.device)
                recent_seq=seq_train['recent'][:,0].nonzero().flatten()
                if len(recent_seq):si[:o.sequence_batch//2]=recent_seq[torch.randint(len(recent_seq),(o.sequence_batch//2,),device=o.device)]
                sequence_loss,_,_=sequence_prediction({k:v[si] for k,v in seq_train.items()})
                loss=loss+o.sequence_weight*sequence_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite calibration loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step%20==0:
                row=dict(step=iteration+step,loss=float(loss.detach()),sequence_loss=float(sequence_loss.detach()) if sequence_loss is not None else None);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    after=evaluate(validation)
    sequence_after=evaluate_sequence()
    after_by_recording={name:evaluate(row) for name,row in validation_sets.items()}
    report=dict(source=source,train_motions=train_ids,held_out_motions=validation_ids,train_samples=len(train['state']),
        validation_samples=len(validation['state']),steps=o.steps,before=before,after=after,
        before_by_recording=before_by_recording,after_by_recording=after_by_recording,
        sequence_before=sequence_before,sequence_after=sequence_after,
        sequence_context='recorded terrain heights; recurrent predicted state and force; no resets crossed',
        recent_samples=len(recent),recent_fraction=.5 if len(recent) else 0.,
        recordings=[str(p.resolve()) for p in (o.recording,*o.additional_recordings)],
        resumed_from=str(o.resume) if o.resume else None,total_updates=iteration+o.steps,
        replay=dict(switch_fraction=.5,hard_fraction=o.hard_fraction,priority='joint and force prediction error; training split only'),
        oracle_terrain_context=True,nominal_physics_context='friction 1; no randomization offsets; runtime solver equivalence not asserted',
        activation_fitted=True,policy_updated=False,simulator_gradient_alignment=None,optimization_allowed=False,
        limitation='one-step trajectory holdout; not multi-step gradient certification or edited-target success')
    torch.save(dict(schema='contact_wm_supermimic_calibration_v1',wm=m.state_dict(),optimizer=optimizer.state_dict(),
        iteration=iteration+o.steps,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        source=source,report=report,options={k:str(v) if isinstance(v,Path) else v for k,v in vars(o).items()}),out/'model.pt')
    (out/'report.json').write_text(json.dumps(report,indent=2));print('CALIBRATION_COMPLETE '+json.dumps(dict(before=before,after=after)),flush=True)

if __name__=='__main__':main()
