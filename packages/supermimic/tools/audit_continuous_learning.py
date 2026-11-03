"""Frozen on-policy audit: direct/continuation entries, cutoff/native endpoints, gradients."""
import sys,json,runpy,inspect
from pathlib import Path
import torch
import supermimic.contact_sequence as seq
from supermimic.contact_dmo import replay_loss
original=seq.stage_batches
output=Path(sys.argv[sys.argv.index('--output')+1])
def audit(buffer,rows,plan,actor,context,default_q,batch_size):
    scope=inspect.currentframe().f_back.f_locals;args=scope['args'];clock=scope['clock'];raw=scope['raw']
    _,native_plan,_=seq.sequence_plan(raw,args.sequence_length,None)
    report={'groups':{},'gradient':{},'native_batch':scope['native_batch'],'duration_filter':scope['duration_filter']};params=list(actor.parameters());grads={}
    for mode,p in [('filtered',plan),('native',native_plan)]:
        for data,stops,history,weight,total in original(buffer,rows,p,actor,context,default_q,batch_size):
            vals={}
            with torch.set_grad_enabled(mode=='filtered'):
                loss,detail=replay_loss(actor,scope['wm'],clock,scope['geometry'],context,scope['normalize_xi'],default_q,data,
                    prior_actor=scope['prior'],prior_weight=args.prior_weight,pose_weight=args.pose_weight,
                    completion_mode=args.completion_mode,stop_steps=stops,initial_history=history,per_attempt_losses=vals)
                whole=sum(vals.values());contact=vals['contact'];root=sum(vals[k] for k in ['root_position','root_orientation','root_velocity'])
            B=len(stops);idx=torch.arange(B,device=stops.device);end=data['done'].long().argmax(0);success=~data['term'][end,idx];continuation=data['ep_step'][0]>0
            g=data['goal_index'][0];new=clock.target_masks[clock.event_ends[g]-1]&~clock.keep_masks[clock.event_ends[g]-1]
            at=data['active'][stops-1,idx]&data['known'][stops-1,idx]
            missing=(new&~at).any(-1)
            hit_by=((data['contact_goal']&(torch.arange(len(data['done']),device=stops.device)[:,None]<stops[None])[...,None]).any(0)|~new).all(-1)
            for incoming in [False,True]:
                mask=continuation==incoming;label='continuation' if incoming else 'direct'
                if mode=='filtered' and mask.any():
                    grad=torch.autograd.grad(whole[mask].sum()/total,params,retain_graph=True,allow_unused=True)
                    vector=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(params,grad)])
                    grads[label]=grads.get(label,0)+vector
                for segment in [-1]+g[mask].unique().tolist():
                    m=mask if segment==-1 else mask&(g==segment);n=int(m.sum())
                    if not n:continue
                    key=f'{mode}/{label}/{segment+1}'
                    sums=dict(n=n,success=int(success[m].sum()),success_duration_sum=float((end[m&success]+1).sum()),failure_duration_sum=float((end[m&~success]+1).sum()),truncated=int((stops[m]<end[m]+1).sum()),successful_truncated=int((success[m]&(stops[m]<end[m]+1)).sum()),cutoff_missing_contact=int(missing[m].sum()),cutoff_goal_achieved=int(hit_by[m].sum()),duration_sum=float((end[m]+1).sum()),cutoff_sum=float(stops[m].sum()),contact_sum=float(contact[m].detach().sum()),root_sum=float(root[m].detach().sum()),loss_sum=float(whole[m].detach().sum()))
                    acc=report['groups'].setdefault(key,{k:0 for k in sums})
                    for k,v in sums.items():acc[k]+=v
            del vals,loss,whole,contact,root
    a,b=grads['direct'],grads['continuation'];report['gradient']=dict(direct_norm=float(a.norm()),continuation_norm=float(b.norm()),cosine=float(torch.dot(a,b)/(a.norm()*b.norm())),total_norm=float((a+b).norm()))
    (output/'learning-audit.json').write_text(json.dumps(report,indent=2));print('AUDIT',json.dumps(report['gradient']),flush=True)
    state=json.loads((output/'status.json').read_text());state.update(phase='complete',audit_only=True,actor_updates_performed=0);(output/'status.json').write_text(json.dumps(state,indent=2));raise SystemExit(0)
seq.stage_batches=audit
runpy.run_module('supermimic.contact_train',run_name='__main__')
