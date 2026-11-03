"""Measure successful versus failed stage durations directly from native rollout."""
import sys,json,inspect,runpy,os
import torch
from pathlib import Path
import supermimic.contact_sequence as seq
out=Path(sys.argv[sys.argv.index('--output')+1])
def audit(buffer,rows,plan,actor,context,default_q,batch_size):
    scope=inspect.currentframe().f_back.f_locals;raw=scope['raw']
    _,p,_=seq.sequence_plan(raw,scope['args'].sequence_length,None)
    stages,worlds=p['enabled'].nonzero(as_tuple=True);ends=p['native_ends'][stages,worlds]-1;starts=p['starts'][stages,worlds]
    lengths=ends-starts+1;success=raw['stage_success'][ends,worlds];groups=raw['goal_index'][ends,worlds];cont=stages>0
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import rot_angle_sq
    from supermimic.contact_control import observation
    demo=clock=scope['clock'];demo=clock.data
    entry=buffer.data['s'][rows[starts],worlds];reference=demo['s'][torch.as_tensor(clock.bounds,device=worlds.device)[groups]]
    S=unflatten_state(entry);R=unflatten_state(reference)
    histidx=(starts[:,None]+torch.arange(-actor.hist+1,1,device=worlds.device)).clamp_min(0)
    hs=buffer.data['s'][rows[histidx],worlds[:,None]];ha=buffer.data['a_prev'][rows[histidx],worlds[:,None]];hx=buffer.data['xi'][rows[histidx],worlds[:,None]]
    history=observation(unflatten_state(hs.flatten(0,1)),ha.flatten(0,1),hx.flatten(0,1),default_q,context).reshape(len(worlds),actor.hist,-1)
    condition=clock.condition(S,torch.as_tensor(clock.bounds,device=worlds.device)[groups],ends=clock.event_ends[groups])
    with torch.no_grad():
        actions=actor(history,condition);repeated=actor(history[:,-1:].expand_as(history),condition)
    features=dict(root_cm=(S[0]-R[0]).norm(dim=-1)*100,angle_rad=rot_angle_sq(S[1],R[1]).sqrt(),velocity=(S[2]-R[2]).norm(dim=-1),joint_rms=(S[4]-R[4]).square().mean(-1).sqrt(),joint_velocity_rms=(S[5]-R[5]).square().mean(-1).sqrt(),history_action_rms=(actions-repeated).square().mean(-1).sqrt())
    before=buffer.data['s1'][rows[(starts-1).clamp_min(0)],worlds]
    continuity=float((entry[cont]-before[cont]).abs().max())
    assert continuity<1e-4
    assert (buffer.data['stage_step'][rows[starts],worlds]==0).all()
    result=[]
    for g in [-1]+groups.unique().tolist():
        for c in [False,True]:
            m=(cont==c)&((groups==g) if g>=0 else True);n=int(m.sum())
            if not n:continue
            a=lengths[m&success].float();f=lengths[m&~success].float()
            result.append(dict(segment=g+1,continuation=c,n=n,successes=len(a),success_duration=float(a.mean()) if len(a) else None,failure_duration=float(f.mean()) if len(f) else None,entry={k:float(v[m].mean()) for k,v in features.items()}))
    (out/'continuity.json').write_text(json.dumps(dict(max_state_jump=continuity,local_step_zero=True,repeat_history=bool(os.environ.get('AUDIT_REPEAT_HISTORY')))))
    (out/'duration-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    state=json.loads((out/'status.json').read_text());state.update(phase='complete',audit_only=True);(out/'status.json').write_text(json.dumps(state));raise SystemExit(0)
if os.environ.get('AUDIT_REPEAT_HISTORY'):
    from supermimic.contact_task import ContactTask
    original_step=ContactTask.step
    @torch.no_grad()
    def repeated_step(task):
        if task.history is not None:
            mask=(task.stage_step==0)&(task.env.ep_step[task.ids]>0)
            if mask.any():
                from supermimic.contact_control import observation
                state=tuple(x[task.ids] for x in task.env.state())
                obs=observation(state,task.env.a_prev[task.ids],task.env.xi[task.ids],task.default_q,task.context)
                task.history[mask]=obs[mask,None].expand(-1,task.actor.hist,-1)
        return original_step(task)
    ContactTask.step=repeated_step
seq.stage_batches=audit
runpy.run_module('supermimic.contact_train',run_name='__main__')
