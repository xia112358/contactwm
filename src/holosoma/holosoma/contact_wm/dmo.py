"""DMO updates for the historical nine-target task, using fresh native episodes."""
import torch
from supermimic.contact_dmo import swap
from holosoma.contact_wm.observations import unpack_history


def native_policy_loss(learner, data, targets):
    first={k:v[0] for k,v in data.items()}
    state=first['state'];history=unpack_history(first['observation'][:,102:])
    x=dict(state=state,history=history,previous=history[:,-1,:29],
           contact=first['previous_active'].float(),known=first['previous_known'],
           force=first['previous_force_w'],target=targets,support=first['support'],
           arrival=26-(first['age']-5),features=state.new_zeros(len(state),63),
           view=state.new_zeros(len(state),4))
    contact,previous,force=x['contact'],x['previous'],x['force']
    before=torch.cat((torch.zeros_like(data['ended'][:1]),data['ended'][:-1]),0)
    live=~before.long().cummax(0).values.bool()
    states=[];actions=[];forces=[];logits=[];priors=[];valid=[];errors=[]
    for t in range(len(data['state'])):
        row={k:v[t] for k,v in data.items()}
        row['dmo_live']=live[t]
        predicted,pcontact,history,action,pforce,logit,prior,ok=learner._imag_step(
            state,contact,history,previous,force,row['previous_known'],targets,x['support'],
            x['arrival']-t,x['state'],x['features'],x['view'],t==0,native=row)
        state=swap(predicted,row['next_state'])
        contact=swap(pcontact,row['active'].float())
        force=swap(pforce,row['force_w'])
        logit=swap(logit,torch.logit(row['active'].float().clamp(1e-4,1-1e-4)))
        previous=action
        states.append(state);actions.append(action);forces.append(force)
        logits.append(logit);priors.append(prior);valid.append(ok&row['valid'])
        errors.append(row['action_error'])
    values,ok=learner._objective(x,*(torch.stack(v) for v in (states,actions,forces,logits,priors,valid)),
                                 native={**data,'live':live})[:2]
    # A stale policy or mismatched actor input invalidates an on-policy DMO update.
    action_error=torch.stack(errors).max()
    if float(action_error)>1e-4:
        raise RuntimeError(f'DMO action replay mismatch: {float(action_error):g}')
    return values,ok,dict(dmo_action_error=float(action_error),
                          dmo_complete_attempts=int(live[-1].sum()),dmo_attempts=len(state))
