import copy
import torch
from holosoma.contact_wm.stream_replay import Replay
from holosoma.contact_wm.dynamics_audit import audit_dynamics,policy_gate,errors


def test_terminal_transition_kept_but_no_reset_crossing():
    r=Replay(32,1,'cpu')
    for t in range(24):
        age=t%6
        r.add(dict(state=torch.tensor([[float(t)]]),age=torch.tensor([age]),
            valid=torch.ones(1,dtype=torch.bool),previous_known=torch.ones(1,6,dtype=torch.bool),
            ended=torch.tensor([age==5]),event=torch.tensor([age==5])))
    torch.manual_seed(4)
    x=r.windows(256,3,event_fraction=1.)
    assert x['ended'][-1].all()
    assert not x['ended'][:-1].any()
    assert (x['state'][1:]-x['state'][:-1]==1).all()


def test_event_sampling_handles_no_events():
    r=Replay(8,1,'cpu')
    for t in range(8):r.add(dict(state=torch.tensor([[float(t)]]),valid=torch.ones(1,dtype=torch.bool),
        previous_known=torch.ones(1,6,dtype=torch.bool),event=torch.zeros(1,dtype=torch.bool)))
    assert r.windows(16,3,event_fraction=1.)['state'].shape==(3,16,1)


def test_motion_sampling_has_explicit_controller_split():
    r=Replay(8,2,'cpu')
    for t in range(8):r.add(dict(state=torch.tensor([[0.],[1.]]),valid=torch.ones(2,dtype=torch.bool),
        previous_known=torch.ones(2,6,dtype=torch.bool),motion_replay=torch.tensor([False,True])))
    assert (r.windows(32,3,motion_replay=True)['state']==1).all()
    assert (r.windows(32,3,motion_replay=False)['state']==0).all()


def test_native_contact_balancing_prevents_rare_positive_suppression():
    from holosoma.contact_wm.batched_learner import native_fit_tensor
    logits=torch.zeros(100,6,requires_grad=True)
    s=torch.zeros(100,71);force=torch.zeros(100,6,3)
    active=torch.zeros(100,6,dtype=torch.bool);active[:2]=True
    prediction=dict(state=s,force_w=force,contact_logits=logits)
    loss=native_fit_tensor(prediction,s,active,force,torch.ones_like(active),torch.ones(100,dtype=torch.bool),balanced_contacts=True)
    loss.backward()
    torch.testing.assert_close(logits.grad[:2].abs().sum(),logits.grad[2:].abs().sum())


class Dynamics:
    bias=0.
    def source_step(self,s,a,physics,height,force):
        value=s.clone();value[:,0]+=a[:,0]+self.bias
        logits=s.new_tensor([2.,-2.,2.,-2.,2.,-2.])[None].expand(len(s),-1)
        return dict(state=value,force_w=force,contact_logits=logits)


class Learner:
    wm=Dynamics()
    source_grid=None
    def geometry(self,s):return s[:,:3,None].transpose(1,2).expand(-1,6,-1)
    def physics(self,s):return s.new_zeros(len(s),33)
    def height(self,s,local):return s.new_zeros(len(s),121),torch.ones(len(s),dtype=torch.bool)


def test_audit_exposes_compounding_error_and_gate_rejects_it():
    r=Replay(64,2,'cpu');active=torch.tensor([[1,0,1,0,1,0]],dtype=torch.bool).expand(2,-1)
    for t in range(50):
        s=torch.zeros(2,71);s[:,0]=t*.01;s[:,6]=1
        target=s.clone();target[:,0]+=.01
        r.add(dict(state=s,next_state=target,action=torch.full((2,29),.01),height=torch.zeros(2,121),
            previous_force_w=torch.zeros(2,6,3),force_w=torch.zeros(2,6,3),
            previous_active=active,active=active,known=torch.ones_like(active),previous_known=torch.ones_like(active),
            valid=torch.ones(2,dtype=torch.bool),age=torch.full((2,),t+1),ended=torch.zeros(2,dtype=torch.bool)))
    learner=Learner();learner.wm=Dynamics()
    good=audit_dynamics(learner,r,batch_size=8)
    assert policy_gate(good)['allowed']
    split=copy.deepcopy(good)
    mode=split['horizons']['8']['predicted_height']
    mode['by_controller']={'motion':copy.deepcopy(mode['metrics']),'policy':copy.deepcopy(mode['metrics'])}
    mode['by_controller']['policy']['all']['root_p95_m']=.06
    assert not policy_gate(split)['allowed']  # Good Motion data must not hide policy errors.
    uncovered=copy.deepcopy(good)
    mode=uncovered['horizons']['35']['predicted_height']
    mode['by_controller']={'motion':mode['metrics'],'policy':None}
    assert 'h35/policy' in policy_gate(uncovered)['unobserved']
    learner.wm.bias=.002
    bad=audit_dynamics(learner,r,batch_size=8)
    e1=bad['horizons']['1']['predicted_height']['metrics']['endpoint']['root_p95_m']
    e35=bad['horizons']['35']['predicted_height']['metrics']['endpoint']['root_p95_m']
    assert e1<.003 and e35>.069
    assert not policy_gate(bad)['allowed']
    assert not policy_gate(good,reference=bad)['allowed']
    assert any(s.startswith('fixed_reference/') for s in policy_gate(good,reference=bad)['reasons'])
    assert not bad['gradient_validated']
    missing=copy.deepcopy(good);missing['horizons']['35']={'available':False}
    assert not policy_gate(missing)['allowed']
    nan=copy.deepcopy(good);nan['horizons']['8']['predicted_height']['metrics']['all']['force_rmse_N']=float('nan')
    assert not policy_gate(nan)['allowed']


def test_dynamics_error_does_not_depend_on_task_contact_witnesses():
    from holosoma.contact_wm.geometry import WitnessGeometry
    from somaforge_core import G1_29DOF_JOINT_ORDER
    s=torch.zeros(1,71);s[:,6]=1
    prediction=s.clone();prediction[:,3]=.01;prediction[:,6]=(1-.01**2)**.5
    contact=torch.zeros(1,1,6,dtype=torch.bool)
    batch=dict(next_state=s[None],known=torch.ones_like(contact),active=contact,
        previous_active=contact,previous_known=torch.ones_like(contact),force_w=torch.zeros(1,1,6,3))
    predictions=[dict(state=prediction,force_w=torch.zeros(1,6,3),contact_logits=torch.full((1,6),-2.))]
    a=WitnessGeometry(G1_29DOF_JOINT_ORDER,torch.zeros(6,3))
    b=WitnessGeometry(G1_29DOF_JOINT_ORDER,torch.full((6,3),100.))
    assert errors(predictions,batch,a)==errors(predictions,batch,b)
