"""Adjacent-stage selection and gradients, using the actual DMO loss."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import runpy
from pathlib import Path
x=runpy.run_path(str(Path(__file__).with_name('check_contact_duration.py')))
globals().update({k:v for k,v in x.items() if not k.startswith('__')})
from supermimic.contact_sequence import sequence_plan,slice_plan
from supermimic.contact_duration import DurationBandHistory
# First stage is longer than its prior and second stage is reached. Its early
# supervision must not cut the graph to the later stage.
dat=dict(done=torch.tensor([[0,0],[0,1],[0,0],[0,0],[1,0]],dtype=torch.bool),
    term=torch.tensor([[0,0],[0,1],[0,0],[0,0],[1,0]],dtype=torch.bool),
    goal_index=torch.tensor([[0,1],[0,1],[0,1],[1,1],[1,1]]),
    stage_done=torch.tensor([[0,0],[0,1],[1,0],[0,0],[1,0]],dtype=torch.bool),
    stage_success=torch.tensor([[0,0],[0,0],[1,0],[0,0],[0,0]],dtype=torch.bool))
ids,plan,report=sequence_plan(dat,2,DurationBandHistory([1,2,2],0),0,8,.6)
assert ids.tolist()==[0,1]
assert plan['ends'][:,0].tolist()==[1,5] and plan['stop'].tolist()==[5,2]
assert plan['enabled'].tolist()==[[True,True],[True,False]]
assert slice_plan(plan,torch.tensor([0]))['ends'].shape==(2,1)
# Three physical steps: stage 1 spans first two, stage 2 the third. Only
# second-stage loss is retained, yet it must differentiate both earlier actions.
clock.bounds=[0,1,2];clock.event_ends=torch.tensor([1,2])
clock.target_masks=new.expand(2,-1);clock.keep_masks=torch.zeros(2,6,dtype=torch.bool)
clock.data={'point':target.expand(2,-1,-1),'s1':ref.expand(2,-1)}
clock.data['s']=clock.data['s1']
d['goal_index']=torch.tensor([[0],[0],[1]])
d['contact_goal'].zero_()
plan=dict(length=2,ends=torch.tensor([[1],[3]]),enabled=torch.tensor([[False],[True]]),truncated=torch.tensor([[True],[False]]),stop=torch.tensor([3]))
with patch('supermimic.contact_dmo.observation',lambda S,*a:S[0][:,:1]):
    actor=Actor();loss,details=replay_loss(actor,WM(),clock,Geometry(),None,lambda x:x,None,d,completion_mode='accumulated_target',stage_plan=plan,check=True)
    loss.backward()
    assert actor.weight.grad.abs().min()>0
    assert abs(details['terminal_contact']-36)<1e-4
    assert details['handoff_terminal_to_first_action_gradient']>0
    plan['enabled'][0]=True
    actor=Actor();_,details=replay_loss(actor,WM(),clock,Geometry(),None,lambda x:x,None,d,completion_mode='accumulated_target',stage_plan=plan)
    # Frame 1 measured point x=.01 => .04; frame 3 x=.3 =>36.
    assert abs(details['terminal_contact']-18.02)<1e-4
print('PASS pairs: local cutoffs, failed-first isolation, no handoff gradient detach, endpoint averaging and exact action replay')
# Replay windows are batch-major (B,K), unlike DMO replay (T,B).
from supermimic.buffer import ReplayBuffer
buffer=ReplayBuffer(3,2,'cpu',extra_fields={'sequence_live':((),torch.bool)})
for step in range(3):
    row={k:torch.zeros_like(v[0]) for k,v in buffer.data.items()}
    row['sequence_live'][0]=True
    buffer.add(row)
window=buffer.sample_windows(128,1)
valid=window['sequence_live'].all(1)
filtered={k:v[valid] for k,v in window.items()}
assert 0<len(filtered['s'])<128 and filtered['s'].shape[1]==1
assert filtered['sequence_live'].all()
print('PASS WM replay filtering: batch-major windows retain active native transitions')
