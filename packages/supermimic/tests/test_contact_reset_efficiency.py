"""CPU contract tests for exact resets and handoff entries, without a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from supermimic.contact_task import ContactTask
from supermimic.contact_arrival import ArrivalWindow
from supermimic.mdp import flatten_state,unflatten_state
from supermimic.rotations import mat_to_quat,so3_exp


def state(n):
    return (torch.randn(n,3),torch.eye(3).expand(n,3,3).clone(),torch.randn(n,3),torch.randn(n,3),torch.randn(n,29)*3,torch.randn(n,29))


def simulator_reset_method():
    # Execute the actual method without importing Isaac/CUDA during CPU tests.
    path=Path(__file__).parents[1]/'supermimic/env.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TrackingSim')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
    scope=dict(torch=torch,mat_to_quat=mat_to_quat,so3_exp=so3_exp)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),scope)
    return scope['reset']


def test_sim_reset_writes_exact_state_once_without_reference_clamping():
    calls={};writes=[];n=4
    class Robot:
        def __getattr__(self,key):
            def write(**kwargs):writes.append(key);calls[key]=kwargs
            return write
    def no_reference(*_):raise AssertionError('Exact state reset must not read motion pose')
    env=SimpleNamespace(device='cpu',motion=SimpleNamespace(frame_bin=torch.arange(20),ref_state=no_reference),
        reset_noise=True,eval_mode=True,robot=Robot(),scene=SimpleNamespace(env_origins=torch.randn(n,3),reset=lambda ids:writes.append('scene.reset')),
        f=torch.zeros(n,dtype=torch.long),ep_step=torch.ones(n,dtype=torch.long),a_prev=torch.ones(n,29),
        start_bin=torch.zeros(n,dtype=torch.long),ep_disc_ret=torch.ones(n),ep_ret=torch.ones(n),xi=torch.ones(n,52))
    ids=torch.tensor([1,3]);S=state(2);frames=torch.tensor([5,8])
    simulator_reset_method()(env,ids,frames,state=S)
    assert len(writes)==6 and len(set(writes))==6
    torch.testing.assert_close(calls['write_root_link_pose_to_sim_index']['root_pose'][:,:3],S[0]+env.scene.env_origins[ids])
    assert torch.equal(calls['write_joint_position_to_sim_index']['position'],S[4])
    assert torch.equal(calls['write_joint_velocity_to_sim_index']['velocity'],S[5])
    assert env.ep_step.tolist()==[1,0,1,0] and env.f.tolist()==[0,5,0,8]
    assert env.a_prev[[0,2]].eq(1).all() and env.a_prev[ids].eq(0).all()


def task_fixture(cache):
    t=ContactTask.__new__(ContactTask);N=4;frames=6;starts=torch.tensor([0,1,0,1])
    t.material_goal_mode=False;t.new_contact_stop=False
    t.ids=torch.arange(N);t.starts=starts;t.segment=starts.clone();t.N=N;t.count=2
    t.clock=SimpleNamespace(segment_starts=torch.tensor([0,1]));t.actor=SimpleNamespace(hist=5)
    t.demo=dict(s=flatten_state(state(frames)),a_prev=torch.randn(frames,29),xi=torch.randn(frames,52),f=torch.arange(frames))
    t._reset_history=torch.randn(2,5,232);t.history=torch.zeros(N,5,232);t.entry_xi=torch.zeros(N,52)
    t.pending_history_reset=torch.zeros(N,dtype=torch.bool);t.previous_contact=torch.zeros(N,6,dtype=torch.bool)
    t.recontact=torch.ones(N,6,dtype=torch.bool);t.valid_contacts=t.recontact.clone();t.onset_point=torch.ones(N,6,3);t.onset_face=torch.zeros(N,6,dtype=torch.long)
    t.arrival=ArrivalWindow((N,6),'cpu',2);t.contact_only=False;t.cache_reset_contacts=cache;t.stable_contact_frames=0
    t._reset_contact_segment=torch.full((N,),-1,dtype=torch.long);t._reset_contact_active=torch.zeros(N,6,dtype=torch.bool)
    calls=[];real=torch.tensor([[True,False]*3,[False,True]*3,[True]*6,[False]*6])
    def read():calls.append('probe');return real.clone()
    t.initial_contact_probe=SimpleNamespace(read=read)
    def reset(ids,frames,*,state):calls.append(('reset',ids.clone(),frames.clone(),flatten_state(state)))
    t.env=SimpleNamespace(reset=reset,a_prev=torch.zeros(N,29),xi=torch.zeros(N,52),refresh=lambda:None,default_q=torch.zeros(29),alpha=.5,
        robot=SimpleNamespace(set_joint_position_target_index=lambda **kw:calls.append(('target',kw))))
    return t,calls,real


def test_cache_is_world_and_segment_specific_and_invalidates():
    t,calls,real=task_fixture(True);slots=torch.tensor([0,2])
    t.reset(slots);t.reset(slots)
    assert calls.count('probe')==1 and torch.equal(t.previous_contact[slots],real[slots])
    assert torch.equal(t.history[slots],t._reset_history[t.segment[slots]])
    t.segment[2]=1;t.starts[2]=1;t.reset(slots)
    assert calls.count('probe')==2
    t.invalidate_reset_contacts();t.reset(slots);assert calls.count('probe')==3
    assert t.pending_history_reset.tolist()==[True,False,True,False]


def test_default_uncached_path_reads_native_evidence_each_reset():
    t,calls,real=task_fixture(False);slots=torch.tensor([1,3]);t.reset(slots);t.reset(slots)
    assert calls.count('probe')==2 and torch.equal(t.previous_contact[slots],real[slots])


def test_handoff_uses_one_reset_and_its_own_history_and_evidence():
    t,calls,_=task_fixture(True);slots=torch.tensor([0,2]);S=state(2)
    entry=dict(s=flatten_state(S),a_prev=torch.randn(2,29),xi=torch.randn(2,52),history=torch.randn(2,5,232),f=torch.tensor([4,5]),active=torch.ones(2,6,dtype=torch.bool))
    t.reset(slots,entry=entry)
    assert 'probe' not in calls and len([c for c in calls if isinstance(c,tuple) and c[0]=='reset'])==1
    assert torch.equal(t.history[slots],entry['history']) and torch.equal(t.entry_xi[slots],entry['xi'])
    assert torch.equal(t.previous_contact[slots],entry['active']) and not t.recontact[slots].any()
    assert (t._reset_contact_segment==-1).all()
    target=next(c[1]['target'] for c in calls if isinstance(c,tuple) and c[0]=='target')
    torch.testing.assert_close(target,entry['xi'][:,1:30]+.5*entry['a_prev'])


def test_new_contact_handoff_probes_actual_entry_and_cannot_poison_reference_cache():
    t,calls,real=task_fixture(True);t.new_contact_stop=True
    t.reference_entry_blocked=torch.zeros(2,6,dtype=torch.bool);t.reference_entry_absence=torch.zeros(2,6,dtype=torch.int32)
    t.ends=torch.ones(4,dtype=torch.long)
    t.clock.target_masks=torch.ones(1,6,dtype=torch.bool);t.clock.keep_masks=torch.zeros(1,6,dtype=torch.bool)
    t.entry_contact=torch.zeros(4,6,dtype=torch.bool);t.contact_confirming=torch.ones(4,dtype=torch.bool)
    t.contact_hold_action=torch.ones(4,29)
    stages=[];t.contact_monitor=SimpleNamespace(begin_stage=lambda *args:stages.append(args))
    def read(*,raw):
        assert raw
        calls.append('raw_probe');return real.clone()
    t.initial_contact_probe=SimpleNamespace(read=read)
    slots=torch.tensor([0,2]);t.reset(slots);assert calls.count('raw_probe')==1
    entry=dict(s=flatten_state(state(2)),a_prev=torch.randn(2,29),xi=torch.randn(2,52),history=torch.randn(2,5,232),f=torch.tensor([4,5]),active=torch.zeros(2,6,dtype=torch.bool))
    t.reset(slots,entry=entry)
    assert torch.equal(t.entry_contact[slots],real[slots])
    assert not t.contact_confirming[slots].any() and not t.contact_hold_action[slots].any()
    assert t._reset_contact_segment[slots].eq(-1).all()
    assert torch.equal(stages[-1][1],real[slots])
    t.reset(slots);assert calls.count('raw_probe')==3


def test_reference_reset_passes_old_phase_and_partial_absence_to_monitor():
    t,calls,real=task_fixture(True);t.new_contact_stop=True
    t.ends=torch.ones(4,dtype=torch.long)
    t.clock.target_masks=torch.ones(1,6,dtype=torch.bool);t.clock.keep_masks=torch.zeros(1,6,dtype=torch.bool)
    t.entry_contact=torch.zeros(4,6,dtype=torch.bool);t.contact_confirming=torch.zeros(4,6,dtype=torch.bool);t.contact_hold_action=torch.zeros(4,29)
    t.reference_entry_blocked=torch.zeros(2,6,dtype=torch.bool);t.reference_entry_blocked[0,1]=True
    t.reference_entry_absence=torch.zeros(2,6,dtype=torch.int32);t.reference_entry_absence[0,1]=8
    stages=[];t.contact_monitor=SimpleNamespace(begin_stage=lambda *args:stages.append(args))
    t.initial_contact_probe=SimpleNamespace(read=lambda *,raw:real.clone())
    t.reset(torch.tensor([0]))
    assert not real[0,1] and t.entry_contact[0,1]
    assert stages[0][3][0,1]==8
    assert stages[0][3][0,0]==0


def native_task_fixture(seed):
    torch.manual_seed(seed)
    t,calls,real=task_fixture(True);t.new_contact_stop=True
    t.ends=torch.ones(4,dtype=torch.long)
    t.clock.target_masks=torch.ones(1,6,dtype=torch.bool);t.clock.keep_masks=torch.zeros(1,6,dtype=torch.bool)
    t.reference_entry_blocked=torch.zeros(2,6,dtype=torch.bool);t.reference_entry_blocked[0,1]=True
    t.reference_entry_absence=torch.zeros(2,6,dtype=torch.int32);t.reference_entry_absence[0,1]=8
    t.entry_contact=torch.zeros(4,6,dtype=torch.bool);t.contact_confirming=torch.ones(4,6,dtype=torch.bool)
    t.contact_hold_action=torch.ones(4,29)
    stages=[]
    t.contact_monitor=SimpleNamespace(begin_stage=lambda *args:stages.append(tuple(x.clone() for x in args)))
    t.env.refresh=lambda:calls.append('refresh')
    def read(*,raw):
        assert raw;calls.append('raw_probe');return real.clone()
    t.initial_contact_probe=SimpleNamespace(read=read)
    return t,calls,stages


def test_batched_native_reset_matches_sequential_mixed_entries_with_one_solve():
    old,old_calls,old_stages=native_task_fixture(31)
    batch,batch_calls,batch_stages=native_task_fixture(31)
    torch.manual_seed(88)
    def payload():
        return dict(s=flatten_state(state(1)),a_prev=torch.randn(1,29),xi=torch.randn(1,52),history=torch.randn(1,5,232),f=torch.tensor([4]),active=torch.zeros(1,6,dtype=torch.bool))
    entries=[(torch.tensor([0]),None),(torch.tensor([1]),payload()),(torch.tensor([2]),payload()),(torch.tensor([3]),None)]
    for slots,entry in entries:old.reset(slots,entry)
    batch.reset_batch(entries)
    assert old_calls.count('raw_probe')==4 and batch_calls.count('raw_probe')==1
    assert old_calls.count('refresh')==4 and batch_calls.count('refresh')==1
    for name in ('history','entry_xi','pending_history_reset','previous_contact','entry_contact','contact_confirming','contact_hold_action','_reset_contact_segment','_reset_contact_active','valid_contacts','onset_point','onset_face'):
        torch.testing.assert_close(getattr(old,name),getattr(batch,name),rtol=0,atol=0)
    for name in ('a_prev','xi'):torch.testing.assert_close(getattr(old.env,name),getattr(batch.env,name),rtol=0,atol=0)
    def stage_by_world(stages):
        return {int(ids[i]):tuple(v[i] for v in values) for ids,*values in stages for i in range(len(ids))}
    old_by_world=stage_by_world(old_stages);batch_by_world=stage_by_world(batch_stages)
    for slot in old_by_world:
        for a,b in zip(old_by_world[slot],batch_by_world[slot]):assert torch.equal(a,b)
    for label in ('reset','target'):
        def packed(calls):
            writes=[x for x in calls if isinstance(x,tuple) and x[0]==label]
            if label=='reset':
                ids=torch.cat([x[1] for x in writes]);values=[torch.cat([x[i] for x in writes]) for i in (2,3)]
            else:
                ids=torch.cat([x[1]['env_ids'] for x in writes]);values=[torch.cat([x[1]['target'] for x in writes])]
            order=ids.argsort();return ids[order],[x[order] for x in values]
        a,av=packed(old_calls);b,bv=packed(batch_calls);assert torch.equal(a,b)
        for x,y in zip(av,bv):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert len([x for x in batch_calls if isinstance(x,tuple) and x[0]=='reset'])==2


def test_batched_cached_references_need_no_new_native_solve():
    t,calls,_=native_task_fixture(32)
    t.reset(torch.arange(4));calls.clear()
    t.reset_batch([(torch.tensor([0,2]),None),(torch.tensor([1,3]),None)])
    assert calls.count('raw_probe')==0 and calls.count('refresh')==1
