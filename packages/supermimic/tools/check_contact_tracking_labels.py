"""CPU mock-physics check for selective tracking labels; no native run or updates."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import ast
import copy
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
pass
from supermimic.mdp import flatten_state
from supermimic.motion import CONTROL_DT


def load_step():
    # Compile the actual method without importing/initializing Isaac Lab.
    tree=ast.parse((ROOT/'supermimic/env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrackingSim')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    scope=dict(torch=torch,flatten_state=flatten_state,CONTROL_DT=CONTROL_DT)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(ROOT/'supermimic/env.py'),'exec'),scope)
    return scope['step']


class Labels:
    def __init__(self):self.reward_sizes=[];self.term_sizes=[]
    def reward(self,state,frames,action,previous,terms=False):
        self.reward_sizes.append(len(frames))
        reward=state[0].sum(-1)+frames*.1+action.square().sum(-1)
        return reward,dict(track=reward*.7,action_rate=(action-previous).square().sum(-1))
    def termination(self,state,frames):
        self.term_sizes.append(len(frames))
        return (state[0][:,0]<.25)|(frames>6)


class MockEnv:
    step=load_step()
    def __init__(self):
        self.N=8;self.eval_mode=True;self.device='cpu';self.gamma=.99;self.max_episode_steps=5
        self.p=torch.arange(self.N,dtype=torch.float32)[:,None].expand(-1,3).clone()*.1
        self.q=torch.zeros(self.N,29);self.velocity=torch.zeros(self.N,3)
        self.f=torch.tensor([0,1,2,3,4,5,6,7]);self.ep_step=torch.arange(self.N)%5
        self.xi=torch.zeros(self.N,52);self.xi[:,0]=1.;self.xi[:,1:30]=.01
        self.a_prev=torch.full((self.N,29),.03);self.default_q=torch.zeros(29);self.alpha=torch.ones(29)*.1
        self.ep_ret=torch.arange(self.N,dtype=torch.float32);self.ep_disc_ret=self.ep_ret.clone()
        self.done_eval=torch.tensor([False,False,True,False,False,False,False,False]);self.failed_eval=torch.zeros(self.N,dtype=torch.bool)
        self.motion=SimpleNamespace(clip_end=torch.tensor([8]),clip_id=torch.zeros(20,dtype=torch.long));self.mdp=Labels()
        self.robot=SimpleNamespace(set_joint_position_target_index=self.set_target)
        self.scene=SimpleNamespace(write_data_to_sim=lambda:None,update=lambda _:None)
        self.sim=SimpleNamespace(step=self.physics)
        self.physics_steps=0
    def set_target(self,*,target):self.target=target.clone()
    def physics(self,*,render=False):
        self.q=self.target.clone();self.velocity=self.q[:,:3]*.01;self.p=self.p+self.velocity
        self.xi[:,34:52]=self.p[:,:1];self.physics_steps+=1
    def state(self):
        return self.p.clone(),torch.eye(3).expand(self.N,3,3).clone(),self.velocity.clone(),torch.zeros(self.N,3),self.q.clone(),torch.zeros_like(self.q)


def main():
    actions=torch.arange(8*29,dtype=torch.float32).reshape(8,29)*.001
    background=torch.tensor([1,4,7]);students=torch.tensor([0,2,3,5,6])
    initial=MockEnv();full=copy.deepcopy(initial);selected=copy.deepcopy(initial)
    for _ in range(2):
        a=full.step(actions,precomputed_state=full.state())
        b=selected.step(actions,precomputed_state=selected.state(),tracking_env_ids=background)
        for key in ('s','s1','a','a_prev','xi','f','ep_step'):assert torch.equal(a[key],b[key]),key
        for key in ('r','done','term'):assert torch.equal(a[key][background],b[key][background]),key
        for key,value in a['terms'].items():
            assert torch.equal(value[background],b['terms'][key][background]),key
            assert b['terms'][key][students].eq(0).all()
        assert b['r'][students].eq(0).all() and not b['term'][students].any()
        assert b['tracking_evaluated'].nonzero().flatten().tolist()==background.tolist()
        assert 'tracking_evaluated' not in a
        for key in ('f','ep_step','a_prev','xi','target'):assert torch.equal(getattr(full,key),getattr(selected,key)),key
        for key in ('ep_ret','ep_disc_ret','done_eval','failed_eval'):
            assert torch.equal(getattr(full,key)[background],getattr(selected,key)[background]),key
        assert torch.equal(selected.done_eval[students],initial.done_eval[students])
        assert torch.equal(selected.ep_ret[students],initial.ep_ret[students])
    assert selected.mdp.reward_sizes==[3,3] and full.mdp.reward_sizes==[8,8]
    assert selected.physics_steps==full.physics_steps==2
    all_worlds=MockEnv();ordinary=MockEnv()
    x=all_worlds.step(actions,tracking_env_ids=torch.arange(8));y=ordinary.step(actions)
    for key in y:
        if isinstance(y[key],dict):
            for name in y[key]:assert torch.equal(x[key][name],y[key][name])
        else:assert torch.equal(x[key],y[key]),key
    empty=MockEnv();z=empty.step(actions,tracking_env_ids=torch.tensor([],dtype=torch.long))
    assert not z['tracking_evaluated'].any() and z['r'].eq(0).all() and not z['term'].any()
    assert torch.equal(z['done'],(z['f']+1>=8)|(z['ep_step']+1>=5))
    rejected=MockEnv();rejected.eval_mode=False
    try:rejected.step(actions,tracking_env_ids=background)
    except ValueError:pass
    else:raise AssertionError('Selective tracking may not change training reset semantics')
    assert rejected.physics_steps==0
    print('PASS identical physics/state/control, background tracking labels/bookkeeping, explicit mask, full and empty selections')

    # Exercise actual endpoint DMO on the existing two-stage fixture. ContactTask
    # replaces raw tracking done/term with native contact failure/timeout before
    # replay insertion; skipped tracking reward/terms are not objective inputs.
    fixture=runpy.run_path(str(ROOT/'tools/check_contact_pairs.py'))
    d=fixture['d'];results=[]
    for reward_scale in (1.,0.):
        actor=fixture['Actor']()
        data=dict(d,r=torch.full_like(d['goal_index'],reward_scale,dtype=torch.float32),
                  tracking_evaluated=torch.full_like(d['done'],bool(reward_scale)))
        with patch('supermimic.contact_dmo.observation',lambda state,*_:state[0][:,:1]):
            loss,_=fixture['replay_loss'](actor,fixture['WM'](),fixture['clock'],fixture['Geometry'](),None,
                lambda value:value,None,data,completion_mode='accumulated_target',stage_plan=fixture['plan'])
        loss.backward();results.append((loss.detach(),[p.grad.clone() for p in actor.parameters()]))
    torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
    for a,b in zip(results[0][1],results[1][1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    print('PASS skipped tracking reward/mask do not change actual contact DMO loss or actor gradient; no optimizer updates')


if __name__=='__main__':main()
