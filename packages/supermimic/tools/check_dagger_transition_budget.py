"""Synthetic dynamics regression for DAgger data ownership and reset boundaries.

This fixture produces no physical contact labels or accepted training data.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from supermimic.contact_dagger import run_dagger
    from supermimic.mdp import flatten_state
    class Actor(torch.nn.Module):
        hist=2
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1))
        def forward(self,h,c):return (self.weight+h[:,-1,:1]*.01).expand(-1,29)
    class Teacher:
        def __call__(self,h,g):return (.2+h[:,-1,:1]*.02).expand(-1,29)
    class Env:
        def __init__(self):
            self.N=6;self.task_ids=torch.zeros(6,dtype=torch.long)
            self.scene=NS(env_origins=torch.zeros(6,3))
            self.p=torch.zeros(6,3);self.q=torch.zeros(6,29)
            self.f=torch.zeros(6,dtype=torch.long);self.ep_step=self.f.clone()
            self.a_prev=torch.zeros(6,29);self.xi=torch.zeros(6,52)
            self.done_eval=torch.zeros(6,dtype=torch.bool);self.failed_eval=self.done_eval.clone()
            self.robot=NS(write_root_link_pose_to_sim_index=lambda root_pose,env_ids:self.p.__setitem__(env_ids,root_pose[:,:3]),
                write_root_link_velocity_to_sim_index=lambda **kw:None,
                write_joint_position_to_sim_index=lambda position,env_ids:self.q.__setitem__(env_ids,position),
                write_joint_velocity_to_sim_index=lambda **kw:None,set_joint_position_target_index=lambda **kw:None)
        def state(self):return self.p.clone(),torch.eye(3).expand(6,3,3),torch.zeros(6,3),torch.zeros(6,3),self.q.clone(),torch.zeros(6,29)
        def reset(self,ids,frames):
            self.p[ids]=0;self.q[ids]=0;self.f[ids]=frames;self.ep_step[ids]=0;self.a_prev[ids]=0;self.xi[ids]=0
        def refresh(self):pass
        def step(self,actions):
            self.p[:,0]+=actions[:,0]*.01;self.a_prev=actions.clone();self.f+=1;self.ep_step+=1
            return dict(s1=flatten_state(self.state()),term=torch.zeros(6,dtype=torch.bool),done=torch.zeros(6,dtype=torch.bool))
    env=Env()
    observation=lambda S,previous,*_:torch.cat((S[0][:,:1],previous[:,:1]),-1)
    mdp=NS(obs=lambda S,previous,xi,f:observation(S,previous),termination=lambda S,f:torch.zeros(6,dtype=torch.bool))
    motion=NS(clip_start=torch.tensor([0]),clip_end=torch.tensor([100]),goal=lambda f,k:f[:,None].float(),ref_state=lambda f:(torch.zeros(len(f),3),))
    clock=NS(event_ends=torch.tensor([2,5]),bounds=[0,2,5],condition=lambda S,frame,**kw:torch.zeros(len(frame),1))
    demo=dict(f=torch.arange(5),s=flatten_state(env.state())[:5],a_prev=torch.zeros(5,29),xi=torch.zeros(5,52),
              ep_step=torch.zeros(5,dtype=torch.long))
    clock.data=demo
    actor=Actor();optimizer=torch.optim.Adam(actor.parameters(),lr=.01)
    args=NS(output=a.output,bc_learning_rate=.01,dagger_min_rounds=10,bc_patience=8,bc_min_improvement=.001,
        bc_relative_improvement=.005,dagger_resume=None,dagger_rounds=2,dagger_updates_per_round=2,bc_batch=8,
        dagger_rollouts_per_segment=3,dagger_action_noise=.03,checkpoint='fixture',recording='fixture',task_manifest='fixture')
    with patch('supermimic.contact_dagger.observation',observation):
        result=run_dagger(args=args,env=env,actor=actor,optimizer=optimizer,tracker=Teacher(),motion=motion,mdp=mdp,
            clock=clock,demo=demo,context=None,default_q=None,chosen=torch.arange(4),teacher_hist=3,goal_frames=1,
            status=lambda **kw:None,stopped=lambda:False)
    assert result['fresh_native_student_transitions']==45
    for file in sorted((a.output/'dagger').glob('rollout_*.pt')):
        saved=torch.load(file,map_location='cpu',weights_only=False)
        assert saved['report']['native_student_transitions']==15
        for i,row in saved['data'].items():
            assert len(row['action'])==([2,3][i]*3)
            assert (row['episode_step']==0).sum()==3
            continuous=row['episode_step'][1:]>row['episode_step'][:-1]
            torch.testing.assert_close(row['history'][1:,-1,1][continuous],row['student_action'][:-1,0][continuous])
            expected=Teacher()(row['teacher_history'],None)
            torch.testing.assert_close(row['action'],expected)
            torch.testing.assert_close(row['next_state'][:,0]-row['state'][:,0],row['student_action'][:,0]*.01,atol=1e-8,rtol=1e-5)
    print('PASS real collection count scaling, no copied transition inflation, executed-action histories and reset boundaries')


if __name__=='__main__':main()
