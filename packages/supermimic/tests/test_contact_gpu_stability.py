"""CPU reference vs graph-replayed GPU row matching, resets and witness points."""
from types import SimpleNamespace
import pytest
import torch
import warp as wp
from supermimic.contact_gpu_stability import _qualify, _score
from supermimic.contact_stability import NativeContactStability, CONTRACT

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph equivalence requires a GPU')


class Pair:
    def __init__(self):
        self.device='cuda:0';self.capacity=16
        alloc=lambda n,d:wp.zeros(n,dtype=d,device=self.device)
        self.a={};self.o={}
        for key,d in [('heads',wp.int32),('next',wp.int32),('local',wp.vec3),('pos',wp.vec3),('load',wp.float32),('robot',wp.int32),('terrain',wp.int32)]:
            self.a[key]=alloc(6 if key=='heads' else 16,d);self.o[key]=wp.clone(self.a[key])
        self.o['heads'].fill_(-1)
        self.used=alloc(16,wp.int32);self.old_used=alloc(16,wp.int32)
        self.part=wp.array(range(6),dtype=wp.int64,device=self.device)
        self.p=alloc((1,6),wp.vec3)
        self.R=wp.from_torch(torch.eye(3,device=self.device).expand(1,6,3,3).contiguous(),dtype=wp.mat33)
        self.targets=alloc(6,wp.vec3);self.origins=alloc(1,wp.vec3)
        self.step=alloc(1,wp.int64);self.old_step=wp.full(6,-2,dtype=wp.int64,device=self.device);self.substep=alloc(1,wp.int32)
        self.f={k:alloc(6,d) for k,d in [('stable',wp.bool),('row',wp.bool),('load_stable',wp.bool),('speed',wp.float32),('load',wp.float32)]}
        self.qualified=alloc(6,wp.bool);self.witness=alloc(6,wp.vec3)
        self.reference=NativeContactStability(.002)
        wp.load_module(module='supermimic.contact_gpu_stability',device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:self.launch()
        self.graph=capture.graph

    def launch(self):
        a=self.a;o=self.o;f=self.f
        self.used.zero_();self.old_used.zero_()
        wp.launch(_qualify,6,inputs=[a['heads'],a['next'],a['local'],a['pos'],a['load'],a['robot'],a['terrain'],o['heads'],o['next'],o['local'],o['pos'],o['load'],o['robot'],o['terrain'],self.used,self.old_used,self.part,self.p,self.R,self.targets,self.origins,self.step,self.old_step,self.substep,1,.002,CONTRACT['min_load_n'],CONTRACT['max_relative_load_change'],CONTRACT['max_material_speed_m_s'],CONTRACT['row_match_distance_m'],f['stable'],f['row'],f['load_stable'],f['speed'],f['load'],self.qualified,self.witness],device=self.device)
        for k in a:wp.copy(o[k],a[k])

    def update(self,step,points,loads,robot=None,terrain=None,shift=0.,target=None):
        n=len(points);pts=torch.tensor(points,device=self.device,dtype=torch.float32).reshape(n,3)
        force=torch.zeros(n,3,device=self.device);force[:,2]=torch.tensor(loads,device=self.device)
        robot=torch.tensor([1]*n if robot is None else robot,device=self.device,dtype=torch.int32)
        terrain=torch.tensor([2]*n if terrain is None else terrain,device=self.device,dtype=torch.int32)
        p=torch.zeros(1,6,3,device=self.device);p[0,0,0]=shift
        wp.to_torch(self.p).copy_(p);wp.to_torch(self.step)[0]=step
        wp.to_torch(self.targets).zero_()
        if target is not None:wp.to_torch(self.targets)[0]=torch.tensor(target,device=self.device)
        for k in self.a:wp.to_torch(self.a[k]).zero_()
        wp.to_torch(self.a['heads']).fill_(-1)
        if n:wp.to_torch(self.a['heads'])[0]=0
        wp.to_torch(self.a['next'])[:n]=torch.arange(1,n+1,device=self.device,dtype=torch.int32)
        if n:wp.to_torch(self.a['next'])[n-1]=-1
        wp.to_torch(self.a['pos'])[:n]=pts;wp.to_torch(self.a['local'])[:n]=pts-p[0,0]
        wp.to_torch(self.a['load'])[:n]=force[:,2];wp.to_torch(self.a['robot'])[:n]=robot;wp.to_torch(self.a['terrain'])[:n]=terrain
        wp.get_stream(self.device).wait_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        wp.capture_launch(self.graph)
        wp.stream_from_torch(torch.cuda.current_stream()).wait_stream(wp.get_stream(self.device))
        view=SimpleNamespace(evidence=SimpleNamespace(known=torch.ones(1,6,dtype=torch.bool,device=self.device),rows=dict(position_w=pts,shape0=terrain.long(),shape1=robot.long())),force_w=force,groups=torch.zeros(n,device=self.device,dtype=torch.long),rows=torch.arange(n,device=self.device),robot_side=torch.ones(n,device=self.device,dtype=torch.long))
        expected=self.reference.update(view,(p,wp.to_torch(self.R)),torch.tensor([step],device=self.device))
        for key,out in [('contact_stable','stable'),('contact_row_stable','row'),('contact_load_stable','load_stable')]:
            assert torch.equal(wp.to_torch(self.f[out]).reshape(1,6),expected[key]),key
        torch.testing.assert_close(wp.to_torch(self.f['speed']).reshape(1,6),expected['contact_material_speed'],atol=1e-5,rtol=1e-5)
        if bool(expected['contact_stable'][0,0]):
            target_point=wp.to_torch(self.targets)[0]
            best=(pts-target_point).square().sum(-1).argmin()
            torch.testing.assert_close(wp.to_torch(self.witness)[0],pts[best],rtol=0,atol=0)
        return bool(expected['contact_stable'][0,0])


def test_graph_row_permutation_switching_load_and_episode_reset():
    p=Pair();a=[[0.,0.,0.],[.03,0.,0.]]
    assert not p.update(0,a,[50,50])
    assert p.update(1,list(reversed(a)),[50,50])
    assert not p.update(2,a,[50,50],terrain=[3,3])
    assert p.update(3,a,[50,50],terrain=[3,3])
    assert not p.update(4,a,[5,5],terrain=[3,3])
    assert not p.update(5,[],[])
    assert not p.update(0,a,[50,50])
    assert p.update(1,a,[50,50])


def test_graph_material_speed_and_row_count_are_not_representative_speed():
    p=Pair();a=[[0.,0.,0.],[.03,0.,0.]]
    p.update(0,a,[50,50]);assert p.update(1,a,[50,50],shift=.0005)
    assert not p.update(2,a,[50,50],shift=.002)
    assert not p.update(3,a[:1],[100],shift=.002)
    assert p.update(4,a[:1],[100],shift=.002)


def test_native_point_ties_use_same_cuda_reduction_and_row_order():
    p=Pair();a=[[-.4582638144493103,.07427710294723511,.7637933492660522],[-.4582620859146118,.07428722083568573,.7637933492660522]]
    p.update(0,a,[50,50]);assert p.update(1,a,[50,50])


def test_greedy_bijection_does_not_double_match_one_old_row():
    p=Pair();p.update(0,[[0,0,0],[.03,0,0]],[50,50])
    assert not p.update(1,[[.001,0,0],[.002,0,0]],[50,50])
