import pytest
import torch
import warp as wp
from supermimic.contact_gpu_stability import _filter_initial_support

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU substep sampling requires CUDA')

class Timeline:
    def __init__(self):
        z=lambda n,t:wp.zeros(n,dtype=t,device='cuda:0')
        self.heads=z(6,wp.int32);self.observed=wp.array([True],dtype=wp.bool,device='cuda:0')
        self.episode=z(1,wp.int64);self.sub=z(1,wp.int32);self.initial=z(6,wp.bool);self.stable=z(6,wp.bool);self.point=z(6,wp.vec3)
        self.target=z(6,wp.vec3);self.origins=z(1,wp.vec3);self.tolerance=wp.full(6,.05,dtype=wp.float32,device='cuda:0')
        self.pending=z(6,wp.bool);self.absence=z(6,wp.int32);self.qualified=z(6,wp.bool);self.witness=z(6,wp.vec3)
        self.i=0
    def step(self,active,stable=False,point=1.,known=True):
        wp.to_torch(self.heads).fill_(-1);wp.to_torch(self.heads)[0]=0 if active else -1
        wp.to_torch(self.observed)[0]=known;wp.to_torch(self.episode)[0]=self.i//10;wp.to_torch(self.sub)[0]=self.i%10
        wp.to_torch(self.stable).fill_(False);wp.to_torch(self.stable)[0]=stable;wp.to_torch(self.point).zero_();wp.to_torch(self.point)[0,0]=point
        wp.get_stream('cuda:0').wait_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        wp.launch(_filter_initial_support,6,inputs=[self.heads,self.observed,self.episode,self.sub,20,self.initial,self.stable,self.point,self.target,self.origins,self.tolerance,self.pending,self.absence,self.qualified,self.witness],device='cuda:0')
        wp.stream_from_torch(torch.cuda.current_stream()).wait_stream(wp.get_stream('cuda:0'));self.i+=1
        return bool(wp.to_torch(self.qualified)[0])


def test_initial_support_and_short_dropout_do_not_become_new_target():
    t=Timeline();assert not t.step(True,True)
    for _ in range(10):assert not t.step(False)
    assert not t.step(True,True)
    for _ in range(19):assert not t.step(False)
    assert not t.step(True,True)
    for _ in range(20):t.step(False)
    assert t.step(True,True)


def test_already_at_target_needs_no_release_and_initial_air_can_land():
    t=Timeline();assert t.step(True,True,point=.01)
    t=Timeline();assert not t.step(False);assert t.step(True,True,point=.2)


def test_unknown_is_not_release_and_reset_clears_history():
    t=Timeline();assert not t.step(True,True)
    for _ in range(21):assert not t.step(False,known=False)
    assert not t.step(True,True)
    t.i=0;assert not t.step(True,True)


def test_reference_contact_with_first_substep_dropout_remains_initial_support():
    t=Timeline();wp.to_torch(t.initial)[0]=True
    assert not t.step(False)
    assert not t.step(True,True,point=1.)
    for _ in range(20):t.step(False)
    assert t.step(True,True,point=.2)
