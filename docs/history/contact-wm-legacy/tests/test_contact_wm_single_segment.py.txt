from types import SimpleNamespace
import torch
from holosoma.contact_wm.single_segment import objective
from holosoma.contact_wm.stable_contact import StableContactTracker, CONTRACT


def tracker_input():
    points=torch.zeros(1,6,32,3);points[:,:,1,0]=.04
    forces=torch.zeros_like(points);forces[:,:,:2,2]=20.
    mask=torch.zeros(1,6,32,dtype=torch.bool);mask[:,:,:2]=True
    return dict(position=torch.zeros(1,6,3),rotation=torch.eye(3).expand(1,6,3,3),
        points=points,forces=forces,mask=mask,normal=torch.tensor([0.,0.,1.]).expand(1,6,3),
        active=torch.ones(1,6,dtype=torch.bool),known=torch.ones(1,6,dtype=torch.bool),
        reset=torch.zeros(1,dtype=torch.bool))


def warm(tracker,data):
    for _ in range(CONTRACT['stable_frames']+1):result=tracker.update(**data)
    assert result['stable_contact_mask'].all()
    return result


def test_row_permutation_is_stable_and_switching_does_not_move_anchor():
    tracker=StableContactTracker(1,'cpu');data=tracker_input();first=warm(tracker,data)
    data['points']=data['points'].clone();data['points'][:,:,:2]=data['points'][:,:,:2].flip(-2)
    result=tracker.update(**data)
    assert result['stable_contact_mask'].all()
    torch.testing.assert_close(result['stable_contact_local'],first['stable_contact_local'])
    data['points'][:,:,0,0]+=.1
    result=tracker.update(**data)
    assert not result['stable_contact_mask'].any()
    torch.testing.assert_close(result['stable_contact_local'],first['stable_contact_local'])
    for _ in range(CONTRACT['stable_frames']-1):
        assert not tracker.update(**data)['stable_contact_mask'].any()
    result=tracker.update(**data)
    assert result['stable_contact_mask'].all()
    torch.testing.assert_close(result['stable_contact_local'],first['stable_contact_local'])


def test_load_change_row_disappearance_and_reset_gate_scoring():
    for change in ('load','count','reset','unknown','inactive'):
        tracker=StableContactTracker(1,'cpu');data=tracker_input();warm(tracker,data)
        if change=='load':data['forces'][:,:,:2,2]=40.
        if change=='count':data['mask'][:,:,1]=False
        if change=='reset':data['reset'][:]=True
        if change=='unknown':data['known'][:]=False
        if change=='inactive':data['active'][:]=False
        assert not tracker.update(**data)['stable_contact_mask'].any(),change


def test_nearest_row_matching_must_be_one_to_one():
    tracker=StableContactTracker(1,'cpu');data=tracker_input();warm(tracker,data)
    data['points'][:,:,1,0]=.001
    assert not tracker.update(**data)['stable_contact_mask'].any()


def task():
    class Geometry:
        def poses(self,state):
            return state[:,:3,None].transpose(1,2).expand(-1,6,-1),torch.eye(3).expand(len(state),6,3,3)
    class Surface:
        geometry=Geometry()
        def region_loss(self,state,target,parts,**kwargs):
            return dict(loss=((state[:,:3,None].transpose(1,2)-target)/.05).square().sum(-1))
    H,B=5,1
    states=torch.zeros(H,B,71,requires_grad=True)
    new=torch.zeros(B,6,dtype=torch.bool);new[:,2]=True
    learner=SimpleNamespace(establish=new,maintain=torch.zeros_like(new),segment_reference=torch.zeros(71),
        segment_surface=Surface(),normal_w=torch.tensor([0.,0.,1.]).expand(B,6,3))
    x=dict(target=torch.zeros(B,6,3),support=torch.zeros(B,6,3),previous=torch.zeros(B,29))
    native=dict(active=torch.ones(H,B,6,dtype=torch.bool),known=torch.ones(H,B,6,dtype=torch.bool),
        stable_contact_mask=torch.zeros(H,B,6,dtype=torch.bool),
        stable_contact_local=torch.zeros(H,B,6,3),live=torch.ones(H,B,dtype=torch.bool))
    return states,learner,x,native


def test_switching_frames_have_zero_contact_value_and_gradient():
    states,learner,x,native=task();H,B=states.shape[:2]
    native['stable_contact_mask'][2:4,:,2]=True
    native['stable_contact_local'][2:4,:,2,0]=.1
    native['stable_contact_local'][0:2,:,2,0]=100. # arbitrarily bad unstable points
    actions=torch.zeros(H,B,29)
    values,ok=objective(learner,x,states,actions,actions,torch.ones(H,B,dtype=torch.bool),native)
    assert ok and torch.isclose(values[1],torch.tensor(4.))
    values[1].backward()
    assert states.grad[2:4,0,0].gt(0).all()
    assert states.grad[[0,1,4]].eq(0).all()
    assert learner.segment_metrics['stable_contact_frames']==2
    assert learner.segment_metrics['terminal_guidance_loss']==0


def test_no_stable_contact_uses_terminal_geometry_not_unstable_rows():
    states,learner,x,native=task();H,B=states.shape[:2]
    x['target'][:,2,0]=.1
    native['stable_contact_local'][...,0]=100.
    actions=torch.zeros(H,B,29)
    values,ok=objective(learner,x,states,actions,actions,torch.ones(H,B,dtype=torch.bool),native)
    assert ok and torch.isclose(values[1],torch.tensor(4.))
    values[1].backward()
    assert states.grad[:-1].eq(0).all() and states.grad[-1,0,0]<0
    assert learner.segment_metrics['stable_contact_loss']==0
    assert learner.segment_metrics['missing_stable_parts']==1


def test_low_total_load_never_qualifies_even_with_unchanged_rows():
    tracker=StableContactTracker(1,'cpu');data=tracker_input()
    data['forces'][:,:,:2,2]=10.
    for _ in range(8):assert not tracker.update(**data)['stable_contact_mask'].any()


def test_body_translation_is_not_hidden_by_following_world_point():
    states,learner,x,native=task();H,B=states.shape[:2]
    with torch.no_grad():states[2:4,0,0]=.05
    native['stable_contact_mask'][2:4,:,2]=True
    actions=torch.zeros(H,B,29)
    values,_=objective(learner,x,states,actions,actions,torch.ones(H,B,dtype=torch.bool),native)
    assert torch.isclose(values[1],torch.tensor(1.))
    values[1].backward()
    assert states.grad[2:4,0,0].gt(0).all()


def test_live_mask_excludes_post_reset_stable_frames():
    states,learner,x,native=task();H,B=states.shape[:2]
    native['stable_contact_mask'][:,:,2]=True
    native['stable_contact_local'][3:,:,2,0]=100.
    native['live'][3:]=False
    actions=torch.zeros(H,B,29)
    values,ok=objective(learner,x,states,actions,actions,torch.ones(H,B,dtype=torch.bool),native)
    assert ok and values[1]==0
    assert learner.segment_metrics['stable_contact_frames']==3
