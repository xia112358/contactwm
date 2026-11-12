from types import SimpleNamespace
import torch
from supermimic.contact_material import world_points,point_error_sq,update_witness_error,mesh_anchors
from supermimic.contact_dmo import loss_aligned_attempt_metrics,swap


def anchors():return torch.tensor([[0.,-.04,0.],[0.,.04,0.],[.2,-.04,0.],[.2,.04,0.]])
def pitch(a):
    c=torch.cos(torch.tensor(a));s=torch.sin(torch.tensor(a))
    return torch.tensor([[c,0.,-s],[0.,1.,0.],[s,0.,c]])

def test_heel_hit_does_not_hide_forefoot_rotation():
    local=anchors();actual=world_points(torch.zeros(3),pitch(.5),local)
    assert torch.allclose(actual[:2],local[:2])
    assert point_error_sq(actual,local).sqrt()>.05
    assert point_error_sq(local,local)==0

def test_translation_and_rotation_have_finite_restoring_gradients():
    p=torch.tensor([.1,.02,0.],requires_grad=True);a=torch.tensor(.3,requires_grad=True)
    c=a.cos();s=a.sin();z=a*0;one=z+1
    R=torch.stack([c,z,-s,z,one,z,s,z,c]).reshape(3,3)
    actual=world_points(p,R,anchors());cost=point_error_sq(actual,anchors())/.05**2
    cost.backward();assert p.grad[0]>0 and a.grad>0

def test_stable_pose_swap_retains_substep_forward_and_fk_gradient():
    p=torch.zeros(3,requires_grad=True);pred=world_points(p,torch.eye(3),anchors())
    measured=world_points(torch.tensor([.1,0.,0.]),pitch(.3),anchors())
    points=swap(pred,measured);assert torch.equal(points,measured)
    point_error_sq(points,anchors()).backward();assert p.grad[0]>0

def test_latest_qualified_witness_overwrites_earlier_success_but_not_jitter():
    target=anchors();previous=torch.tensor([float('inf')]);good=target[None];bad=good+.1
    previous=update_witness_error(previous,torch.tensor([True]),good,good)
    assert previous.item()==0
    assert update_witness_error(previous,torch.tensor([False]),bad,good).item()==0
    assert update_witness_error(previous,torch.tensor([True]),bad,good).sqrt().item()>.05

def test_metrics_latch_completed_fixed_correspondence_and_count_missing_failure():
    T,B=3,3;state=torch.zeros(T,B,71);state[...,2]=.8;state[...,3]=1
    mask=torch.zeros(1,6,dtype=torch.bool);mask[0,0]=True
    local=anchors().expand(6,4,3).clone()
    clock=SimpleNamespace(material_goal_mode=True,material_local=local[None],material_targets=local[None],event_ends=torch.tensor([1]),target_masks=mask,keep_masks=torch.zeros_like(mask),data={'point':torch.zeros(1,6,3),'s1':state[0,:1]})
    stable=torch.zeros(T,B,6,dtype=torch.bool);stable[0,:2,0]=True;stable[1,0,0]=True
    R=torch.eye(3).expand(T,B,6,3,3).clone();R[1,0,0]=pitch(.5)
    data=dict(done=torch.tensor([[False]*B,[False]*B,[True]*B]),goal_index=torch.zeros(T,B,dtype=torch.long),s1=state,contact_stable=stable,contact_loss_point=torch.zeros(T,B,6,3),contact_loss_position=torch.zeros(T,B,6,3),contact_loss_rotation=R)
    result=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert result['success_rate']==2/3 and result['missing_witness_attempts']==1
    expected=0.
    assert abs(result['best_contact_error_cm']-expected)<1e-5
    assert result['contact_metric']=='stable_hand_contact_fixed_foot_knee_material'

@__import__('pytest').mark.parametrize('part',[0,2,4])
def test_replay_fixed_points_use_stable_pose_and_missing_endpoint_gradient(monkeypatch,part):
    from torch import nn
    import supermimic.contact_dmo as dmo
    class Actor(nn.Module):
        hist=1
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.zeros(2))
        def forward(self,history,condition):return self.weight[:,None].expand(-1,29)
    class Model(nn.Module):
        def step(self,state,action,xi):return (state[0]+action[:,:1]*.01,*state[1:])
        def next_context(self,state,action,xi,xi_n):return xi
    class Region:
        def region_loss(self,state,target,parts,**kwargs):
            loss=((state[0][:,None]-target)/.05).square().sum(-1)
            return dict(loss=loss,normal=loss,tangent=torch.zeros_like(loss))
    monkeypatch.setattr(dmo,'observation',lambda state,*a:state[0][:,:1]*0)
    T,B=3,2;state=torch.zeros(T,B,71);state[...,3]=1;state[...,:3]=.2
    active=torch.zeros(T,B,6,dtype=torch.bool);stable=active.clone();stable[1,0,part]=True
    done=torch.zeros(T,B,dtype=torch.bool);done[-1]=True
    masks=torch.zeros(T,6,dtype=torch.bool);masks[:,part]=True;local=anchors().expand(6,4,3).clone()
    clock=SimpleNamespace(material_contact_tolerances=torch.tensor([.05]),segment_starts=torch.tensor([0]),segment_stops=torch.tensor([T]),event_ends=torch.tensor([T]),target_masks=masks,keep_masks=torch.zeros_like(masks),fixed_goal_mode=True,material_goal_mode=True,material_local=local[None],material_targets=local[None]+.1,data={'point':torch.full((T,6,3),.1),'s1':state[:,0]},condition=lambda state,*a,**kw:state[0][:,:1]*0)
    data=dict(s=state,s1=state,a=torch.zeros(T,B,29),a_prev=torch.zeros(T,B,29),xi=torch.zeros(T,B,52),next_xi=torch.zeros(T,B,52),ep_step=torch.arange(T)[:,None].expand(T,B),goal_index=torch.zeros(T,B,dtype=torch.long),done=done,active=active,known=torch.ones_like(active),contact_stable=stable,contact_first=active.clone(),contact_goal=active.clone(),point=torch.full((T,B,6,3),1e6),contact_loss_point=torch.full((T,B,6,3),1e6),contact_loss_position=torch.full((T,B,6,3),.3),contact_loss_rotation=torch.eye(3).expand(T,B,6,3,3).clone())
    if part==2:data['contact_loss_point'][:]=.3  # hand uses the stable native witness
    if part==4:data['contact_loss_point'][:]=100  # knee still uses the fixed material point
    geometry=SimpleNamespace(poses=lambda s:(s[0][:,None].expand(-1,6,-1),torch.eye(3).expand(B,6,3,3)))
    actor=Actor();loss,details=dmo.replay_loss(actor,Model(),clock,geometry,None,lambda x:x,None,data,contact_surface=Region(),pose_weight=0,completion_mode='accumulated_target',stable_contact_frames=2)
    loss.backward()
    # Stable witness (.3-.1)^2: 48; missing endpoint (.2-.1)^2: 12.
    assert abs(details['terminal_contact']-30)<1e-4
    assert details['stable_contact_terms']==1 and details['terminal_guidance_terms']==1
    assert torch.isfinite(actor.weight.grad).all() and actor.weight.grad.gt(0).all()


def test_native_filter_latches_same_solve_pose_and_excludes_old_support():
    import warp as wp
    from supermimic.contact_gpu_stability import _filter_initial_support
    wp.init();device='cpu'
    def arr(v,d):return wp.array(v,dtype=d,device=device)
    heads=arr([0]*6,wp.int32);observed=arr([True],wp.bool);step=arr([0],wp.int64);substep=arr([0],wp.int32)
    reference=arr([True]*6,wp.bool);raw_stable=arr([True]*6,wp.bool)
    point=arr([[0.,0.,0.]]*6,wp.vec3);target=arr([[1.,0.,0.]]*6,wp.vec3);origin=arr([[0.,0.,0.]],wp.vec3)
    tolerance=arr([.05]*6,wp.float32);initial=arr([False]*6,wp.bool);absence=arr([0]*6,wp.int32)
    qualified=arr([False]*6,wp.bool);witness=wp.zeros(6,dtype=wp.vec3,device=device)
    body=arr(list(range(6)),wp.int64)
    p=wp.from_torch(torch.full((1,6,3),.2),dtype=wp.vec3);R=wp.from_torch(torch.eye(3).expand(1,6,3,3).contiguous(),dtype=wp.mat33)
    saved_p=wp.zeros(6,dtype=wp.vec3,device=device);saved_R=wp.zeros(6,dtype=wp.mat33,device=device)
    material_enabled=arr([False]*6,wp.bool);material_target=arr([[0.,0.,0.]]*6,wp.vec3);material_goal=arr([[1.,0.,0.]]*6,wp.vec3);material_valid=arr([True]*6,wp.bool)
    next_row=arr([-1],wp.int32);row_local=arr([[0.,0.,0.]],wp.vec3);row_load=arr([1.],wp.float32);row_pos=arr([[0.,0.,0.]],wp.vec3)
    inputs=[heads,observed,step,substep,2,reference,raw_stable,point,target,origin,tolerance,initial,absence,qualified,witness,body,p,R,saved_p,saved_R,arr([0]*6,wp.int32),arr([False]*6,wp.bool),wp.zeros(6,dtype=wp.vec3,device=device),wp.zeros(6,dtype=wp.vec3,device=device),wp.zeros(6,dtype=wp.mat33,device=device),material_enabled,material_target,material_goal,material_valid,1,.01,next_row,row_local,row_load,row_pos]
    inputs.extend([arr([False]*6,wp.bool),wp.zeros(6,dtype=wp.int32,device=device),wp.zeros(6,dtype=wp.bool,device=device)])
    wp.launch(_filter_initial_support,6,inputs=inputs,device=device)
    assert not wp.to_torch(qualified).any()  # original support far from the next goal
    substep.fill_(1);heads.fill_(-1);raw_stable.fill_(False)
    for _ in range(2):wp.launch(_filter_initial_support,6,inputs=inputs,device=device)
    heads.fill_(0);raw_stable.fill_(True)
    wp.launch(_filter_initial_support,6,inputs=inputs,device=device)
    assert wp.to_torch(qualified).all();assert torch.allclose(wp.to_torch(saved_p),torch.full((6,3),.2))
    raw_stable.fill_(False);wp.to_torch(p).fill_(.8)
    wp.launch(_filter_initial_support,6,inputs=inputs,device=device)
    assert torch.allclose(wp.to_torch(saved_p),torch.full((6,3),.2))  # later jitter cannot move the witness
    # A stable shin row must not qualify a different fixed knee material.
    heads.fill_(0);raw_stable.fill_(True);material_enabled.fill_(True);wp.to_torch(row_local).fill_(.1)
    substep.fill_(0);step.fill_(1)
    wp.launch(_filter_initial_support,6,inputs=inputs,device=device)
    assert not wp.to_torch(qualified).any()
    wp.to_torch(row_local).zero_();substep.fill_(1)
    wp.launch(_filter_initial_support,6,inputs=inputs,device=device)
    assert wp.to_torch(qualified).all()


def test_sequence_metrics_use_same_latched_material_witness():
    from supermimic.contact_sequence import sequence_attempt_metrics
    T,B=3,1;state=torch.zeros(T,B,71);state[...,2]=.8;state[...,3]=1
    local=anchors().expand(6,4,3).clone();mask=torch.zeros(1,6,dtype=torch.bool);mask[0,0]=True
    clock=SimpleNamespace(material_contact_tolerances=torch.tensor([.05]),material_goal_mode=True,material_local=local[None],material_targets=local[None],event_ends=torch.tensor([1]),target_masks=mask,keep_masks=torch.zeros_like(mask),data={'point':torch.zeros(1,6,3),'s1':state[0]})
    stable=torch.zeros(T,B,6,dtype=torch.bool);stable[0,0,0]=stable[1,0,0]=True
    R=torch.eye(3).expand(T,B,6,3,3).clone();R[1,0,0]=pitch(.5)
    done=torch.tensor([[False],[False],[True]])
    d=dict(done=done,term=torch.zeros_like(done),goal_index=torch.zeros(T,B,dtype=torch.long),s1=state,active=torch.ones_like(stable),known=torch.ones_like(stable),point=torch.zeros(T,B,6,3),contact_loss_point=torch.zeros(T,B,6,3),stage_done=done,stage_success=done.clone(),stage_step=torch.arange(T)[:,None],contact_stable=stable,contact_loss_position=torch.zeros(T,B,6,3),contact_loss_rotation=R)
    result=sequence_attempt_metrics(d,clock,1)
    expected=0.
    assert result['success_rate']==1 and abs(result['best_contact_error_cm']-expected)<1e-5


def test_material_goal_reset_drops_predecessor_witness():
    from test_contact_reset_efficiency import task_fixture
    task,_,_=task_fixture(True);task.material_goal_mode=True
    task.material_error_sq=torch.zeros(task.N,6)
    task.reset(torch.tensor([0,2]))
    assert torch.isinf(task.material_error_sq[[0,2]]).all()
    assert (task.material_error_sq[[1,3]]==0).all()

def test_hands_use_native_contact_while_feet_and_knees_keep_material():
    from supermimic.contact_material import mixed_error_sq
    local=anchors().expand(6,4,3).clone();target=local.clone();native=torch.zeros(6,3);goals=torch.zeros(6,3)
    actual=local.clone();actual[2:4,:,0]+=.03
    assert torch.equal(mixed_error_sq(actual,target,native,goals),torch.zeros(6))
    native[2:4,0]=.02;actual[4:,:,0]+=.03
    result=mixed_error_sq(actual,target,native,goals)
    torch.testing.assert_close(result[2:4],torch.full((2,),.02**2))
    torch.testing.assert_close(result[4:],torch.full((2,),.03**2))
    actual[0]=world_points(torch.zeros(3),pitch(.5),local[0])
    assert mixed_error_sq(actual,target,native,goals)[0].sqrt()>.05


def test_pose_phase_uses_aligned_time_not_late_point_medoid():
    from supermimic.contact_material_targets import aligned_pose_index
    # Raw clocks differ across executions; late support can roll onto a toe.
    a=dict(ticks=[740,740,1080],rows=[2,3,4],aligned_frames=[74.,74.,108.])
    b=dict(ticks=[910,1250],rows=[5,6],aligned_frames=[74.,108.])
    assert aligned_pose_index(a,74)==0
    assert aligned_pose_index(b,74)==0


def test_pose_consensus_keeps_rigid_observed_pose_and_equal_execution_votes():
    from supermimic.contact_material_targets import pose_medoid
    local=anchors()
    candidates=torch.stack((local,local+.001,world_points(torch.zeros(3),pitch(.7),local)))
    chosen,_=pose_medoid(candidates)
    assert chosen in (0,1)
    assert torch.equal(candidates[chosen],local) or torch.equal(candidates[chosen],local+.001)
    # Repeated native rows must not duplicate a successful execution's vote.
    from supermimic.contact_material_targets import aligned_pose_index
    c=dict(ticks=[100]*100+[200],rows=list(range(100))+[1],aligned_frames=[10.]*100+[20.])
    assert aligned_pose_index(c,20)==100


def test_material_pose_keeps_population_motion_carrier_despite_footfall_drift():
    from supermimic.contact_material_targets import representative_pose_index
    candidates=[dict(source='/pool/capture-seed4/motion-006.pt'),dict(source='/pool/capture-seed0/motion-006.pt')]
    assert representative_pose_index(candidates,'capture-seed0/motion-006')==1


def test_source_alignment_maps_consensus_time_back_to_actual_carrier_frames():
    import numpy as np
    from supermimic.contact_source_pool import carrier_aligned_frames
    # The carrier clock itself is not identity. A carrier witness at physical
    # frame 3 must remain frame 3; a second source maps through consensus.
    carrier=np.array([0.,1.5,3.,4.5,6.]);other=np.array([0.,1.,2.,3.,4.])
    np.testing.assert_allclose(carrier_aligned_frames([3.],carrier,carrier),[3.])
    np.testing.assert_allclose(carrier_aligned_frames([2.],other,carrier),[3.])
    np.testing.assert_allclose(carrier_aligned_frames([6.9],carrier,carrier),[6.9])
