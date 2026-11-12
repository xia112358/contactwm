import torch
from supermimic.contact_paired_material import paired_material_error_sq,matched_material_index


def test_stable_witness_keeps_material_and_goal_paired():
    p=torch.zeros(1,3);R=torch.eye(3)[None];local=torch.tensor([[[0.,0.,0.],[.03,0.,0.]]]);target=torch.tensor([[[0.,0.,0.],[.10,0.,0.]]]);valid=torch.ones(1,2,dtype=torch.bool)
    loss,which=paired_material_error_sq(p,R,torch.tensor([[.03,0.,0.]]),torch.tensor([True]),local,target,valid)
    assert which.item()==1
    torch.testing.assert_close(loss,torch.tensor([.07**2]))


def test_guidance_uses_complete_pair_and_backpropagates():
    p=torch.tensor([[.02,0.,0.]],requires_grad=True);R=torch.eye(3)[None];local=torch.tensor([[[0.,0.,0.],[.03,0.,0.]]]);target=torch.tensor([[[.10,0.,0.],[.06,0.,0.]]]);valid=torch.ones(1,2,dtype=torch.bool)
    loss,which=paired_material_error_sq(p,R,torch.zeros(1,3),torch.tensor([False]),local,target,valid)
    assert which.item()==1
    loss.sum().backward();assert p.grad[0,0]<0


def test_invalid_candidates_cannot_match_even_when_closer():
    p=torch.zeros(1,3);R=torch.eye(3)[None];local=torch.tensor([[[0.,0.,0.],[.03,0.,0.]]]);valid=torch.tensor([[True,False]])
    assert matched_material_index(torch.tensor([[.03,0.,0.]]),p,R,local,valid).item()==0


def test_attempt_metric_does_not_cross_material_and_goal():
    from types import SimpleNamespace
    from supermimic.contact_dmo import loss_aligned_attempt_metrics
    T=B=1;state=torch.zeros(T,B,71);state[...,2]=.8;state[...,3]=1
    mask=torch.zeros(1,6,dtype=torch.bool);mask[0,5]=True
    candidate=torch.zeros(1,6,2,3);candidate[0,5,1,0]=.03
    goals=torch.zeros_like(candidate);goals[0,5,1,0]=.10
    stable=torch.zeros(T,B,6,dtype=torch.bool);stable[0,0,5]=True
    witness=torch.zeros(T,B,6,3);witness[0,0,5,0]=.03
    clock=SimpleNamespace(material_goal_mode=True,material_local=torch.zeros(1,6,4,3),material_targets=torch.zeros(1,6,4,3),paired_material_local=candidate,paired_material_targets=goals,paired_material_valid=torch.ones(1,6,2,dtype=torch.bool),event_ends=torch.tensor([1]),target_masks=mask,keep_masks=torch.zeros_like(mask),data={'point':torch.zeros(1,6,3),'s1':state[0]})
    data=dict(done=torch.ones(T,B,dtype=torch.bool),goal_index=torch.zeros(T,B,dtype=torch.long),s1=state,contact_stable=stable,contact_loss_point=witness,contact_loss_position=torch.zeros(T,B,6,3),contact_loss_rotation=torch.eye(3).expand(T,B,6,3,3).clone(),contact_loss_candidate=torch.tensor([[[0,0,0,0,0,1]]]))
    result=loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)
    assert result['success_rate']==0
    assert abs(result['best_contact_error_cm']-7.)<1e-4


def test_gpu_initial_support_cannot_bypass_release_even_at_its_own_goal():
    import warp as wp
    from supermimic.contact_gpu_stability import _filter_initial_support
    wp.init();device='cpu'
    def arr(v,d):return wp.array(v,dtype=d,device=device)
    heads=arr([0]*6,wp.int32);observed=arr([True],wp.bool);step=arr([0],wp.int64);substep=arr([0],wp.int32)
    reference=arr([True]*6,wp.bool);raw=arr([True]*6,wp.bool);point=arr([[0.,0.,0.]]*6,wp.vec3);target=arr([[0.,0.,0.]]*6,wp.vec3);origin=arr([[0.,0.,0.]],wp.vec3);tol=arr([.05]*6,wp.float32);initial=arr([False]*6,wp.bool);absence=arr([0]*6,wp.int32);qualified=arr([False]*6,wp.bool);witness=wp.zeros(6,dtype=wp.vec3,device=device);body=arr(list(range(6)),wp.int64);position=wp.from_torch(torch.zeros(1,6,3),dtype=wp.vec3);rotation=wp.from_torch(torch.eye(3).expand(1,6,3,3).contiguous(),dtype=wp.mat33);savep=wp.zeros(6,dtype=wp.vec3,device=device);saver=wp.zeros(6,dtype=wp.mat33,device=device);enabled=arr([True]*6,wp.bool)
    local=arr([[0.,0.,0.],[.03,0.,0.]]*6,wp.vec3);goal=arr([[0.,0.,0.],[1.,0.,0.]]*6,wp.vec3);valid=arr([True]*12,wp.bool);nxt=arr([-1],wp.int32);rowlocal=arr([[.03,0.,0.]],wp.vec3);load=arr([40.],wp.float32);rowpos=arr([[0.,0.,0.]],wp.vec3)
    args=[heads,observed,step,substep,2,reference,raw,point,target,origin,tol,initial,absence,qualified,witness,body,position,rotation,savep,saver,arr([0]*6,wp.int32),arr([False]*6,wp.bool),wp.zeros(6,dtype=wp.vec3,device=device),wp.zeros(6,dtype=wp.vec3,device=device),wp.zeros(6,dtype=wp.mat33,device=device),enabled,local,goal,valid,2,.01,nxt,rowlocal,load,rowpos]
    args.extend([arr([False]*6,wp.bool),wp.zeros(6,dtype=wp.int32,device=device),wp.zeros(6,dtype=wp.bool,device=device)])
    wp.launch(_filter_initial_support,6,inputs=args,device=device)
    assert not wp.to_torch(qualified).any()
    wp.to_torch(goal)[1::2]=0
    wp.launch(_filter_initial_support,6,inputs=args,device=device)
    assert not wp.to_torch(qualified).any()
    # A genuinely airborne entry may qualify the same material after landing.
    wp.to_torch(reference).zero_()
    wp.launch(_filter_initial_support,6,inputs=args,device=device)
    assert wp.to_torch(qualified).all()
    assert wp.to_torch(args[20]).tolist()==[1]*6


def test_unmatched_stable_native_row_guides_without_becoming_success_witness():
    from types import SimpleNamespace
    from supermimic.contact_dmo import _replay_events
    mask=torch.zeros(1,6,dtype=torch.bool);mask[0,5]=True
    clock=SimpleNamespace(fixed_goal_mode=True,paired_material_local=torch.zeros(1,6,2,3),event_ends=torch.tensor([1]),target_masks=mask,keep_masks=torch.zeros_like(mask))
    stable=torch.zeros(4,1,6,dtype=torch.bool);raw=stable.clone();raw[1,0,5]=True
    data={'contact_stable':stable,'contact_raw_stable':raw}
    events={};selected,terminal,_=_replay_events(data,clock,torch.tensor([0]),torch.tensor([4]),torch.tensor([False]),'accumulated_target',None,native_events=events,stable_contact_frames=2)
    assert selected[:,0,5].tolist()==[False,True,False,False]
    assert not events['mask'].any() and terminal[3,0]
    stable[2,0,5]=True;events={};selected,_,_=_replay_events(data,clock,torch.tensor([0]),torch.tensor([4]),torch.tensor([False]),'accumulated_target',None,native_events=events,stable_contact_frames=2)
    assert selected[:,0,5].tolist()==[False,False,True,False] and events['mask'][2,0,5]


def test_unmatched_real_row_still_produces_material_alignment_guidance():
    from supermimic.contact_paired_material import paired_raw_guidance_error_sq
    position=torch.zeros(1,3,requires_grad=True);rotation=torch.eye(3)[None]
    local=torch.tensor([[[.03,0.,0.]]]);target=local.clone();raw=torch.zeros(1,3);valid=torch.ones(1,1,dtype=torch.bool)
    loss=paired_raw_guidance_error_sq(position,rotation,raw,local,target,valid)
    torch.testing.assert_close(loss,torch.tensor([.03**2]))
    loss.sum().backward();assert position.grad[0,0]>0
