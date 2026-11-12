import numpy as np
import torch
from torch import nn
from holosoma.contact_wm.core import ContactPolicy,ContactWorldModel,ContactGoal,fit_loss,unroll,gradient_alignment
from holosoma.contact_wm.validation import contact_transitions,native_success


def goal():
    target=torch.zeros(1,6,3);target[:,2,0]=.3
    normal=torch.zeros_like(target);normal[...,2]=1
    return ContactGoal(target,normal,torch.tensor([[0,0,1,0,0,0]],dtype=torch.bool),
        torch.tensor([[1,1,0,0,0,0]],dtype=torch.bool),torch.zeros_like(target),torch.ones(1,6),3,2)


def test_direct_policy_preserves_prior_and_trains_all_layers():
    prior=nn.Sequential(nn.Linear(567,32),nn.ELU(),nn.Linear(32,29))
    policy=ContactPolicy(prior);obs=torch.randn(3,567);extra=torch.randn(3,25)
    torch.testing.assert_close(policy(obs,extra),prior(obs),atol=1e-6,rtol=1e-6)
    policy(obs,extra).square().mean().backward()
    assert policy.net[0].weight.grad[:,567:].abs().sum()>0
    assert policy.net[-1].weight.grad.abs().sum()>0


class AnalyticTestDynamics(nn.Module):
    # Unit-test dynamics only, never used as a physical robot WM.
    def __init__(self):super().__init__();self.gain=nn.Parameter(torch.tensor(.1))
    def forward(self,s,a,c,x):
        next_state=torch.cat((s[:,:1]+self.gain*a[:,:1],s[:,1:]),-1)
        force=a[:,:1,None].expand(-1,6,3)*0+2
        return dict(state=next_state,contact_logits=a[:,:1].expand(-1,6),contact=torch.sigmoid(a[:,:1]).expand(-1,6),force_w=force)


def test_bptt_reaches_every_early_action_and_freezes_only_wm_weights():
    torch.manual_seed(3);policy=ContactPolicy(nn.Sequential(nn.Linear(567,16),nn.Tanh(),nn.Linear(16,29)))
    wm=AnalyticTestDynamics();state=torch.zeros(1,71);state[:,6]=1
    def observe(s,c,h,a,g,t):
        obs=torch.cat((s,a,torch.zeros(1,467)),1)
        return obs,torch.cat((g.normal_w.flatten(1),g.maintain.float(),torch.ones(1,1)*(g.arrival_step-t)),1),h
    def geometry(s):return s[:,:3,None].transpose(1,2).expand(-1,6,-1)
    result=unroll(policy,wm,state,torch.zeros(1,6),None,torch.zeros(1,29),goal(),horizon=4,
        observation=observe,geometry=geometry,context=lambda s,c,p:torch.zeros(1,1),
        stability=lambda s,p,g:s[:,0].square().mean())
    result['losses']['loss'].backward()
    assert all(a.grad is not None and a.grad.abs().sum()>0 for a in result['actions'])
    assert wm.gain.grad is None and wm.gain.requires_grad
    assert torch.isfinite(result['losses']['loss'])


def test_native_success_requires_real_activation_load_and_hold():
    n=5;a=np.ones((n,6),bool);k=a.copy();p=np.zeros((n,6,3));f=p.copy();f[...,2]=10
    g=goal();g.position_w.zero_()
    fields={name:getattr(g,name)[0].numpy() for name in ('position_w','normal_w','support_position_w','establish','maintain','min_normal_force')}
    kwargs=dict(active=a,known=k,force_w=f,point_w=p,point_velocity_w=p,failed=np.zeros(n,bool),goal=fields,arrival=2,hold=2)
    assert native_success(**kwargs)['success']
    a[3,2]=False
    assert not native_success(**kwargs)['success'] # Distance zero is insufficient.
    a[:]=True;f[3,2]=0
    assert not native_success(**kwargs)['success']
    f[...,2]=10;k[0,0]=False
    assert not native_success(**kwargs)['success'] # Missing initial support evidence.


def test_native_event_has_preexisting_support_and_no_left_censored_onset():
    a=np.zeros((25,6),bool);a[:,:2]=True;a[10:,2]=True;k=np.ones_like(a)
    events=contact_transitions(a,k)
    assert len(events)==1 and events[0]['arrival']==10
    assert events[0]['establish'].tolist()==[False,False,True,False,False,False]
    k[9,2]=False
    assert not contact_transitions(a,k)


def test_wm_unknown_force_and_reset_masks_are_finite():
    m=ContactWorldModel(4,hidden=16);s=torch.zeros(2,71);s[:,6]=1
    pred=m(s,torch.zeros(2,29),torch.zeros(2,6),torch.zeros(2,4))
    f=torch.zeros(2,6,3);f[:,2]=float('nan');known=torch.ones(2,6,dtype=torch.bool);known[:,2]=False
    loss=fit_loss(pred,s,torch.zeros(2,6),f,known,torch.tensor([True,False]));loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_gradient_probe_reports_alignment_not_contact_success():
    grad=torch.tensor([1.,2.]);directions=torch.eye(2)
    result=gradient_alignment(grad,directions,torch.tensor([.01,.02]),torch.tensor([-.01,-.02]),.01)
    torch.testing.assert_close(result['cosine'],torch.tensor(1.))
    assert result['sign_agreement']==1


def test_php_history_layout_roundtrip_and_witness_gradient():
    from holosoma.contact_wm.observations import pack_history,unpack_history
    from holosoma.contact_wm.geometry import WitnessGeometry
    from somaforge_core import G1_29DOF_JOINT_ORDER
    h=torch.randn(2,5,93)
    torch.testing.assert_close(unpack_history(pack_history(h)),h)
    state=torch.zeros(2,71);state[:,6]=1;state.requires_grad_()
    geometry=WitnessGeometry(list(G1_29DOF_JOINT_ORDER),torch.zeros(6,3))
    points=geometry(state);points.square().sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad[:,13:42].abs().sum()>0
def test_world_height_raster_pose_gradient():
    from holosoma.contact_wm.height_context import HeightRaster
    import torch
    y,x=torch.meshgrid(torch.linspace(-3,3,121),torch.linspace(-3,3,121),indexing='ij')
    reader=HeightRaster((.2*x+.1*y)[None],torch.tensor([[-3.,-3.]]),.05)
    state=torch.zeros(1,71);state[:,6]=1;state[:,2]=.5;state.requires_grad_()
    actual=reader(state)
    expected=.2*reader.local[:,0]+.1*reader.local[:,1]-.5
    torch.testing.assert_close(actual[0],expected,atol=1e-6,rtol=1e-5)
    gradient=torch.autograd.grad(actual.sum(),state)[0]
    torch.testing.assert_close(gradient[0,:3],torch.tensor([24.2,12.1,-121.]),atol=1e-4,rtol=1e-5)


def test_changing_solver_centroid_is_not_material_slip():
    from holosoma.contact_wm.analyze import material_point_velocity
    pos=torch.zeros(3,1,3)
    rotation=torch.eye(3).expand(3,1,3,3)
    points=torch.tensor([[[0.,0.,0.]],[[.1,0.,0.]],[[.2,0.,0.]]])
    # A static foot's contact patch may move while its material stays still.
    torch.testing.assert_close(material_point_velocity(points,pos,rotation,.02),torch.zeros_like(points))
    pos[:,0,1]=torch.tensor([0.,.02,.04])
    velocity=material_point_velocity(points,pos,rotation,.02)
    torch.testing.assert_close(velocity[1:,0],torch.tensor([[0.,1.,0.],[0.,1.,0.]]))


def test_wm_selection_checks_each_holdout_and_rejects_nan():
    import copy
    from holosoma.contact_wm.learn_loop import candidate_gate
    metrics=dict(joint_rmse_rad=.01,force_rmse_N=20.)
    before={'motion0':{'all':metrics.copy(),'switch':metrics.copy()},'motion7':{'all':metrics.copy(),'switch':None}}
    report=dict(before_by_recording=before,after_by_recording=copy.deepcopy(before))
    assert candidate_gate(report,.1)['accepted']
    report['after_by_recording']['motion7']['all']['force_rmse_N']=23.
    assert not candidate_gate(report,.1)['accepted']
    report['after_by_recording']['motion7']['all']['force_rmse_N']=float('nan')
    assert not candidate_gate(report,.1)['accepted']


def test_runtime_goal_replaces_reference_condition_only():
    from holosoma.contact_wm.runtime import goal_observation
    g=goal();s=torch.zeros(1,71);s[:,6]=1
    obs=torch.randn(1,567);hint=torch.tensor([[.2,.1,0.]])
    conditioned,extra=goal_observation(obs,s,g,hint,0)
    torch.testing.assert_close(conditioned[:,:12],obs[:,:12])
    torch.testing.assert_close(conditioned[:,39:],obs[:,39:])
    torch.testing.assert_close(conditioned[:,36:39],hint)
    torch.testing.assert_close(conditioned[:,30:36],(g.maintain|g.establish).float())
    torch.testing.assert_close(extra[:,-1],torch.tensor([.06]))


def test_execution_failure_reads_proxy_and_does_not_use_reference():
    from types import SimpleNamespace
    from holosoma.contact_wm.runtime import execution_failure
    class Proxy:
        def __getitem__(self,key):return root[key]
    root=torch.zeros(2,13);root[:,2]=.8;root[:,6]=1
    env=SimpleNamespace(simulator=SimpleNamespace(robot_root_states=Proxy()))
    assert not execution_failure(env).any()
    root[1,2]=.1
    assert execution_failure(env).tolist()==[False,True]


def test_sustained_support_contract_keeps_activation_load_and_slip_checks():
    g=goal();g.position_w.zero_()
    fields={name:getattr(g,name)[0].numpy() for name in ('position_w','normal_w','support_position_w','establish','maintain','min_normal_force')}
    a=np.ones((5,6),bool);p=np.zeros((5,6,3));p[:,:2,0]=.08;f=np.zeros_like(p);f[...,2]=10
    args=dict(active=a,known=a.copy(),force_w=f,point_w=p,point_velocity_w=np.zeros_like(p),
              failed=np.zeros(5,bool),goal=fields,arrival=2,hold=2)
    assert not native_success(**args)['success']
    assert native_success(**args,position_required=fields['establish'])['success']
    a[0,0]=False
    assert not native_success(**args,position_required=fields['establish'])['success']


def test_target_evaluation_reports_position_error_and_gain():
    from holosoma.contact_wm.target_response import target_grid,response_metrics
    targets=target_grid(.04)
    following=targets+torch.tensor([.01,0.,0.])
    metrics=response_metrics(following,targets,exclude_center=True)
    torch.testing.assert_close(torch.tensor(metrics['target_to_endpoint_jacobian_xy']),torch.eye(2),atol=1e-6,rtol=1e-5)
    assert 4 not in metrics['evaluated_indices'] and abs(metrics['mean_target_error_m']-.01)<1e-6



def test_supported_climbing_inclination_does_not_trigger_gross_failure():
    from types import SimpleNamespace
    from holosoma.contact_wm.runtime import execution_failure
    root=torch.zeros(3,13);root[:,2]=1.2
    root[:,3]=torch.sin(torch.tensor(.75));root[:,6]=torch.cos(torch.tensor(.75))
    root[1,2]=.1;root[2,7]=float('nan')
    env=SimpleNamespace(simulator=SimpleNamespace(robot_root_states=root))
    assert execution_failure(env).tolist()==[False,True,True]


def test_planned_points_are_absolute_and_require_explicit_frame(tmp_path):
    import json
    import pytest
    from holosoma.contact_wm.core import load_planned_contact_points
    from holosoma.contact_wm.geometry import LINKS
    points=torch.arange(18,dtype=torch.float32).reshape(6,3)/10
    data=dict(frame='world',links=list(LINKS),position_w=points.tolist())
    path=tmp_path/'planned.json';path.write_text(json.dumps(data))
    torch.testing.assert_close(load_planned_contact_points(path,device='cpu'),points[None],rtol=0,atol=0)
    data['frame']='root';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='world frame'):load_planned_contact_points(path,device='cpu')
    data['frame']='world';data['position_w'][2][0]=float('nan');path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='finite'):load_planned_contact_points(path,device='cpu')


def test_contact_geometry_penalizes_absolute_miss_at_arrival_and_hold():
    from holosoma.contact_wm.core import contact_objective
    g=goal();g.arrival_step=3;g.hold_steps=2
    points=g.position_w[None].expand(4,-1,-1,-1).clone()
    predictions=[dict(contact_logits=torch.zeros(1,6),force_w=torch.ones(1,6,3)) for _ in range(4)]
    actions=torch.zeros(4,1,29)
    def loss(p):return contact_objective(predictions,p,p[0],actions,actions[0],g)['geometry']
    assert loss(points)==0
    # Common translation of the new contact is an absolute miss, even if
    # inter-goal displacement would be unchanged under a response objective.
    missed=points.clone();missed[:,:,2,0]+=.04;missed.requires_grad_()
    value=loss(missed);value.backward()
    assert value>0 and (missed.grad[2:,:,2,0]>0).all()
    assert (missed.grad[:2,:,2]==0).all()  # approach need not already be at target


def test_executor_does_not_shift_absolute_planned_point(tmp_path,monkeypatch):
    import json
    from holosoma.contact_wm import runtime
    from holosoma.contact_wm.geometry import LINKS
    def make_policy(_):
        return ContactPolicy(nn.Sequential(nn.Linear(567,8),nn.ELU(),nn.Linear(8,29))),None,None
    monkeypatch.setattr(runtime,'load_php_policy',make_policy)
    policy,_,_=make_policy(None);g=goal()
    checkpoint=tmp_path/'policy.pt'
    torch.save(dict(schema='contact_wm_direct_policy_v1',prior={'source':'unit-test-prior'},policy=policy.state_dict(),
        goal=vars(g),reference_origin_w=[0.,0.,0.],goal_hint=torch.zeros(1,3),start_step=0,horizon=4,motion_index=0),checkpoint)
    planned=torch.arange(18,dtype=torch.float32).reshape(6,3)/10
    path=tmp_path/'planned.json';path.write_text(json.dumps(dict(frame='world',links=list(LINKS),position_w=planned.tolist())))
    executor=runtime.ContactExecutor(checkpoint,'cpu',torch.tensor([100.,200.,0.]),planned_contact_points=path)
    torch.testing.assert_close(executor.goal.position_w[0,2],planned[2],rtol=0,atol=0)
    state=torch.zeros(1,71);state[:,6]=1
    action=executor.act(torch.zeros(1,567),state,0,torch.zeros(1,6,3))
    assert action.shape==(1,29) and torch.isfinite(action).all()
    torch.testing.assert_close(executor.goal.position_w[0,2],planned[2],rtol=0,atol=0)


def test_new_contact_geometry_is_not_diluted_by_support_duration():
    from holosoma.contact_wm.core import contact_geometry
    for horizon in (15,35):
        normal=torch.zeros(horizon,1,6)
        tangent=torch.zeros(horizon,1,6,3);tangent[:,:,2,0]=.04;tangent.requires_grad_()
        new=torch.zeros_like(normal,dtype=torch.bool);new[-10:,:,2]=True
        support=torch.zeros_like(new);support[:,:,3:5]=True
        target,anchor=contact_geometry(normal,tangent,new,support,.05)
        torch.testing.assert_close(target,torch.tensor(.64));assert anchor==0
        grad5=torch.autograd.grad(target,tangent,retain_graph=True)[0]
        target2,_=contact_geometry(normal,tangent,new,support,.02)
        grad2=torch.autograd.grad(target2,tangent)[0]
        torch.testing.assert_close(target2,torch.tensor(4.))
        torch.testing.assert_close(grad2,grad5*6.25)
        assert (grad2[:-10]==0).all()
