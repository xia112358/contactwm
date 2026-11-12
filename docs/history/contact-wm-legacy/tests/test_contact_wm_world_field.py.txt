import torch
from holosoma.contact_wm.world_contact_field import WorldContactField,pack_native_contacts


def field():
    y,x=torch.meshgrid(torch.linspace(-.1,.1,11),torch.linspace(-.1,.1,11),indexing='ij')
    points=torch.stack((x.flatten(),y.flatten(),torch.full((121,),.7)),-1)
    return WorldContactField(points[None].repeat(6,1,1))


def test_native_pair_packing_preserves_positions_force_sign_and_multiple_contacts():
    snapshot=dict(count=4,position_w=torch.tensor([[3.,2.,1.],[4.,2.,1.],[9.,9.,9.],[2.,1.,1.]]),
        force_on_body1_w=torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[2.,3.,4.]]))
    selected=dict(eligible=torch.tensor([1,1,0,1],dtype=torch.bool),groups=torch.tensor([2,2,0,9]),robot_side=torch.tensor([1,0,1,1]))
    result=pack_native_contacts(snapshot,selected,torch.tensor([[1.,0.,0.],[0.,1.,0.]]),2,2)
    assert result['spatial_mask'].sum()==3
    torch.testing.assert_close(result['spatial_points_w'][0,2,:2],torch.tensor([[2.,2.,1.],[3.,2.,1.]]))
    torch.testing.assert_close(result['spatial_forces_w'][0,2].sum(0),torch.tensor([-3.,-3.,-3.]))
    import pytest
    with pytest.raises(ValueError,match='capacity'):pack_native_contacts(snapshot,selected,torch.zeros(2,3),2,1)


def test_force_splat_conserves_resultant_and_unknown_nan_is_masked():
    model=field();points=torch.zeros(2,6,2,3);points[...,2]=.7
    points[:,:,0,0]=-.03;points[:,:,1,0]=.04
    forces=torch.zeros_like(points);forces[:,:,0,2]=10;forces[:,:,1,2]=30
    mask=torch.ones(2,6,2,dtype=torch.bool);known=torch.ones(2,6,dtype=torch.bool)
    known[1]=False;points[1]=float('nan');forces[1]=float('nan')
    mass,force,observed,inside=model.targets(points,forces,mask,known,torch.ones(2,dtype=torch.bool))
    torch.testing.assert_close(mass[0].sum(-1),torch.ones(6))
    torch.testing.assert_close(force[0].sum(-2),torch.tensor([0.,0.,40.]).expand(6,-1))
    assert torch.isfinite(mass).all() and torch.isfinite(force).all()
    assert not observed[1].any() and inside.all()
    centroid=(mass[0,...,None]*model.points_w).sum(-2)
    torch.testing.assert_close(centroid[:,0],torch.full((6,),.0225),atol=1e-5,rtol=1e-5)


def test_world_queries_stay_fixed_and_spatial_loss_reaches_dynamics_state():
    torch.manual_seed(5);model=field();nodes=model.points_w.clone()
    state=torch.randn(2,71,requires_grad=True)
    p=model(state);p['force_w']=torch.zeros(2,6,3)
    points=model.points_w[:,60][None,:,None].repeat(2,1,1,1)
    forces=torch.ones_like(points);mask=torch.ones(2,6,1,dtype=torch.bool)
    target=model.targets(points,forces,mask,torch.ones(2,6,dtype=torch.bool),torch.ones(2,dtype=torch.bool))
    model.fit_loss(p,*target[:3]).backward()
    assert state.grad.abs().sum()>0 and torch.isfinite(state.grad).all()
    torch.testing.assert_close(model.points_w,nodes)
    correction=model.dynamics_correction(state,torch.ones(2,6,3),torch.eye(3)[None].repeat(2,1,1))
    torch.testing.assert_close(correction,torch.zeros_like(correction))


def test_predicted_world_field_conserves_force_probability_and_moment():
    model=field();state=torch.randn(2,71)
    p=model(state);p.update(force_w=torch.randn(2,6,3),contact_logits=torch.randn(2,6))
    nodes=model.on_nodes(p)
    torch.testing.assert_close(nodes['force_on_robot_w'].sum(-2),p['force_w'])
    torch.testing.assert_close(nodes['contact_probability'].sum(-1),p['contact_logits'].sigmoid())
    torque=torch.linalg.cross(nodes['position_w'][None]-state[:,None,None,:3],nodes['force_on_robot_w'],dim=-1).sum(-2)
    equivalent=torch.linalg.cross(p['spatial_point_w']-state[:,None,:3],p['force_w'],dim=-1)
    torch.testing.assert_close(torque,equivalent,atol=1e-6,rtol=1e-5)


def test_expected_target_error_penalizes_two_wrong_peaks():
    model=field()
    with torch.no_grad():
        for p in model.encoder.parameters():p.zero_()
        model.encoder[-1].bias.fill_(-100.)
        model.encoder[-1].bias.reshape(6,-1)[:,[55,65]]=0
    state=torch.zeros(1,71);target=torch.zeros(1,6,3);target[:,:,2]=.7
    normal=torch.zeros_like(target);normal[:,:,2]=1
    cost,center=model.expected_target_cost(state,target,normal,.02)
    torch.testing.assert_close(center,target,atol=1e-6,rtol=1e-6)
    assert (cost>24.9).all()  # +/-10cm peaks have the correct centroid, but miss the goal.
