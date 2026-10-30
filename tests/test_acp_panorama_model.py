import io
import math

import pytest
import torch

from holosoma.acp.panorama_model import PanoramaContactPlanner, SphericalConv, panorama_planner_loss


def inputs(b=2, h=8, w=16):
    torch.manual_seed(42)
    return dict(history=torch.randn(b, 465), goal=torch.randn(b, 3),
                depth=torch.rand(b, h, w)+.4, valid=torch.rand(b, h, w)>.3,
                age=torch.rand(b, h, w)*5, base_position_w=torch.randn(b, 3),
                view_origin_w=torch.randn(b, 3), yaw=torch.randn(b))


def model():
    return PanoramaContactPlanner(height=8, width=16)


def test_exact_surface_selection_and_world_geometry():
    m=model(); x=inputs(); p=m(**x)
    assert p['positions'].shape==(2,6,3)
    assert (p['affordance'].masked_select(~p['valid'][:,None].expand(-1,6,-1))==0).all()
    for b in range(2):
        for part in range(6):
            idx=p['selected_indices'][b,part]; row,col=divmod(int(idx),16)
            assert x['valid'][b,row,col]
            ray=m.directions[row,col]*x['depth'][b,row,col]
            c,s=x['yaw'][b].cos(),x['yaw'][b].sin()
            expected=torch.stack((c*ray[0]-s*ray[1],s*ray[0]+c*ray[1],ray[2]))+x['view_origin_w'][b]
            torch.testing.assert_close(p['ungated_positions_w'][b,part],expected)
    # Translate world poses only: local predictions must stay unchanged.
    y={**x,'base_position_w':x['base_position_w']+10,'view_origin_w':x['view_origin_w']+10}
    q=m(**y)
    torch.testing.assert_close(p['affordance'],q['affordance'])
    torch.testing.assert_close(q['ungated_positions_w'],p['ungated_positions_w']+10)


def test_unknown_placeholders_and_empty_batch_are_safe():
    m=model(); x=inputs(); x['valid'][0]=False
    x['depth'][~x['valid']]=float('nan');x['age'][~x['valid']]=float('inf')
    p=m(**x)
    assert not p['occupancy'][0].any()
    assert not p['positions_w'][0].any()
    assert (p['selected_indices'][0]==-1).all()
    assert not p['affordance'][0].any()
    target=p['ungated_positions_w'].detach()
    loss=panorama_planner_loss(p,target,torch.ones(2,6,dtype=torch.bool),torch.ones(2,6,dtype=torch.bool))
    loss['loss'].backward()
    assert not loss['supervised'][0].any()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
    assert m.image_encoder[0].conv.weight.grad.abs().sum()>0
    assert m.attention.in_proj_weight.grad.abs().sum()>0


def test_missing_surface_positive_and_unknown_labels_not_negatives():
    m=model(); p=m(**inputs()); target=p['ungated_positions_w'].detach().clone()
    active=torch.ones(2,6,dtype=torch.bool); known=torch.ones_like(active)
    target[:,0]=100; known[:,1]=False;target[:,1]=float('nan')
    active[:,2]=False;target[:,2]=float('nan')
    loss=panorama_planner_loss(p,target,active,known)
    assert not loss['supervised'][:,:2].any()
    assert loss['supervised'][:,2:].all()
    assert torch.isfinite(loss['loss'])
    zero=panorama_planner_loss(p,target,active,torch.zeros_like(known))
    assert zero['loss']==0


def test_longitude_wrap_and_stale_observations():
    layer=SphericalConv(1,2); x=torch.randn(1,1,8,16)
    torch.testing.assert_close(layer(x.roll(3,-1)),layer(x).roll(3,-1))
    m=model(); x=inputs();x['age'][:]=16
    p=m(**x)
    assert not p['available'].any()
    assert not p['occupancy_probability'].any()
    p['occupancy_logits'].sum().backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())


def test_validation_and_checkpoint_roundtrip():
    m=model(); x=inputs();p=m(**x)
    out=io.BytesIO();torch.save(dict(config=m.config,state_dict=m.state_dict()),out);out.seek(0)
    saved=torch.load(out,weights_only=True);clone=PanoramaContactPlanner(**saved['config']);clone.load_state_dict(saved['state_dict'])
    torch.testing.assert_close(p['affordance'],clone(**x)['affordance'])
    x['depth'][x['valid']]=10
    with pytest.raises(ValueError,match='Known pixels'):m(**x)


def test_optimizer_reduces_fixed_batch_loss():
    torch.manual_seed(7); m=model();x=inputs();p=m(**x)
    target=p['ungated_positions_w'].detach();active=torch.ones(2,6,dtype=torch.bool);known=torch.ones_like(active)
    opt=torch.optim.Adam(m.parameters(),lr=.002);values=[]
    for _ in range(12):
        opt.zero_grad();loss=panorama_planner_loss(m(**x),target,active,known)['loss'];values.append(loss.item());loss.backward();opt.step()
    assert values[-1]<values[0]


def test_contact_schema_is_six_and_checkpoint_explicit():
    m=model()
    assert m.contact_names==('left_foot','right_foot','left_knee','right_knee','left_hand','right_hand')
    assert m.config['num_contacts']==6
    nine=PanoramaContactPlanner(height=8,width=16,num_contacts=9)
    assert nine(**inputs())['positions_w'].shape==(2,9,3)
    with pytest.raises(RuntimeError):nine.load_state_dict(m.state_dict())
