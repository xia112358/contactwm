import numpy as np
import torch

from generator.next_interaction_heightmap_v2 import (
    RootYawHeightmapInteractionPredictor, heightmap_supersample_for_architecture,
    render_root_yaw_box_heightmaps,
)
from generator.next_interaction_heightmap_v3 import DenseHeightmapInteractionPredictor
from generator.next_interaction_heightmap_v4 import (
    BoundHeightmapInteractionPredictor, JointBoundHeightmapInteractionPredictor,
)
from generator.next_interaction_heightmap import heightmap_grid


def _q(batch=1):
    q=np.zeros((batch,36),np.float32);q[:,2]=.8;q[:,3]=1
    return q


def test_root_yaw_render_is_invariant_to_common_planar_transform():
    center=np.array([[.45,.05,.2]],np.float32);rotation=np.eye(3,dtype=np.float32)[None]
    half=np.array([[.2,.15,.2]],np.float32);ground=np.array([0.],np.float32);q=_q()
    expected=render_root_yaw_box_heightmaps(center,rotation,half,ground,q)
    angle=.7;c,s=np.cos(angle),np.sin(angle)
    planar=np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32);shift=np.array([.3,-.2,0],np.float32)
    moved_center=center@planar.T+shift;moved_rotation=planar[None]@rotation
    moved_q=q.copy();moved_q[:,:3]+=shift;moved_q[:,3]=np.cos(angle/2);moved_q[:,6]=np.sin(angle/2)
    actual=render_root_yaw_box_heightmaps(moved_center,moved_rotation,half,ground,moved_q)
    np.testing.assert_allclose(actual,expected,atol=1e-6)


def test_v2_forward_is_finite_and_pose_loss_does_not_update_categorical_heads():
    model=RootYawHeightmapInteractionPredictor(width=48,layers=1)
    q=torch.from_numpy(_q(2));contact=torch.zeros(2,6,dtype=torch.bool)
    anchor=torch.zeros(2,6,3);surface=torch.full((2,6),-1,dtype=torch.long)
    height=torch.full((2,71,61),-.8)
    prediction=model(q,contact,anchor,surface,height)
    assert prediction.qpos.shape==(2,36)
    assert torch.isfinite(prediction.qpos).all()
    prediction.qpos.sum().backward()
    assert model.role_head.weight.grad is None
    assert model.surface_head.weight.grad is None


def test_root_yaw_pose_representation_round_trips():
    model=RootYawHeightmapInteractionPredictor(width=48,layers=1)
    current=torch.from_numpy(_q(2));target=current.clone()
    target[:,0]=torch.tensor([.2,-.1]);target[:,1]=torch.tensor([-.15,.3])
    target[:,3:7]=torch.tensor([[.8,.1,.2,.3],[.7,-.2,.1,.4]])
    target[:,3:7]=torch.nn.functional.normalize(target[:,3:7],dim=-1)
    target[:,7:]=torch.randn_like(target[:,7:])
    torch.testing.assert_close(model.decode_pose(model.encode_pose(target,current),current),target)


def test_v3_keeps_observation_only_contract_and_dense_gradient():
    model=DenseHeightmapInteractionPredictor(width=48,layers=1)
    q=torch.from_numpy(_q(2));contact=torch.zeros(2,6,dtype=torch.bool)
    anchor=torch.zeros(2,6,3);surface=torch.full((2,6),-1,dtype=torch.long)
    height=torch.full((2,71,61),-.8,requires_grad=True)
    prediction=model(q,contact,anchor,surface,height)
    assert prediction.qpos.shape==(2,36)
    assert prediction.role_logits.shape==(2,6,4)
    assert model.height(height).shape==(2,36*31,48)
    (prediction.qpos.square().sum()+prediction.role_logits.square().sum()).backward()
    assert height.grad is not None
    assert torch.isfinite(height.grad).all()
    assert float(height.grad.abs().sum())>0


def test_v3_coupled_pose_backpropagates_into_contact_heads():
    model=DenseHeightmapInteractionPredictor(width=48,layers=1,couple_contact_pose=True)
    q=torch.from_numpy(_q(2));contact=torch.zeros(2,6,dtype=torch.bool)
    anchor=torch.zeros(2,6,3);surface=torch.full((2,6),-1,dtype=torch.long)
    height=torch.full((2,71,61),-.8)
    prediction=model(q,contact,anchor,surface,height)
    prediction.qpos.square().sum().backward()
    assert model.role_head.weight.grad is not None
    assert model.surface_head.weight.grad is not None
    assert float(model.role_head.weight.grad.abs().sum())>0
    assert float(model.surface_head.weight.grad.abs().sum())>0


def test_supersampled_heightmap_reduces_subpixel_edge_jump():
    center=np.array([[.46,0.,.45]],np.float32);rotation=np.eye(3,dtype=np.float32)[None]
    half=np.array([[.3,.3,.45]],np.float32);ground=np.array([0.],np.float32)
    q=_q(1);shifted=q.copy();shifted[:,0]-=.001
    hard_a=render_root_yaw_box_heightmaps(center,rotation,half,ground,q)
    hard_b=render_root_yaw_box_heightmaps(center,rotation,half,ground,shifted)
    smooth_a=render_root_yaw_box_heightmaps(center,rotation,half,ground,q,supersample=2)
    smooth_b=render_root_yaw_box_heightmaps(center,rotation,half,ground,shifted,supersample=2)
    assert np.abs(smooth_b-smooth_a).max() < np.abs(hard_b-hard_a).max()


def test_single_point_heightmap_contains_no_fabricated_edge_height():
    center=np.array([[.461,.013,.45]],np.float32);rotation=np.eye(3,dtype=np.float32)[None]
    half=np.array([[.303,.287,.45]],np.float32);ground=np.array([0.],np.float32);q=_q()
    height=render_root_yaw_box_heightmaps(center,rotation,half,ground,q,supersample=1)
    ground_local=ground[0]-q[0,2]
    top_local=center[0,2]+half[0,2]-q[0,2]
    assert np.logical_or(
        np.isclose(height,ground_local,atol=1e-6),
        np.isclose(height,top_local,atol=1e-6),
    ).all()


def test_bound_architectures_require_single_point_heightmaps():
    assert heightmap_supersample_for_architecture("heightmap_v4_bound") == 1
    assert heightmap_supersample_for_architecture("heightmap_v5_joint_bound") == 1
    assert heightmap_supersample_for_architecture("heightmap_v3_robust") == 2


def test_v4_contact_and_pose_share_a_hard_spatial_binding():
    torch.set_num_threads(1);torch.manual_seed(11)
    model=BoundHeightmapInteractionPredictor(width=48,layers=1)
    q=torch.from_numpy(_q(2));contact=torch.zeros(2,6,dtype=torch.bool)
    anchor=torch.zeros(2,6,3);surface=torch.full((2,6),-1,dtype=torch.long)
    height=torch.full((2,71,61),-.8)
    height[:,20:40,20:40]=.2;height.requires_grad_()
    prediction=model(q,contact,anchor,surface,height)
    assert prediction.qpos.shape==(2,36)
    assert prediction.binding_logits.shape==(2,6,36*31)
    assert prediction.binding_points_local.shape==(2,6,3)
    assert prediction.binding_confidence.shape==(2,6)
    assert prediction.binding_grid_local.shape==(2,36*31,3)
    grid=torch.from_numpy(heightmap_grid()[::2,::2]).reshape(-1,2)
    selected=prediction.binding_logits.argmax(-1)
    torch.testing.assert_close(prediction.binding_points_local[...,:2],grid[selected])
    prediction.qpos.square().sum().backward(retain_graph=True)
    assert model.role_head.weight.grad is not None
    assert float(model.role_head.weight.grad.abs().sum())>0
    assert model.binding_query.weight.grad is not None
    assert float(model.binding_query.weight.grad.abs().sum())==0
    model.zero_grad(set_to_none=True)
    prediction.binding_points_local.square().sum().backward()
    assert model.binding_query.weight.grad is not None
    assert float(model.binding_query.weight.grad.abs().sum())>0


def test_v4_zero_binding_residual_preserves_coupled_v3_output():
    torch.set_num_threads(1);torch.manual_seed(17)
    old=DenseHeightmapInteractionPredictor(width=48,layers=1,couple_contact_pose=True)
    new=BoundHeightmapInteractionPredictor(width=48,layers=1)
    incompatible=new.load_state_dict(old.state_dict(),strict=False)
    assert 'binding_scale' in incompatible.missing_keys
    q=torch.from_numpy(_q());contact=torch.zeros(1,6,dtype=torch.bool)
    anchor=torch.zeros(1,6,3);surface=torch.full((1,6),-1,dtype=torch.long)
    height=torch.full((1,71,61),-.8)
    expected=old(q,contact,anchor,surface,height)
    actual=new(q,contact,anchor,surface,height)
    torch.testing.assert_close(actual.qpos,expected.qpos)
    torch.testing.assert_close(actual.role_logits,expected.role_logits)
    torch.testing.assert_close(actual.surface_logits,expected.surface_logits)


def test_v5_zero_initialized_global_decoder_preserves_coupled_v3_output():
    torch.set_num_threads(1);torch.manual_seed(23)
    old=DenseHeightmapInteractionPredictor(width=48,layers=1,couple_contact_pose=True)
    new=JointBoundHeightmapInteractionPredictor(width=48,layers=1)
    incompatible=new.load_state_dict(old.state_dict(),strict=False)
    assert 'constraint_pose_head.weight' in incompatible.missing_keys
    q=torch.from_numpy(_q());contact=torch.zeros(1,6,dtype=torch.bool)
    anchor=torch.zeros(1,6,3);surface=torch.full((1,6),-1,dtype=torch.long)
    height=torch.full((1,71,61),-.8)
    expected=old(q,contact,anchor,surface,height)
    actual=new(q,contact,anchor,surface,height)
    torch.testing.assert_close(actual.qpos,expected.qpos)
    torch.testing.assert_close(actual.role_logits,expected.role_logits)
    torch.testing.assert_close(actual.surface_logits,expected.surface_logits)
    actual.qpos.square().sum().backward()
    assert float(new.constraint_pose_head.weight.grad.abs().sum())>0
