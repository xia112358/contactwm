import inspect
import torch
from generator.next_interaction import NextInteractionPredictor,nearest_face,joint_feasibility_loss


def test_joint_feasibility_is_soft_and_not_diluted():
    joints=torch.zeros(2,29,requires_grad=True)
    with torch.no_grad():joints[0,0]=1.1;joints[1,0]=-1.1
    cost,violation=joint_feasibility_loss(joints,-torch.ones(29),torch.ones(29))
    assert (cost>1.).all()  # One 0.1-rad violation cannot disappear in a DOF average.
    cost.sum().backward()
    assert joints.grad[0,0]>0 and joints.grad[1,0]<0
    assert torch.count_nonzero(joints.grad[:,1:])==0
    torch.testing.assert_close(violation[:,0],torch.full((2,),.1))
    valid=torch.zeros(1,29,requires_grad=True)
    zero,_=joint_feasibility_loss(valid,-torch.ones(29),torch.ones(29))
    zero.sum().backward()
    assert zero.item()==0 and torch.count_nonzero(valid.grad)==0


def test_unbounded_pose_and_gradients():
    torch.set_num_threads(1)
    model=NextInteractionPredictor(width=48,layers=1)
    current=torch.zeros(2,36);current[:,3]=1
    raw=torch.zeros(2,36,requires_grad=True)
    with torch.no_grad():raw[:,3]=1;raw[:,:3]=3.;raw[:,7:]=4.
    q=model.decode_pose(raw,current)
    torch.testing.assert_close(q[:,7:],torch.full((2,29),4.),rtol=0,atol=0)
    torch.testing.assert_close(q[:,:3],torch.full((2,3),3.),rtol=0,atol=0)
    q[:,7:].sum().backward()
    torch.testing.assert_close(raw.grad[:,7:],torch.ones(2,29),rtol=0,atol=0)


def test_face_geometry_is_finite_and_rotation_aware():
    angle=torch.tensor(.7);u=torch.tensor([angle.cos(),angle.sin(),0.]);v=torch.tensor([-angle.sin(),angle.cos(),0.])
    face=torch.cat((torch.zeros(3),torch.tensor([0.,0.,1.]),u,v,torch.tensor([1.,.2,0.])))[None,None]
    p=(u*2+v*.5+torch.tensor([0.,0.,.3]))[None,None]
    closest,signed=nearest_face(p,face)
    torch.testing.assert_close(closest[0,0,0],u+v*.2)
    torch.testing.assert_close(signed,torch.tensor([[[.3]]]))


def test_forward_has_only_observations_and_no_offset_head():
    torch.set_num_threads(1);torch.manual_seed(6)
    model=NextInteractionPredictor(width=48,layers=1)
    assert list(inspect.signature(model.forward).parameters)==['current_q','current_contact','current_anchor','current_surface','faces']
    assert not any('offset' in n or 'phase' in n or 'velocity' in n for n,_ in model.named_parameters())
    q=torch.zeros(2,36);q[:,3]=1.;q[:,2]=.8
    faces=torch.tensor([0,0,0,0,0,1,1,0,0,0,1,0,1,1,1.])[None,None].repeat(2,2,1)
    faces[:,1,2]=.7;faces[:,1,14]=0
    a=model(q,torch.zeros(2,6,dtype=torch.bool),torch.zeros(2,6,3),torch.full((2,6),-1),faces)
    assert a.qpos.shape==(2,36) and a.role_logits.shape==(2,6,4) and a.surface_logits.shape==(2,6,2)
    torch.testing.assert_close(a.qpos[:,3:7].norm(dim=-1),torch.ones(2))
    assert torch.isfinite(a.qpos).all() and (a.duration>0).all()
    points=model.geometry_points(a,faces)
    cloud,parts=model.geometry(model.fk,a.qpos[:,None])
    for part in range(6):
        distance=(cloud[:,0,parts==part]-points[:,part,None]).norm(dim=-1)
        torch.testing.assert_close(distance.amin(-1),torch.zeros(2),rtol=0,atol=0)
