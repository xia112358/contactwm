import torch
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from contact_solver.collision_geometry import CanonicalG1CollisionPoints


def test_fk_positions_rotations_and_q_gradients():
    torch.set_num_threads(1);torch.manual_seed(4)
    old=CanonicalG1ForwardKinematics();new=BatchedG1ForwardKinematics()
    new.load_state_dict(old.state_dict())
    q=torch.randn(2,4,36,requires_grad=True)
    geometry=CanonicalG1CollisionPoints(24)
    for links in (geometry.link_names,('torso_link',),('left_wrist_yaw_link','right_knee_link')):
        a=old.link_poses(q,links);b=new.link_poses(q,links)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        ga=torch.autograd.grad(sum(x.square().sum() for x in a),q,retain_graph=True)[0]
        gb=torch.autograd.grad(sum(x.square().sum() for x in b),q,retain_graph=True)[0]
        torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-5)
    assert len(new._plans[('torso_link',)]) < len(new.joint_records)
