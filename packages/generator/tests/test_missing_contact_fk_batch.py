from types import SimpleNamespace

import torch

from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from generator.conditioned_pose_predictor import _missing_intent_surface_distance


def test_missing_contact_reuses_one_fk_across_parts_surfaces_and_shapes(monkeypatch):
    fk = CanonicalG1ForwardKinematics()
    original = fk.link_poses
    calls = []
    def counted(q, names):
        calls.append((len(q),names))
        return original(q,names)
    monkeypatch.setattr(fk,'link_poses',counted)
    model = SimpleNamespace(fk=fk)
    q = torch.zeros(3,36); q[:,2] = .8; q[:,3] = 1; q.requires_grad_()
    surfaces = torch.tensor([[0,1,0,1,0,1],[1,0,1,0,1,0],[-1,-1,0,0,1,1]])
    scene = {'box_center':torch.tensor([[0.,0.,.4]]).expand(3,-1),
             'box_rotation':torch.eye(3).expand(3,-1,-1),
             'box_half_extents':torch.tensor([[.4,.5,.4]]).expand(3,-1),
             'ground_height':torch.zeros(3)}
    loss = _missing_intent_surface_distance(model,q,surfaces,scene,torch.tensor([.02,.018,.021]))
    assert len(calls) == 1 and calls[0][0] == 3
    assert loss.shape == (3,6) and torch.isfinite(loss).all()
    assert loss[2,:2].eq(0).all()
    loss.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
