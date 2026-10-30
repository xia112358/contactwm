import torch

from contact_solver.contact_constrained_projector import CanonicalContactCollisionGeometry
from somaforge_core.motion_contracts import BODY_NAMES
from generator.full1000_position_predictor import Full1000PositionPredictor
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from generator.next_interaction_heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS
from generator.next_interaction_heightmap_v2 import _root_yaw_basis
from contact_solver.part_collision_geometry import PartCollisionGeometry


def test_compact_support_preserves_complete_mesh_and_sphere_vertical_extrema():
    torch.manual_seed(37)
    fk = CanonicalG1ForwardKinematics()
    geometry = PartCollisionGeometry(fk)
    q = torch.randn(3, 36)*.2
    q[:, 2] += .8
    q[:, 3:7] /= q[:, 3:7].norm(dim=-1, keepdim=True)
    position, rotation = fk.link_poses(q, BODY_NAMES[1:7])
    basis, _ = _root_yaw_basis(q)
    local_position = torch.einsum('bij,bpj->bpi', basis.transpose(1, 2), position-q[:, None, :3])
    local_rotation = basis[:, None].transpose(-1, -2) @ rotation
    actual = (local_position+geometry.lowest_points(local_rotation))[..., 2]+q[:, None, 2]
    shapes = CanonicalContactCollisionGeometry().shapes
    p, r = fk.link_poses(q, tuple(s.link_name for s in shapes))
    expected = q.new_full((len(q), 6), torch.inf)
    for i, shape in enumerate(shapes):
        if shape.kind == 'sphere':
            center = p[:, i]+torch.einsum('bij,j->bi', r[:, i], shape.local_center)
            z = center[:, 2]-shape.radius
        else:
            points = p[:, i, None]+torch.einsum('bij,vj->bvi', r[:, i], shape.local_points)
            z = points[..., 2].amin(-1)
        expected[:, shape.part] = torch.minimum(expected[:, shape.part], z)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-7)
    assert sum(c['vertices'] for c in geometry.asset_counts) == sum(
        len(s.local_points) for s in shapes if s.local_points is not None)
    heightmap = q.new_full((len(q), HEIGHTMAP_ROWS, HEIGHTMAP_COLS), -.8)
    features = geometry(local_position, local_rotation, heightmap)
    assert features.shape == (3, 6, 19) and torch.isfinite(features).all()
    outside = local_position.clone(); outside[..., 0] += 100
    masked = geometry(outside, local_rotation, heightmap)
    assert masked[..., -2:].count_nonzero() == 0


def test_token_geometry_migration_zero_start_and_trainable_geometry_dependency():
    torch.manual_seed(31)
    base = Full1000PositionPredictor(24, 1, 8).eval()
    model = Full1000PositionPredictor(24, 1, 8, part_geometry=True).eval()
    migrated = model.load_state_dict(base.state_dict(), strict=False)
    assert not migrated.unexpected_keys
    assert migrated.missing_keys and all(k.startswith(('part_geometry_', 'execution_geometry_encoder.')) for k in migrated.missing_keys)
    q = torch.zeros(1, 36); q[:, 2] = .8; q[:, 3] = 1
    observation = dict(current_q=q, current_contact=torch.zeros(1, 6, dtype=torch.bool),
        current_anchor=torch.zeros(1, 6, 3), heightmap=torch.full((1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS), -.8))
    a, b = base(**observation), model(**observation)
    torch.testing.assert_close(a.qpos, b.qpos, rtol=0, atol=0)
    torch.testing.assert_close(a.role_logits, b.role_logits, rtol=0, atol=0)
    b.qpos.square().sum().backward()
    assert model.part_geometry_encoder[-1].weight.grad is None
    assert model.execution_geometry_encoder.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.execution_geometry_encoder.weight.normal_(std=.01)
        before = model(**observation)
        model.part_geometry_observation.local_support[0] += .01
        after = model(**observation)
    assert not torch.equal(before.qpos, after.qpos)
    shifted = {k: v.clone() for k, v in observation.items()}
    delta = torch.tensor([.2, -.3, 0.])
    shifted['current_q'][:, :3] += delta
    shifted['current_anchor'] += delta
    with torch.no_grad():
        a, b = model(**observation), model(**shifted)
    torch.testing.assert_close(a.qpos[:, :3]+delta, b.qpos[:, :3], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a.qpos[:, 3:], b.qpos[:, 3:], rtol=1e-4, atol=1e-5)
    assert not any('surface' in k for k in model.state_dict())
