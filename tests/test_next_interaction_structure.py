import torch

from generator.next_interaction import structural_pose_loss
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _matrix_from_rotation6d


def supervision(fk, q):
    position, rotation6d = fk(q[:, None])
    return position[:, 0], _matrix_from_rotation6d(rotation6d[:, 0]), {
        'q': q.clone(),
        'body_position': position[:, 0].clone(),
    }


def test_structural_pose_is_zero_at_demonstrated_pose():
    fk = CanonicalG1ForwardKinematics()
    q = torch.zeros(1, 36); q[:, 2] = .8; q[:, 3] = 1
    position, rotation, target = supervision(fk, q)
    loss, metrics = structural_pose_loss(type('Model', (), {'fk': fk})(), q, position, rotation, target)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert all(float(value.max()) == 0 for value in metrics.values())


def test_structure_keeps_worst_limb_and_legacy_joint_max_is_explicit():
    fk = CanonicalG1ForwardKinematics()
    target_q = torch.zeros(1, 36); target_q[:, 2] = .8; target_q[:, 3] = 1
    _, _, target = supervision(fk, target_q)
    predicted_q = target_q.clone(); predicted_q[:, 10] = 1.0
    position6d, rotation6d = fk(predicted_q[:, None])
    position = position6d[:, 0]; rotation = _matrix_from_rotation6d(rotation6d[:, 0])
    loss, metrics = structural_pose_loss(type('Model', (), {'fk': fk})(), predicted_q, position, rotation, target)
    # Every limb's maximum FK error remains emphasized, while joint-space
    # imitation is counted only by the separate pose objective.
    assert float(loss) > 0
    torch.testing.assert_close(metrics['joint_error_max_rad'], torch.ones(1))
    legacy, _ = structural_pose_loss(type('Model', (), {'fk': fk})(),
        predicted_q, position, rotation, target, joint_max_weight=1.)
    torch.testing.assert_close(legacy-loss, torch.full_like(loss, 4.))


def test_structure_does_not_supervise_joints_a_second_time():
    fk = CanonicalG1ForwardKinematics()
    q = torch.zeros(1, 36); q[:, 2] = .8; q[:, 3] = 1
    position, rotation, target = supervision(fk, q)
    predicted = q.clone(); predicted[:, 10] = 1.
    # Hold FK quantities at their target values to isolate the removed
    # joint-space term from the retained, physically distinct FK supervision.
    loss, metrics = structural_pose_loss(type('Model', (), {'fk': fk})(),
        predicted, position, rotation, target)
    assert loss.item() == 0
    assert metrics['joint_error_max_rad'].item() == 1
