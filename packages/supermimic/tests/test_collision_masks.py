from supermimic.lowstep import collision_masks_match


def test_mujoco_collision_pair_uses_either_mask_direction():
    assert collision_masks_match(3, 0, 1, 2)
    assert collision_masks_match(1, 2, 3, 0)
    assert not collision_masks_match(1, 0, 2, 0)
