import json
import numpy as np
import pytest
from somaforge_core.motion_contracts import BODY_NAMES, CONTACT_PARTS
from generator.teacher_data import TeacherDataset


def test_generated_teacher_contract(tmp_path) -> None:
    # Synthetic loader fixture, not a source of physical contact truth.
    directory = tmp_path
    motion = directory / 'teacher.npz'
    position = np.zeros((3, len(BODY_NAMES), 3), dtype=np.float32)
    quaternion = np.zeros((3, len(BODY_NAMES), 4), dtype=np.float32)
    quaternion[..., 0] = 1
    contact = np.zeros((3, len(CONTACT_PARTS)), dtype=bool)
    contact[:, :2] = True
    rotation = np.broadcast_to(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
                               (1, len(BODY_NAMES), 6)).copy()
    np.savez(motion, body_pos_w=position, body_quat_w=quaternion,
             contact_force_part_mask=contact, body_names=BODY_NAMES,
             contact_force_part_order=CONTACT_PARTS, boundary_frames=[0, 2], fps=50.)
    np.savez(directory / 'teacher_events.npz',
             start_position=position[:1], end_position=position[-1:],
             start_rotation=rotation, end_rotation=rotation,
             start_contact=contact[:1], end_contact=contact[-1:],
             touchdown=np.zeros((1, len(CONTACT_PARTS))),
             target_contact=np.zeros((1, len(CONTACT_PARTS), 3)),
             target_surface=np.zeros((1, len(CONTACT_PARTS)), dtype=np.int64),
             duration=[.04], geometry=np.zeros((1, 6)), action=[0],
             trajectory_id=[0], event_index=[0])
    (directory / 'report.json').write_text(json.dumps({
        'schema': 'climb00_privileged_wide_event_teacher_v1',
        'samples': [{'motion_file': str(motion)}]}))
    dataset = TeacherDataset(directory)
    assert dataset.trajectory_count == 1
    assert len(dataset) == 1
    segment = dataset.segment(0, 0)
    assert segment.position.shape[1:] == (len(BODY_NAMES), 3)
    assert segment.contact.shape[1] == len(CONTACT_PARTS)
    assert segment.position.shape[0] >= 2
    assert segment.start.rotation6d.shape == (len(BODY_NAMES), 6)
    assert segment.start.vector63.shape == (63,)
    with pytest.raises(ValueError, match="sparse-only teacher"):
        dataset.full_body_segment(0, 0)
