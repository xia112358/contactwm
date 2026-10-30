import numpy as np
import pytest

from somaforge_core.contact_order_acceptance import audit_contact_order


def contract(*requirements):
    return dict(actions=[dict(id=str(i), start=i*10, end=(i+1)*10-1,
                              requirements=list(reqs)) for i, reqs in enumerate(requirements)],
                retained_blocks=[[0, len(requirements)*10]],
                policy=dict(stable_frames=3, max_dropout_frames=1))


def req(kind, part, surface=0):
    return dict(kind=kind, part=part, surface=surface)


def audit(c, faces):
    return audit_contact_order(c, faces >= 0, faces)


def test_independent_parts_can_swap_and_ignore_reference_clock():
    c = contract([req("establish", 0)], [req("establish", 1)])
    faces = np.full((100, 6), -1)
    faces[70:, 0] = 0
    faces[20:, 1] = 0
    assert audit(c, faces)["passed"]
    assert audit(c, np.repeat(faces, 3, axis=0))["passed"]


def test_dependent_contact_requires_support_before_and_through_acquisition():
    c = contract([req("establish", 0)], [req("keep", 0), req("establish", 1)])
    faces = np.full((100, 6), -1)
    faces[10:50, 0] = 0
    faces[30:33, 1] = 0
    assert audit(c, faces)["passed"]
    faces[20:30, 0] = -1
    # A new support episode begins simultaneously: it can support the event.
    assert audit(c, faces)["passed"]
    faces[30:35, 0] = -1
    assert not audit(c, faces)["passed"]


def test_continuous_contact_cannot_fake_explicit_release_and_retouch():
    c = contract([req("establish", 0)], [req("release", 0, -1)], [req("establish", 0)])
    faces = np.full((80, 6), -1)
    faces[:, 0] = 0
    assert not audit(c, faces)["passed"]
    faces[20:30, 0] = -1
    assert audit(c, faces)["passed"]


def test_reestablish_does_not_invent_an_unrequested_release():
    c = contract([req("establish", 0)], [req("establish", 0)])
    faces = np.full((80, 6), -1)
    faces[:, 0] = 0
    assert audit(c, faces)["passed"]


def test_same_part_surface_order_and_release_are_preserved():
    c = contract([req("establish", 0, 0)], [req("establish", 0, 1)])
    faces = np.full((60, 6), -1)
    faces[:20, 0] = 1
    faces[30:, 0] = 0
    assert not audit(c, faces)["passed"]
    assert audit(c, faces[::-1])["passed"]
    c = contract([req("establish", 0)], [req("release", 0, -1)])
    faces[:] = -1
    faces[:20, 0] = 0
    assert audit(c, faces)["passed"]
    faces[20:, 0] = 0
    assert not audit(c, faces)["passed"]


def test_keep_across_two_dependent_events_cannot_hide_long_gap():
    c = contract([req("establish", 0)],
                 [req("keep", 0), req("establish", 1)],
                 [req("keep", 0), req("establish", 2)])
    faces = np.full((100, 6), -1)
    faces[5:90, 0] = 0
    faces[20:30, 1] = 0
    faces[70:, 2] = 0
    assert audit(c, faces)["passed"]
    faces[40:50, 0] = -1
    assert not audit(c, faces)["passed"]
    faces[40:50, 0] = 0
    faces[40, 0] = -1
    assert audit(c, faces)["passed"]
    faces[40, 0] = 1  # Wrong-surface contact is never filled as sensor dropout.
    assert not audit(c, faces)["passed"]


def test_keep_only_phase_does_not_extend_into_an_unrequested_next_action():
    c = contract([req("establish", 0)], [req("keep", 0), req("establish", 2)], [req("keep", 0)],
                 [req("establish", 1)])
    faces = np.full((80, 6), -1)
    faces[:70, 0] = 0
    faces[20:25, 2] = 0
    faces[60:, 1] = 0
    assert audit(c, faces)["passed"]
    faces[30:60, 0] = -1
    assert audit(c, faces)["passed"]


def test_missing_contact_and_invalid_evidence_are_not_success():
    c = contract([req("establish", 0)])
    faces = np.full((40, 6), -1)
    assert audit(c, faces)["failed_checks"] == 1
    faces[20:22, 0] = 0  # Fails existing stable-evidence rule, not a task deadline.
    assert not audit(c, faces)["passed"]
    with pytest.raises(ValueError, match="surface"):
        audit_contact_order(c, np.ones((40, 6), bool), np.full((40, 6), -1))


def test_support_handover_can_touch_but_cannot_leave_a_gap():
    c = contract([req("establish", 0)], [req("keep", 0), req("establish", 1)])
    faces = np.full((50, 6), -1)
    faces[5:20, 0] = 0
    faces[20:, 1] = 0
    assert audit(c, faces)["passed"]
    faces[19, 0] = -1
    assert not audit(c, faces)["passed"]
