import numpy as np

from somaforge_core.consensus_event_alignment import align_observed_events, contact_intent
from somaforge_core.contact_order_acceptance import audit_contact_order


def report(first, second, n=100):
    contract = dict(policy=dict(stable_frames=2, max_dropout_frames=1), retained_blocks=[[0, n]],
        actions=[dict(id='acquire', start=0, end=n-1, requirements=[
            dict(part=0, surface=0, kind='establish'), dict(part=2, surface=1, kind='establish')])])
    mask = np.zeros((n, 6), bool); surface = np.full((n, 6), -1)
    mask[first:, 0] = True; surface[first:, 0] = 0
    mask[second:, 2] = True; surface[second:, 2] = 1
    return audit_contact_order(contract, mask, surface), mask, surface


def test_actual_events_align_across_large_time_shifts_without_reference_deadlines():
    reports = [report(a, b)[0] for a, b in [(10, 30), (40, 60), (60, 80)]]
    maps, audit = align_observed_events(reports, 100)
    # Soft event residuals are preferable to zero-speed plateaus or arbitrary
    # compression. The middle execution still has the identity event clock.
    np.testing.assert_allclose(maps[1, [40, 60]], [40, 60])
    assert (np.diff(maps, axis=1) >= .25-1e-8).all()
    assert (np.diff(maps, axis=1) <= 4+1e-8).all()
    assert (maps[:, 0] == 0).all() and (maps[:, -1] == 99).all()
    assert audit['reference_frame_deadlines'] is False


def test_independent_close_event_reversal_can_share_a_monotone_clock():
    reports = [report(a, b)[0] for a, b in [(40, 42), (41, 39), (40, 41)]]
    maps, audit = align_observed_events(reports, 100)
    assert (np.diff(maps, axis=1) >= 0).all()
    assert maps[1, 41] > maps[1, 40]
    assert any(len(group) > 1 for group in audit['event_groups'])


def test_episode_intent_keeps_bounded_holes_separate_from_actual_labels():
    r, mask, surface = report(10, 30)
    mask[45, 0] = False; surface[45, 0] = -1
    original = mask.copy()
    wanted, faces, release = contact_intent(r, mask, surface)
    assert wanted[45, 0] and faces[45, 0] == 0
    assert not release.any()
    np.testing.assert_array_equal(mask, original)


def test_severe_event_compression_is_bounded_and_residual_is_recorded():
    reports = [report(a, b)[0] for a, b in [(5, 6), (40, 41), (40, 90)]]
    maps, audit = align_observed_events(reports, 100)
    assert np.diff(maps, axis=1).min() >= .25-1e-8
    assert np.diff(maps, axis=1).max() <= 4+1e-8
    assert any(d['identity_blend'] < 1 for d in audit['clock_diagnostics'])


def test_pose_resampling_uses_shortest_quaternion_arc_without_integer_skips():
    from somaforge_core.consensus_event_alignment import resample_motion
    q = np.zeros((4, 36), np.float32); q[:, 0] = np.arange(4); q[:, 3] = [1, -1, 1, -1]
    out = resample_motion(q, np.array([0., .5, 2.5, 3.]))
    np.testing.assert_allclose(out[:, 0], [0, .5, 2.5, 3])
    np.testing.assert_allclose(np.abs(out[:, 3]), 1)


def test_original_pose_dp_searches_observed_source_clock_not_reference_deadline():
    from somaforge_core.contact_aggregation import match_pose_anchors
    reference = np.zeros((100, 36)); reference[:, 3] = 1
    source = reference.copy()
    reference[20, 7] = 1
    source[62, 7] = 1
    selected = match_pose_anchors(reference, source, [0, 20, 99], [0, 60, 99], band=3)
    np.testing.assert_array_equal(selected, [0, 62, 99])


def test_pose_guided_event_alignment_retains_identity_and_bounded_clock():
    reports = [report(a, b)[0] for a, b in [(10, 30), (40, 60), (60, 80)]]
    q = np.zeros((3, 100, 36)); q[:, :, 3] = 1
    maps, audit = align_observed_events(reports, 100, motions=q, reference_index=1)
    np.testing.assert_allclose(maps[1], np.arange(100))
    assert np.diff(maps, axis=1).min() >= .25-1e-8
    assert np.diff(maps, axis=1).max() <= 4+1e-8
    assert audit['clock_diagnostics'][0]['pose_matching'].startswith('original_ordered_pose_dp')
