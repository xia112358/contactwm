import numpy as np

from somaforge_core.staged_consensus import (
    compatible_sources, pose_residual, apply_residual, fusion_stages, localized_transfer,
)


def poses(n=100):
    q = np.zeros((n, 36), np.float32)
    q[:, 3] = 1
    return q


def test_same_contact_episode_confirmations_are_not_different_strategies():
    def report(times):
        return dict(passed=True, checks=[dict(action=str(i), kind='establish', part=0,
                    surface=0, episode=dict(interval=t)) for i, t in enumerate(times)])
    groups, audit = compatible_sources([report([[10, 30], [10, 30]]),
                                        report([[40, 60], [40, 60]]), report([[10, 30], [40, 60]])])
    assert groups == [[0, 1], [2]]
    assert not audit['cross_limb_total_order']


def test_quaternion_signs_do_not_create_fusion_rotation_or_jump():
    base = poses()
    sources = np.stack([base, base, base])
    sources[1, ::2, 3] = -1
    assert np.max(np.abs(pose_residual(sources, base))) == 0
    b, delta, smooth, _ = fusion_stages(sources, 0, [0, 50, 99], fps=50)
    np.testing.assert_allclose(apply_residual(b, smooth), base)


def test_phase_weights_are_shared_and_residual_noise_is_reduced():
    base = poses()
    sources = np.stack([base, base, base])
    noise = .01*(-1.)**np.arange(len(base))
    sources[1:, :, 0] = noise
    sources[1:, :, 7] = noise
    b, delta, smooth, audit = fusion_stages(sources, 0, [0, 30, 70, 99], fps=50)
    np.testing.assert_allclose(delta[:, 0], delta[:, 6])
    assert np.linalg.norm(np.diff(smooth[:, 0], n=2)) < np.linalg.norm(np.diff(delta[:, 0], n=2))
    np.testing.assert_allclose(np.sum(audit['phase_weights'], axis=1), 1)


def test_local_correction_preserves_outside_and_refuses_large_changes():
    base = poses(); candidate = base.copy(); candidate[:, 0] = .005
    failed = dict(checks=[dict(passed=False, episode=dict(interval=[45, 48]))])
    corrected, audit = localized_transfer(candidate, base, failed, fps=50)
    np.testing.assert_array_equal(corrected[:39], candidate[:39])
    np.testing.assert_array_equal(corrected[55:], candidate[55:])
    assert corrected[46, 0] == 0
    candidate[:, 0] = .2
    assert localized_transfer(candidate, base, failed, fps=50)[0] is None


def test_keep_does_not_add_a_timing_anchor():
    from somaforge_core.consensus_event_alignment import event_times
    r = dict(checks=[dict(passed=True, action='hold', kind='keep', part=0,
                        support_episode=dict(interval=[10, 80]))])
    assert event_times(r) == {}


def test_local_support_repair_uses_actual_gap_in_corresponding_representative_episode():
    base = poses(); candidate = base.copy(); candidate[:, 0] = .005
    row = dict(action='handover', kind='keep', part=0, surface=0)
    failed = dict(checks=[dict(row, passed=False, support_episode=dict(interval=[0, 10]))])
    reference = dict(checks=[dict(row, passed=True, support_episode=dict(interval=[30, 70]))])
    mask = np.ones((100, 6), bool); mask[48:51, 0] = False
    surface = np.zeros((100, 6), int)
    out, audit = localized_transfer(candidate, base, failed, fps=50, representative_report=reference,
                                    candidate_mask=mask, candidate_surface=surface)
    assert out[49, 0] == 0
    np.testing.assert_array_equal(out[:42], candidate[:42])
    np.testing.assert_array_equal(out[57:], candidate[57:])
