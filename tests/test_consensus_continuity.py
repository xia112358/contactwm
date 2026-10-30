import numpy as np
import pytest
import torch

from somaforge_core.consensus_continuity import ContinuityGuard, acceptance_status, continuous_source_clock
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics


def fixture(n=81, fps=50, count=3):
    q = np.zeros((count, n, 36), np.float32)
    q[:, :, 2] = .8; q[:, :, 3] = 1
    t = np.arange(n)/fps
    q[:, :, 0] = .2*t
    q[:, :, 7] = .1*np.sin(t)
    anchors = [0, (n-1)/2, n-1]
    alignment = dict(output_anchors=anchors, source_anchors=[anchors]*count,
                     event_keys=[['touch', 'establish', 0, 'start']], common_event_frames=[anchors[1]])
    return q, alignment, CanonicalG1ForwardKinematics()


@pytest.mark.parametrize('channel', ['root_position', 'root_orientation', 'left_hip_pitch_joint'])
def test_isolated_jump_has_exact_frame_and_channel_even_in_long_quiet_motion(channel):
    q, alignment, fk = fixture(n=501)
    guard = ContinuityGuard(q, alignment, range(3), 50, fk)
    motion = q[0].copy()
    if channel == 'root_position': motion[250, 0] += .1
    elif channel == 'root_orientation': motion[250, 3:7] = [np.cos(.2), 0, 0, np.sin(.2)]
    else: motion[250, 7] += .3
    report = guard.check(motion, fk)
    assert not report['passed']
    assert any(r['channel'] == channel and 250 in range(r['stencil'][0], r['stencil'][1]+1)
               for r in report['violations'])
    assert any('/marker_' in r['channel'] for r in report['violations'])
    assert len(guard.channels) == 31+4*len(fk.joint_records)+4


def test_quaternion_sign_changes_are_not_rotation_jumps():
    q, alignment, fk = fixture()
    guard = ContinuityGuard(q, alignment, range(3), 50, fk)
    q[0, ::2, 3:7] *= -1
    assert guard.check(q[0], fk)['passed']


def test_one_source_outlier_cannot_hide_output_jump():
    q, alignment, fk = fixture()
    q[2, 40, 0] += 20
    guard = ContinuityGuard(q, alignment, range(3), 50, fk)
    motion = q[0].copy(); motion[40, 0] += .1
    assert not guard.check(motion, fk)['passed']


def test_fast_motion_supported_by_source_phase_passes_but_not_in_quiet_phase():
    q, alignment, fk = fixture()
    q[:, 40:, 7] += .8*np.sin(np.linspace(0, 2*np.pi, 41))
    guard = ContinuityGuard(q, alignment, range(3), 50, fk)
    assert guard.check(q[0], fk)['passed']
    moved = q[0].copy(); moved[10:20, 7] += .2*np.sin(np.linspace(0, 2*np.pi, 10))
    report = guard.check(moved, fk)
    assert not report['passed'] and any(r['phases'] == [0] for r in report['violations'])


def test_fps_changes_units_not_acceptance_and_loss_has_finite_gradient():
    q, alignment, fk = fixture()
    a = ContinuityGuard(q, alignment, range(3), 25, fk)
    b = ContinuityGuard(q, alignment, range(3), 50, fk)
    np.testing.assert_allclose(b.bounds[0], a.bounds[0]*2)
    np.testing.assert_allclose(b.bounds[1], a.bounds[1]*4)
    motion = torch.tensor(q[0], requires_grad=True)
    with torch.no_grad(): motion[30, 0] += .05
    cost, count, _ = b.objective(motion, fk)
    cost.backward()
    assert count > 0 and torch.isfinite(motion.grad).all() and motion.grad[30, 0] > 0
    assert a.check(motion.detach(), fk)['violation_count'] == b.check(motion.detach(), fk)['violation_count']


def test_continuous_clock_does_not_calibrate_using_rounded_repeat_skip_jumps():
    q, alignment, fk = fixture()
    q[:, :, 7] = 0
    alignment['source_anchors'] = [[0, 20, 80]]*3
    mapped = continuous_source_clock(q[0], alignment['output_anchors'], alignment['source_anchors'][0])
    guard = ContinuityGuard(q, alignment, range(3), 50, fk)
    assert guard.check(mapped, fk)['passed']
    # All channels are calibrated from the original continuous clock.
    assert not np.array_equal(mapped[:, 0], q[0, np.rint(np.interp(np.arange(81), [0, 40, 80], [0, 20, 80])).astype(int), 0])


def test_acceptance_requires_contacts_and_local_continuity_without_rms_gate():
    assert acceptance_status(dict(passed=True), dict(passed=True, global_rms=999)) == 'accepted'
    assert acceptance_status(dict(passed=True), dict(passed=False)) == 'local_continuity_unresolved'
    assert acceptance_status(dict(passed=False), dict(passed=True)) == 'contact_guard_unresolved'


def test_pooled_independent_events_do_not_fabricate_a_stationary_source_phase():
    q, alignment, fk = fixture()
    q[:, :, 7] = 0
    alignment.update(output_anchors=[0, 20, 60, 80], source_anchors=[[0, 40, 40, 80]]*3,
        event_keys=[['a', 'establish', 0, 'start'], ['b', 'establish', 1, 'start']], common_event_frames=[20, 60])
    report = dict(checks=[dict(action=action, part=part, kind='establish', passed=True,
                              episode=dict(interval=[start, 81])) for action, part, start in [('a', 0, 50), ('b', 1, 30)]])
    guard = ContinuityGuard(q, alignment, range(3), 50, fk, source_reports=[report]*3)
    assert guard.native_intervals[0][1] == [30, 50]
    assert guard.check(q[0], fk)['passed']  # Clock is frozen; original motion is not.
