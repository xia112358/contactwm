"""Continuous geometry, chain-rule derivatives and jointly constrained clocks."""
import numpy as np
from scipy.spatial.transform import Rotation
from somaforge_core.continuous_motion import ContinuousMotion, clock_basis, clock_rate_certificate, joint_motion_clocks, certify_clocks


def signal(n=61):
    t = np.linspace(0, 1, n)
    q = np.zeros((n, 36)); q[:, 0] = .1*t*t; q[:, 2] = .8
    q[:, 3:7] = Rotation.from_euler('z', (4*np.pi*t)[:,None]).as_quat()[:, [3,0,1,2]]
    q[:, 7] = .2*t*t
    return q


def test_continuous_curve_reproduces_quadratic_and_chain_rule():
    q = signal(); curve = ContinuousMotion(q); t = np.array([.3, 1.8, 23.4, 59.7])
    pose, v, a = curve.evaluate(t, rate=np.full(4,1.2), acceleration=np.full(4,.03))
    np.testing.assert_allclose(pose[:, 7], .2*(t/60)**2, atol=1e-12)
    np.testing.assert_allclose(v[:, 7], .4*t/60**2*1.2, atol=1e-12)
    np.testing.assert_allclose(a[:, 7], .4/60**2*1.2**2+.4*t/60**2*.03, atol=1e-12)
    # Quaternion value and both derivatives remain tangent to the unit sphere.
    np.testing.assert_allclose((pose[:,3:7]*v[:,3:7]).sum(-1), 0, atol=1e-12)
    np.testing.assert_allclose((pose[:,3:7]*a[:,3:7]).sum(-1)+(v[:,3:7]**2).sum(-1), 0, atol=1e-12)


def test_continuous_pose_commutes_with_world_rotation_and_quaternion_signs():
    q = signal(); world = Rotation.from_euler('xyz',[.2,-.1,.7]); shift=np.array([2.,3.,.1])
    transformed=q.copy(); transformed[:,:3]=world.apply(q[:,:3])+shift
    transformed[:,3:7]=(world*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    transformed[::2,3:7]*=-1
    t=np.linspace(.1,59.9,200);a=ContinuousMotion(q).evaluate(t)[0];b=ContinuousMotion(transformed).evaluate(t)[0]
    np.testing.assert_allclose(b[:,:3],world.apply(a[:,:3])+shift,atol=1e-11)
    relative=(world*Rotation.from_quat(a[:,[4,5,6,3]])).inv()*Rotation.from_quat(b[:,[4,5,6,3]])
    assert relative.magnitude().max()<1e-11


def test_clock_boundary_constraints_are_exact_for_arbitrary_controls():
    basis,knots,curve=clock_basis(61,3)
    control=np.sin(knots/7);control[[0,-1]]=0
    for order in (0,1,2):
        np.testing.assert_allclose(curve([0,60],nu=order)@control,0,atol=1e-11)
    assert basis.shape==(61,len(knots)-2)
    certificate=clock_rate_certificate(curve)
    dense=curve(np.linspace(0,60,4000),nu=1)[:,1:-1]@control[1:-1]
    bound=certificate@control[1:-1]
    assert dense.min()>=bound.min()-1e-10 and dense.max()<=bound.max()+1e-10


def test_joint_registration_preserves_gauge_bounds_and_reduces_joint_objective():
    motions=np.array([signal() for _ in range(3)])
    t=np.arange(61.);maps=np.array([t+.15*t*(1-t/60),t,t-.15*t*(1-t/60)])
    anchors=[(np.array([0.,30.,60.]),np.array([0.,30.,60.]))]*3
    fitted,report=joint_motion_clocks(motions,maps,anchors,resolution=3,steps=15)
    np.testing.assert_array_equal(fitted[:,[0,-1]],[[0,60]]*3)
    assert np.diff(fitted,axis=1).min()>=.25-1e-8
    assert np.diff(fitted,axis=1).max()<=4+1e-8
    assert report['best_loss']<report['history'][0]['loss']
    assert report['native_contact_validation_required']
    proof=certify_clocks(fitted,report['clock_control_knots'],report['clock_displacement_controls'],(.25,4.))
    assert proof['passed']
    import pytest
    changed=fitted.copy();changed[0,10]+=.01
    with pytest.raises(ValueError,match='different sampled clocks'):
        certify_clocks(changed,report['clock_control_knots'],report['clock_displacement_controls'],(.25,4.))


def test_same_clock_audit_accepts_smooth_retiming_but_rejects_an_added_spike():
    from somaforge_core.event_continuity import EventContinuityGuard
    from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
    n=101;u=np.linspace(0,1,n);q=np.zeros((n,36));q[:,2]=.8;q[:,3]=1;q[:,7]=.2*u*u
    clock=u+.2*u*(1-u);warped=q.copy();warped[:,7]=.2*clock*clock
    report=dict(checks=[]);fk=CanonicalG1ForwardKinematics()
    raw=EventContinuityGuard(np.array([q]*3),[report]*3,50,fk)
    common=EventContinuityGuard(np.array([warped]*3),[report]*3,50,fk,reference_clock='transported')
    assert not raw.check(warped,fk,report)['passed']
    assert common.check(warped,fk,report)['passed']
    damaged=warped.copy();damaged[50,7]+=.1
    assert not common.check(damaged,fk,report)['passed']
    assert common.check(warped,fk,report)['schema']=='transported_source_information_continuity_v3'


def test_final_pose_fit_basis_preserves_boundary_states_without_row_cropping():
    from somaforge_core.continuous_motion import pose_correction_basis
    basis,_=pose_correction_basis(101,6)
    np.testing.assert_allclose(basis[[0,1,-2,-1]],0,atol=1e-12)
    assert 0<basis.shape[1]<101
