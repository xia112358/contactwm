import numpy as np
from scipy.spatial.transform import Rotation
from somaforge_core.motion_signal import smooth_motion_signal


def motion():
    q=np.zeros((101,36),np.float64);q[:,0]=np.linspace(0,1,101);q[:,2]=.8
    rotation=Rotation.from_euler('z',np.linspace(0,4*np.pi,101)[:,None])
    q[:,3:7]=rotation.as_quat()[:,[3,0,1,2]]
    q[1:-1,7]=.03*np.where(np.arange(99)%2,1,-1)
    return q


def test_smooths_signal_noise_preserves_endpoints_and_source():
    q=motion();original=q.copy();out=smooth_motion_signal(q,fps=50)
    np.testing.assert_array_equal(q,original)
    np.testing.assert_allclose(out[[0,-1]],q[[0,-1]],atol=1e-12)
    assert np.diff(out[:,7],n=2).std()<np.diff(q[:,7],n=2).std()/10
    np.testing.assert_allclose(np.linalg.norm(out[:,3:7],axis=1),1,atol=1e-12)


def test_world_rotation_rotates_heading_and_path_together():
    q=motion();yaw=Rotation.from_euler('z',[.7]);shift=np.array([2.,3.,.2])
    transformed=q.copy();transformed[:,:3]=yaw.apply(q[:,:3])+shift
    transformed[:,3:7]=(yaw*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    a=smooth_motion_signal(q,fps=50);b=smooth_motion_signal(transformed,fps=50)
    np.testing.assert_allclose(b[:,:3],yaw.apply(a[:,:3])+shift,atol=1e-10)
    expected=yaw*Rotation.from_quat(a[:,[4,5,6,3]])
    actual=Rotation.from_quat(b[:,[4,5,6,3]])
    assert (expected.inv()*actual).magnitude().max()<1e-10


def test_local_smoothing_preserves_every_pose_outside_compact_support():
    from somaforge_core.motion_signal import local_smooth_motion_signal
    q=motion();out,window=local_smooth_motion_signal(q,[50],fps=50,support_time_scale=.08)
    np.testing.assert_array_equal(out[window==0],q[window==0])
    assert window[50]==1 and (window[:38]==0).all() and (window[62:]==0).all()
    assert abs(out[50,7]) < abs(q[50,7])/3
    np.testing.assert_allclose(np.linalg.norm(out[:,3:7],axis=1),1,atol=1e-12)


def test_boundary_support_does_not_collapse_and_preserves_initial_velocity():
    from somaforge_core.motion_signal import local_smooth_motion_signal
    q=motion()
    a,wa=local_smooth_motion_signal(q,[1,2],fps=50,support_time_scale=.12)
    b,wb=local_smooth_motion_signal(q,[1,2],fps=50,support_time_scale=.24)
    assert (wa>0).sum()>10 and (wb>0).sum()>(wa>0).sum()
    np.testing.assert_array_equal(a[:2],q[:2])
    np.testing.assert_array_equal(a[wa==0],q[wa==0])
    # The same boundary conditions apply under time reversal.
    reversed_a,_=local_smooth_motion_signal(q[::-1],[99,98],fps=50,support_time_scale=.12)
    np.testing.assert_allclose(a,reversed_a[::-1],atol=1e-10)
