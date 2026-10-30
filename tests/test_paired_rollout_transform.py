import numpy as np
from generator.research.archived_experiments.paired_pose_transform import transform_q,yaw_basis


def test_roundtrip_and_scene_frame():
    q=np.zeros(36,np.float32);q[:3]=[.7,.8,.9];q[3:7]=[np.cos(.4),0,0,np.sin(.4)]
    origin=np.array([.3,-.2,.5]);basis=yaw_basis(np.array([np.cos(.7),0,0,np.sin(.7)]))
    restored=transform_q(transform_q(q,origin,basis,True),origin,basis)
    np.testing.assert_allclose(restored,q,atol=2e-7)
