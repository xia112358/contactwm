import numpy as np
import torch
from generator.rollout_termination import (
    pose_failure, geometry_failure, ProgressWatch, training_rollout_safety_mask,
)

def test_pose():
    q=np.zeros(36);q[3]=1;lo=np.full(29,-1.);hi=-lo
    assert pose_failure(q,lo,hi) is None
    q[7]=1.2;assert pose_failure(q,lo,hi)=='joint_limit'
    q[7]=np.nan;assert pose_failure(q,lo,hi)=='nonfinite_pose'

def test_depth_and_unknown():
    s=dict(terrain_penetration_m=.03,self_penetration_m=0)
    assert geometry_failure(s) is None
    s['terrain_penetration_m']=.0301
    assert geometry_failure(s)=='large_penetration'
    s['worst_self']=dict(normal_w=[0,0,0])
    assert geometry_failure(s)=='invalid_geometry'

def test_stall_resets_on_progress():
    w=ProgressWatch(0)
    for _ in range(7):assert w.update(0) is None
    assert w.update(1) is None
    for _ in range(7):assert w.update(None) is None
    assert w.update(None)=='stalled'

def test_training_rollout_keeps_contact_mismatch_until_a_safety_boundary():
    q=torch.zeros((4,36));q[:,3]=1
    safe=training_rollout_safety_mask(
        q,
        torch.tensor([0.,2.99,3.01,0.]),
        torch.tensor([0.,0.,0.,1.]),
        torch.tensor([0.,.14,0.,0.]),
    )
    assert safe.tolist()==[True,True,False,False]
