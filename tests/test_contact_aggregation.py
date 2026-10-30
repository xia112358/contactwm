import itertools
import numpy as np
import pytest
from somaforge_core.contact_aggregation import (binary_consensus,align_pose,
    group_observations,equal_demo_patch_cloud,AggregationConfig)


def test_exact_dynamic_program():
    votes=np.array([[[0],[1],[0],[1]],[[1],[1],[0],[0]],[[0],[1],[0],[0]]],bool)
    result,p=binary_consensus(votes,.4)
    energy=lambda s: np.abs(np.asarray(s,dtype=float)-votes[:,:,0]).mean(0).sum()+.4*np.count_nonzero(np.diff(s))
    assert energy(result[:,0])==pytest.approx(min(energy(x) for x in itertools.product((0,1),repeat=4)))


def test_absence_votes_and_temporal_denoising():
    v=np.zeros((5,15,1),bool); v[0,:,0]=1
    assert not binary_consensus(v,0)[0].any()
    v[:]=1; v[:,7]=0
    smoothed,support=binary_consensus(v,2)
    assert smoothed.all() and support[7,0]==0
    assert not v[:,7].any()  # immutable observations


def test_points_do_not_multiply_votes_or_mix_faces():
    p=dict(part=0,robot_shape=1,surface=0,allocated=True)
    keys,v=group_observations([[[p]*100],[[dict(p,surface=1)]],[[]]])
    assert keys==[(0,1,0),(0,1,1)] and v.sum()==2
    assert not binary_consensus(v,0)[0].any()
    c=equal_demo_patch_cloud([np.zeros((100,3)),np.ones((1,3))])
    assert c['point_weights'][:100].sum()==pytest.approx(.5)
    assert c['point_weights'][100]==pytest.approx(.5)
    assert np.array_equal(c['representative_cloud'],np.zeros((100,3)))


def test_pose_alignment_identity_and_bounds():
    q=np.zeros((21,36)); q[:,3]=1; q[:,0]=np.linspace(0,1,21)
    m,path=align_pose(q,q,3)
    assert np.array_equal(m,np.arange(21))
    flipped=q.copy(); flipped[:,3:7]*=-1
    assert np.array_equal(align_pose(q,flipped,3)[0],m)
    shifted=q[np.maximum(np.arange(21)-2,0)]
    m,_=align_pose(q,shifted,3)
    assert (np.diff(m)>=0).all() and np.abs(m-np.arange(21)).max()<=3
    assert (m[0],m[-1])==(0,20)


def test_contract_not_solver_labels():
    c=AggregationConfig().contract(fps=50,source_contract={'schema':'test'})
    assert c['semantics']=='offline_interaction_hypothesis_not_realized_contact'
    with pytest.raises(ValueError): AggregationConfig(switch_cost=-1)
