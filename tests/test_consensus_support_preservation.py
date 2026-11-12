import numpy as np
from somaforge_core.consensus_support_preservation import support_reference,support_loss
from somaforge_core.observed_event_consensus import build_consensus,audit_consensus

def test_matching_one_fragment_does_not_preserve_continuous_support():
    mask=np.zeros((3,40,6),bool);mask[:,1:35,1]=True
    face=np.where(mask,1,-1);c=build_consensus(mask,face)
    reports=[audit_consensus(c,m,s) for m,s in zip(mask,face)]
    ref=support_reference(c,reports,mask,face,mask[0],face[0],np.tile(np.arange(40),(3,1)),range(3))
    broken=mask[0].copy();broken[16:21,1]=False
    assert audit_consensus(c,broken,face[0])['passed']
    assert support_loss(ref,broken,face[0])['lost_part_frames']==5
    assert support_loss(ref,mask[0],face[0])['passed']

def test_no_contact_is_invented_where_fusion_or_population_lacks_it():
    mask=np.zeros((3,40,6),bool);mask[:,1:35,1]=True
    face=np.where(mask,1,-1);c=build_consensus(mask,face)
    reports=[audit_consensus(c,m,s) for m,s in zip(mask,face)]
    fused=mask[0].copy();fused[10,1]=False;mask[:2,20,1]=False
    ref=support_reference(c,reports,mask,face,fused,face[0],np.tile(np.arange(40),(3,1)),range(3))
    assert not ref['wanted'][10,1] and not ref['wanted'][20,1]

def test_pose_audit_fragment_cannot_shorten_measured_support_interval():
    mask=np.zeros((3,40,6),bool);mask[:,1:35,1]=True
    face=np.where(mask,1,-1);c=build_consensus(mask,face)
    reports=[audit_consensus(c,m,s) for m,s in zip(mask,face)]
    for report in reports:report['checks'][0]['episode']['interval']=[22,26]
    ref=support_reference(c,reports,mask,face,mask[0],face[0],np.tile(np.arange(40),(3,1)),range(3))
    assert ref['wanted'][1:35,1].all()
