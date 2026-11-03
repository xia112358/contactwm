"""The legacy endpoint loader remains isolated from offline episode labels."""
import torch


def trace(sequence,stable):
    a=torch.zeros(len(sequence),6,dtype=torch.bool);a[:,0]=torch.tensor(sequence,dtype=torch.bool)
    q=torch.zeros_like(a);q[:,0]=torch.tensor(stable,dtype=torch.bool)
    return dict(active=a,known=torch.ones_like(a),contact_stable=q,point=torch.zeros(len(a),6,3),contact_load=a.float()*100)


def test_stable_slice_loader_rejects_incompatible_goals_not_restart_failure(tmp_path):
    import copy
    import pytest
    from supermimic.contact_library import load_slice_clock
    from supermimic.contact_stability import CONTRACT
    from supermimic.contact_phase import VERSION
    d=trace([0,0,1,1],[0,0,0,1]);d['s']=torch.zeros(4,71);d['xi']=torch.zeros(4,52)
    meta=dict(task_manifest_sha256='task',source_checkpoint_sha256='checkpoint',selection_manifest_sha256='selection',stable_support_contract=CONTRACT,contact_phase_version=VERSION,recording_spans=[dict(motion=0,begin=0,end=4)])
    rec=dict(data=d,metadata=meta);target=torch.zeros(1,6,dtype=torch.bool);target[0,0]=True
    packed=dict(schema='native_stable_contact_slices_v1',metadata=copy.deepcopy(meta),starts=torch.tensor([0]),stops=torch.tensor([4]),goals=torch.tensor([3]),motion_ids=torch.tensor([0]),case_ids=torch.tensor([1]),native_restart_passed=torch.tensor([True]),target_masks=target,keep_masks=torch.zeros_like(target),support_masks=torch.zeros_like(target))
    path=tmp_path/'cases.pt';torch.save(packed,path)
    assert load_slice_clock(rec,path).reference_support_contract==CONTRACT
    packed['native_restart_passed'][0]=False;torch.save(packed,path)
    assert load_slice_clock(rec,path).reference_support_contract==CONTRACT
    packed['native_restart_passed'][0]=True;packed['metadata']['stable_support_contract']['min_load_n']=1.;torch.save(packed,path)
    with pytest.raises(ValueError,match='contracts differ'):load_slice_clock(rec,path)
    packed['metadata']=copy.deepcopy(meta);torch.save(packed,path);d['contact_stable'][3,0]=False
    with pytest.raises(ValueError,match='unqualified'):load_slice_clock(rec,path)
