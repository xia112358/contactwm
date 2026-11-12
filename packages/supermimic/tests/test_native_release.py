import torch,warp as wp,numpy as np
from supermimic.contact_onset import NativeContactOnset
from supermimic.contact_release import native_release_history


def test_short_real_airborne_motion_releases_but_contact_roll_and_jitter_do_not():
    with wp.ScopedDevice('cpu'):
        m=NativeContactOnset(1,'cpu',20);m.begin_stage(torch.tensor([0]),torch.ones(1,6,dtype=torch.bool),torch.ones(1,6,dtype=torch.bool))
        groups=wp.array([0],dtype=int);query=wp.ones(1,dtype=wp.bool);known=wp.ones(1,dtype=wp.bool);sub=wp.zeros(1,dtype=int)
        def step(active,x):
            wp.to_torch(query)[0]=active;wp.to_torch(m.current)[0,0]=x
            m.begin_control();m.sample(groups,query,known,sub)
            return bool(wp.to_torch(m.blocked)[0]),bool(wp.to_torch(m.event)[0])
        assert step(True,0)==(True,False)
        assert step(True,.05)==(True,False)  # motion within sustained contact
        assert step(False,.0503)==(True,False)
        assert step(True,.0505)==(True,False)  # one-tick scrape/dropout
        assert step(False,.056)==(True,False)
        assert step(False,.062)==(False,False)  # real motion without native contact
        assert step(True,.068)==(False,True)


def test_reference_release_uses_same_native_motion_and_no_future_evidence():
    a=np.array([[1],[0],[0],[1]],dtype=bool);p=np.zeros((4,1,3));p[:,0,0]=[0,.006,.012,.018]
    phase,absence,onset=native_release_history(a,np.ones_like(a),p,20)
    assert phase[:,0].tolist()==[True,True,False,True]
    assert onset[:,0].tolist()==[0,0,0,3]
    short=native_release_history(a[:2],np.ones_like(a[:2]),p[:2],20)
    assert short[0][-1,0]  # no look-ahead at the later displacement


def test_boundary_repair_moves_shared_cut_once_without_overlap():
    from supermimic.stable_contact_slices import move_partition_before_native_onsets
    p=dict(case_ids=torch.tensor([3,7,9]),trace_ids=torch.zeros(3,dtype=torch.long),starts=torch.tensor([0,10,20]),stops=torch.tensor([10,20,30]),goals=torch.tensor([9,19,29]))
    result,changes=move_partition_before_native_onsets(p,{7:9,9:22})
    assert result['starts'].tolist()==[0,9,20]
    assert result['stops'].tolist()==[9,20,30]
    assert result['goals'].tolist()==[8,19,29]
    assert p['starts'].tolist()==[0,10,20]
    assert len(changes)==1


def test_boundary_repair_cannot_invent_prefix_or_erase_previous_event():
    import pytest
    from supermimic.stable_contact_slices import move_partition_before_native_onsets
    p=dict(case_ids=torch.tensor([3,7]),trace_ids=torch.zeros(2,dtype=torch.long),starts=torch.tensor([0,10]),stops=torch.tensor([10,20]),goals=torch.tensor([9,19]))
    with pytest.raises(ValueError,match='retained source edge'):move_partition_before_native_onsets(p,{3:-1})
    with pytest.raises(ValueError,match='erase'):move_partition_before_native_onsets(p,{7:0})
