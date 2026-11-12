import torch
import pytest
from holosoma.contact_wm.stream_replay import Replay


@pytest.mark.parametrize('batches',[1,2,10])
def test_frozen_control_and_alternating_update_ratio(batches):
    from holosoma.contact_wm.stream_train import wm_update_count
    assert wm_update_count('frozen',batches)==0
    assert wm_update_count('alternating',batches)==batches
    with pytest.raises(ValueError):wm_update_count('alternating',0)


def test_resume_counts_use_saved_ratio_and_exact_new_counts():
    from holosoma.contact_wm.stream_train import restored_wm_updates
    legacy=dict(iteration=1878,options=dict(wm_batches=10))
    assert restored_wm_updates(legacy,2)==18780
    assert restored_wm_updates(dict(legacy,wm_updates=18900),2)==18900


def test_ring_windows_never_cross_reset_or_wrap_discontinuity():
    r=Replay(10,4,'cpu')
    for step in range(30):
        age=step%7
        r.add(dict(state=torch.full((4,1),float(step)),valid=torch.full((4,),age!=6),
                   previous_known=torch.ones(4,6,dtype=torch.bool)))
    for _ in range(10):
        x=r.windows(20,3)
        assert x['valid'].all()
        assert torch.equal(x['state'][1:]-x['state'][:-1],torch.ones_like(x['state'][1:]))
        assert x['state'].min()>=20


def test_unknown_contacts_excluded_from_windows():
    r=Replay(10,2,'cpu')
    for step in range(10):
        known=torch.ones(2,6,dtype=torch.bool);known[0]=False
        r.add(dict(state=torch.tensor([[0.],[1.]]),valid=torch.ones(2,dtype=torch.bool),previous_known=known))
    assert (r.windows(32,3)['state']==1).all()


def test_native_monitor_does_not_turn_proximity_into_contact(tmp_path):
    import json
    import numpy as np
    from holosoma.contact_wm.native_monitor import NativeMonitor
    from holosoma.contact_wm.core import ContactGoal
    from holosoma.contact_wm.geometry import LINKS
    from somaforge_core import G1_29DOF_JOINT_ORDER
    mesh=tmp_path/'robot.json';mesh.write_text(json.dumps({'links':LINKS}))
    atlas=tmp_path/'atlas.npz';np.savez(atlas,height=np.full((1,3,3),np.inf),minimum=np.zeros((1,2)),resolution=.05)
    m=NativeMonitor(tmp_path,G1_29DOF_JOINT_ORDER,mesh,atlas)
    xyz=torch.zeros(9,6,3);normal=xyz.clone();normal[:,:,2]=1
    new=torch.zeros(9,6,dtype=torch.bool);new[:,2]=True
    support=torch.zeros_like(new);support[:,3:5]=True
    goal=ContactGoal(xyz,normal,new,support,xyz,torch.ones(9,6),26,10)
    m.begin(0,goal);state=torch.zeros(9,71);state[:,6]=1
    for _ in range(40):m.add(state,torch.zeros(9,29),xyz,torch.zeros_like(new),torch.ones_like(new),xyz,torch.zeros(9,dtype=torch.bool))
    report=m.finish()
    assert report['successes']==0
    assert all(not r['checks']['activation'] for r in report['results'])
    exported=json.loads((tmp_path/'native-latest.json').read_text())
    assert exported['terrain']['height']==[[None]]
    assert len(exported['episodes'][0]['frames'])==40


def test_first_post_reset_solve_is_not_used_as_previous_contact():
    r=Replay(4,2,'cpu')
    r.add(dict(state=torch.zeros(2,1),age=torch.tensor([0,5]),valid=torch.ones(2,dtype=torch.bool)))
    assert r.data['valid'][0].tolist()==[False,True]
