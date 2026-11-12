import torch
import pytest
from types import SimpleNamespace
from supermimic.contact_substep_support import SubstepSupport, SubstepContactMonitor, substep_support_contract


def sample(monitor,t,active=True,load=100.,speed=0.,position=0.,worlds=1):
    shape=(worlds,6)
    obs=dict(active=torch.full(shape,active),known=torch.ones(shape,dtype=torch.bool),surface_local=torch.zeros(shape,dtype=torch.long))
    features=dict(contact_load=torch.full(shape,load),contact_material_speed=torch.full(shape,speed))
    p=torch.zeros(worlds,6,3);p[:,:,0]=position
    R=torch.eye(3).expand(worlds,6,3,3).clone()
    step=torch.full((worlds,),t,dtype=torch.long)
    return monitor.update(obs,features,(p,R),step)


def test_confirmation_is_five_physics_solves():
    m=SubstepSupport(.002)
    for t in range(4):assert not sample(m,t)['state'].any()
    event=sample(m,4)
    assert event['established'].all()
    assert (event['onset_step']==0).all()
    assert (event['confirmed_step']==4).all()
    assert substep_support_contract(.002,10)['confirmation_span_s']==.008
    assert not sample(m,5)['established'].any()


def test_scrape_does_not_become_support_in_shorter_window():
    m=SubstepSupport(.002)
    for t in range(20):
        assert not sample(m,t,speed=.3,position=t*.002*.3)['state'].any()


def test_pose_tolerance_scales_with_physics_dt():
    m=SubstepSupport(.002)
    for t in range(20):
        assert not sample(m,t,speed=.1,position=t*.0005)['state'].any()


def test_measured_short_gap_is_continuous_and_exit_keeps_physical_time():
    m=SubstepSupport(.002)
    for t in range(25):sample(m,t)
    for t in range(25,40):assert sample(m,t,False,0.,position=.004)['state'].all()
    assert not sample(m,40,position=.004)['established'].any()
    for t in range(41,60):assert sample(m,t,False,0.,position=.004)['state'].all()
    result=sample(m,60,False,0.,position=.004)
    assert result['exited'].all() and (result['exit_step']==41).all()


def test_real_departure_does_not_wait_out_stationary_gap_timer():
    m=SubstepSupport(.002)
    for t in range(25):sample(m,t)
    assert sample(m,25,False,0.,position=.003)['state'].all()
    result=sample(m,26,False,0.,position=.03)
    assert result['exited'].all() and (result['exit_step']==25).all()


def test_online_confirmation_and_independent_event_are_distinct():
    m=SubstepSupport(.002)
    for t in range(5):result=sample(m,t)
    assert result['established'].all() and not result['event_ready'].any()
    for t in range(5,24):assert not sample(m,t,False,0.)['event_ready'].any()
    assert sample(m,24,False,0.)['exited'].all()
    m=SubstepSupport(.002)
    for t in range(20):result=sample(m,t)
    assert result['event_ready'].all()


def test_loaded_internal_rolling_preserves_support():
    m=SubstepSupport(.002)
    for t in range(25):sample(m,t)
    for t in range(25,50):
        result=sample(m,t,position=(t-24)*.01,speed=1.)
        assert result['state'].all() and not result['established'].any()


def test_world_reset_cannot_reuse_support_history():
    m=SubstepSupport(.002)
    for t in range(6):sample(m,t)
    assert not sample(m,0)['state'].any()
    for t in range(1,4):assert not sample(m,t)['established'].any()
    assert sample(m,4)['established'].all()


def test_loaded_but_inactive_is_not_contact():
    m=SubstepSupport(.002)
    for t in range(10):assert not sample(m,t,False)['state'].any()


def test_monitor_retains_intra_control_events_and_restores_hook(monkeypatch):
    import supermimic.contact_substep_support as module
    class Quality:
        def __init__(self,dt):pass
        def update(self,view,poses,step):
            return dict(contact_load=torch.full((1,6),100. if reader.i<5 else 0.),contact_material_speed=torch.zeros(1,6))
    monkeypatch.setattr(module,'NativeContactStability',Quality)
    calls=[]
    original=lambda:calls.append('solve')
    manager=SimpleNamespace(_step_solver=original,_cfg=SimpleNamespace(use_cuda_graph=True),
        _invalidate_graph=lambda:None,get_solver_dt=lambda:.002,_num_substeps=10)
    env=SimpleNamespace(sim=SimpleNamespace(backend_manager=manager),ep_step=torch.zeros(1,dtype=torch.long),scene=SimpleNamespace(env_origins=torch.zeros(1,3)))
    class Reader:
        direct=True;primary_faces=True;i=-1;last_view=None
        last_part_poses=(torch.zeros(1,6,3),torch.eye(3).expand(1,6,3,3))
        def read_device(self,*,substep,same_solve):
            self.i+=1
            assert substep==self.i%10 and same_solve
            self.evidence_binding=object()
            self.last_evidence=SimpleNamespace(terrain_view=lambda binding:SimpleNamespace(reduce=lambda:{'contact_count':torch.full((1,6),int(self.i<5))}))
            return dict(active=torch.full((1,6),self.i<5),known=torch.ones(1,6,dtype=torch.bool),surface_local=torch.zeros(1,6,dtype=torch.long))
    reader=Reader()
    with SubstepContactMonitor(env,reader) as monitor:
        assert not manager._cfg.use_cuda_graph
        with pytest.raises(ValueError,match='all actual'):monitor.summary()
        for _ in range(10):manager._step_solver()
        summary=monitor.summary()
        assert summary['substep_support_established'][:,4].all()
        assert not summary['substep_support_exited'].any()
        assert not summary['substep_support_event_ready'].any()
        assert summary['support_state'].all()
        assert summary['contact_active_any'].all()
        assert torch.equal(summary['substep_raw_contact_active'],summary['substep_contact_active'])
        assert summary['substep_part_position'].shape==(1,10,6,3)
    assert manager._step_solver is original
    assert manager._cfg.use_cuda_graph
    assert len(calls)==10


def recording(start=0,frames=6,on=7,off=35):
    m=SubstepSupport(.002);results=[sample(m,t,on<=t<off,100. if on<=t<off else 0.) for t in range(frames*10)]
    names=dict(event_ready='event_ready',state='state',established='established',exited='exited',onset_step='onset_step',confirmed_step='confirmed_step',exit_step='exit_step')
    data={'substep_support_'+key:torch.stack([r[value][0] for r in results]).reshape(frames,10,6) for key,value in names.items()}
    data.update(ep_step=torch.arange(frames),surface_local=torch.zeros(frames,6,dtype=torch.long))
    return {key:value[start:] for key,value in data.items()}


def test_projection_uses_substep_onset_without_waiting_five_control_frames():
    from supermimic.contact_substep_support import control_support_episodes
    episodes=control_support_episodes(recording())
    assert len(episodes)==6
    assert all((e['start'],e['confirmed'],e['stop'])==(0,1,3) for e in episodes)


def test_projection_clips_history_to_source_slice():
    from supermimic.contact_substep_support import control_support_episodes
    episodes=control_support_episodes(recording(start=1))
    assert all((e['start'],e['confirmed'],e['stop'])==(0,0,2) for e in episodes)
    assert len(episodes)==6


def test_intraframe_episode_retained_without_creating_zero_frame_segment():
    from supermimic.contact_substep_support import control_support_episodes
    data=recording(on=10,off=15)
    assert data['substep_support_established'].any()
    assert data['substep_support_exited'].any()
    assert not control_support_episodes(data)


def test_first_post_reset_solve_is_known_only_in_observed_worlds():
    from supermimic.contact_native import NativeContacts
    r=NativeContacts.__new__(NativeContacts)
    r.env=SimpleNamespace(N=2,device='cpu',ep_step=torch.zeros(2,dtype=torch.long))
    r._observed_world_mask=torch.tensor([True,False])
    known=r._known_mask(solved=True)
    assert known[0].all() and not known[1].any()
    out=r._mask_unknown(dict(active=torch.ones(2,6,dtype=torch.bool)),known=known)
    assert out['active'][0].all() and not out['active'][1].any()
