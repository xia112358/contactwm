from types import SimpleNamespace
import torch
from supermimic.contact_substep_support import SubstepSupport
from supermimic.contact_support_events import SupportEventContactMonitor


def setup(initial=True):
    m=SupportEventContactMonitor.__new__(SupportEventContactMonitor)
    detector=SubstepSupport(.005)
    p=torch.zeros(1,6,3);R=torch.eye(3).expand(1,6,3,3)
    m.monitor=SimpleNamespace(support=detector,reader=SimpleNamespace(last_part_poses=(p,R)))
    m.exit_steps=detector.exit_steps;m.blocked=torch.full((1,6),initial);m.delivered=torch.zeros(1,6,dtype=torch.bool);m.new_event=m.delivered.clone()
    tick=0
    def feed(active,n):
        nonlocal tick
        emitted=False
        for _ in range(n):
            a=torch.full((1,6),active);obs=dict(active=a,known=torch.ones_like(a),surface_local=torch.zeros(1,6,dtype=torch.long),point=p)
            features=dict(contact_load=a.float()*80,contact_material_speed=torch.zeros(1,6),contact_stable=a)
            support=detector.update(obs,features,(p,R),torch.tensor([tick]))
            m.begin_control();m._sample(tick,0,obs,features,support);emitted |= bool(m.new_event.any());tick+=1
        return emitted
    return m,feed


def test_entry_support_and_one_control_frame_gap_never_create_event():
    m,feed=setup(True)
    assert not feed(True,12)
    assert not feed(False,4)  # 20ms dropout, below the existing 40ms exit contract
    assert not feed(True,12)


def test_confirmed_exit_then_new_support_emits_once_without_position_goal():
    m,feed=setup(True)
    assert not feed(True,12)
    assert not feed(False,m.exit_steps)
    assert feed(True,12)
    assert not feed(True,12)


def test_handoff_cannot_reuse_predecessor_support():
    m,feed=setup(False)
    assert feed(True,12)
    m.begin_stage(torch.tensor([0]),torch.ones(1,6,dtype=torch.bool))
    assert not feed(True,12)
    assert not feed(False,4)
    assert not feed(True,12)
    feed(False,m.exit_steps)
    assert feed(True,12)


def test_handoff_during_bridged_gap_keeps_existing_support_blocked():
    m,feed=setup(False)
    assert feed(True,12)
    assert not feed(False,4)
    m.begin_stage(torch.tensor([0]),torch.zeros(1,6,dtype=torch.bool),continuing=True)
    assert not feed(True,12)
    feed(False,m.exit_steps)
    assert feed(True,12)
