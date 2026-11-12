"""Failed handoff endpoints retain body supervision, not success witnesses."""
import importlib.util
from pathlib import Path
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('stop',[1,2,3])
def test_interrupted_endpoint_has_body_loss_and_stays_failed(stop):
    dmo=load('supermimic.handoff_endpoint_test',ROOT/'packages/supermimic/supermimic/handoff/contact_dmo.py')
    fixture=load('completion_fixture',ROOT/'packages/supermimic/tests/test_contact_completion_latch.py').fixture
    clock,data=fixture()
    data['contact_stable'].zero_()
    data['body_endpoint_valid']=torch.zeros_like(data['done'])
    data['contact_interrupted']=torch.ones_like(data['done'])
    ev,terminal,schedule=dmo._replay_events(data,clock,torch.tensor([0]),torch.tensor([stop]),torch.tensor([stop<3]),'accumulated_target',None,stable_contact_frames=2)
    assert terminal[:,0].tolist()==[False]*(stop-1)+[True]
    assert schedule[stop-1]['root'].tolist()==[0]
    assert int(ev.sum())==2
    assert dmo.loss_aligned_attempt_metrics(data,clock,torch.tensor([.05]),.25,.4)['success_rate']==0
