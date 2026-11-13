import importlib.util
from pathlib import Path
import torch


def test_resume_restores_entries_capacity_and_exact_next_draw():
    path=Path(__file__).parents[1]/'supermimic/handoff/stratified_pool.py'
    spec=importlib.util.spec_from_file_location('pool_resume_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    original=mod.StratifiedPool(capacity=16)
    for origin,depth in [(0,2),(1,1)]:
        payload=dict(s=torch.arange(12).reshape(6,2).float()+origin*100,origin=torch.full((6,),origin),depth=torch.full((6,),depth))
        original.add(2,payload,10)
    original.sample(2,5,10)
    restored=mod.StratifiedPool(capacity=1);restored.load_state_dict(original.state_dict())
    assert restored.capacity==original.capacity
    assert restored.composition()==original.composition()
    for _ in range(3):
        a=original.sample(2,9,11);b=restored.sample(2,9,11)
        for name in a:assert torch.equal(a[name],b[name])
