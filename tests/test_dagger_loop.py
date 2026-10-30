import pytest
import torch
from generator.dagger_loop import append_dataset, model_digest, verify_iteration


def test_accumulation_preserves_exact_old_prefix():
    old=[{'x':torch.tensor([[1.],[2.]])}]
    new=[{'x':torch.tensor([[3.]])}]
    merged=append_dataset(old,new)
    assert merged[0]['x'].tolist()==[[1.],[2.],[3.]]
    assert old[0]['x'].tolist()==[[1.],[2.]]


def test_reject_stale_model_reset_optimizer_or_lost_dataset():
    previous=dict(updated_model_sha256='new',optimizer_step_end=200,dataset_size_after=8)
    verify_iteration(previous,'new',200,8)
    for args in [('old',200,8),('new',0,8),('new',200,0)]:
        with pytest.raises(ValueError):verify_iteration(previous,*args)


def test_model_digest_changes_on_parameter_update():
    model=torch.nn.Linear(1,1)
    before=model_digest(model)
    with torch.no_grad():model.weight.add_(1)
    assert model_digest(model)!=before
