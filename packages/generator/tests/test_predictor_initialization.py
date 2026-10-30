from types import SimpleNamespace

import pytest
import torch

from generator.predictor_initialization import (
    initialize_predictor, initialize_training_joint_bias, state_fingerprint,
)


def test_scratch_never_reads_checkpoint_and_preserves_seeded_weights(monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError('Scratch must never read a checkpoint')
    monkeypatch.setattr(torch, 'load', forbidden_load)
    torch.manual_seed(13)
    first = torch.nn.Linear(4, 36)
    before = state_fingerprint(first)
    initial, loaded, fresh, metadata = initialize_predictor(
        first, initialization='scratch', seed=13)
    torch.manual_seed(13)
    second = torch.nn.Linear(4, 36)
    assert before == state_fingerprint(first) == state_fingerprint(second)
    assert initial == {} and loaded == [] and fresh == ['bias', 'weight']
    assert metadata['predictor_checkpoints_loaded'] == []
    assert metadata['source'] is None
    assert metadata['random_state_sha256'] == before


@pytest.mark.parametrize('argument', ['stage_a_checkpoint', 'initial_position_checkpoint'])
def test_scratch_rejects_inherited_checkpoints(argument):
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        initialize_predictor(torch.nn.Linear(4, 36), initialization='scratch',
                             **{argument: 'unwanted.pt'})


def test_retired_initialization_is_rejected_before_reading_weights(monkeypatch):
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Must not load weights'))
    with pytest.raises(ValueError, match='scratch initialization only'):
        initialize_predictor(torch.nn.Linear(4,36), initialization='stage_a')


def test_joint_bias_excludes_validation_and_does_not_change_other_weights():
    model = SimpleNamespace(pose_head=torch.nn.Linear(4, 36))
    weights = model.pose_head.weight.detach().clone()
    root_bias = model.pose_head.bias[:7].detach().clone()
    q = torch.zeros(4, 36)
    q[0, 7:] = 1
    q[2, 7:] = 3
    q[1, 7:] = q[3, 7:] = 1000
    metadata = initialize_training_joint_bias(model, q, torch.tensor([0, 2]))
    torch.testing.assert_close(model.pose_head.bias[7:], torch.full((29,), 2.))
    torch.testing.assert_close(model.pose_head.bias[:7], root_bias)
    torch.testing.assert_close(model.pose_head.weight, weights)
    assert metadata['validation_or_test_used'] is False
    assert metadata['training_samples'] == 2
