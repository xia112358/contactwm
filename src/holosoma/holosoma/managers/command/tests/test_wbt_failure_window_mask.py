import torch

from holosoma.managers.command.terms.wbt import _get_bad_tracking_done_mask


def test_bad_tracking_done_mask_matches_standard_term() -> None:
    env_ids = torch.tensor([0, 2, 3])
    term_dones = {
        "bad_tracking": torch.tensor([True, False, True, False]),
        "timeout": torch.tensor([False, True, False, True]),
    }

    mask = _get_bad_tracking_done_mask(term_dones, env_ids)

    assert mask is not None
    assert mask.tolist() == [True, True, False]


def test_bad_tracking_done_mask_matches_relaxed_term() -> None:
    env_ids = torch.tensor([0, 1, 3])
    term_dones = {
        "bad_tracking_relaxed": torch.tensor([False, True, False, True]),
        "timeout": torch.tensor([True, True, True, True]),
    }

    mask = _get_bad_tracking_done_mask(term_dones, env_ids)

    assert mask is not None
    assert mask.tolist() == [False, True, True]


def test_bad_tracking_done_mask_unions_matching_terms() -> None:
    env_ids = torch.tensor([0, 1, 2, 3])
    term_dones = {
        "bad_tracking": torch.tensor([True, False, False, False]),
        "bad_tracking_relaxed": torch.tensor([False, False, True, False]),
        "severe_state_invalid": torch.tensor([False, True, False, True]),
    }

    mask = _get_bad_tracking_done_mask(term_dones, env_ids)

    assert mask is not None
    assert mask.tolist() == [True, False, True, False]


def test_bad_tracking_done_mask_returns_none_without_matching_term() -> None:
    env_ids = torch.tensor([0, 1])
    term_dones = {"timeout": torch.tensor([True, False])}

    assert _get_bad_tracking_done_mask(term_dones, env_ids) is None
