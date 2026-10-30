from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.agents.callbacks.recording import EvalRecordingCallback
from holosoma.config_types.eval_callback import RecordingConfig






def test_full_recording_preserves_explicit_initial_state_setting() -> None:
    disabled = EvalRecordingCallback(
        RecordingConfig(profile="full", record_initial_state=False),
        SimpleNamespace(device="cpu"),
    )
    enabled = EvalRecordingCallback(
        RecordingConfig(profile="full", record_initial_state=True),
        SimpleNamespace(device="cpu"),
    )

    assert disabled._records_initial_state() is False
    assert enabled._records_initial_state() is True


def test_all_env_recording_can_continue_across_episode_resets() -> None:
    callback = EvalRecordingCallback(
        RecordingConfig(profile="full", env_id=-1, stop_when_done=False),
        SimpleNamespace(device="cpu"),
    )
    callback._update_all_env_done_metadata = lambda actor_state: None
    callback._all_envs_done = lambda: True
    recorded: list[dict] = []
    callback._record_step = recorded.append

    state = {"dones": torch.tensor([True, True])}
    callback.on_post_eval_env_step(state)

    assert recorded == [state]
    assert callback._stopped_after_done is False
