"""Config types for eval callbacks."""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import Literal

from holosoma.config_types.video import VideoConfig
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RecordingConfig:
    """Settings for trajectory recording during evaluation."""

    enabled: bool = False
    """Whether to enable trajectory recording."""

    output_path: str = "eval_recording.npz"
    """Path to save NPZ recording."""

    env_id: int = 0
    """Environment ID to record. Use -1 to record all eval environments."""

    record_initial_state: bool = False
    """Record the reset state before the first evaluation environment step."""

    profile: Literal["full"] = "full"
    """Channel profile for full physics/contact replay."""

    stop_when_done: bool = True
    """Stop recording after the selected environment(s) finish their first episode."""

    chunk_frames: int = 0
    """Flush bounded lossless chunks; zero retains the existing single NPZ format.
    Chunked recordings publish <output_path>.json and retain every raw contact.
    """


@dataclass(frozen=True)
class RecordingCallbackConfig:
    """Instantiation config for EvalRecordingCallback."""

    _target_: str = "holosoma.agents.callbacks.recording.EvalRecordingCallback"
    """Class to instantiate."""

    config: RecordingConfig = RecordingConfig()
    """Recording settings."""


@dataclass(frozen=True)
class AcceptanceConfig:
    """Settings for lightweight per-motion acceptance evaluation."""

    enabled: bool = False
    """Whether to enable acceptance evaluation."""

    output_path: str = "acceptance.csv"
    """Path to save per-motion acceptance CSV."""

    summary_path: str = ""
    """Path to save JSON summary. Defaults to output_path with _summary.json."""

    fail_output_path: str = ""
    """Path to save failed rows only. Defaults to output_path with _fail.csv."""

    require_one_env_per_motion: bool = True
    """Require num_envs to equal num_motions so full acceptance is one env per motion."""

    stop_when_complete: bool = True
    """Stop evaluation once every tracked environment has pass/fail status."""

    repeats: int = 1
    """Number of acceptance evaluation repeats to run in one simulator process."""


@dataclass(frozen=True)
class AcceptanceCallbackConfig:
    """Instantiation config for EvalAcceptanceCallback."""

    _target_: str = "holosoma.agents.callbacks.acceptance.EvalAcceptanceCallback"
    """Class to instantiate."""

    config: AcceptanceConfig = AcceptanceConfig()
    """Acceptance evaluation settings."""


@dataclass(frozen=True)
class PushConfig:
    """Settings for push perturbation during evaluation."""

    enabled: bool = False
    """Enable push perturbations."""

    force_range: tuple[float, float] = (50.0, 200.0)
    """Min and max force magnitude in Newtons."""

    duration_s: tuple[float, float] = (0.1, 0.3)
    """Min and max push duration in seconds."""

    interval_s: tuple[float, float] = (3.0, 8.0)
    """Min and max interval between pushes in seconds."""

    body_names: str = "torso_link,pelvis"
    """Comma-separated body names to push."""

    env_id: int = 0
    """Environment ID to apply pushes."""


@dataclass(frozen=True)
class PushCallbackConfig:
    """Instantiation config for EvalPushCallback."""

    _target_: str = "holosoma.agents.callbacks.push.EvalPushCallback"
    """Class to instantiate."""

    config: PushConfig = PushConfig()
    """Push perturbation settings."""


@dataclass(frozen=True)
class PayloadConfig:
    """Settings for wrist payload simulation during evaluation.

    Applies a constant downward force on wrist/elbow links to simulate
    the robot holding a payload (force = mass * 9.81).
    """

    enabled: bool = False
    """Enable wrist payload forces."""

    mass_kg: float = 1.0
    """Payload mass in kg. Force is split evenly across resolved bodies."""

    body_names: str = "left_wrist_yaw_link,right_wrist_yaw_link"
    """Comma-separated wrist body names."""

    env_id: int = 0
    """Environment ID to apply payload forces."""


@dataclass(frozen=True)
class PayloadCallbackConfig:
    """Instantiation config for EvalPayloadCallback."""

    _target_: str = "holosoma.agents.callbacks.payload.EvalPayloadCallback"
    """Class to instantiate."""

    config: PayloadConfig = PayloadConfig()
    """Payload simulation settings."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Runtime settings and outputs for one policy evaluation.

    Simulator visualization is intentionally not represented here. Isaac Lab 3
    owns that selection through ``--visualizer`` (for example
    ``--visualizer kit``).
    """

    num_envs: int = 1
    """Number of evaluation environments."""

    max_steps: int | None = None
    """Maximum policy steps. ``None`` runs until a callback stops evaluation."""

    max_episode_length_s: float = 100000.0
    """Evaluation episode horizon in seconds."""

    randomize_tiles: bool = False
    """Randomize terrain tiles during evaluation."""

    xy_offset_range: float = 0.0
    """Terrain spawn XY offset range in meters."""

    export_onnx: bool = False
    """Export an ONNX artifact into the evaluation output directory."""

    video: VideoConfig = field(default_factory=lambda: VideoConfig(enabled=False, upload_to_wandb=False))
    """Optional rendered video output. Disabled by default."""

    recording: RecordingCallbackConfig = RecordingCallbackConfig()
    """Trajectory recording callback."""

    acceptance: AcceptanceCallbackConfig = AcceptanceCallbackConfig()
    """Lightweight per-motion acceptance callback."""

    push: PushCallbackConfig = PushCallbackConfig()
    """Push perturbation callback."""

    payload: PayloadCallbackConfig = PayloadCallbackConfig()
    """Wrist payload simulation callback."""

    def collect_active_callbacks(self) -> dict:
        """Collect callback configs where config.enabled is True."""
        cb_configs = {}
        callback_names = ("recording", "acceptance", "push", "payload")
        for f in dataclasses.fields(self):
            if f.name not in callback_names:
                continue
            cfg = getattr(self, f.name)
            if not hasattr(cfg, "_target_"):
                raise ValueError(f"Callback config '{f.name}' missing _target_ field")
            if not hasattr(cfg.config, "enabled"):
                raise ValueError(f"Callback config '{f.name}' missing config.enabled field")
            if cfg.config.enabled:
                cb_configs[f.name] = cfg
        return cb_configs
