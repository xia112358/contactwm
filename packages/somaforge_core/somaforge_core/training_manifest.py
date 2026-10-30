"""Shared manifest contract for Predictor and WBT training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

TRAINING_MANIFEST_SCHEMA = "somaforge_training_pipeline_v2"
STAGES = ("wbt_baseline", "predictor", "holosoma_wbt")


def default_training_manifest_path() -> Path:
    from somaforge_core.robot_assets import somaforge_root

    return somaforge_root() / "configs" / "training_pipeline_manifest.json"


def load_training_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path else default_training_manifest_path()
    with manifest_path.expanduser().resolve().open(encoding="utf-8") as stream:
        payload = json.load(stream)
    validate_training_manifest(payload, context=str(manifest_path))
    payload["_path"] = str(manifest_path.expanduser().resolve())
    return payload


def validate_training_manifest(payload: Mapping[str, Any], *, context: str = "training manifest") -> None:
    if payload.get("schema") != TRAINING_MANIFEST_SCHEMA:
        raise ValueError(f"{context} has unsupported schema: {payload.get('schema')!r}")
    if payload.get("asset_manifest") != "configs/assets_manifest.json":
        raise ValueError(f"{context} must use configs/assets_manifest.json")
    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError(f"{context} has no stages mapping")
    missing = [stage for stage in STAGES if stage not in stages]
    if missing:
        raise ValueError(f"{context} is missing stages: {', '.join(missing)}")
    for stage in STAGES:
        spec = stages[stage]
        if not isinstance(spec, Mapping):
            raise ValueError(f"{context} stage {stage} must be an object")
        for key in ("inputs", "outputs"):
            if not isinstance(spec.get(key), list) or not spec[key]:
                raise ValueError(f"{context} stage {stage} requires non-empty {key}")


def stage_spec(stage: str, path: str | Path | None = None) -> Mapping[str, Any]:
    if stage not in STAGES:
        raise KeyError(f"unknown training stage: {stage}")
    return load_training_manifest(path)["stages"][stage]
