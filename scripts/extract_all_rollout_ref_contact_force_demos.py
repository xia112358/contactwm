#!/usr/bin/env python3
"""Run parallel eval recordings and extract rollout-ref contact-force demos."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
from somaforge_core import (
    NEWTON_ROLLOUT_KINEMATICS_BACKEND,
    decode_contact_force_provenance,
    decode_kinematics_provenance,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIMB_RE = re.compile(r"climb[_-](\d+)")

DEFAULT_BASE_MANIFEST = REPO_ROOT / "runtime/current/manifests/omniretarget_baseline_29.json"
DEFAULT_WORK_DIR = REPO_ROOT / "runtime/current/rollout/newton_contact_force"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/current/motions/newton_contact_force"
DEFAULT_MANIFEST_OUTPUT = REPO_ROOT / "runtime/current/manifests/newton_contact_force_8part.json"
DEFAULT_FAILURE_REPORT = REPO_ROOT / "runtime/current/rollout/newton_contact_force/failures.json"
DEFAULT_PARALLEL_ENVS = 16


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _resolve_manifest_path(path: Path, manifest_path: Path) -> Path:
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _climb_id(path_or_entry: str) -> int:
    match = CLIMB_RE.search(path_or_entry)
    if match is None:
        raise ValueError(f"Could not infer climb id from: {path_or_entry}")
    return int(match.group(1))


def _single_motion_manifest(
    base: dict[str, Any],
    base_manifest_path: Path,
    motion_entry: dict[str, Any],
    climb_id: int,
) -> dict[str, Any]:
    terrain_id = motion_entry["terrain_id"]
    terrains = [entry for entry in base["terrains"] if entry["terrain_id"] == terrain_id]
    if len(terrains) != 1:
        raise ValueError(
            f"Expected exactly one terrain for climb {climb_id} terrain_id={terrain_id}, got {len(terrains)}"
        )
    terrain = dict(terrains[0])
    terrain_file = Path(str(terrain["terrain_file"]))
    if not terrain_file.is_absolute():
        terrain["terrain_file"] = str((base_manifest_path.parent / terrain_file).resolve())
    motion = dict(motion_entry)
    motion["motion_file"] = str(_resolve_manifest_path(Path(str(motion["motion_file"])), base_manifest_path))
    if motion.get("source_file"):
        motion["source_file"] = str(_resolve_manifest_path(Path(str(motion["source_file"])), base_manifest_path))
    return {
        "schema": base.get("schema"),
        "schema_version": base.get("schema_version", 1),
        "description": f"Single-motion rollout-ref extraction manifest for climb {climb_id:02d}.",
        "robot_asset_id": base.get("robot_asset_id"),
        "robot_asset_sha256": base.get("robot_asset_sha256"),
        "robot_asset_bundle_sha256": base.get("robot_asset_bundle_sha256"),
        "robot_asset_usd_bundle_sha256": base.get("robot_asset_usd_bundle_sha256"),
        "kinematics_backend": base.get("kinematics_backend"),
        "terrains": [terrain],
        "motion_files": [motion],
    }


def _default_python() -> str:
    from somaforge_core.runtime_paths import newton_python
    return str(newton_python())

def _subprocess_env() -> dict[str, str]:
    from somaforge_core.runtime_paths import newton_subprocess_environment
    return newton_subprocess_environment()


def _run(cmd: list[str], dry_run: bool) -> bool:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, env=_subprocess_env())
    return result.returncode == 0


def _eval_command(
    python_executable: str,
    checkpoint: Path,
    single_manifest: Path,
    recording_path: Path,
    max_eval_steps: int,
    parallel_envs: int,
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "holosoma.eval_agent",
        "termination:g1-29dof-wbt",
        "--headless",
        "--device",
        "cuda:0",
        "--checkpoint",
        str(checkpoint),
        "--recording.config.enabled",
        "--recording.config.record-initial-state",
        "--recording.config.env-id",
        str(-1 if parallel_envs > 1 else 0),
        "--recording.config.output-path",
        str(recording_path),
        "--num-envs",
        str(parallel_envs),
        "--max-steps",
        str(max_eval_steps),
        "--export-onnx",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.motion-manifest",
        str(single_manifest),
        "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob",
        "1.0",
        "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob",
        "0.0",
        "--command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale",
        "0.0",
        "--terrain.terrain-term.motion-matched-manifest",
        str(single_manifest),
    ]
    return cmd


def _extract_command(python_executable: str, motion_path: Path, recording_path: Path, output_path: Path) -> list[str]:
    return [
        python_executable,
        str(REPO_ROOT / "scripts/extract_rollout_ref_contact_force_demo.py"),
        "--motion",
        str(motion_path),
        "--recording",
        str(recording_path),
        "--output",
        str(output_path),
    ]


def _build_output_manifest(
    base: dict[str, Any],
    base_manifest_path: Path,
    output_dir: Path,
    output_manifest: Path,
    allow_missing: bool,
) -> None:
    demo_by_id = {
        _climb_id(str(path)): path for path in sorted(output_dir.glob("climb_*_rollout_ref_contact_force.npz"))
    }
    motion_files = []
    missing = []
    for entry in base["motion_files"]:
        cid = _climb_id(str(entry["motion_file"]))
        demo = demo_by_id.get(cid)
        if demo is None:
            missing.append(cid)
            if allow_missing:
                continue
            continue
        with np.load(demo, allow_pickle=False) as data:
            contact_provenance = decode_contact_force_provenance(
                data["contact_force_provenance_json"] if "contact_force_provenance_json" in data else None,
                context=str(demo),
            )
            kinematics_provenance = decode_kinematics_provenance(
                data["kinematics_provenance_json"] if "kinematics_provenance_json" in data else None,
                context=str(demo),
            )
            source_recording = Path(str(np.asarray(data["rollout_ref_source_recording"]).item())).resolve()
        source_motion = _resolve_manifest_path(Path(str(entry["motion_file"])), base_manifest_path)
        motion_files.append(
            {
                "motion_id": entry.get("motion_id", cid),
                "motion_file": os.path.relpath(demo.resolve(), output_manifest.parent.resolve()),
                "terrain_id": entry["terrain_id"],
                "weight": entry.get("weight", 1.0),
                "source_file": os.path.relpath(source_recording, output_manifest.parent.resolve()),
                "source_sha256": sha256_file(source_recording),
                "reference_motion_file": os.path.relpath(source_motion, output_manifest.parent.resolve()),
                "reference_motion_sha256": sha256_file(source_motion),
                "motion_sha256": sha256_file(demo),
                "kinematics_schema": kinematics_provenance["schema"],
                "kinematics_backend": kinematics_provenance["kinematics_backend"],
                "velocity_derivation": kinematics_provenance["velocity_derivation"],
                "contact_force_schema": contact_provenance["schema"],
                "contact_force_backend": contact_provenance["source_backend"],
                "contact_solver_sha256": contact_provenance["solver_config_sha256"],
            }
        )
    if missing and not allow_missing:
        raise ValueError(f"Missing rollout-ref demos for climb ids: {sorted(missing)}")
    terrains = []
    for entry in base["terrains"]:
        terrain = dict(entry)
        terrain_path = _resolve_manifest_path(Path(str(terrain["terrain_file"])), base_manifest_path)
        terrain["terrain_file"] = os.path.relpath(terrain_path, output_manifest.parent.resolve())
        terrains.append(terrain)
    _write_json(
        output_manifest,
        {
            "schema": "somaforge_motion_terrain_manifest_v1",
            "schema_version": base.get("schema_version", 1),
            "description": "Canonical 8-part contact-force manifest generated from successful Newton policy rollouts.",
            "source_kind": "newton_policy_rollout_contact_force_8part",
            "contact_force_schema": "somaforge_contact_force_8part_v1",
            "contact_force_backend": "isaaclab3_newton_mjwarp",
            "collision_pipeline": "newton",
            "kinematics_backend": NEWTON_ROLLOUT_KINEMATICS_BACKEND,
            "robot_asset_id": base.get("robot_asset_id", "robot.g1.spherehand"),
            "robot_asset": base.get("robot_asset", "g1_29dof_spherehand"),
            "robot_asset_sha256": base.get("robot_asset_sha256"),
            "robot_asset_bundle_sha256": base.get("robot_asset_bundle_sha256"),
            "robot_asset_usd_bundle_sha256": base.get("robot_asset_usd_bundle_sha256"),
            "terrains": terrains,
            "motion_files": motion_files,
        },
    )
    print(f"Wrote {output_manifest} motions={len(motion_files)} terrains={len(base['terrains'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Current canonical baseline WBT checkpoint; not required with --extract-only.",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--failure-report", type=Path, default=DEFAULT_FAILURE_REPORT)
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--motion-id", type=int, action="append", help="Only process selected climb id; repeatable.")
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=DEFAULT_PARALLEL_ENVS,
        help=(
            "Run this many eval envs per attempt and extract the first complete rollout. "
            "The standard extraction flow records all envs with env_id=-1; pass 1 only for single-env debugging."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1.")
    if args.parallel_envs < 1:
        raise ValueError("--parallel-envs must be at least 1.")
    if args.checkpoint is None and not args.extract_only:
        raise ValueError("--checkpoint is required unless --extract-only is used.")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    if checkpoint is not None and not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(f"Missing baseline checkpoint: {checkpoint}")

    base_manifest_path = args.base_manifest.expanduser().resolve()
    base = load_motion_terrain_manifest(str(base_manifest_path))
    selected = set(args.motion_id or [])
    manifest_dir = args.work_dir / "manifests"
    recording_dir = args.work_dir / "recordings"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recording_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []

    for motion_entry in base["motion_files"]:
        motion_path = Path(motion_entry["motion_file"])
        climb_id = _climb_id(str(motion_path))
        if selected and climb_id not in selected:
            continue
        motion_npz_path = _resolve_manifest_path(motion_path, base_manifest_path)
        with np.load(motion_npz_path) as motion_npz:
            motion_frame_count = int(motion_npz["joint_pos"].shape[0])
        eval_steps = int(args.max_eval_steps) if args.max_eval_steps is not None else max(motion_frame_count - 1, 1)

        single_manifest = manifest_dir / f"climb_{climb_id:02d}_manifest.json"
        output_path = args.output_dir / f"climb_{climb_id:02d}_rollout_ref_contact_force.npz"
        _write_json(single_manifest, _single_motion_manifest(base, base_manifest_path, motion_entry, climb_id))

        if args.skip_existing and output_path.exists():
            print(f"Skip existing {output_path}")
            successes.append({"climb_id": climb_id, "attempt": 0, "output": str(output_path)})
            continue

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to replace existing rollout output: {output_path}. "
                "Move it to the system trash or pass --skip-existing."
            )

        attempt_errors = []
        success = False
        for attempt in range(1, args.attempts + 1):
            recording_path = recording_dir / f"climb_{climb_id:02d}_attempt_{attempt:02d}_eval_recording.npz"
            attempt_output_path = args.work_dir / "extracted" / f"climb_{climb_id:02d}_attempt_{attempt:02d}.npz"
            if recording_path.exists() and not args.extract_only:
                raise FileExistsError(
                    f"Refusing to replace existing eval recording: {recording_path}. "
                    "Move it to the system trash before rerunning this attempt."
                )
            if attempt_output_path.exists():
                raise FileExistsError(
                    f"Refusing to replace existing extraction attempt: {attempt_output_path}. "
                    "Move it to the system trash before rerunning this attempt."
                )

            print(f"=== climb_{climb_id:02d} attempt {attempt}/{args.attempts} ===", flush=True)
            if not args.extract_only:
                assert checkpoint is not None
                eval_ok = _run(
                    _eval_command(
                        args.python, checkpoint, single_manifest, recording_path, eval_steps, args.parallel_envs
                    ),
                    args.dry_run,
                )
                if not eval_ok:
                    attempt_errors.append({"attempt": attempt, "stage": "eval", "error": "eval command failed"})
                    continue
                if not args.dry_run and not recording_path.exists():
                    attempt_errors.append({"attempt": attempt, "stage": "eval", "error": "recording was not produced"})
                    continue
            elif not recording_path.exists():
                attempt_errors.append(
                    {"attempt": attempt, "stage": "extract", "error": "existing recording was not found"}
                )
                continue

            if args.record_only:
                success = True
                successes.append(
                    {
                        "climb_id": climb_id,
                        "attempt": attempt,
                        "checkpoint": str(checkpoint) if checkpoint is not None else None,
                        "recording": str(recording_path),
                    }
                )
                break

            extract_ok = _run(
                _extract_command(args.python, motion_npz_path, recording_path, attempt_output_path),
                args.dry_run,
            )
            if extract_ok and (args.dry_run or attempt_output_path.exists()):
                if not args.dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(attempt_output_path, output_path)
                success = True
                successes.append(
                    {
                        "climb_id": climb_id,
                        "attempt": attempt,
                        "checkpoint": str(checkpoint) if checkpoint is not None else None,
                        "recording": str(recording_path),
                        "output": str(output_path),
                    }
                )
                break
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "stage": "extract",
                    "error": "recording was not a complete successful rollout",
                }
            )

        if not success:
            failures.append(
                {
                    "climb_id": climb_id,
                    "motion_file": str(motion_npz_path),
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "attempts": args.attempts,
                    "errors": attempt_errors,
                }
            )
            print(f"FAILED climb_{climb_id:02d} after {args.attempts} attempts", flush=True)

    if not selected and not args.record_only:
        _build_output_manifest(
            base,
            base_manifest_path,
            args.output_dir,
            args.output_manifest,
            allow_missing=args.allow_missing,
        )
        _write_json(
            args.failure_report,
            {
                "successes": successes,
                "failures": failures,
            },
        )
        print(f"Wrote {args.failure_report} successes={len(successes)} failures={len(failures)}")


if __name__ == "__main__":
    main()
