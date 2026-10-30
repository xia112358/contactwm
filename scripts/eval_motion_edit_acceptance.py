#!/usr/bin/env python3
"""Run the unified motion-edit acceptance eval.

The standard is one eval environment per manifest motion, starting from frame 0.
Training samplers are disabled so hotspot/group-probe state cannot affect eval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "runtime/current/manifests/motion_edit_ref_v1.json"
DEFAULT_PROJECT = "MotionEditFinetune"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / DEFAULT_PROJECT / "eval_csv"
STANDARD_ID = "motion_edit_full_start0_v1"
STANDARD_REPEATS = 3
MOTION_EDIT_ROBOT_CONFIG = "robot:g1-29dof"


def _load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("motion_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no motion_files: {path}")
    return entries


def _motion_length(path: str) -> int:
    with np.load(path) as data:
        if "body_pos_w" in data:
            return int(data["body_pos_w"].shape[0])
        if "joint_pos" in data:
            return int(data["joint_pos"].shape[0])
    raise ValueError(f"Cannot infer motion length from {path}")


def _manifest_stats(path: Path) -> tuple[int, int]:
    entries = _load_manifest(path)
    lengths = [_motion_length(str(entry["motion_file"])) for entry in entries]
    return len(entries), max(lengths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_if_exists(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def _git_metadata() -> dict[str, object]:
    status = _git_output(["status", "--short"])
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "branch": _git_output(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    src_path = str(REPO_ROOT / "src/holosoma")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _build_command(args: argparse.Namespace, motion_count: int, max_motion_len: int) -> list[str]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.name}_acceptance.csv"
    summary_path = output_dir / f"{args.name}_acceptance_summary.json"
    fail_path = output_dir / f"{args.name}_fail.csv"
    max_eval_steps = args.max_eval_steps if args.max_eval_steps is not None else max_motion_len + args.eval_step_margin

    cmd = [
        args.python,
        "-m",
        "holosoma.eval_agent",
        "termination:g1-29dof-wbt",
        MOTION_EDIT_ROBOT_CONFIG,
        "--headless",
        "--device",
        args.device,
        "--checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--training.project",
        args.project,
        "--acceptance.config.enabled",
        "True",
        "--acceptance.config.output-path",
        str(csv_path),
        "--acceptance.config.summary-path",
        str(summary_path),
        "--acceptance.config.fail-output-path",
        str(fail_path),
        "--acceptance.config.repeats",
        str(args.repeats),
        "--num-envs",
        str(motion_count),
        "--max-steps",
        str(max_eval_steps),
        "--export-onnx",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.motion-manifest",
        str(args.motion_manifest.expanduser().resolve()),
        "--command.setup-terms.motion-command.params.motion-config.canonicalize-motion-order-on-load",
        "True",
        "--command.setup-terms.motion-command.params.motion-config.reset-sampler",
        "uniform",
        "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob",
        "1.0",
        "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob",
        "0.0",
        "--command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale",
        "0.0",
        "--command.setup-terms.motion-command.params.motion-config.use-start-probe-envs",
        "True",
        "--command.setup-terms.motion-command.params.motion-config.probe-env-per-motion",
        "1",
        "--command.setup-terms.motion-command.params.motion-config.use-group-probe-envs",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.chain-motion-segments",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.hold-at-motion-end-in-eval",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.local-motion-segment-reference",
        "False",
        "--terrain.terrain-term.motion-matched-manifest",
        str(args.motion_manifest.expanduser().resolve()),
        "--video.enabled",
        "False",
    ]
    if args.bad_motion_body_pos_threshold is not None:
        cmd.extend(
            [
                "--termination.terms.bad-tracking.params.bad-motion-body-pos-threshold",
                str(args.bad_motion_body_pos_threshold),
            ]
        )
    return cmd


def _is_standard_manifest(path: Path) -> bool:
    return path.expanduser().resolve() == DEFAULT_MANIFEST.resolve()


def _validate_standard_args(args: argparse.Namespace) -> None:
    if int(args.repeats) <= 0:
        raise SystemExit("--repeats must be positive")

    diagnostic_reasons = []
    if not _is_standard_manifest(args.motion_manifest):
        diagnostic_reasons.append("motion manifest is not the full standard manifest")
    if args.max_eval_steps is not None:
        diagnostic_reasons.append("--max-eval-steps overrides the standard max length + margin")
    if int(args.repeats) != STANDARD_REPEATS:
        diagnostic_reasons.append(f"--repeats must be {STANDARD_REPEATS} for the formal standard")

    if diagnostic_reasons and not args.allow_diagnostic:
        reason_text = "; ".join(diagnostic_reasons)
        raise SystemExit(
            "Refusing non-standard motion-edit acceptance eval. "
            f"{reason_text}. Use --allow-diagnostic for temporary diagnostics; "
            "diagnostic outputs are not comparable with formal results."
        )


def _summary_path_for_acceptance_csv(path: Path) -> Path:
    return path.with_name(f"{path.stem}_summary.json")


def _validate_formal_acceptance_csv(path: Path, *, allow_diagnostic: bool, role: str) -> None:
    path = path.expanduser().resolve()
    if "legacy_unstandardized" in path.parts and not allow_diagnostic:
        raise SystemExit(
            f"Refusing legacy unstandardized {role}: {path}. Use --allow-diagnostic only for temporary diagnostics."
        )
    summary_path = _summary_path_for_acceptance_csv(path)
    if allow_diagnostic:
        return
    if not summary_path.exists():
        raise SystemExit(
            f"Refusing {role} without formal summary metadata: {path}. "
            f"Expected {summary_path} with eval_standard.formal=true."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Refusing {role} with unreadable summary metadata: {summary_path}: {exc}") from exc
    standard = summary.get("eval_standard")
    if (
        not isinstance(standard, dict)
        or standard.get("standard_id") != STANDARD_ID
        or standard.get("formal") is not True
    ):
        raise SystemExit(
            f"Refusing non-standard {role}: {path}. "
            f"Expected eval_standard.standard_id={STANDARD_ID!r} and formal=true in {summary_path}."
        )


def _standard_metadata(
    args: argparse.Namespace,
    motion_count: int,
    max_motion_len: int,
    command: list[str],
) -> dict[str, object]:
    manifest_path = args.motion_manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    max_eval_steps = args.max_eval_steps if args.max_eval_steps is not None else max_motion_len + args.eval_step_margin
    diagnostic_reasons = []
    if not _is_standard_manifest(args.motion_manifest):
        diagnostic_reasons.append("non_standard_manifest")
    if args.max_eval_steps is not None:
        diagnostic_reasons.append("manual_max_eval_steps")
    if int(args.repeats) != STANDARD_REPEATS:
        diagnostic_reasons.append("non_standard_repeats")

    return {
        "standard_id": STANDARD_ID,
        "formal": not diagnostic_reasons,
        "diagnostic_only": bool(diagnostic_reasons),
        "diagnostic_reasons": diagnostic_reasons,
        "motion_manifest": str(manifest_path),
        "motion_manifest_sha256": _sha256(manifest_path),
        "standard_manifest": str(DEFAULT_MANIFEST.resolve()),
        "standard_manifest_sha256": _sha256_if_exists(DEFAULT_MANIFEST.resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "motion_count": int(motion_count),
        "max_motion_len": int(max_motion_len),
        "max_eval_steps": int(max_eval_steps),
        "eval_step_margin": int(args.eval_step_margin),
        "repeats": int(args.repeats),
        "num_envs": int(motion_count),
        "start_at_timestep_zero_prob": 1.0,
        "freeze_at_timestep_zero_prob": 0.0,
        "initial_pose_noise_scale": 0.0,
        "use_start_probe_envs": True,
        "probe_env_per_motion": 1,
        "use_group_probe_envs": False,
        "chain_motion_segments": False,
        "hold_at_motion_end_in_eval": False,
        "local_motion_segment_reference": False,
        "terrain_randomize_tiles": False,
        "terrain_xy_offset_range": 0.0,
        "canonicalize_motion_order_on_load": True,
        "device": args.device,
        "project": args.project,
        "command": command,
        "command_string": " ".join(shlex.quote(part) for part in command),
        "git": _git_metadata(),
    }


def _write_standard_metadata(output_dir: Path, name: str, metadata: dict[str, object]) -> None:
    path = output_dir / f"{name}_eval_standard.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _augment_summary(summary_path: Path, metadata: dict[str, object]) -> None:
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["eval_standard"] = metadata
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _augment_summaries(args: argparse.Namespace, metadata: dict[str, object]) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = output_dir / f"{args.name}_acceptance_summary.json"
    if args.repeats <= 1:
        _augment_summary(summary_path, metadata)
        return

    for rep_index in range(1, args.repeats + 1):
        rep_summary = _with_repeat_suffix(summary_path, rep_index)
        rep_metadata = dict(metadata)
        rep_metadata["repeat_index"] = rep_index
        _augment_summary(rep_summary, rep_metadata)


def _with_repeat_suffix(path: Path, rep_index: int) -> Path:
    stem = path.stem
    suffix = path.suffix
    replacements = (
        ("_acceptance_summary", f"_rep{rep_index}_acceptance_summary"),
        ("_acceptance", f"_rep{rep_index}_acceptance"),
        ("_fail", f"_rep{rep_index}_fail"),
    )
    for old, new in replacements:
        if stem.endswith(old):
            return path.with_name(f"{stem[: -len(old)]}{new}{suffix}")
    return path.with_name(f"{stem}_rep{rep_index}{suffix}")


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["motion_file"]: row for row in rows}


def _compare(new_csv: Path, baseline_csv: Path, output_dir: Path, name: str) -> None:
    new_rows = _read_rows(new_csv)
    baseline_rows = _read_rows(baseline_csv)
    transitions: Counter[str] = Counter()
    compare_rows: list[dict[str, str]] = []
    for motion_file, new_row in new_rows.items():
        baseline_row = baseline_rows.get(motion_file)
        if baseline_row is None:
            transitions["missing_baseline"] += 1
            continue
        transitions[f"old_{baseline_row['status']}_new_{new_row['status']}"] += 1
        if baseline_row["status"] != new_row["status"]:
            compare_rows.append(
                {
                    "motion_file": motion_file,
                    "old_status": baseline_row["status"],
                    "new_status": new_row["status"],
                    "old_progress": baseline_row["progress"],
                    "new_progress": new_row["progress"],
                    "old_terms": baseline_row["terms"],
                    "new_terms": new_row["terms"],
                }
            )

    compare_csv = output_dir / f"{name}_compare.csv"
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["motion_file", "old_status", "new_status", "old_progress", "new_progress", "old_terms", "new_terms"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compare_rows)
    compare_summary = output_dir / f"{name}_compare_summary.json"
    compare_summary.write_text(json.dumps({"transitions": dict(transitions), "csv": str(compare_csv)}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--motion-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name", default="motion_edit_acceptance")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Local log project under logs/.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bad-motion-body-pos-threshold", type=float, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--eval-step-margin", type=int, default=8)
    parser.add_argument(
        "--repeats",
        type=int,
        default=STANDARD_REPEATS,
        help="Formal standard repeats. Use --allow-diagnostic to run a different value.",
    )
    parser.add_argument("--compare-against", type=Path, default=None)
    parser.add_argument(
        "--allow-diagnostic",
        action="store_true",
        help="Allow subset manifests or non-standard repeat/max-step settings. Outputs are marked diagnostic_only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _validate_standard_args(args)
    if args.compare_against is not None:
        _validate_formal_acceptance_csv(
            args.compare_against,
            allow_diagnostic=args.allow_diagnostic,
            role="--compare-against CSV",
        )
    motion_count, max_motion_len = _manifest_stats(args.motion_manifest.expanduser().resolve())
    cmd = _build_command(args, motion_count, max_motion_len)
    output_dir = args.output_dir.expanduser().resolve()
    metadata = _standard_metadata(args, motion_count, max_motion_len, cmd)
    print("Command:")
    print(" ".join(cmd), flush=True)
    print(f"Eval standard: {STANDARD_ID} formal={metadata['formal']}", flush=True)
    if args.dry_run:
        return

    _write_standard_metadata(output_dir, args.name, metadata)
    subprocess.run(cmd, cwd=REPO_ROOT, env=_subprocess_env(), check=True)
    _augment_summaries(args, metadata)

    acceptance_csv = output_dir / f"{args.name}_acceptance.csv"
    if args.compare_against is not None:
        if args.repeats <= 1:
            _compare(acceptance_csv, args.compare_against.expanduser().resolve(), output_dir, args.name)
        else:
            for rep_index in range(1, args.repeats + 1):
                rep_csv = _with_repeat_suffix(acceptance_csv, rep_index)
                _compare(
                    rep_csv,
                    args.compare_against.expanduser().resolve(),
                    output_dir,
                    f"{args.name}_rep{rep_index}",
                )


if __name__ == "__main__":
    main()
