"""One balanced policy batch, lossless recording, then existing consensus fitting.

Raw executions and solver evidence remain distinct from fitted kinematics.
The controller never trains, stops training, or runs a preliminary evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from . import trajectory_refinement as workflow
from . import extraction_standard as standard


def batch_command(plan, checkpoint, recording, *, envs_per_motion=standard.ENVS_PER_MOTION, chunk_frames=standard.CHUNK_FRAMES,
                  device="cuda:0", visualizer=None, headless=True):
    if envs_per_motion < 1 or chunk_frames < 1:
        raise ValueError("Positive environment and chunk budgets required")
    cfg = plan["config"]["collection"]
    command = workflow.recording_command(plan, plan["cases"][0], checkpoint, recording,
        seed=cfg["seed"], noise=cfg["initial_pose_noise"], device=device,
        visualizer=visualizer, headless=headless)
    for option, value in (("--num-envs", len(plan["cases"])*envs_per_motion),
                          ("--recording.config.env-id", -1),
                          ("--max-steps", max(c["frames"] for c in plan["cases"])+10)):
        command[command.index(option)+1] = str(value)
    single = str(Path(plan["output_dir"])/"references"/f"{plan['cases'][0]['key']}.json")
    manifest = str(Path(plan["output_dir"])/"references.json")
    command = [manifest if item == single else item for item in command]
    prefix = "--command.setup-terms.motion-command.params.motion-config."
    command += [prefix+"fixed-motion-assignment-in-eval", "True",
                prefix+"use-completion-learning-sampler", "False",
                "--recording.config.chunk-frames", str(chunk_frames)]
    return command


def collect(plan, checkpoint, *, execute=False, output=None, envs_per_motion=None,
            chunk_frames=None, aggregate=True, **launcher):
    cfg = plan["config"]["collection"]
    repeats = envs_per_motion or cfg.get("envs_per_motion", standard.ENVS_PER_MOTION)
    chunk = chunk_frames or cfg.get("chunk_frames", standard.CHUNK_FRAMES)
    checkpoint = Path(checkpoint).resolve()
    policy, _ = workflow.policy_run(plan, checkpoint)
    root = Path(output).resolve() if output else Path(plan["output_dir"])/"collections"/policy["sha256"]
    command = batch_command(plan, checkpoint, root/"recording.npz",
                            envs_per_motion=repeats, chunk_frames=chunk, **launcher)
    if not execute:
        return dict(command=command, num_envs=len(plan["cases"])*repeats,
                    envs_per_motion=repeats, output=str(root), preliminary_evaluation=False)
    root.mkdir(parents=True, exist_ok=True)
    files = [workflow.fingerprint(plan["output_dir"]+"/references.json")]
    for case in plan["cases"]:
        files += [dict(path=case["reference"]["motion_file"], sha256=case["reference"]["motion_sha256"]),
                  dict(path=case["terrain"]["terrain_file"], sha256=case["terrain"]["terrain_sha256"])]
    implementation = [Path(__file__), Path(workflow.__file__),
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/checkpoint_inference.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/rollout_consensus.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/rollout_reaggregation.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/native_rollout_consensus.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/observed_event_consensus.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/population_event_clock.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/motion_signal.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/consensus_event_targets.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/consensus_event_alignment.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/consensus_continuity.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/event_continuity.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/staged_consensus.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/contact_order_acceptance.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/contact_aggregation.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/contact_face_selection.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/newton_support.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/newton_contacts.py",
        workflow.somaforge_root()/"packages/somaforge_core/somaforge_core/contact_motion.py",
        workflow.somaforge_root()/"src/holosoma/holosoma/config_types/command.py",
        workflow.somaforge_root()/"src/holosoma/holosoma/config_types/eval_callback.py",
        workflow.somaforge_root()/"src/holosoma/holosoma/agents/callbacks/recording.py",
        workflow.somaforge_root()/"src/holosoma/holosoma/managers/command/terms/wbt.py"]
    implementation = list(dict.fromkeys(implementation+standard.implementation_paths()))
    receipt = dict(extraction_standard=standard.definition(plan["config"].get("aggregation")), schema="somaforge_parallel_rollout_collection_v1", checkpoint=policy,
        training_plan=workflow.fingerprint(Path(plan["output_dir"])/"plan.json"),
        reference_count=len(plan["cases"]), envs_per_motion=repeats,
        num_envs=len(plan["cases"])*repeats, chunk_frames=chunk,
        assignment="environment_id % reference_count; fixed across resets",
        inputs=files, implementation=[workflow.fingerprint(p) for p in implementation],
        preliminary_evaluation=False, aggregation=standard.fit_options(plan["config"].get("aggregation")))
    workflow.verify_files(files)
    workflow.write_json(root/"collection_plan.json", receipt, immutable=True)
    inference = workflow.prepare_inference_policy(checkpoint, policy, root/"policy")
    recording = root/"recording.npz.json"
    if not recording.exists() or not workflow.read_json(recording)["complete"]:
        if recording.exists():
            raise ValueError("Incomplete recording retained; use a new output for a new policy batch")
        command = batch_command(plan, inference["path"], root/"recording.npz",
            envs_per_motion=repeats, chunk_frames=chunk, **launcher)
        environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        workflow.run_command(command, root/"capture"/uuid.uuid4().hex, environment=environment)
    workflow.verify_files(receipt["inputs"]+receipt["implementation"]+[inference])
    extract_batch(plan, recording, root, repeats=repeats)
    if aggregate:
        from .rollout_consensus import aggregate_collection
        aggregate_collection(plan, root, device=launcher.get("device", "cuda:0"))
    return workflow.read_json(root/"collection_summary.json")


def save_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError(f"Existing extracted chunk differs: {path}")
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def extract_batch(plan, recording, root, *, repeats):
    """Read each global chunk once; retain native clocks and original row IDs."""
    from .newton_contact_data import require_current_newton_manifest
    from .newton_contacts import constraint_decision
    from .robot_assets import validate_g1_asset_metadata
    source = workflow.read_json(recording)
    if source["schema"] != "somaforge_chunked_recording_v1" or not source["complete"]:
        raise ValueError("A complete chunk recording is required")
    meta = source["metadata"]
    require_current_newton_manifest(meta, context=str(recording))
    validate_g1_asset_metadata(meta["robot_asset"], context=str(recording))
    count = len(plan["cases"])
    if meta["env_id"] != -1 or meta["num_envs"] != count*repeats:
        raise ValueError("Recording does not cover the complete balanced batch")
    body_env = {int(k): int(v) for k, v in meta["newton_scene_binding"]["body_env"].items()}
    total_frames = 0
    for number, chunk in enumerate(source["chunks"]):
        workflow.verify_files([chunk])
        if chunk["start"] != total_frames:
            raise ValueError("Missing or reordered native recording chunk")
        total_frames = chunk["end"]
        with np.load(chunk["path"], allow_pickle=False) as z:
            arrays = {k: z[k] for k in ("root_pos", "root_quat_xyzw", "dof_pos", "body_pos_w",
                "body_quat_xyzw", "env_origins", "motion_id", "motion_time_step", "motion_start_step",
                "terminated", "timeout")}
            # NPZ members decompress on every access; read each contact channel
            # once per chunk rather than once per frame.
            z = {key: z[key] for key in z.files if key.startswith("solver_contact_")}
            if not np.all(arrays["motion_id"] == np.arange(meta["num_envs"])[None] % count):
                raise ValueError("Motion assignment changed; no reference may be silently omitted")
            pair_columns = {k: [] for k in ("frame", "env", "part", "surface", "eligible", "point", "shape", "row")}
            for frame, n_raw in enumerate(z["solver_contact_count"]):
                n = int(n_raw)
                active, allocated = constraint_decision(z["solver_contact_dist"][frame],
                    z["solver_contact_includemargin"][frame], z["solver_contact_type"][frame],
                    z["solver_contact_efc_address"][frame], n, z["solver_contact_constraint_rows"][frame])
                if np.any(active[:n] & ~allocated[:n]):
                    raise ValueError("Unallocated active solver constraint; raw evidence retained")
                group = z["solver_contact_task_group"][frame, :n]
                selected = np.flatnonzero(active[:n] & (group >= 0))
                envs = group[selected] // 6
                b0 = z["solver_contact_body0"][frame, selected]
                b1 = z["solver_contact_body1"][frame, selected]
                side0 = np.array([int(b) in body_env for b in b0])
                side1 = np.array([int(b) in body_env for b in b1])
                if np.any(side0 == side1):
                    raise ValueError("Task contact includes self/cross-robot collision")
                shapes = np.where(side0, z["solver_contact_shape0"][frame, selected],
                                  z["solver_contact_shape1"][frame, selected])
                point = z["solver_contact_position_w"][frame, selected] - arrays["env_origins"][frame, envs]
                values = (np.full(len(selected), frame), envs, group[selected] % 6,
                    z["solver_contact_task_surface"][frame, selected],
                    z["solver_contact_task_eligible"][frame, selected], point, shapes, selected)
                for key, value in zip(pair_columns, values):
                    pair_columns[key].append(value)
            pairs = {k: np.concatenate(v) for k, v in pair_columns.items()}
            for index, case in enumerate(plan["cases"]):
                env_ids = np.arange(index, meta["num_envs"], count)
                subset = {k: v[:, env_ids] for k, v in arrays.items()}
                subset["root_pos"] = subset["root_pos"] - subset["env_origins"]
                subset["body_pos_w"] = subset["body_pos_w"] - subset["env_origins"][:, :, None]
                choose = pairs["env"] % count == index
                subset.update({"pair_"+k: v[choose] for k, v in pairs.items()})
                subset["env_ids"] = env_ids
                subset["recording_start"] = np.asarray(chunk["start"])
                save_arrays(root/"candidates"/case["key"]/f"chunk_{number:05d}.npz", subset)
        print(f"Extracted chunk {number+1}/{len(source['chunks'])}, frames={total_frames}", flush=True)
    workflow.write_json(root/"extraction.json", dict(recording=workflow.fingerprint(recording),
        chunks=source["chunks"], frames=total_frames,
        transform="subtract recorded terrain origin from root/body/contact positions; orientations unchanged",
        raw_evidence="complete original recording chunks retained; solver row IDs are chunk-local"), immutable=True)


def load_candidates(directory, *, fields=None):
    paths = sorted(Path(directory).glob("chunk_*.npz"))
    if not paths:
        raise ValueError("No extracted candidate chunks")
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            names = z.files if fields is None else set(fields) | {'env_ids', 'recording_start'}
            if set(names)-set(z.files):
                raise ValueError(f'Missing candidate fields: {sorted(set(names)-set(z.files))}')
            parts.append({k: z[k] for k in names})
    output = {}
    for key in parts[0]:
        if key in ("env_ids", "recording_start"):
            continue
        values = [p[key] + p["recording_start"] if key == "pair_frame" else p[key] for p in parts]
        output[key] = np.concatenate(values)
    output["env_ids"] = parts[0]["env_ids"]
    return output
