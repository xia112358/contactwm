"""CPU extraction of unchanged Newton executions and same-solve observations.

No trajectory averaging, interpolation, geometric contact thresholds or physics
queries. All active pairs remain in the original content-addressed recording.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .asset_registry import sha256_file
from .contact_face_selection import process_face_contacts
from .motion_schema import G1_29DOF_JOINT_ORDER, newton_rollout_kinematics_provenance
from .newton_contact_data import require_current_newton_manifest
from .newton_contacts import FIELDS, PARTS, SCHEMA as CONTACT_SCHEMA
from .newton_support import SUPPORT_FIELDS, SUPPORT_SCHEMA, observed_support, select_support_contacts
from .robot_assets import encode_robot_asset_json, validate_g1_asset_metadata

OBSERVATION_SCHEMA = "somaforge_executed_contact_observations_v1"
STATE_FIELDS = ("root_pos", "root_quat_xyzw", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel",
                "body_pos_w", "body_quat_xyzw", "body_lin_vel_w", "body_ang_vel_w")


def complete_segment(steps, terminated, timeout, frames):
    """Only the first episode; tolerate the recorder's duplicate reset bootstrap.

    A later successful episode must not hide a failed first trial. The returned
    indices are contiguous original samples, never stitched/resampled states.
    """
    steps, terminated, timeout = map(np.asarray, (steps, terminated, timeout))
    if steps.ndim != 1 or terminated.shape != steps.shape or timeout.shape != steps.shape:
        raise ValueError("Expected one recorded environment with matching clock/done channels")
    start = 1 if len(steps) > 1 and steps[0] == steps[1] == 0 else 0
    end = start + frames
    if (len(steps) < end or not np.array_equal(steps[start:end], np.arange(frames))
            or terminated[:end].any() or timeout[:end-1].any()):
        return None
    return np.arange(start, end)


def map_surfaces(expected, actual):
    """Match explicit scene planes, not contacts, with numerical roundoff tolerance."""
    mapping = {}
    for observed in actual:
        matches = [int(target["surface"]) for target in expected
                   if np.allclose(observed["normal_w"], target["normal_w"], atol=1e-8, rtol=0)
                   and np.isclose(observed["plane_offset"], target["plane_offset"], atol=1e-8, rtol=0)]
        if len(matches) > 1:
            raise ValueError("Ambiguous intent-to-actual surface mapping")
        if matches:
            mapping[int(observed["surface"])] = matches[0]
    return mapping


def task_check(case, contacts, actual_catalog):
    if case["task"]["kind"] == "completion_only":
        return dict(passed=None, status="not_requested", scope="complete execution only; task intent unverified")
    from .contact_order_acceptance import audit_contact_order
    mapping = map_surfaces(case["expected_surface_catalog"], actual_catalog)
    required = {r["surface"] for a in case["contract"]["actions"] for r in a["requirements"] if r["surface"] >= 0}
    if not required.issubset(mapping.values()):
        raise ValueError("An intended task surface is missing from the actual scene")
    surface = np.asarray(contacts["contact_surface"])
    mapped = np.full(surface.shape, -1, dtype=np.int64)
    for old, new in mapping.items():
        mapped[surface == old] = new
    mask = np.asarray(contacts["contact_part_mask"], bool)
    if np.any(mask & (mapped < 0)):
        raise ValueError("An observed contact has no explicit task surface mapping")
    report = audit_contact_order(case["contract"], mask, mapped)
    if report["check_count"] <= 0:
        raise ValueError("An empty contact-event contract cannot certify task acceptance")
    report["surface_mapping"] = mapping
    report["contact_truth"] = "actual activation/allocation; loads/slip are separate observations"
    return report


def read_execution(recording, case):
    with np.load(recording, allow_pickle=False) as z:
        meta = json.loads(z["_metadata_json"].item())
        require_current_newton_manifest(meta, context=str(recording))
        validate_g1_asset_metadata(meta.get("robot_asset"), context=str(recording))
        if int(meta.get("env_id", -1)) != 0 or int(meta["num_envs"]) != 1:
            raise ValueError("This adapter records one motion/environment per trial")
        if meta.get("solver_contact_semantics", {}).get("schema") != CONTACT_SCHEMA:
            raise ValueError("Missing solver activation/allocation semantics")
        if meta.get("solver_support_semantics", {}).get("schema") != SUPPORT_SCHEMA:
            raise ValueError("Missing same-solve force and velocity semantics")
        if tuple(meta["dof_names"]) != G1_29DOF_JOINT_ORDER:
            raise ValueError("Noncanonical recorded joint order")
        if not np.isclose(1 / meta["dt"], case["fps"], rtol=0, atol=1e-6):
            raise ValueError("Recording and reference clocks differ; resampling is not allowed")
        indices = complete_segment(z["motion_time_step"], z["terminated"], z["timeout"], case["frames"])
        if indices is None:
            return None, dict(status="failed", execution_passed=False, task_passed=None,
                              reason="First episode does not contain a complete native frame-zero execution")
        if meta.get("episode_done") and not meta.get("done_is_timeout"):
            return None, dict(status="failed", execution_passed=False, task_passed=None,
                              reason="First episode terminated", done_terms=meta.get("done_terms", []))
        if not np.all(z["motion_id"][indices] == 0):
            raise ValueError("Single-reference recording changed motion identity")
        states = {key: z[key][indices].copy() for key in STATE_FIELDS}
        if any(not np.isfinite(value).all() for value in states.values()):
            raise ValueError("Nonfinite recorded robot state")
        for key in ("root_quat_xyzw", "body_quat_xyzw"):
            if not np.allclose(np.linalg.norm(states[key], axis=-1), 1, atol=1e-4, rtol=0):
                raise ValueError("Invalid recorded quaternion; extraction does not repair poses")
        # Require complete raw audit and solver evidence, even if no active
        # contact happens to be present at a particular sample.
        for key in ("raw_contact_count", "raw_contact_shape0", "raw_contact_shape1", "raw_contact_body0",
                    "raw_contact_body1", "raw_contact_point0_w", "raw_contact_point1_w", "raw_contact_normal_w"):
            if key not in z or len(z[key]) <= indices[-1]:
                raise ValueError(f"Missing/truncated raw contact audit: {key}")
        channels = {key: z["solver_contact_" + key] for key in (*FIELDS, *SUPPORT_FIELDS)}
    binding, labels = meta["newton_scene_binding"], meta["newton_body_labels"]
    pairs, support_rows = [], []
    for frame in indices:
        count = int(channels["count"][frame])
        snapshot = {key: value[frame, :count] for key, value in channels.items() if key != "count"}
        snapshot["count"] = count
        selected = select_support_contacts(snapshot, binding, labels)
        support_rows.append({k: v[0] for k, v in observed_support(snapshot, binding, labels, 1).items()})
        rows = []
        for i in np.flatnonzero(selected["groups"] >= 0):
            rows.append(dict(part=int(selected["groups"][i]), surface=int(selected["surfaces"][i]),
                             allocated=bool(snapshot["constraint_allocated"][i]),
                             position_w=snapshot["position_w"][i].tolist(), dist=float(snapshot["dist"][i]),
                             includemargin=float(snapshot["includemargin"][i]), solver_row=int(i)))
        pairs.append(rows)
    contacts = process_face_contacts(pairs, binding["surface_catalog"], fps=case["fps"])
    task = task_check(case, contacts, binding["surface_catalog"])
    abnormal = sum(len(row) for row in contacts["abnormal_contact_pairs"])
    passed = task["passed"] is not False and abnormal <= case["task"]["max_abnormal_contacts"]
    report = dict(status="passed" if passed else "failed", execution_passed=True, task_passed=task["passed"],
                  task=task, abnormal_contact_count=abnormal, frames=len(indices), native_record_frames=indices.tolist(),
                  sampling=meta["solver_support_semantics"], support_scope="measured last solve; no undeclared load/slip acceptance")
    support = {key: np.asarray([row[key] for row in support_rows]) for key in support_rows[0]}
    payload = dict(meta=meta, states=states, indices=indices, contacts=contacts, support=support)
    return payload, report


def inspect_recording(recording, case):
    _, report = read_execution(recording, case)
    return report


def export_recording(recording, case, destination, accepted, policy):
    from .trajectory_refinement import fingerprint, read_json, verify_files, write_json
    destination = Path(destination)
    provenance = dict(recording=fingerprint(recording), checkpoint=policy,
                      reference=fingerprint(case["reference"]["motion_file"]), context=accepted["context"])
    receipt = destination / "export.json"
    if receipt.exists():
        saved = read_json(receipt)
        if saved["provenance"] != provenance:
            raise ValueError("Existing export belongs to another execution")
        verify_files(saved["artifacts"])
        return
    payload, report = read_execution(recording, case)
    if report["status"] != "passed" or report["native_record_frames"] != accepted["native_record_frames"]:
        raise ValueError("Execution no longer passes its recorded acceptance")
    destination.mkdir(parents=True, exist_ok=True)
    states, meta = payload["states"], payload["meta"]
    clock = dict(native_record_frames=payload["indices"].tolist(), dt_s=meta["dt"], sim_dt_s=meta["sim_dt"],
                 pose_sampling="recorded reset/bootstrap and post-integration states; unchanged",
                 contact_sampling=meta["solver_support_semantics"], strict_pose_contact_same_state=False)
    kinematics = newton_rollout_kinematics_provenance(
        source_path=str(recording), source_sha256=provenance["recording"]["sha256"],
        output_fps=case["fps"], body_names=meta["body_names"])
    kinematics["boundary_sampling"] = clock["pose_sampling"]
    motion = dict(fps=np.asarray(case["fps"]), joint_names=np.asarray(meta["dof_names"]),
                  body_names=np.asarray(meta["body_names"]),
                  joint_pos=np.concatenate((states["root_pos"], states["root_quat_xyzw"][:, [3, 0, 1, 2]], states["dof_pos"]), axis=-1),
                  joint_vel=np.concatenate((states["root_lin_vel"], states["root_ang_vel"], states["dof_vel"]), axis=-1),
                  body_pos_w=states["body_pos_w"], body_quat_w=states["body_quat_xyzw"][..., [3, 0, 1, 2]],
                  body_lin_vel_w=states["body_lin_vel_w"], body_ang_vel_w=states["body_ang_vel_w"],
                  robot_asset_json=np.asarray(encode_robot_asset_json(meta["robot_asset"])),
                  kinematics_provenance_json=np.asarray(json.dumps(kinematics)),
                  rollout_provenance_json=np.asarray(json.dumps(dict(provenance, clock=clock))))
    # Exclusive writes plus content comparison permit safe recovery of partial
    # exports without deleting or silently replacing an earlier trajectory.
    def save_npz(path, arrays):
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                if set(existing.files) != set(arrays) or any(
                        not np.array_equal(existing[k], v, equal_nan=True) if np.asarray(v).dtype.kind in "fc"
                        else not np.array_equal(existing[k], v) for k, v in arrays.items()):
                    raise ValueError(f"Partial export differs from source: {path}")
        else:
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
    save_npz(destination / "motion.npz", motion)
    contacts = payload["contacts"]
    metadata = dict(schema=OBSERVATION_SCHEMA, provenance=provenance, clock=clock,
                    motion_sha256=sha256_file(destination / "motion.npz"), newton_runtime=meta["newton_runtime"],
                    robot_asset=meta["robot_asset"], scene=binding_summary(meta["newton_scene_binding"]),
                    contact_semantics=meta["solver_contact_semantics"],
                    contact_label_contract=contacts["contact_label_contract"],
                    raw_pair_audit="complete original recording retained; no pair deletion")
    observations = {k: contacts[k] for k in ("contact_part_mask", "contact_surface", "contact_position_w")}
    observations.update({"observed_support_" + k: v for k, v in payload["support"].items()})
    observations.update(native_record_frames=payload["indices"], part_order=np.asarray(PARTS),
                        contact_pairs_json=np.asarray(json.dumps(contacts["contact_pairs"])),
                        abnormal_contact_pairs_json=np.asarray(json.dumps(contacts["abnormal_contact_pairs"])),
                        metadata_json=np.asarray(json.dumps(metadata)))
    save_npz(destination / "observations.npz", observations)
    write_json(receipt, dict(provenance=provenance, clock=clock,
                            artifacts=[fingerprint(destination / name) for name in ("motion.npz", "observations.npz")]),
               immutable=True)


def binding_summary(binding):
    return {key: binding[key] for key in ("model_fingerprint", "body_env", "shape_surface", "surface_catalog")}
