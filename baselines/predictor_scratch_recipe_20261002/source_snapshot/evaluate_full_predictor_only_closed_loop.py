#!/usr/bin/env python3
"""Autoregress a full next-interaction predictor without infiller or projection."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import tyro
from isaaclab.app import AppLauncher


ROOT = Path.cwd().resolve()
for package in (ROOT / "packages/climb00_pipeline", ROOT / "packages/somaforge_core", ROOT / "tmp"):
    sys.path.insert(0, str(package))

from climb00_pipeline.full_interaction_predictor import FullHeightmapInteractionPredictor
from climb00_pipeline.full1000_position_predictor import Full1000PositionPredictor
from climb00_pipeline.planned_contact_predictor import (
    PlannedHeightmapContactPredictor, audit_intended_surfaces,
)
from climb00_pipeline.conditioned_pose_predictor import ConditionedHeightmapPosePredictor
from climb00_pipeline.contact_location_predictor import (
    HeightmapContactLocationPredictor,
    contact_points_to_heightmap_cells,
    contact_points_to_heightmap_map,
    contact_points_to_observed_heightmap_map,
)
from climb00_pipeline.next_interaction_heightmap_v2 import render_root_yaw_box_heightmaps
from climb00_pipeline.next_interaction_heightmap_v2 import _root_yaw_basis
from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.newton_contact_query import query_contacts
from somaforge_core.loaded_material_motion import unknown_material_path
from somaforge_core.robot_assets import decode_robot_asset_json
from generator.training_data import box_obbs
from evaluate_contact_constrained_projector_oracle import yaw_frame
from climb00_pipeline.interaction_acceptance import (
    DEFAULT_ACCEPTANCE, acceptance_contract, contact_acceptance,
)
from climb00_pipeline.rollout_termination import (
    TerminationLimits, geometry_failure, pose_failure,
)
from climb00_pipeline.recursive_selection import recursive_selection
from climb00_pipeline.contact_layout import representative_pairs, relative_layout_statistics, persistent_role_consistent, nearest_spatial_pairs


DEFAULT_AUDIT = ROOT / "tmp/newton_contact_sources_v1/1789326506727043514"
DEFAULT_POLICY = ROOT / "runtime/current/holosoma/logs/WholeBodyTracking/20260727_085520-g1_29dof_wbt_single_climb00_completionema_horizon50_ncon160_from4k_to10k-locomotion/model_06000.pt"


def observe(q_world: np.ndarray, endpoint: str) -> dict:
    raw = query_contacts(q_world[None], None, endpoint=endpoint)
    selected = select_contact_pairs(raw["pairs"], raw["surface_catalog"])
    separation = raw["full_robot_separation"][0]
    candidate_audit = []
    for pair in raw.get("candidate_pairs", [[]])[0]:
        if not 0 <= int(pair.get("part", -1)) < 6:
            continue
        candidate_audit.append({
            key: pair.get(key)
            for key in (
                "part", "surface", "body_name", "dist", "includemargin",
                "constraint_active", "allocated", "efc_address", "normal_w",
                "position_w", "terrain_position_w",
            )
        })
    return {
        "surface_catalog": raw["surface_catalog"],
        "mask": np.asarray(selected["contact_part_mask"][0], dtype=bool),
        "surface": np.asarray(selected["contact_surface"][0], dtype=np.int64),
        "position_w": np.asarray(selected["contact_position_w"][0], dtype=np.float32),
        "pairs": selected["contact_pairs"][0],
        "raw_pairs": raw["pairs"][0],
        "abnormal_pairs": selected["abnormal_contact_pairs"][0],
        "terrain_penetration_m": float(separation["terrain_penetration_m"]),
        "self_penetration_m": float(separation["self_penetration_m"]),
        "worst_terrain": separation["worst_terrain"],
        "candidate_audit": candidate_audit,
        "separation": separation,
    }


def event_names(entry: dict) -> dict[tuple[int, int], str]:
    rows = [json.loads(line) for line in Path(entry["event_segments_file"]).read_text().splitlines() if line.strip()]
    return {(int(row["start_frame"]), int(row["end_frame"])): str(row["terminal_event"]) for row in rows}


def evaluate(args: argparse.Namespace, endpoint: str) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    payload = torch.load(args.data_cache, map_location="cpu", weights_only=False)
    samples, data = payload["samples"], payload["data"]
    with np.load(args.q_cache, allow_pickle=False) as archive:
        q_segments = np.asarray(archive["target_q"], dtype=np.float32)
        q_motion_ids = np.asarray(archive["motion_ids"], dtype=np.int64)
        q_current_frames = np.asarray(archive["current_frames"], dtype=np.int64)
        q_target_frames = np.asarray(archive["target_frames"], dtype=np.int64)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    schema = checkpoint.get("schema")
    if schema not in {
        "full_interaction_predictor_v1",
        "full1000_position_predictor_v1",
        "contact_location_predictor_v1",
        "embodied_interaction_predictor_v1",
        "planned_contact_predictor_v1",
    }:
        raise ValueError("unsupported predictor-only checkpoint schema")
    decode_robot_asset_json(checkpoint["robot_asset_json"], context="predictor-only rollout")
    sample_motion_ids = np.asarray([int(row["motion_id"]) for row in samples])
    sample_current_frames = np.asarray([int(row["current_frame"]) for row in samples])
    sample_target_frames = np.asarray([int(row["target_frame"]) for row in samples])
    if not (
        np.array_equal(q_motion_ids, sample_motion_ids)
        and np.array_equal(q_current_frames, sample_current_frames)
        and np.array_equal(q_target_frames, sample_target_frames)
    ):
        raise ValueError("data-cache and q-cache sample timelines differ")
    if sample_motion_ids.min(initial=0) < 0 or sample_motion_ids.max(initial=-1) >= len(manifest["motion_files"]):
        raise ValueError("cache motion ids are outside the checkpoint manifest")
    device = torch.device(args.device)
    if schema == "full1000_position_predictor_v1":
        model = Full1000PositionPredictor(
            int(checkpoint["config"]["width"]), int(checkpoint["config"]["layers"]),
            int(checkpoint["config"].get("location_width", 32)),
            body_geometry=bool(checkpoint["config"].get("body_geometry", False)),
            part_geometry=bool(checkpoint["config"].get("part_geometry", False)),
            region_plan=bool(checkpoint["config"].get("region_plan", False)),
            unified_contact=bool(checkpoint["config"].get("unified_contact", False)),
            event_roles=bool(checkpoint["config"].get("event_roles", False)),
            execution_observation_gradients=bool(checkpoint["config"].get("execution_observation_gradients", False)),
        ).to(device)
    elif schema in {
        "contact_location_predictor_v1",
        "embodied_interaction_predictor_v1",
        "planned_contact_predictor_v1",
    }:
        model_class = PlannedHeightmapContactPredictor if schema == "planned_contact_predictor_v1" else HeightmapContactLocationPredictor
        model = model_class(
            int(checkpoint["config"]["width"]),
            int(checkpoint["config"]["layers"]),
            int(checkpoint["config"]["location_width"]),
            joint_residual_output=bool(
                checkpoint["config"].get("joint_residual_output", False)
            ),
            relational_terrain=bool(
                checkpoint["config"].get("relational_terrain", False)
            ),
        ).to(device)
    else:
        model = FullHeightmapInteractionPredictor(
            int(checkpoint["config"]["width"]), int(checkpoint["config"]["layers"])
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diagnostic_teacher = None
    if args.diagnostic_teacher_checkpoint is not None:
        teacher_checkpoint = torch.load(
            args.diagnostic_teacher_checkpoint, map_location=device, weights_only=False
        )
        diagnostic_teacher = ConditionedHeightmapPosePredictor(
            int(teacher_checkpoint["config"]["width"]),
            int(teacher_checkpoint["config"]["layers"]),
        ).to(device)
        teacher_destination = diagnostic_teacher.state_dict()
        teacher_source = {
            key: value for key, value in teacher_checkpoint["model"].items()
            if key in teacher_destination and teacher_destination[key].shape == value.shape
        }
        missing = sorted(set(teacher_destination) - set(teacher_source))
        if missing:
            raise ValueError(f"diagnostic teacher base is incomplete: {missing}")
        diagnostic_teacher.load_state_dict(teacher_source)
        diagnostic_teacher.eval()

    centers, rotations, half_extents, grounds = box_obbs(data)
    rows = [index for index, sample in enumerate(samples) if int(sample["motion_id"]) == args.motion_id]
    rows.sort(key=lambda index: int(samples[index]["current_frame"]))
    if not rows:
        raise ValueError(f"motion {args.motion_id} has no predictor samples")
    groups = [[rows[0]]]
    for row in rows[1:]:
        previous = groups[-1][-1]
        if int(samples[previous]["target_frame"]) == int(samples[row]["current_frame"]):
            groups[-1].append(row)
        else:
            groups.append([row])
    if args.continuous_group is not None:
        if not 1 <= args.continuous_group <= len(groups):
            raise ValueError(
                f"continuous group must be in [1,{len(groups)}], got {args.continuous_group}"
            )
        rows = groups[args.continuous_group - 1]
    if args.start_frame is not None:
        matches = [i for i, row in enumerate(rows) if int(samples[row]["current_frame"]) == args.start_frame]
        if len(matches) != 1:
            raise ValueError(f"start frame {args.start_frame} must match exactly one event start")
        rows = rows[matches[0]:]
    if any(int(samples[a]["target_frame"]) != int(samples[b]["current_frame"])
           for a, b in zip(rows[:-1], rows[1:])):
        raise ValueError("predictor-only rollout requires one continuous event chain")
    if args.total_steps is not None:
        rows = rows[:args.total_steps]
        args.extra_steps = max(0, args.total_steps - len(rows))
    entry = manifest["motion_files"][args.motion_id]
    terminals = event_names(entry)
    with np.load(entry["motion_file"], allow_pickle=False) as motion:
        decode_robot_asset_json(motion["robot_asset_json"], context="predictor-only reference")
        world_motion = np.asarray(motion["joint_pos"], dtype=np.float32)
        joint_names = motion["joint_names"].astype(str).tolist()

    first = rows[0]
    current_frame = int(samples[first]["current_frame"])
    current_world = world_motion[current_frame].copy()
    from generator.forward_progress import ForwardProgressGate
    task_direction = (np.asarray(args.global_forward_xy) if args.global_forward_xy is not None
                      else world_motion[-1,:2]-world_motion[0,:2])
    progress_gate = ForwardProgressGate(current_world[:2], task_direction,
        window=args.progress_window, minimum_m=args.progress_minimum_m) if args.require_forward_progress else None
    origin, basis = yaw_frame(q_segments[first, 0], current_world)
    world_center = origin + basis @ centers[first]
    world_rotation = basis @ rotations[first]
    world_half = half_extents[first : first + 1]
    world_ground = np.asarray([grounds[first] + origin[2]], dtype=np.float32)
    # Establish the fixed physical scene BEFORE perturbing the robot. Moving
    # both would leave the relative observation unchanged and invalidate this
    # robustness test. Only the initial root translation changes.
    reference_initial_world = current_world.copy()
    with torch.no_grad():
        initial_yaw_basis, _ = _root_yaw_basis(
            torch.as_tensor(current_world[None], device=device)
        )
    initial_forward = initial_yaw_basis[0, :, 0].cpu().numpy()
    initial_offset = -args.initial_backward_m * initial_forward
    if args.initial_offset_world_m is not None:
        initial_offset = np.asarray(args.initial_offset_world_m, dtype=np.float32)
    current_world[:3] += initial_offset

    def infer(q_world: np.ndarray, observed: dict, heightmap: np.ndarray, repair=None):
        q_tensor = torch.as_tensor(q_world[None], device=device)
        contact_tensor = torch.as_tensor(observed["mask"][None], device=device)
        point_tensor = torch.as_tensor(observed["position_w"][None], device=device)
        with torch.no_grad():
            if schema in {
                "contact_location_predictor_v1",
                "embodied_interaction_predictor_v1",
                "planned_contact_predictor_v1",
            }:
                if checkpoint.get("dataset", {}).get("raster_label_semantics") == "nearest_observed_3d_cell_v1":
                    contact_map, _, _ = contact_points_to_observed_heightmap_map(
                        q_tensor, contact_tensor, point_tensor, torch.as_tensor(heightmap, device=device),
                    )
                else:
                    contact_map, _, _ = contact_points_to_heightmap_map(q_tensor, contact_tensor, point_tensor)
                prediction = model(
                    current_q=q_tensor,
                    current_contact=contact_tensor,
                    current_contact_map=contact_map,
                    heightmap=torch.as_tensor(heightmap, device=device),
                )
                if schema == "embodied_interaction_predictor_v1":
                    return prediction, {
                        "contact": None,
                        "surface": None,
                        "cell": None,
                        "point_world": None,
                    }
                local = prediction.contact_points_local
                yaw_basis, _ = _root_yaw_basis(q_tensor)
                point_world = q_tensor[:, None, :3] + torch.einsum(
                    "bij,bpj->bpi", yaw_basis, local
                )
                return prediction, {
                    "contact": prediction.contact[0].cpu().numpy(),
                    "surface": (
                        audit_intended_surfaces(
                            point_world, prediction.contact,
                            {"surface_catalog": observed["surface_catalog"]},
                        )[0].cpu().numpy() if schema == "planned_contact_predictor_v1" else None
                    ),
                    "cell": prediction.cell[0].cpu().numpy(),
                    "point_world": point_world[0].cpu().numpy(),
                }
            if schema == "full1000_position_predictor_v1":
                prediction = model(current_q=q_tensor, current_contact=contact_tensor,
                                   current_anchor=point_tensor,
                                   heightmap=torch.as_tensor(heightmap, device=device), **(repair or {}))
                basis, _ = _root_yaw_basis(q_tensor)
                points = q_tensor[:, None, :3] + torch.einsum("bij,bpj->bpi", basis, prediction.conditioned_points_local)
                return prediction, {
                    "contact": prediction.conditioned_contact[0].cpu().numpy(),
                    "surface": audit_intended_surfaces(points, prediction.conditioned_contact,
                        {"surface_catalog": observed["surface_catalog"]})[0].cpu().numpy(),
                    "cell": prediction.cell[0].cpu().numpy(),
                    "point_world": points[0].cpu().numpy(),
                }
            prediction = model(
                current_q=q_tensor,
                current_contact=contact_tensor,
                current_anchor=point_tensor,
                current_surface=torch.as_tensor(observed["surface"][None], device=device),
                heightmap=torch.as_tensor(heightmap, device=device),
            )
            return prediction, {
                "contact": prediction.contact[0].cpu().numpy(),
                "surface": prediction.surface[0].cpu().numpy(),
                "cell": None,
                "point_world": None,
            }

    def cells_in_frame(q_world: np.ndarray, mask: np.ndarray, points_world: np.ndarray):
        with torch.no_grad():
            cell, valid = contact_points_to_heightmap_cells(
                torch.as_tensor(q_world[None], device=device),
                torch.as_tensor(mask[None], device=device),
                torch.as_tensor(points_world[None], device=device),
            )
        return cell[0].cpu().numpy(), valid[0].cpu().numpy()

    predicted_q = [current_world.tolist()]
    reference_q = [reference_initial_world.tolist()]
    contact_masks = []
    contact_surfaces = []
    contact_points = []
    records = []
    initial_observation = None
    for step, row in enumerate(rows, 1):
        if args.teacher_forced:
            current_world = world_motion[int(samples[row]["current_frame"])].copy()
        observed = observe(current_world, endpoint)
        heightmap = render_root_yaw_box_heightmaps(
            world_center[None], world_rotation[None], world_half, world_ground,
            current_world[None], supersample=args.heightmap_supersample,
        )
        if initial_observation is None:
            initial_observation = {
                "q_world": current_world.tolist(),
                "contact_mask": observed["mask"].tolist(),
                "contact_surface": observed["surface"].tolist(),
                "contact_position_w": observed["position_w"].tolist(),
                "heightmap": heightmap[0].tolist(),
            }
        prediction, intention = infer(current_world, observed, heightmap)
        next_world = prediction.qpos[0].detach().cpu().numpy()
        actual = observe(next_world, endpoint)
        attempt_records = []
        issued = dict(repair_mask=torch.ones(1, dtype=torch.bool, device=device),
            repair_role=prediction.conditioned_role.detach() if getattr(prediction, 'conditioned_role', None) is not None else None,
            repair_points_world=torch.as_tensor(intention['point_world'][None], device=device) if intention['point_world'] is not None else None,
            repair_regions=prediction.planned_regions.detach() if getattr(prediction, 'planned_regions', None) is not None else None)
        if args.execution_attempts > 1:
            from climb00_pipeline.next_interaction_heightmap import HEIGHTMAP_RESOLUTION_M
            for attempt in range(args.execution_attempts):
                enabled, face = intention['contact'], intention['surface']
                selected = representative_pairs(actual['pairs'], enabled, face)
                complete = bool(enabled.any() and all(pair is not None for part, pair in enumerate(selected) if enabled[part]))
                error = float('inf')
                if complete:
                    delta = np.asarray([pair['position_w'][:2] for part, pair in enumerate(selected) if enabled[part]])-intention['point_world'][enabled,:2]
                    error = float(np.sqrt(np.mean(np.sum(delta**2,axis=-1))))
                limits = TerminationLimits(penetration_m=DEFAULT_ACCEPTANCE.shallow_penetration_m)
                failure = pose_failure(next_world,model.fk.joint_lower.cpu().numpy(),model.fk.joint_upper.cpu().numpy(),limits)
                failure = failure or geometry_failure(actual['separation'],limits)
                region_ok = True
                if issued['repair_regions'] is not None:
                    with torch.no_grad():
                        region_ok = model.region_geometry.audit_pairs(model.fk,torch.as_tensor(next_world[None],device=device),
                            actual['pairs'],face,issued['repair_regions'][0])['realized']
                own_done = bool(complete and error <= 2*HEIGHTMAP_RESOLUTION_M and failure is None and region_ok
                    and contact_acceptance(actual['pairs'],enabled,face)['contact_accepted'])
                attempt_records.append(dict(attempt=attempt+1,q_world=next_world.tolist(),own_endpoint_completed=own_done,
                    position_rms_cm=100*error if np.isfinite(error) else None,safety_failure=failure,
                    raw_pairs=actual['raw_pairs']))
                if own_done or attempt+1==args.execution_attempts: break
                repair_heightmap=render_root_yaw_box_heightmaps(world_center[None],world_rotation[None],world_half,
                    world_ground,next_world[None],supersample=args.heightmap_supersample)
                prediction, _ = infer(next_world, actual, repair_heightmap, repair=issued)
                next_world=prediction.qpos[0].detach().cpu().numpy()
                actual=observe(next_world,endpoint)
        reference = world_motion[int(samples[row]["target_frame"])]
        intended_contact = intention["contact"]
        intended_surface = intention["surface"]
        gt_contact = np.asarray(data["end_contact"])[row].astype(bool)
        gt_surface = np.asarray(data["target_surfaces"])[row].astype(np.int64)
        if schema == "embodied_interaction_predictor_v1":
            intended_contact = actual["mask"]
            intended_surface = actual["surface"]
            intended_cell = None
            predicted_plan_matches_gt = bool(
                np.array_equal(actual["mask"], gt_contact)
                and np.array_equal(actual["surface"][actual["mask"]], gt_surface[gt_contact])
            )
            # This architecture has no separately emitted plan.  Comparing
            # the realized contact to itself would be a tautology, not an
            # implementation metric.
            predicted_plan_realized = None
            actual_contact_matches_gt = predicted_plan_matches_gt
        elif schema == "contact_location_predictor_v1":
            # Every cached event is independently recentered.  Using the
            # first event's frame for later GT witnesses corrupts exact-cell
            # diagnostics even though the actual recursive rollout is fine.
            row_current = world_motion[int(samples[row]["current_frame"])]
            row_origin, row_basis = yaw_frame(q_segments[row, 0], row_current)
            target_points_world = row_origin + np.einsum(
                "ij,pj->pi", row_basis, np.asarray(data["target_contacts"])[row]
            )
            gt_cell, gt_cell_valid = cells_in_frame(
                current_world, gt_contact, target_points_world
            )
            actual_cell, actual_cell_valid = cells_in_frame(
                current_world, actual["mask"], actual["position_w"]
            )
            intended_cell = intention["cell"]
            predicted_plan_matches_gt = bool(
                np.array_equal(intended_contact, gt_contact)
                and np.all(gt_cell_valid[gt_contact])
                and np.array_equal(intended_cell[gt_contact], gt_cell[gt_contact])
            )
            predicted_plan_realized = bool(
                np.array_equal(intended_contact, actual["mask"])
                and np.all(actual_cell_valid[actual["mask"]])
                and np.array_equal(
                    intended_cell[intended_contact], actual_cell[actual["mask"]]
                )
            )
            actual_contact_matches_gt = bool(
                np.array_equal(actual["mask"], gt_contact)
                and np.all(actual_cell_valid[actual["mask"]])
                and np.all(gt_cell_valid[gt_contact])
                and np.array_equal(actual_cell[actual["mask"]], gt_cell[gt_contact])
            )
        else:
            intended_cell = intention["cell"]
            predicted_plan_matches_gt = bool(
                np.array_equal(intended_contact, gt_contact)
                and np.array_equal(intended_surface[intended_contact], gt_surface[gt_contact])
            )
            predicted_plan_realized = bool(
                np.array_equal(intended_contact, actual["mask"])
                and np.array_equal(intended_surface[intended_contact], actual["surface"][actual["mask"]])
            )
            actual_contact_matches_gt = bool(
                np.array_equal(actual["mask"], gt_contact)
                and np.array_equal(actual["surface"][actual["mask"]], gt_surface[gt_contact])
            )
        with torch.no_grad():
            predicted_body, _ = model.fk(torch.as_tensor(next_world[None, None], device=device))
            reference_body, _ = model.fk(torch.as_tensor(reference[None, None], device=device))
        frame_pair = (int(samples[row]["current_frame"]), int(samples[row]["target_frame"]))
        record = {
            "step": step,
            "row": int(row),
            "frames": list(frame_pair),
            "terminal_event": terminals.get(frame_pair, "unknown"),
            "predicted_contact": intended_contact.tolist(),
            "predicted_surface": None if intended_surface is None else intended_surface.tolist(),
            "predicted_heightmap_cell": None if intended_cell is None else intended_cell.tolist(),
            "predicted_contact_point_w": (
                None if intention["point_world"] is None else intention["point_world"].tolist()
            ),
            "actual_contact": actual["mask"].tolist(),
            "actual_surface": actual["surface"].tolist(),
            "gt_contact": gt_contact.tolist(),
            "gt_surface": gt_surface.tolist(),
            "predicted_plan_matches_gt": predicted_plan_matches_gt,
            "predicted_plan_realized": predicted_plan_realized,
            "actual_contact_matches_gt": actual_contact_matches_gt,
            "actual_candidate_audit": actual["candidate_audit"],
            "actual_contact_pairs": actual["pairs"],
            "actual_raw_contact_pairs": actual["raw_pairs"],
            "actual_abnormal_contact_pairs": actual["abnormal_pairs"],
            "full_robot_separation": actual["separation"],
            "terrain_penetration_cm": 100.0 * actual["terrain_penetration_m"],
            "self_penetration_cm": 100.0 * actual["self_penetration_m"],
            "root_error_cm": 100.0 * float(np.linalg.norm(next_world[:3] - reference[:3])),
            "joint_rmse_rad": float(np.sqrt(np.mean((next_world[7:] - reference[7:]) ** 2))),
            "body_error_cm": 100.0 * float(torch.linalg.vector_norm(
                predicted_body - reference_body, dim=-1
            ).mean()),
            "predicted_q_world": next_world.tolist(),
            "reference_q_world": reference.tolist(),
        }
        acceptance = contact_acceptance(actual["pairs"], gt_contact, gt_surface)
        limits = TerminationLimits(penetration_m=DEFAULT_ACCEPTANCE.shallow_penetration_m)
        failure = pose_failure(next_world, model.fk.joint_lower.cpu().numpy(),
                               model.fk.joint_upper.cpu().numpy(), limits)
        failure = failure or geometry_failure(actual["separation"], limits)
        record["task_acceptance"] = dict(
            **acceptance, safety_failure=failure,
            accepted=acceptance["contact_accepted"] and failure is None,
        )
        if schema == "full_interaction_predictor_v1":
            own_acceptance = contact_acceptance(actual["pairs"], intended_contact, intended_surface)
            record["own_plan_acceptance"] = dict(
                **own_acceptance, safety_failure=failure,
                accepted=bool(intended_contact.any() and own_acceptance["contact_accepted"] and failure is None),
            )
        if schema in ("planned_contact_predictor_v1", "full1000_position_predictor_v1"):
            shared = intended_contact & actual["mask"] & (intended_surface == actual["surface"])
            record["predicted_contact_mask_matches_gt"] = bool(np.array_equal(intended_contact, gt_contact))
            record["predicted_contact_mask_realized"] = bool(np.array_equal(intended_contact, actual["mask"]))
            record["intended_point_error_cm"] = (
                float(100 * np.linalg.norm(intention["point_world"][shared] - actual["position_w"][shared], axis=-1).mean())
                if shared.any() else None
            )
            own_acceptance = contact_acceptance(actual["pairs"], intended_contact, intended_surface)
            record["own_plan_acceptance"] = dict(
                **own_acceptance, safety_failure=failure,
                accepted=bool(intended_contact.any() and own_acceptance["contact_accepted"] and failure is None),
            )
            location_realized = []
            nearest_errors = []
            for part in range(6):
                if not intended_contact[part]:
                    location_realized.append(None)
                    nearest_errors.append(None)
                    continue
                witnesses = [np.asarray(pair["terrain_position_w"], dtype=np.float32)
                             for pair in actual["pairs"]
                             if int(pair["part"]) == part and int(pair["surface"]) == intended_surface[part]]
                matched = False
                for witness in witnesses:
                    point = np.zeros((6, 3), dtype=np.float32)
                    mask = np.zeros(6, dtype=bool)
                    point[part], mask[part] = witness, True
                    if checkpoint.get("dataset", {}).get("raster_label_semantics") == "nearest_observed_3d_cell_v1":
                        _, cell_tensor, valid_tensor = contact_points_to_observed_heightmap_map(
                            torch.as_tensor(current_world[None], device=device),
                            torch.as_tensor(mask[None], device=device),
                            torch.as_tensor(point[None], device=device),
                            torch.as_tensor(heightmap, device=device),
                        )
                        cell, valid = cell_tensor[0].cpu().numpy(), valid_tensor[0].cpu().numpy()
                    else:
                        cell, valid = cells_in_frame(current_world, mask, point)
                    matched |= bool(valid[part] and cell[part] == intended_cell[part])
                location_realized.append(matched)
                nearest_errors.append(
                    float(100 * min(np.linalg.norm(w - intention["point_world"][part]) for w in witnesses))
                    if witnesses else None
                )
            record["own_spatial_contact_cells_realized"] = location_realized
            record["own_nearest_contact_point_error_cm"] = nearest_errors
            record["own_spatial_plan_realized"] = bool(
                intended_contact.any() and predicted_plan_realized
                and all(value is not False for value in location_realized)
            )
            record["own_spatial_plan_safe"] = record["own_spatial_plan_realized"] and failure is None
            # Centered layout is a historical diagnostic only; it cannot
            # excuse displacement in scene-fixed endpoint acceptance.
            chosen = representative_pairs(actual['pairs'], intended_contact, intended_surface)
            observed_layout = np.asarray([pair is not None for pair in chosen])
            complete_layout = bool(intended_contact.any() and (observed_layout | ~intended_contact).all())
            if complete_layout:
                witnesses = np.asarray([pair['position_w'] if pair is not None else [0., 0., 0.] for pair in chosen])
                error, shift, _ = relative_layout_statistics(torch.as_tensor(witnesses[None]),
                    torch.as_tensor(intention['point_world'][None]), torch.as_tensor(observed_layout[None]))
                record["common_contact_offset_xy_cm"] = (100 * shift[0]).tolist()
                record["relative_layout_rms_cm"] = float(100 * error[0].sqrt())
            else:
                record["common_contact_offset_xy_cm"] = None
                record["relative_layout_rms_cm"] = None
            record["exact_cell_match_required_for_contact_acceptance"] = False
            record['relative_layout_complete'] = complete_layout
            record['relative_layout_parts'] = int(observed_layout.sum())
            record['relative_layout_contract'] = 'newton_representative_witness_xy_v1'
        if getattr(prediction, 'planned_regions', None) is not None:
            with torch.no_grad():
                audit = model.region_geometry.audit_pairs(model.fk,
                    torch.as_tensor(next_world[None], device=device), actual['pairs'], intended_surface,
                    prediction.planned_regions[0])
            record['region_plan_acceptance'] = dict(audit,
                accepted=bool(audit['realized'] and record['own_plan_acceptance']['accepted']))
        if getattr(prediction, 'conditioned_role', None) is not None:
            record['predicted_role'] = prediction.conditioned_role[0].cpu().tolist()
            record['temporal_retouch_verified'] = False
        record['execution_attempts'] = attempt_records
        from climb00_pipeline.endpoint_termination import endpoint_failure_reasons
        from climb00_pipeline.next_interaction_heightmap import HEIGHTMAP_RESOLUTION_M
        position_contract = checkpoint.get('dataset', {}).get('contact_position_contract')
        record['endpoint_position_contract'] = position_contract
        record['spatial_match_policy'] = ('nearest_actual_pair_in_predicted_region'
            if position_contract == 'observed_endpoint_region_geometry_v3' else 'legacy_representative')
        support_contract = position_contract in ('persistent_support_observed_scene_v2', 'observed_endpoint_region_geometry_v3')
        require_spatial = bool(support_contract or checkpoint['config'].get('anchored_plan_execution') or checkpoint['config'].get('pending_plan_repair'))
        if support_contract:
            record['persistent_role_consistent'] = bool(
                getattr(prediction, 'conditioned_role', None) is not None and
                persistent_role_consistent(prediction.conditioned_role.cpu(), torch.as_tensor(observed['mask'][None]))[0])
        spatial_pairs = actual['pairs']
        spatial_regions = getattr(prediction, 'planned_regions', None)
        if position_contract == 'observed_endpoint_region_geometry_v3' and spatial_regions is not None:
            # Copy for diagnostic annotations; preserve raw Newton evidence.
            spatial_pairs = [dict(pair) for pair in actual['pairs']]
            if spatial_pairs:
                with torch.no_grad():
                    ids = model.region_geometry.witness_regions(model.fk,
                        torch.as_tensor(next_world[None], device=device),
                        torch.zeros(len(spatial_pairs), device=device, dtype=torch.long),
                        torch.tensor([p['part'] for p in spatial_pairs], device=device),
                        torch.tensor([p['position_w'] for p in spatial_pairs], device=device))
                for pair, region in zip(spatial_pairs, ids.tolist(), strict=True):
                    pair['contact_region'] = region
            spatial_regions = spatial_regions[0].cpu().numpy()
        else:
            spatial_regions = None
        record['own_endpoint_position_rms_cm'] = None
        if intention['point_world'] is not None and intended_contact.any():
            witnesses = (nearest_spatial_pairs(spatial_pairs, intended_contact, intended_surface,
                intention['point_world'], regions=spatial_regions)
                if position_contract == 'observed_endpoint_region_geometry_v3'
                else representative_pairs(actual['pairs'], intended_contact, intended_surface))
            if all(witnesses[p] is not None for p in np.flatnonzero(intended_contact)):
                delta = np.asarray([witnesses[p]['position_w'][:2] for p in np.flatnonzero(intended_contact)])-intention['point_world'][intended_contact,:2]
                record['own_endpoint_position_rms_cm'] = float(100*np.sqrt(np.mean(np.sum(delta**2,axis=-1))))
        record['support_motion'] = unknown_material_path()
        record['actual_support_evidence_known'] = False
        record['termination_reasons'] = endpoint_failure_reasons(record, require_spatial=require_spatial,
            tolerance_cm=200*HEIGHTMAP_RESOLUTION_M, require_observed_support=args.require_observed_support)
        record['endpoint_accepted'] = not record['termination_reasons']
        if diagnostic_teacher is not None and step < len(rows):
            next_row = rows[step]
            next_heightmap = render_root_yaw_box_heightmaps(
                world_center[None], world_rotation[None], world_half, world_ground,
                next_world[None], supersample=args.heightmap_supersample,
            )
            with torch.no_grad():
                teacher_action = diagnostic_teacher(
                    current_q=torch.as_tensor(next_world[None], device=device),
                    current_contact=torch.as_tensor(actual["mask"][None], device=device),
                    current_anchor=torch.as_tensor(actual["position_w"][None], device=device),
                    current_surface=torch.as_tensor(actual["surface"][None], device=device),
                    heightmap=torch.as_tensor(next_heightmap, device=device),
                    planned_contact=torch.as_tensor(
                        np.asarray(data["end_contact"])[next_row][None], device=device
                    ).bool(),
                    planned_surface=torch.as_tensor(
                        np.asarray(data["target_surfaces"])[next_row][None], device=device
                    ).long(),
                )
            teacher_actual = observe(
                teacher_action.qpos[0].detach().cpu().numpy(), endpoint
            )
            teacher_gt_contact = np.asarray(data["end_contact"])[next_row].astype(bool)
            teacher_gt_surface = np.asarray(data["target_surfaces"])[next_row].astype(np.int64)
            record["diagnostic_teacher_next_gt_realized"] = bool(
                np.array_equal(teacher_actual["mask"], teacher_gt_contact)
                and np.array_equal(
                    teacher_actual["surface"][teacher_actual["mask"]],
                    teacher_gt_surface[teacher_gt_contact],
                )
            )
            record["diagnostic_teacher_next_actual_contact"] = teacher_actual["mask"].tolist()
            record["diagnostic_teacher_next_gt_contact"] = teacher_gt_contact.tolist()
        if progress_gate is not None:
            record.update(progress_gate.update(next_world[:2]))
            if not record['forward_progress_valid']:
                record['termination_reasons'].append('no_forward_progress')
                record['endpoint_accepted'] = False
        records.append(record)
        predicted_q.append(next_world.tolist())
        reference_q.append(reference.tolist())
        contact_masks.append(actual["mask"].tolist())
        contact_surfaces.append(actual["surface"].tolist())
        contact_points.append(actual["position_w"].tolist())
        if record.get('forward_progress_valid') is False:
            break
        if not args.teacher_forced and not args.diagnostic_continue and not record['endpoint_accepted']:
            break  # Record the failed output; never feed it to the next event.
        current_world = next_world
        if not args.diagnostic_continue and attempt_records and not attempt_records[-1]['own_endpoint_completed']:
            break  # Unfinished issued intent must not become a new planning event.
        print(json.dumps({key: record[key] for key in (
            "step", "frames", "terminal_event", "predicted_plan_matches_gt",
            "predicted_plan_realized", "root_error_cm", "body_error_cm"
        )}), flush=True)

    final_reference = world_motion[int(samples[rows[-1]]["target_frame"])]
    for extra_index in range(1, (0 if records[-1].get('forward_progress_valid') is False else args.extra_steps) + 1):
        observed = observe(current_world, endpoint)
        heightmap = render_root_yaw_box_heightmaps(
            world_center[None], world_rotation[None], world_half, world_ground,
            current_world[None], supersample=args.heightmap_supersample,
        )
        prediction, intention = infer(current_world, observed, heightmap)
        next_world = prediction.qpos[0].detach().cpu().numpy()
        actual = observe(next_world, endpoint)
        intended_contact = intention["contact"]
        intended_surface = intention["surface"]
        intended_cell = intention["cell"]
        if schema == "embodied_interaction_predictor_v1":
            intended_contact = actual["mask"]
            intended_surface = actual["surface"]
            intended_cell = None
            predicted_plan_realized = None
        elif schema == "contact_location_predictor_v1":
            actual_cell, actual_cell_valid = cells_in_frame(
                current_world, actual["mask"], actual["position_w"]
            )
            predicted_plan_realized = bool(
                np.array_equal(intended_contact, actual["mask"])
                and np.all(actual_cell_valid[actual["mask"]])
                and np.array_equal(
                    intended_cell[intended_contact], actual_cell[actual["mask"]]
                )
            )
        else:
            predicted_plan_realized = bool(
                np.array_equal(intended_contact, actual["mask"])
                and np.array_equal(
                    intended_surface[intended_contact], actual["surface"][actual["mask"]]
                )
            )
        with torch.no_grad():
            predicted_body, _ = model.fk(torch.as_tensor(next_world[None, None], device=device))
            reference_body, _ = model.fk(torch.as_tensor(final_reference[None, None], device=device))
        step = len(records) + 1
        record = {
            "step": step,
            "row": None,
            "frames": [f"extra-{extra_index - 1}", f"extra-{extra_index}"],
            "terminal_event": "autonomous_extension",
            "extra_prediction": True,
            "predicted_contact": intended_contact.tolist(),
            "predicted_surface": None if intended_surface is None else intended_surface.tolist(),
            "predicted_heightmap_cell": None if intended_cell is None else intended_cell.tolist(),
            "predicted_contact_point_w": (
                None if intention["point_world"] is None else intention["point_world"].tolist()
            ),
            "actual_contact": actual["mask"].tolist(),
            "actual_surface": actual["surface"].tolist(),
            "gt_contact": None,
            "gt_surface": None,
            "predicted_plan_matches_gt": None,
            "predicted_plan_realized": predicted_plan_realized,
            "actual_contact_matches_gt": None,
            "terrain_penetration_cm": 100.0 * actual["terrain_penetration_m"],
            "self_penetration_cm": 100.0 * actual["self_penetration_m"],
            "root_error_cm": 100.0 * float(np.linalg.norm(next_world[:3] - final_reference[:3])),
            "joint_rmse_rad": float(np.sqrt(np.mean((next_world[7:] - final_reference[7:]) ** 2))),
            "body_error_cm": 100.0 * float(torch.linalg.vector_norm(
                predicted_body - reference_body, dim=-1
            ).mean()),
            "predicted_q_world": next_world.tolist(),
            "reference_q_world": final_reference.tolist(),
        }
        record['support_motion'] = unknown_material_path()
        record['actual_support_evidence_known'] = False
        record['actual_contact_pairs'] = actual['pairs']
        record['diagnostic_only'] = True
        if progress_gate is not None:
            record.update(progress_gate.update(next_world[:2]))
            record['termination_reasons'] = [] if record['forward_progress_valid'] else ['no_forward_progress']
        records.append(record)
        predicted_q.append(next_world.tolist())
        reference_q.append(final_reference.tolist())
        contact_masks.append(actual["mask"].tolist())
        contact_surfaces.append(actual["surface"].tolist())
        contact_points.append(actual["position_w"].tolist())
        current_world = next_world
        if record.get('forward_progress_valid') is False:
            break
        print(json.dumps({key: record[key] for key in (
            "step", "frames", "terminal_event", "predicted_plan_realized",
            "root_error_cm", "body_error_cm"
        )}), flush=True)

    roots = [row["root_error_cm"] for row in records]
    bodies = [row["body_error_cm"] for row in records]
    accepted_prefix = 0
    for row in records[:len(rows)]:
        if not row["task_acceptance"]["accepted"]:
            break
        accepted_prefix += 1
    summary = {
        "empty_contact_steps": sum(not any(row["actual_contact"]) for row in records),
        "first_empty_contact_step": next((row["step"] for row in records if not any(row["actual_contact"])), None),
        "task_accepted_prefix": accepted_prefix,
        "task_accepted_steps": sum(row["task_acceptance"]["accepted"] for row in records[:len(rows)]),
        "steps": len(records),
        "continuous_groups": 1,
        "demonstrated_steps": len(rows),
        "extra_steps": args.extra_steps,
        "predicted_plan_matches_gt_steps": sum(
            row["predicted_plan_matches_gt"] is True for row in records
        ),
        "predicted_plan_realized_steps": (
            None
            if schema == "embodied_interaction_predictor_v1"
            else sum(row["predicted_plan_realized"] is True for row in records)
        ),
        "actual_contact_matches_gt_steps": sum(
            row["actual_contact_matches_gt"] is True for row in records
        ),
        "mean_terrain_penetration_cm": float(np.mean([
            row["terrain_penetration_cm"] for row in records
        ])),
        "max_terrain_penetration_cm": float(np.max([
            row["terrain_penetration_cm"] for row in records
        ])),
        "penetrating_steps": sum(
            row["terrain_penetration_cm"] > 0.0 for row in records
        ),
        "settled_pose_steps": [row["step"] for row in records if row["terminal_event"] == "settled_pose"],
        "mean_root_error_cm": float(np.mean(roots)),
        "max_root_error_cm": float(np.max(roots)),
        "final_root_error_cm": roots[-1],
        "mean_body_error_cm": float(np.mean(bodies)),
        "max_body_error_cm": float(np.max(bodies)),
    }
    if schema in ("planned_contact_predictor_v1", "full1000_position_predictor_v1"):
        own_prefix = 0
        for row in records[:len(rows)]:
            if not row.get("own_plan_acceptance", {}).get("accepted", False):
                break
            own_prefix += 1
        relative_errors = [row["relative_layout_rms_cm"] for row in records
                           if row.get("relative_layout_rms_cm") is not None and row.get('relative_layout_parts', 0) >= 2]
        summary.update(
            own_plan_safe_prefix=own_prefix,
            own_plan_safe_steps=sum(row.get("own_plan_acceptance", {}).get("accepted", False) for row in records),
            own_spatial_plan_realized_steps=sum(row.get("own_spatial_plan_realized", False) for row in records),
            own_spatial_plan_safe_steps=sum(row.get("own_spatial_plan_safe", False) for row in records),
            empty_predicted_plan_steps=sum(not any(row["predicted_contact"]) for row in records),
            spatial_realization_contract="nonempty plan; actual Newton primary-face active+allocated terrain witness in each intended observation raster cell; topology exact",
            exact_cell_metrics_diagnostic_only=True,
            mean_relative_layout_rms_cm_on_realized_plans=float(np.mean(relative_errors)) if relative_errors else None,
            relative_layout_evaluable_steps=len(relative_errors),
            relative_layout_minimum_parts=2,
        )
    report = {
        "schema": "interaction_predictor_only_closed_loop_v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "data_contract": {
            "manifest": str(args.manifest.resolve()),
            "data_cache": str(args.data_cache.resolve()),
            "q_cache": str(args.q_cache.resolve()),
            "source": "checkpoint config; explicit CLI overrides must match",
        },
        "motion_id": args.motion_id,
        "joint_names": joint_names,
        "contract": {
            "support_motion": unknown_material_path(),
            "forward_progress": dict(enabled=args.require_forward_progress,
                direction_xy=list(progress_gate.direction) if progress_gate else None,
                direction_source='explicit_global_task_direction' if args.global_forward_xy is not None else 'whole_demonstration_start_to_end_xy',
                window_steps=args.progress_window, minimum_m=args.progress_minimum_m,
                hard_stop_even_in_diagnostic_continue=True),
            "start_frame": int(samples[rows[0]]["current_frame"]),
            "initial_perturbation": {
                "backward_m": args.initial_backward_m,
                "direction": (
                    "explicit world translation" if args.initial_offset_world_m is not None
                    else "negative initial root-yaw forward axis"
                ),
                "world_offset_m": initial_offset.tolist(),
                "reference_initial_q": reference_initial_world.tolist(),
                "terrain_unchanged": True,
                "contacts_and_heightmap_requeried": True,
            },
            "task_acceptance": acceptance_contract(),
            "recurrence": (
                "reference event start is restored before every prediction"
                if args.teacher_forced
                else "raw predicted q becomes next current q"
            ),
            "contact_observation": "fresh Newton/MJWarp query of raw predicted q",
            "contact_intention": (
                "contact transition physically realized by the single emitted action"
                if schema == "embodied_interaction_predictor_v1"
                else "predicted six-part role and observed heightmap cell condition the pose decoder"
                if schema in ("planned_contact_predictor_v1", "full1000_position_predictor_v1")
                else "limb role + observed 2cm heightmap cell"
                if schema == "contact_location_predictor_v1"
                else "limb role + semantic surface"
            ),
            "infiller": False,
            "projector": False,
            "penetration_correction": False,
            "teacher_pose_input": False,
            "event_index_input": False,
        },
        "summary": summary,
        "events": records,
        "visualization": {
            "initial_observation": initial_observation,
            "predicted_q_world": predicted_q,
            "reference_q_world": reference_q,
            "actual_contact_mask": contact_masks,
            "actual_contact_surface": contact_surfaces,
            "actual_contact_position_w": contact_points,
        },
    }
    if checkpoint.get('dataset', {}).get('relative_layout_contract'):
        report['contract']['relative_layout_contract'] = checkpoint['dataset']['relative_layout_contract']
    report['contract']['failure_policy'] = ('independent_teacher_inputs' if args.teacher_forced else
        'diagnostic_continue_after_failure' if args.diagnostic_continue else 'stop_on_first_failure')
    passed = next((i for i, r in enumerate(records) if not r.get('endpoint_accepted', False)), len(records))
    summary.update(planned_events=len(rows), attempted_events=len(records), successful_prefix=passed,
        first_failed_event=passed+1 if passed<len(records) else None,
        completed_chain=passed==len(rows), retries=0)
    if (schema in ("planned_contact_predictor_v1", "full1000_position_predictor_v1", "full_interaction_predictor_v1") and not args.teacher_forced
            and not args.diagnostic_continue and not args.extra_steps and not np.any(initial_offset) and args.execution_attempts == 1
            ):
        report["recursive_selection"] = recursive_selection(report)
    if args.execution_attempts > 1:
        report['execution_controller'] = dict(max_attempts_per_plan=args.execution_attempts,
            completed_events=sum(bool(r.get('execution_attempts')) and r['execution_attempts'][-1]['own_endpoint_completed'] for r in records),
            stopped_on_unfinished_plan=bool(records and records[-1].get('execution_attempts') and not records[-1]['execution_attempts'][-1]['own_endpoint_completed']),
            selection_eligible=False, temporal_retouch_verified=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def run(args: argparse.Namespace) -> None:
    if args.total_steps is not None and (args.total_steps < 1 or not args.diagnostic_continue or args.teacher_forced or args.extra_steps):
        raise ValueError('total_steps requires positive diagnostic self-fed rollout, without extra_steps')
    if args.execution_attempts != 1 or (args.extra_steps and not args.diagnostic_continue):
        raise ValueError('One prediction per next-contact event; failure stops recursion. Retries and extra continuation steps are disabled.')
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") not in {
        "full_interaction_predictor_v1",
        "full1000_position_predictor_v1",
        "contact_location_predictor_v1",
        "embodied_interaction_predictor_v1",
        "planned_contact_predictor_v1",
    }:
        raise ValueError("unsupported predictor-only checkpoint schema")
    config = checkpoint.get("config", {})
    if args.execution_attempts < 1:
        raise ValueError('execution_attempts must be positive')
    if args.execution_attempts > 1 and (checkpoint['schema'] != 'full1000_position_predictor_v1'
            or not config.get('event_roles') or args.teacher_forced or args.extra_steps):
        raise ValueError('Stored-plan repair requires an autonomous event-role model without extra steps')
    if args.heightmap_supersample is None:
        args.heightmap_supersample = int(config.get("heightmap_supersample", 1))
    for name in ("manifest", "data_cache", "q_cache"):
        if name not in config:
            if getattr(args, name) is None:
                raise ValueError(f"checkpoint lacks required data contract path: {name}")
            continue
        trained = Path(config[name]).resolve()
        supplied = getattr(args, name)
        if (
            supplied is not None
            and supplied.resolve() != trained
            and not args.allow_diagnostic_cache_mismatch
        ):
            raise ValueError(
                f"{name.replace('_', '-')} mismatch: checkpoint={trained}, CLI={supplied.resolve()}"
            )
        if supplied is None or not args.allow_diagnostic_cache_mismatch:
            setattr(args, name, trained)
    candidates = None
    comparison = bool(args.comparison_checkpoints)
    if comparison and args.checkpoint_directory is not None:
        raise ValueError('choose explicit comparison or directory selection')
    if args.checkpoint_directory is not None or comparison:
        if (args.teacher_forced or args.extra_steps or args.initial_backward_m
                or args.initial_offset_world_m is not None
                or args.allow_diagnostic_cache_mismatch
                or args.diagnostic_teacher_checkpoint is not None):
            raise ValueError("checkpoint selection requires unperturbed autonomous evaluation")
        if checkpoint["schema"] not in ("planned_contact_predictor_v1", "full1000_position_predictor_v1"):
            raise ValueError("recursive selection requires an explicit contact plan")
        candidates = list(args.comparison_checkpoints) if comparison else sorted(args.checkpoint_directory.glob("step_*.pt"))
        if comparison and len({p.parent.name+'_'+p.stem for p in candidates}) != len(candidates):
            raise ValueError('comparison checkpoint output names collide')
        if not candidates:
            raise ValueError("no milestone checkpoints to evaluate")
        if args.output.exists():
            raise FileExistsError(args.output)
        for candidate in candidates:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            if (payload["schema"] != checkpoint["schema"]
                    or payload["robot_asset_json"] != checkpoint["robot_asset_json"]):
                raise ValueError(f"checkpoint asset/schema differs: {candidate}")
            for name in ("manifest", "data_cache", "q_cache"):
                if Path(payload["config"][name]).resolve() != Path(config[name]).resolve():
                    raise ValueError(f"checkpoint data contract differs: {candidate}")
            if payload.get("dataset", {}).get("raster_label_semantics") != checkpoint.get("dataset", {}).get("raster_label_semantics"):
                raise ValueError(f"checkpoint contact raster contract differs: {candidate}")
        args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    entry = manifest["motion_files"][args.motion_id]
    terrain_ids = [row["terrain_id"] for row in manifest["terrains"]]
    source = args.audit / f"scene_{terrain_ids.index(entry['terrain_id'])}"
    worker_dir = args.output.parent / f"{args.output.name}_worker_{time.time_ns()}"
    worker_dir.mkdir(parents=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    log_path = worker_dir / "worker.log"
    with log_path.open("w") as log:
        worker = subprocess.Popen([
            sys.executable, "scripts/serve_newton_contact_queries.py",
            "--checkpoint", str(args.policy_checkpoint),
            "--motion-manifest", str(source / "manifest.json"),
            "--binding", str(worker_dir / "binding.json"), "--create-native-binding",
            "--inspection-output", str(worker_dir / "model.json"),
            "--query-nconmax", "2048", "--query-njmax", "16384",
            "--port", str(port), "--device", "cuda:0", "--capture-contact-sources",
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180.0
            while not log_path.exists() or '"ready": true' not in log_path.read_text():
                if worker.poll() is not None:
                    raise RuntimeError(f"Newton worker exited; see {log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Newton worker startup timeout")
                time.sleep(2.0)
            if candidates is None:
                evaluate(args, endpoint)
            else:
                rankings = []
                for candidate in candidates:
                    child = argparse.Namespace(**vars(args))
                    child.checkpoint = candidate
                    child.output = args.output / (candidate.parent.name+'_'+candidate.stem if comparison else candidate.stem)
                    evaluate(child, endpoint)
                    report = json.loads((child.output / "report.json").read_text())
                    selected = recursive_selection(report)
                    rankings.append({
                        "checkpoint": str(candidate.resolve()),
                        "step": report["checkpoint_step"],
                        "report": str((child.output / "report.json").resolve()),
                        **selected,
                    })
                    print(json.dumps({"checkpoint_selection_candidate": rankings[-1]}), flush=True)
                if comparison:
                    (args.output / 'comparison.json').write_text(json.dumps({
                        'schema': 'predeclared_checkpoint_comparison_v1', 'motion_id': args.motion_id,
                        'continuous_group': args.continuous_group, 'rankings': rankings,
                        'selected_checkpoint': None}, indent=2)+'\n')
                    return
                # Stable sort retains the earlier checkpoint for exact ties.
                rankings.sort(key=lambda row: tuple(row["key"]), reverse=True)
                winner = rankings[0]
                destination = args.output / "selected.pt"
                shutil.copy2(winner["checkpoint"], destination)
                selection = {
                    "schema": "recursive_checkpoint_rescore_v1",
                    "motion_id": args.motion_id,
                    "selected": winner,
                    "selected_checkpoint": str(destination.resolve()),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "held_out_evaluations_read": False,
                    "rankings": rankings,
                }
                (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
                print(json.dumps({"recursive_selection_complete": winner}), flush=True)
        finally:
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=30)


@dataclass
class EvaluationConfig:
    checkpoint: Path
    output: Path
    checkpoint_directory: Path | None = None
    comparison_checkpoints: tuple[Path, ...] = ()
    manifest: Path | None = None
    data_cache: Path | None = None
    q_cache: Path | None = None
    audit: Path = DEFAULT_AUDIT
    policy_checkpoint: Path = DEFAULT_POLICY
    motion_id: int = 8
    continuous_group: int | None = None
    start_frame: int | None = None
    extra_steps: int = 0
    total_steps: int | None = None
    require_forward_progress: bool = True
    global_forward_xy: tuple[float, float] | None = None
    progress_window: int = 3
    progress_minimum_m: float = 0.03615079075098038
    diagnostic_continue: bool = False
    require_observed_support: bool = False
    execution_attempts: int = 1
    initial_backward_m: float = 0.0
    initial_offset_world_m: tuple[float, float, float] | None = None
    teacher_forced: bool = False
    heightmap_supersample: int | None = None
    diagnostic_teacher_checkpoint: Path | None = None
    allow_diagnostic_cache_mismatch: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    official, remaining = parser.parse_known_args()
    cfg = tyro.cli(EvaluationConfig, args=remaining)
    args = argparse.Namespace(**asdict(cfg), device="cpu" if official.cpu else official.device)
    if args.extra_steps < 0:
        parser.error("--extra-steps must be non-negative")
    if args.teacher_forced and args.extra_steps:
        parser.error("--teacher-forced cannot be combined with --extra-steps")
    if not np.isfinite(args.initial_backward_m) or args.initial_backward_m < 0:
        parser.error("--initial-backward-m must be finite and nonnegative")
    if args.teacher_forced and args.initial_backward_m:
        parser.error("initial perturbation requires self-fed rollout")
    if args.initial_offset_world_m is not None:
        if not np.isfinite(args.initial_offset_world_m).all():
            parser.error("initial world offset must be finite")
        if args.initial_backward_m or args.teacher_forced:
            parser.error("explicit world offset requires self-fed rollout and no backward offset")
    if args.heightmap_supersample is not None and args.heightmap_supersample < 1:
        parser.error("--heightmap-supersample must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
