"""Full1000 dataset preparation, extracted without changing sample/split semantics."""
from __future__ import annotations
import numpy as np
import torch
from generator.conditioned_pose_predictor import ConditionedHeightmapPosePredictor
from contact_solver.heightmap_contact_projector import observed_surface_indices
from somaforge_core.g1_kinematics import _matrix_from_rotation6d
from generator.next_interaction_heightmap_v2 import render_root_yaw_box_heightmaps

def split_name(height: float) -> str:
    if height >= 1.075:
        return "test_height_110"
    if height <= 0.925:
        return "validation_height_090"
    return "train_height_095_100_105"

def box_obbs(data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    polygon = data["box_edge_start"]
    if polygon.ndim != 3 or polygon.shape[1:] != (4, 2):
        raise ValueError(f"box top polygons must have shape [N,4,2], got {polygon.shape}")
    basis = np.asarray(data['box_basis'])
    height = np.asarray(data['box_height'])
    origin = np.asarray(data['box_origin'])
    if (basis.shape != (len(polygon), 3, 3) or origin.shape != (len(polygon), 3)
            or height.shape != (len(polygon),) or not all(np.isfinite(v).all()
                for v in (polygon, basis, origin, height)) or np.any(height <= 0)):
        raise ValueError('Invalid cached box geometry; rebuild predictor cache')
    if (not np.allclose(basis[:, :, 2], [0., 0., 1.], atol=1e-6, rtol=0)
            or not np.allclose(basis.transpose(0, 2, 1) @ basis, np.eye(3), atol=1e-6, rtol=0)):
        raise ValueError('Cached box top must be horizontal and upward; rebuild predictor cache')

    # Surface tangent coordinates are an arbitrary orthonormal chart; they are
    # not guaranteed to align with the rectangular terrain edges.  Taking UV
    # min/max therefore constructs a circumscribed rectangle (twice the true
    # area for a 45-degree diamond).  Use adjacent polygon edges as the OBB
    # axes so the returned prism matches the actual terrain mesh.
    edge_u = polygon[:, 1] - polygon[:, 0]
    edge_v = polygon[:, 2] - polygon[:, 1]
    length_u = np.linalg.norm(edge_u, axis=-1)
    length_v = np.linalg.norm(edge_v, axis=-1)
    if np.any(length_u <= 1.0e-8) or np.any(length_v <= 1.0e-8):
        raise ValueError("box top polygons contain a degenerate edge")
    axis_u = edge_u / length_u[:, None]
    axis_v = edge_v / length_v[:, None]
    if np.any(np.abs(np.sum(axis_u * axis_v, axis=-1)) > 1.0e-4):
        raise ValueError("box top polygons must be rectangular")

    uv_center = polygon.mean(axis=1)
    top_center = data["box_origin"] + np.einsum("bij,bj->bi", data["box_basis"][:, :, :2], uv_center)
    world_u = np.einsum("bij,bj->bi", data["box_basis"][:, :, :2], axis_u)
    world_v = np.einsum("bij,bj->bi", data["box_basis"][:, :, :2], axis_v)
    rotation = np.stack((world_u, world_v, data["box_basis"][:, :, 2]), axis=-1)
    center = top_center - 0.5 * data["box_height"][:, None] * data["box_basis"][:, :, 2]
    half = np.stack((0.5 * length_u, 0.5 * length_v, 0.5 * data["box_height"]), axis=-1)
    ground = data["box_origin"][:, 2] - data["box_height"]
    return (
        center.astype(np.float32),
        rotation.astype(np.float32),
        half.astype(np.float32),
        ground.astype(np.float32),
    )

def take(values: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in values.items()}

def prepare(model: ConditionedHeightmapPosePredictor, cfg, device: torch.device):
    payload = torch.load(cfg.data_cache, map_location="cpu", weights_only=False)
    data, samples = payload["data"], payload["samples"]
    from generator.support_path_acceptance import validate_support_paths
    from somaforge_core.contact_dataset import validate_contact_cache
    support_path_acceptance = validate_support_paths(cfg.manifest, samples)
    validate_contact_cache(payload["summary"], cfg.manifest)
    with np.load(cfg.q_cache, allow_pickle=False) as cache:
        validate_contact_cache(cache, cfg.manifest)
        if len(cache['target_q']) != len(samples):
            raise ValueError('Pose cache/sample count mismatch')
        for key, sample_key in (('motion_ids', 'motion_id'),
                                ('current_frames', 'current_frame'),
                                ('target_frames', 'target_frame')):
            if key not in cache or not np.array_equal(
                    cache[key], [s[sample_key] for s in samples]):
                raise ValueError(f'Pose cache/sample boundary mismatch: {key}')
        q = np.asarray(cache["target_q"][:, (0, -1)], dtype=np.float32)
    center, rotation, half, ground = box_obbs(data)
    depth_observation = getattr(model, 'depth_observation', False)
    heightmap = None if depth_observation else render_root_yaw_box_heightmaps(
        center, rotation, half, ground, q[:, 0],
        supersample=cfg.heightmap_supersample,
    )
    tensor = lambda value, **kwargs: torch.as_tensor(value, device=device, **kwargs)
    current_contact = tensor(data["start_contacts"]).bool()
    current_surface = data['current_surfaces'] if depth_observation else observed_surface_indices(
        heightmap,
        q[:, 0],
        np.asarray(data["start_contacts"], dtype=bool),
        np.asarray(data["current_contacts"], dtype=np.float32),
    )
    planned_contact = tensor(data["end_contact"]).bool()
    planned_surface = tensor(data["target_surfaces"], dtype=torch.long)
    inputs = {
        "current_q": tensor(q[:, 0]),
        "current_contact": current_contact,
        "current_anchor": tensor(data["current_contacts"]),
        "current_surface": tensor(current_surface, dtype=torch.long),
        "planned_contact": planned_contact,
        "planned_surface": planned_surface,
    }
    if not depth_observation:
        inputs['heightmap'] = tensor(heightmap)
    target_q = tensor(q[:, 1])
    with torch.no_grad():
        body_position, body_rotation6d = model.fk(target_q[:, None])
        body_position = body_position[:, 0]
        body_rotation = _matrix_from_rotation6d(body_rotation6d[:, 0])
        material_local = torch.einsum(
            "bpji,bpj->bpi",
            body_rotation[:, 1:7],
            tensor(data["target_contacts"]) - body_position[:, 1:7],
        )
    target = {
        "current_q": inputs["current_q"],
        "q": target_q,
        "body_position": body_position,
        "material_local": material_local,
        "anchor": tensor(data["target_contacts"]),
        "planned_contact": planned_contact,
        "planned_surface": planned_surface,
        "current_contact": current_contact,
        "current_anchor": tensor(data["current_contacts"]),
        "current_surface": tensor(current_surface, dtype=torch.long),
    }
    from generator.support_supervision import prepare_support_supervision
    support_supervision = prepare_support_supervision(cfg.manifest, target, samples)
    from types import SimpleNamespace
    from generator.newton_query_supervision import prepare_query_metadata
    scene = {
        "box_center": tensor(center),
        "box_rotation": tensor(rotation),
        "box_half_extents": tensor(half),
        "ground_height": tensor(ground),
    }
    query_metadata = prepare_query_metadata(
        cfg.manifest,
        inputs,
        target,
        [SimpleNamespace(**sample) for sample in samples],
        scene=scene,
    )
    split = {
        name: torch.tensor([
            index for index, sample in enumerate(samples)
            if split_name(float(sample["height"])).startswith(name)
        ], device=device)
        for name in ("train", "validation", "test")
    }
    if any(len(indices) == 0 for indices in split.values()):
        raise ValueError("empty train/validation/test split")
    if cfg.mode == "overfit":
        motion_id = cfg.overfit_motion_id
        if motion_id is None:
            motion_id = int(samples[int(split["train"][0])]["motion_id"])
        overfit = [
            int(index) for index in split["train"].cpu().tolist()
            if int(samples[index]["motion_id"]) == motion_id
        ]
        if not overfit:
            raise ValueError(f"overfit motion {motion_id} is not in the training split")
        split["overfit"] = torch.tensor(overfit, device=device)
    summary = {
        "schema": "predictor_demonstration_dataset_v1",
        "samples": len(samples),
        "support_path_acceptance": support_path_acceptance,
        "support_supervision": support_supervision,
        "query_scene_validation": query_metadata,
        "splits": {key: len(value) for key, value in split.items()},
        "condition_contract": "decoder consumes one explicit topology+surface plan",
        "pose_input": ("current q + actual contact/anchor + calibrated 2D depth scan + explicit plan"
            if depth_observation else "current q + actual contact/anchor + heightmap-derived surface index + 2cm root-yaw heightmap + explicit plan"),
        "excluded": ["binding", "predicted topology features", "event index", "future pose"],
        "training": "pose-only; no topology classification objective",
        "failure_states_used_for_training": False,
        "realization_contract": (
            "every supplied limb+surface plan is optimized against fresh Newton "
            "activation/allocation/margin witnesses"
        ),
        "geometric_distance_fallback": False,
        "projector_required_for_hard_guarantee": True,
    }
    return inputs, target, scene, split, samples, summary
