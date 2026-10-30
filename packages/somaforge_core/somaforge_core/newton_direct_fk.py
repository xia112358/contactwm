from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from somaforge_core import (
    DIRECT_NEWTON_KINEMATICS_BACKEND,
    G1_29DOF_JOINT_ORDER,
    body_velocities_from_pose,
    canonical_g1_asset_metadata,
    canonical_g1_urdf_path,
    direct_newton_kinematics_provenance,
    encode_kinematics_provenance,
    encode_robot_asset_json,
    sha256_file,
    validate_pose_velocity_consistency,
    validate_root_body_consistency,
)
from somaforge_core.kinematics import angular_velocity_wxyz


@dataclass(frozen=True)
class DirectNewtonFkResult:
    output_path: Path
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    frame_count: int
    backend_metadata: dict[str, Any]


_DIRECT_FK_MODEL_CACHE: dict[
    tuple[str, int, int, str],
    tuple[Any, Any, Any],
] = {}


def canonicalize_motion_with_direct_newton_fk(
    input_path: str | Path,
    output_path: str | Path,
    *,
    robot_urdf: str | Path | None = None,
    device: str = "cpu",
    overwrite: bool = False,
    joint_velocity_filter_window: int | None = None,
    joint_velocity_filter_polyorder: int = 3,
) -> DirectNewtonFkResult:
    """Canonicalize a Holosoma q trajectory with direct Newton ``eval_fk``.

    This function imports Newton and Warp only. It never imports Isaac Lab,
    launches Kit, creates a SimulationContext, or steps a physics solver.
    """

    import newton
    import warp as wp

    source_path = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True to replace it")
    with np.load(source_path, allow_pickle=True) as data:
        source = {key: data[key] for key in data.files}

    fps = _fps(source)
    joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] < 7:
        raise ValueError(f"joint_pos must have shape [T,7+J], got {joint_pos.shape}")
    source_joint_names = _string_list(source.get("joint_names"))
    if not source_joint_names:
        raise ValueError("direct Newton FK requires joint_names in the input motion")
    if joint_pos.shape[1] != 7 + len(source_joint_names):
        raise ValueError(
            f"joint_pos has {joint_pos.shape[1]} columns but joint_names has {len(source_joint_names)} entries"
        )
    missing_source = [name for name in G1_29DOF_JOINT_ORDER if name not in source_joint_names]
    if missing_source:
        raise ValueError(f"input motion is missing canonical G1 joints: {missing_source}")
    canonical_joint_indices = [source_joint_names.index(name) for name in G1_29DOF_JOINT_ORDER]
    canonical_joint_pos = joint_pos[:, 7:][:, canonical_joint_indices]
    canonical_qpos = np.concatenate([joint_pos[:, :7], canonical_joint_pos], axis=1)
    canonical_qpos[:, 3:7] = _normalize_quat_wxyz(canonical_qpos[:, 3:7])
    canonical_qvel = _holosoma_joint_velocities(
        canonical_qpos,
        fps,
        filter_window=joint_velocity_filter_window,
        filter_polyorder=joint_velocity_filter_polyorder,
    )

    urdf_path = Path(robot_urdf).expanduser().resolve() if robot_urdf is not None else canonical_g1_urdf_path()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Newton robot URDF does not exist: {urdf_path}")

    urdf_stat = urdf_path.stat()
    model_key = (
        str(urdf_path),
        int(urdf_stat.st_mtime_ns),
        int(urdf_stat.st_size),
        str(device),
    )
    cached_model = _DIRECT_FK_MODEL_CACHE.get(model_key)
    if cached_model is None:
        builder = newton.ModelBuilder(up_axis="Z")
        import_result = builder.add_urdf(
            str(urdf_path),
            floating=True,
        )
        model = builder.finalize(device=device)
        state = model.state()
        _DIRECT_FK_MODEL_CACHE.clear()
        _DIRECT_FK_MODEL_CACHE[model_key] = (
            model,
            state,
            import_result,
        )
    else:
        model, state, import_result = cached_model

    joint_labels = tuple(str(value) for value in model.joint_label)
    body_labels = tuple(str(value) for value in model.body_label)
    joint_types = np.asarray(model.joint_type.numpy(), dtype=np.int32)
    joint_q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int64)
    joint_qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int64)
    joint_dof_dim = np.asarray(model.joint_dof_dim.numpy(), dtype=np.int64)

    free_type = int(newton.JointType.FREE)
    free_joint_ids = np.flatnonzero(joint_types == free_type)
    if free_joint_ids.size != 1:
        raise ValueError(
            f"direct Newton G1 import must contain exactly one FREE root joint, got {free_joint_ids.tolist()} labels={joint_labels}"
        )
    free_joint_id = int(free_joint_ids[0])
    free_q_start = int(joint_q_start[free_joint_id])
    free_q_end = int(joint_q_start[free_joint_id + 1])
    free_qd_start = int(joint_qd_start[free_joint_id])
    free_qd_end = int(joint_qd_start[free_joint_id + 1])
    if free_q_end - free_q_start != 7 or free_qd_end - free_qd_start != 6:
        raise ValueError("Newton FREE root joint does not use the expected 7-coordinate/6-velocity layout")

    canonical_to_newton_joint = _resolve_canonical_joint_ids(joint_labels, joint_types, newton)
    body_names = tuple(_label_leaf(label) for label in body_labels)
    if len(set(body_names)) != len(body_names):
        duplicates = sorted({name for name in body_names if body_names.count(name) > 1})
        raise ValueError(f"direct Newton body labels are not uniquely resolvable: {duplicates}")
    if "pelvis" not in body_names:
        raise ValueError(f"direct Newton model has no pelvis body: {body_names}")

    body_pos_w = np.empty((canonical_qpos.shape[0], len(body_names), 3), dtype=np.float32)
    body_quat_w = np.empty((canonical_qpos.shape[0], len(body_names), 4), dtype=np.float32)
    q_template = np.asarray(state.joint_q.numpy(), dtype=np.float32)
    qd_template = np.zeros_like(np.asarray(state.joint_qd.numpy(), dtype=np.float32))

    for frame in range(canonical_qpos.shape[0]):
        q = q_template.copy()
        qd = qd_template.copy()
        root = canonical_qpos[frame, :7]
        q[free_q_start:free_q_end] = np.asarray(
            [root[0], root[1], root[2], root[4], root[5], root[6], root[3]], dtype=np.float32
        )
        root_velocity = canonical_qvel[frame, :6]
        qd[free_qd_start:free_qd_end] = root_velocity.astype(np.float32)

        for canonical_index, joint_id in enumerate(canonical_to_newton_joint):
            q_start = int(joint_q_start[joint_id])
            q_end = int(joint_q_start[joint_id + 1])
            qd_start = int(joint_qd_start[joint_id])
            qd_end = int(joint_qd_start[joint_id + 1])
            dof_count = int(joint_dof_dim[joint_id].sum())
            if q_end - q_start != 1 or qd_end - qd_start != 1 or dof_count != 1:
                raise ValueError(
                    f"canonical joint {G1_29DOF_JOINT_ORDER[canonical_index]} is not scalar in Newton: "
                    f"q={q_end - q_start} qd={qd_end - qd_start} dof={dof_count}"
                )
            q[q_start] = np.float32(canonical_qpos[frame, 7 + canonical_index])
            qd[qd_start] = np.float32(canonical_qvel[frame, 6 + canonical_index])

        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.synchronize_device(device)
        body_q = np.asarray(state.body_q.numpy(), dtype=np.float32)
        if body_q.shape != (len(body_names), 7):
            raise ValueError(f"Newton body_q has unexpected shape {body_q.shape}")
        body_pos_w[frame] = body_q[:, :3]
        body_quat_w[frame] = body_q[:, [6, 3, 4, 5]]

    body_quat_w = _normalize_quat_wxyz(body_quat_w).astype(np.float32)
    body_lin_vel_w, body_ang_vel_w = body_velocities_from_pose(body_pos_w, body_quat_w, fps)
    validate_root_body_consistency(
        canonical_qpos,
        body_pos_w,
        body_quat_w,
        root_body_index=body_names.index("pelvis"),
    )
    validate_pose_velocity_consistency(
        body_pos_w,
        body_quat_w,
        body_lin_vel_w,
        body_ang_vel_w,
        fps,
    )

    backend_metadata = {
        "newton_version": str(getattr(newton, "__version__", "unknown")),
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "device": str(device),
        "asset_importer": "ModelBuilder.add_urdf(floating=True)",
        "robot_urdf": str(urdf_path),
        "joint_coord_count": int(model.joint_coord_count),
        "joint_dof_count": int(model.joint_dof_count),
        "joint_count": int(model.joint_count),
        "body_count": int(model.body_count),
        "import_result_keys": sorted(str(key) for key in import_result) if isinstance(import_result, dict) else [],
        "joint_velocity_derivation": (
            "pose_finite_difference"
            if joint_velocity_filter_window is None
            else "pose_local_polynomial_derivative"
        ),
        "joint_velocity_filter_window": joint_velocity_filter_window,
        "joint_velocity_filter_polyorder": (
            None if joint_velocity_filter_window is None else int(joint_velocity_filter_polyorder)
        ),
    }
    provenance = direct_newton_kinematics_provenance(
        source_path=str(source_path),
        source_sha256=sha256_file(source_path),
        output_fps=fps,
        body_names=list(body_names),
        metadata=backend_metadata,
    )
    provenance["joint_velocity_derivation"] = backend_metadata["joint_velocity_derivation"]
    if joint_velocity_filter_window is not None:
        provenance["joint_velocity_filter"] = {
            "window": int(joint_velocity_filter_window),
            "polyorder": int(joint_velocity_filter_polyorder),
            "zero_phase": True,
            "boundary_mode": "shifted_local_polynomial",
        }

    generated = _strip_stale_kinematics(source)
    from somaforge_core.support_evidence import reference_only_support
    reference_only_support(generated, reason='FK output is a pose trajectory, not a new physical execution')
    generated.update(
        {
            "fps": np.asarray(fps, dtype=np.float32),
            "joint_pos": canonical_qpos.astype(np.float32),
            "joint_vel": canonical_qvel.astype(np.float32),
            "joint_names": np.asarray(G1_29DOF_JOINT_ORDER),
            "body_names": np.asarray(body_names),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "robot_asset_json": np.asarray(encode_robot_asset_json(canonical_g1_asset_metadata())),
            "kinematics_provenance_json": np.asarray(encode_kinematics_provenance(provenance)),
            "kinematics_backend": np.asarray(DIRECT_NEWTON_KINEMATICS_BACKEND),
            "newton_canonicalization_required": np.asarray(False),
            "newton_direct_fk_metadata_json": np.asarray(json.dumps(backend_metadata, sort_keys=True)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **generated)
    return DirectNewtonFkResult(
        output_path=output,
        body_names=body_names,
        joint_names=tuple(G1_29DOF_JOINT_ORDER),
        frame_count=int(canonical_qpos.shape[0]),
        backend_metadata=backend_metadata,
    )


def _resolve_canonical_joint_ids(joint_labels: Sequence[str], joint_types: np.ndarray, newton_module: Any) -> list[int]:
    scalar_types = {
        int(newton_module.JointType.REVOLUTE),
        int(newton_module.JointType.PRISMATIC),
    }
    result: list[int] = []
    for canonical_name in G1_29DOF_JOINT_ORDER:
        candidates = [
            index
            for index, label in enumerate(joint_labels)
            if _label_leaf(label) == canonical_name and int(joint_types[index]) in scalar_types
        ]
        if len(candidates) != 1:
            candidates = [
                index
                for index, label in enumerate(joint_labels)
                if canonical_name in str(label) and int(joint_types[index]) in scalar_types
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"cannot uniquely resolve canonical Newton joint {canonical_name!r}: "
                f"matches={[joint_labels[index] for index in candidates]}"
            )
        result.append(int(candidates[0]))
    if len(set(result)) != len(result):
        raise ValueError("canonical Newton joint mapping contains duplicates")
    return result


def _strip_stale_kinematics(source: dict[str, Any]) -> dict[str, Any]:
    generated = dict(source)
    stale_exact = {
        "kinematics_provenance_json",
        "kinematics_backend",
        "newton_direct_fk_metadata_json",
    }
    for key in list(generated):
        if key in stale_exact:
            generated.pop(key, None)
    return generated


def _label_leaf(label: str) -> str:
    value = str(label).rstrip("/").split("/")[-1]
    return value.split(":")[-1]


def _fps(source: dict[str, Any]) -> float:
    if "fps" in source:
        fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    elif "dt" in source:
        dt = float(np.asarray(source["dt"]).reshape(-1)[0])
        fps = 1.0 / dt
    else:
        fps = 50.0
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"motion fps must be finite and positive, got {fps}")
    return fps


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    array = np.asarray(value, dtype=object)
    return [str(item) for item in array.reshape(-1).tolist()]


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)


def _local_polynomial_derivative(
    values: np.ndarray,
    fps: float,
    *,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Differentiate sampled values with a zero-phase local polynomial fit."""
    x = np.asarray(values, dtype=np.float64)
    if window < 3 or window % 2 == 0:
        raise ValueError(f"joint velocity filter window must be odd and >= 3, got {window}")
    if polyorder < 1 or polyorder >= window:
        raise ValueError(f"joint velocity filter polyorder must be in [1, window), got {polyorder}")
    if x.shape[0] < window:
        raise ValueError(f"motion has {x.shape[0]} frames, fewer than velocity filter window {window}")

    half = window // 2
    output = np.empty_like(x)
    for frame in range(x.shape[0]):
        start = min(max(frame - half, 0), x.shape[0] - window)
        indices = np.arange(start, start + window)
        time = (indices - frame) / float(fps)
        design = np.vander(time, N=polyorder + 1, increasing=True)
        derivative_weights = np.linalg.pinv(design)[1]
        output[frame] = np.tensordot(derivative_weights, x[indices], axes=(0, 0))
    return output


def _holosoma_joint_velocities(
    qpos: np.ndarray,
    fps: float,
    *,
    filter_window: int | None = None,
    filter_polyorder: int = 3,
) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape[0] == 1:
        return np.zeros((1, q.shape[1] - 1), dtype=np.float32)
    dt = 1.0 / float(fps)
    root_linear = np.gradient(q[:, :3], dt, axis=0)
    root_angular = angular_velocity_wxyz(_normalize_quat_wxyz(q[:, 3:7]), dt)
    if filter_window is None:
        joint_velocity = np.gradient(q[:, 7:], dt, axis=0)
    else:
        joint_velocity = _local_polynomial_derivative(
            q[:, 7:], fps, window=filter_window, polyorder=filter_polyorder
        )
    return np.concatenate([root_linear, root_angular, joint_velocity], axis=1).astype(np.float32)
