from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from loguru import logger
from somaforge_core import validate_g1_asset_metadata


_G1_SPHEREHAND_STEM = "g1_29dof_spherehand"
_ROBOT_ASSET_SIDECAR = "somaforge_robot_asset.json"


def _validate_preconverted_robot_usd(usd_path: Path, robot_stem: str) -> Path:
    resolved = usd_path.resolve()
    if robot_stem != _G1_SPHEREHAND_STEM:
        return resolved

    sidecar = resolved.parent / _ROBOT_ASSET_SIDECAR
    if not sidecar.is_file():
        raise ValueError(
            f"Canonical G1 USD is missing its asset fingerprint: {sidecar}. "
            "Regenerate it with scripts/convert_g1_spherehand_usd.py."
        )
    with sidecar.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    validate_g1_asset_metadata(metadata, context=str(resolved))
    return resolved


def resolve_bool_attr_or_method(obj: Any, name: str) -> bool:
    value = getattr(obj, name, False)
    if callable(value):
        value = value()
    return bool(value)


def build_newton_physics_cfg(simulator_config: Any):
    from somaforge_core.newton_collision_compat import (
        install,
        install_mesh_query_launch_tuning,
        install_simulation_guard,
    )
    install()
    install_simulation_guard()
    install_mesh_query_launch_tuning()
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

    mjwarp = simulator_config.mujoco_warp
    nconmax = mjwarp.nconmax_per_env
    njmax = mjwarp.njmax_per_env or max(nconmax * 16, 2048)
    use_cuda_graph_env = os.environ.get("HOLOSOMA_NEWTON_USE_CUDA_GRAPH", "1").strip().lower()
    use_cuda_graph = use_cuda_graph_env not in {"0", "false", "no", "off"}

    logger.info(
        "Using Isaac Lab 3.0 Newton MJWarp physics backend "
        f"(nconmax={nconmax}, njmax={njmax}, substeps={simulator_config.sim.substeps}, "
        f"use_cuda_graph={use_cuda_graph})."
    )

    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=njmax,
            nconmax=nconmax,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=simulator_config.sim.substeps,
        debug_mode=False,
        use_cuda_graph=use_cuda_graph,
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
    )


def resolve_robot_usd(asset_root: str, robot_asset_cfg: Any) -> Path:
    profile = getattr(robot_asset_cfg, "collision_profile", "legacy_v1")
    if profile != "legacy_v1":
        if robot_asset_cfg.urdf_file != "g1/g1_29dof_spherehand.urdf":
            raise ValueError("Surface collision profiles require the authoritative G1 source URDF")
        from somaforge_core.robot_assets import canonical_g1_asset_metadata, canonical_g1_usd_path
        metadata = canonical_g1_asset_metadata(collision_profile=profile)
        path = canonical_g1_usd_path(collision_profile=profile)
        # Resolve the versioned asset through the authority, never a caller's
        # unrelated USD override carrying a compatible-looking sidecar.
        if robot_asset_cfg.usd_file is not None:
            configured = (Path(asset_root)/robot_asset_cfg.usd_file).resolve()
            if configured != path.resolve():
                raise ValueError("Explicit robot USD differs from the selected collision profile")
        validate_g1_asset_metadata(metadata, context=str(path))
        return path
    if robot_asset_cfg.usd_file is not None:
        usd_path = Path(asset_root) / robot_asset_cfg.usd_file
        if usd_path.is_file():
            return _validate_preconverted_robot_usd(usd_path, Path(robot_asset_cfg.urdf_file).stem)
        raise FileNotFoundError(f"Configured robot USD does not exist: {usd_path}")

    robot_stem = Path(robot_asset_cfg.urdf_file).stem
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    candidates = [
        Path(asset_root) / "isaaclab3_newton" / robot_stem / f"{robot_stem}.usda",
        Path(asset_root) / f"converted_rank{local_rank}" / robot_stem / f"{robot_stem}.usda",
        Path(asset_root) / "converted_rank0" / robot_stem / f"{robot_stem}.usda",
        Path(asset_root) / f"converted_rank{local_rank}" / f"{robot_stem}.usd",
        Path(asset_root) / "converted_rank0" / f"{robot_stem}.usd",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return _validate_preconverted_robot_usd(candidate, robot_stem)

    candidates_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Newton-only robot loading requires a pre-converted USD. "
        f"No USD found for '{robot_asset_cfg.urdf_file}'. Checked:\n{candidates_text}"
    )


def resolve_usd_next_to_asset(asset_path: str | Path) -> Path:
    source = Path(asset_path)
    stem = source.stem
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    candidates = [
        source.with_suffix(".usda"),
        source.with_suffix(".usd"),
        source.parent / "isaaclab3_newton" / stem / f"{stem}.usda",
        source.parent / "isaaclab3_newton" / stem / f"{stem}.usd",
        source.parent / f"converted_rank{local_rank}" / stem / f"{stem}.usda",
        source.parent / "converted_rank0" / stem / f"{stem}.usda",
        source.parent / f"converted_rank{local_rank}" / f"{stem}.usd",
        source.parent / "converted_rank0" / f"{stem}.usd",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    candidates_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Newton-only asset loading requires a pre-converted USD. "
        f"No USD found for '{source}'. Checked:\n{candidates_text}"
    )


def clone_newton_body_coms(asset: Any, num_envs: int, num_bodies: int, device: str) -> torch.Tensor:
    body_com_pos_b = getattr(getattr(asset, "data", None), "body_com_pos_b", None)
    if body_com_pos_b is not None:
        tensor = getattr(body_com_pos_b, "torch", body_com_pos_b)
        return tensor.clone()
    return torch.zeros((num_envs, num_bodies, 3), dtype=torch.float, device=device)


def env_ids_for_newton(env_ids: torch.Tensor | None, num_envs: int, device: torch.device | str) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.int32)
    return env_ids.to(device=device, dtype=torch.int32)


def body_ids_for_newton(
    asset: Any,
    body_ids: Sequence[int] | torch.Tensor | slice | None,
    body_names: Sequence[str] | str | None,
    device: torch.device | str,
) -> torch.Tensor:
    if body_ids == slice(None) or body_ids is None:
        if body_names is not None:
            resolved, _ = asset.find_bodies(body_names)
            return torch.as_tensor(resolved, dtype=torch.int32, device=device)
        return torch.arange(asset.num_bodies, dtype=torch.int32, device=device)
    return torch.as_tensor(body_ids, dtype=torch.int32, device=device)


def set_newton_body_coms(asset: Any, coms: torch.Tensor, env_ids: torch.Tensor, body_ids: torch.Tensor) -> None:
    if not hasattr(asset, "set_coms_index"):
        raise RuntimeError(f"Asset '{type(asset).__name__}' does not expose Newton set_coms_index().")
    asset.set_coms_index(
        coms=coms[env_ids[:, None], body_ids, :3],
        body_ids=body_ids,
        env_ids=env_ids,
    )
