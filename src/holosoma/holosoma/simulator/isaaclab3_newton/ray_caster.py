from __future__ import annotations

from isaaclab_newton.physics import NewtonManager
# Preserve the trained policy's configured terrain-mesh ray queries.
from isaaclab_newton.sensors.ray_caster import LegacyRayCaster as NewtonRayCaster
from loguru import logger
import math
import torch


class HolosomaNewtonRayCaster(NewtonRayCaster):
    """Newton RayCaster variant that resolves one scanner frame per environment.

    Isaac Lab's Newton RayCaster expands every registered label for every env.
    In this manually assembled scene path the registered regex can already map
    to all env sensor sites, so the stock resolver produces num_envs squared
    frames. Height scanning expects exactly one scanner frame per env.
    """

    def _initialize_pose_tracking(self) -> None:
        super()._initialize_pose_tracking()
        if self._view_count != self._num_envs:
            raise RuntimeError(
                f"HolosomaNewtonRayCaster resolved {self._view_count} scanner frames for "
                f"{self._num_envs} envs; expected one frame per env."
            )
        logger.info(
            "HolosomaNewtonRayCaster resolved {} scanner frames for {} envs.",
            self._view_count,
            self._num_envs,
        )

    def _initialize_rays_impl(self) -> None:
        from isaaclab.sensors.ray_caster import patterns
        if isinstance(self.cfg.pattern_cfg, patterns.PinholeCameraPatternCfg):
            # Newton's generic pattern API supplies (cfg, device), while the
            # camera API also requires one intrinsic matrix per sensor.
            def pinhole(cfg, device):
                fx = cfg.focal_length / cfg.horizontal_aperture * cfg.width
                vertical = cfg.vertical_aperture or cfg.horizontal_aperture * cfg.height / cfg.width
                fy = cfg.focal_length / vertical * cfg.height
                intrinsic = torch.tensor([[fx, 0., cfg.width/2], [0., fy, cfg.height/2], [0., 0., 1.]], device=device)
                starts, directions = patterns.pinhole_camera_pattern(cfg, intrinsic[None], device)
                return starts[0], directions[0]
            self.cfg.pattern_cfg.func = pinhole
        super()._initialize_rays_impl()
        self._php_canonical_ray_directions = self.ray_directions.torch.clone()

    def reset(self, env_ids=None, env_mask=None):
        super().reset(env_ids=env_ids, env_mask=env_mask)
        max_angle = float(getattr(self.cfg, "angular_drift_degrees", 0.0))
        if max_angle <= 0.0 or not hasattr(self, "_php_canonical_ray_directions"):
            return
        if env_ids is None:
            ids = torch.arange(self._view_count, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        radians = math.radians(max_angle)
        angles = torch.empty((ids.numel(), 3), device=self.device).uniform_(-radians, radians)
        cx, cy, cz = torch.cos(angles).unbind(-1)
        sx, sy, sz = torch.sin(angles).unbind(-1)
        rotation = torch.stack(
            (
                cy * cz,
                cz * sx * sy - cx * sz,
                sx * sz + cx * cz * sy,
                cy * sz,
                cx * cz + sx * sy * sz,
                cx * sy * sz - cz * sx,
                -sy,
                cy * sx,
                cx * cy,
            ),
            dim=-1,
        ).reshape(-1, 3, 3)
        canonical = self._php_canonical_ray_directions.index_select(0, ids)
        self.ray_directions.torch[ids] = torch.einsum("eij,erj->eri", rotation, canonical)

    @staticmethod
    def _resolve_site_indices(labels: list[str], prim_expr: str, num_envs: int) -> list[int]:
        site_map = NewtonManager._cl_site_index_map
        site_indices: list[int] = []

        for env_idx in range(num_envs):
            env_candidates: list[int] = []
            for label in labels:
                error_prefix = f"RayCaster target '{prim_expr}' site label '{label}'"
                if label not in site_map:
                    raise ValueError(f"{error_prefix} was not found in NewtonManager._cl_site_index_map.")

                global_idx, per_world = site_map[label]
                if per_world is None:
                    env_candidates.append(global_idx)
                    continue

                if env_idx >= len(per_world):
                    raise ValueError(
                        f"{error_prefix} has {len(per_world)} world entries, expected at least {num_envs}."
                    )

                world_sites = list(per_world[env_idx])
                if len(world_sites) == 1:
                    env_candidates.append(world_sites[0])
                elif len(world_sites) == num_envs:
                    env_candidates.append(world_sites[env_idx])
                else:
                    raise ValueError(
                        f"{error_prefix} resolved {len(world_sites)} sites for env {env_idx}; "
                        "expected either one site or one site per env."
                    )

            if len(env_candidates) == 1:
                site_indices.append(env_candidates[0])
            elif len(env_candidates) == num_envs:
                site_indices.append(env_candidates[env_idx])
            else:
                raise ValueError(
                    f"RayCaster target '{prim_expr}' resolved {len(env_candidates)} candidate sites for env "
                    f"{env_idx}; expected one candidate."
                )

        return site_indices
