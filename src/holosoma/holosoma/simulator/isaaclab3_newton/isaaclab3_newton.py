from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import find_matching_prims, get_current_stage
from loguru import logger
from pxr import PhysxSchema, Usd, UsdPhysics

from holosoma.simulator.isaaclab3_newton.backend import build_newton_physics_cfg, resolve_robot_usd
from holosoma.simulator.isaacsim.isaacsim import IsaacSim


_SELF_COLLISION_FILTER_HANDLE = None


def _register_self_collision_filters(
    exclude_kinematic_distance: int = 3,
) -> None:
    """Register pre-finalize adjacent/cosmetic self-collision filters."""
    global _SELF_COLLISION_FILTER_HANDLE
    if _SELF_COLLISION_FILTER_HANDLE is not None:
        return

    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from isaaclab.physics import PhysicsEvent
    except ImportError:
        return

    def _apply(_event=None):
        from isaaclab.sim import SimulationContext
        from isaaclab_newton.physics import NewtonBuilderCfg

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("Newton self-collision filtering requires an active simulation context.")
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        from somaforge_core.newton_collision_filter import apply_self_collision_filters_to_builder

        body_pairs, shape_pairs = apply_self_collision_filters_to_builder(
            builder, exclude_kinematic_distance=exclude_kinematic_distance
        )
        logger.info(
            f"Newton ModelBuilder self-collision filter: excluded {body_pairs} body pair(s), "
            f"{shape_pairs} shape pair(s) (movable-joint distance threshold={exclude_kinematic_distance})."
        )

    _SELF_COLLISION_FILTER_HANDLE = NewtonManager.register_callback(
        _apply, PhysicsEvent.MODEL_INIT, order=100, name="holosoma_self_collision_filter", wrap_weak_ref=False
    )


def spawn_newton_usd_with_floating_root(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)

    articulation_props = getattr(cfg, "articulation_props", None)
    if articulation_props is None:
        return prim

    stage = get_current_stage()
    self_collision_count = 0
    disabled_count = 0
    for robot_prim in find_matching_prims(prim_path, stage=stage):
        if articulation_props.enabled_self_collisions is not None:
            for descendant in Usd.PrimRange(robot_prim):
                if not descendant.HasAPI(UsdPhysics.ArticulationRootAPI):
                    continue
                physx_articulation = PhysxSchema.PhysxArticulationAPI(descendant)
                if not physx_articulation:
                    physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(descendant)
                physx_articulation.CreateEnabledSelfCollisionsAttr().Set(
                    bool(articulation_props.enabled_self_collisions)
                )
                self_collision_count += 1

        if articulation_props.fix_root_link is not False:
            continue
        root_joint_path = f"{robot_prim.GetPath()}/Physics/root_joint"
        root_joint_prim = stage.GetPrimAtPath(root_joint_path)
        if not root_joint_prim.IsValid():
            continue

        root_joint = UsdPhysics.Joint(root_joint_prim)
        if not root_joint:
            continue

        root_joint.GetJointEnabledAttr().Set(False)
        disabled_count += 1

    if self_collision_count:
        logger.info(
            f"Set Newton self collisions to {bool(articulation_props.enabled_self_collisions)} "
            f"on {self_collision_count} articulation root(s)."
        )
    if disabled_count:
        logger.info(f"Disabled {disabled_count} Newton root fixed joint(s) for floating-base articulation.")

    return prim


class IsaacLab3Newton(IsaacSim):
    """Isaac Lab 3.0 wrapper using Newton's MJWarp physics backend."""

    def __init__(self, tyro_config, terrain_manager, device):
        if tyro_config.robot.asset.enable_self_collisions:
            _register_self_collision_filters(exclude_kinematic_distance=3)
        super().__init__(tyro_config, terrain_manager, device)

    def _build_physics_cfg(self):
        return build_newton_physics_cfg(self.simulator_config)

    def _build_robot_spawn_cfg(
        self,
        asset_root: str,
        robot_asset_cfg: Any,
        robot_rigid_props: sim_utils.RigidBodyPropertiesCfg,
        robot_articulation_props: sim_utils.ArticulationRootPropertiesCfg,
    ):
        usd_path = resolve_robot_usd(asset_root, robot_asset_cfg)
        logger.info(f"Using pre-converted Lab3/Newton robot USD: {usd_path}")
        return sim_utils.UsdFileCfg(
            func=spawn_newton_usd_with_floating_root,
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=robot_rigid_props,
            articulation_props=robot_articulation_props,
        )
