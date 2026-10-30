#!/usr/bin/env python3
"""Canonicalize raw or Motion Edit G1 poses with the training Newton articulation FK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SOMAFORGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOMAFORGE_ROOT / "packages" / "somaforge_core"))
sys.path.insert(0, str(SOMAFORGE_ROOT / "packages" / "motion_edit"))
sys.path.insert(0, str(SOMAFORGE_ROOT / "src" / "holosoma"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("inputs", nargs="*", type=Path, help="Input G1 motion NPZ files.")
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--output-fps", type=float, default=50.0)
parser.add_argument(
    "--input-format",
    choices=("omniretarget_qpos", "holosoma_joint_pos"),
    default="omniretarget_qpos",
    help="Input pose layout. Physics-retarget candidates use holosoma_joint_pos.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="Newton FK instance count. Only 1 is supported until batched articulation FK is reliable.",
)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.inputs = [path.expanduser().resolve() for path in args_cli.inputs]
if args_cli.output_dir is not None:
    args_cli.output_dir = args_cli.output_dir.expanduser().resolve()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from holosoma.config_values.robot import g1_29dof  # noqa: E402
from holosoma.simulator.isaaclab3_newton.backend import resolve_robot_usd  # noqa: E402
from holosoma.simulator.isaaclab3_newton.isaaclab3_newton import (  # noqa: E402
    spawn_newton_usd_with_floating_root,
)
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab_newton.assets import Articulation  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from somaforge_core.canonical_input import load_canonical_qpos_input  # noqa: E402
from somaforge_core import (  # noqa: E402
    G1_29DOF_JOINT_ORDER,
    body_velocities_from_pose,
    canonical_g1_asset_metadata,
    encode_kinematics_provenance,
    encode_robot_asset_json,
    newton_kinematics_provenance,
    sha256_file,
    validate_pose_velocity_consistency,
    validate_root_body_consistency,
)


def _build_robot(batch_size: int, device: str) -> Articulation:
    usd_path = resolve_robot_usd(
        str(SOMAFORGE_ROOT / "src" / "holosoma" / "holosoma" / "data" / "robots"),
        g1_29dof.asset,
    )
    for index in range(batch_size):
        sim_utils.create_prim(f"/World/envs/env_{index}", "Xform")
    cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_newton_usd_with_floating_root,
            usd_path=str(usd_path),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
        ),
        actuators={
            "all": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
                armature=0.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=1.0e6,
                velocity_limit=1.0e6,
                effort_limit_sim=1.0e9,
                velocity_limit_sim=1.0e6,
            )
        },
    )
    return Articulation(cfg)


def _canonicalize(path: Path, output: Path, robot: Articulation, device: str) -> None:
    qpos, qvel = load_canonical_qpos_input(
        path,
        input_format=args_cli.input_format,
        output_fps=args_cli.output_fps,
    )
    robot_joint_names = list(robot.joint_names)
    missing = [name for name in G1_29DOF_JOINT_ORDER if name not in robot_joint_names]
    if missing:
        raise ValueError(f"Newton articulation is missing canonical joints: {missing}")
    source_to_robot = [G1_29DOF_JOINT_ORDER.index(name) for name in robot_joint_names]
    body_names = list(robot.body_names)
    body_pos = np.empty((qpos.shape[0], len(body_names), 3), dtype=np.float32)
    body_quat = np.empty((qpos.shape[0], len(body_names), 4), dtype=np.float32)

    env_ids = torch.zeros(1, dtype=torch.int32, device=device)
    for frame in range(qpos.shape[0]):
        root_pose = np.concatenate((qpos[frame : frame + 1, 4:7], qpos[frame : frame + 1, [1, 2, 3, 0]]), axis=1)
        robot.write_root_pose_to_sim_index(root_pose=torch.as_tensor(root_pose, device=device), env_ids=env_ids)
        robot.write_joint_position_to_sim_index(
            position=torch.as_tensor(qpos[frame : frame + 1, 7:][:, source_to_robot], device=device), env_ids=env_ids
        )
        poses = robot.data.body_link_pose_w.torch[0].detach().cpu().numpy()
        body_pos[frame] = poses[..., :3]
        body_quat[frame] = poses[..., [6, 3, 4, 5]]

    body_lin_vel, body_ang_vel = body_velocities_from_pose(body_pos, body_quat, args_cli.output_fps)
    holosoma_joint_pos = np.concatenate((qpos[:, 4:7], qpos[:, :4], qpos[:, 7:]), axis=1)
    validate_root_body_consistency(
        holosoma_joint_pos,
        body_pos,
        body_quat,
        root_body_index=body_names.index("pelvis"),
    )
    validate_pose_velocity_consistency(
        body_pos,
        body_quat,
        body_lin_vel,
        body_ang_vel,
        args_cli.output_fps,
    )

    provenance = newton_kinematics_provenance(
        source_path=str(path.resolve()),
        source_sha256=sha256_file(path),
        output_fps=args_cli.output_fps,
        body_names=body_names,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fps=np.asarray([args_cli.output_fps], dtype=np.float32),
        joint_pos=holosoma_joint_pos,
        joint_vel=qvel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        joint_names=np.asarray(G1_29DOF_JOINT_ORDER),
        body_names=np.asarray(body_names),
        robot_asset_json=np.asarray(encode_robot_asset_json(canonical_g1_asset_metadata())),
        kinematics_provenance_json=np.asarray(encode_kinematics_provenance(provenance)),
    )
    print(f"Wrote {output} frames={qpos.shape[0]} bodies={len(body_names)} backend=NewtonFK")


def main() -> None:
    if not args_cli.inputs:
        raise ValueError("at least one input motion is required")
    if args_cli.output_dir is None:
        raise ValueError("--output-dir is required")
    if args_cli.batch_size != 1:
        raise ValueError("--batch-size must be 1: batched Newton articulation FK can return stale poses")
    outputs = [args_cli.output_dir.expanduser() / path.name for path in args_cli.inputs]
    for output in outputs:
        if output.exists() and not args_cli.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    device = str(args_cli.device)
    sim_cfg = SimulationCfg(
        dt=1.0 / float(args_cli.output_fps),
        device=device,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=64,
                nconmax=1,
                disable_contacts=True,
                use_mujoco_contacts=False,
                integrator="implicitfast",
            ),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )
    completed_outputs: list[Path] = []
    with build_simulation_context(sim_cfg=sim_cfg, device=device) as sim:
        robot = _build_robot(1, device)
        sim.reset()
        for source, output in zip(args_cli.inputs, outputs, strict=True):
            _canonicalize(source.expanduser(), output, robot, device)
            completed_outputs.append(output)
    if completed_outputs != outputs:
        raise RuntimeError("Newton canonicalization did not complete; inspect the simulation error above")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
