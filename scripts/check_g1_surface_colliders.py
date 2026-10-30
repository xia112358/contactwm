"""Inspect the authoritative surface revision; optionally realize a CPU solver model.

The CPU model check verifies geometry only. It supplies no contact truth/labels.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packages/somaforge_core'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--realize',action='store_true',help='Also import the authority USD into Newton and compile a CPU MuJoCo model')
    args=parser.parse_args()
    from pxr import Usd,UsdPhysics
    from somaforge_core.robot_assets import canonical_g1_asset_metadata
    from somaforge_core.collision_profiles import audit_surface_stage,audit_surface_mujoco
    asset=canonical_g1_asset_metadata(collision_profile='surface_v2')
    stage=Usd.Stage.Open(asset['usd_path']);stage.SetEditTarget(stage.GetSessionLayer())
    report=dict(robot_asset=asset,variants={})
    for variant in ('physx','mujoco','physics'):
        stage.GetDefaultPrim().GetVariantSet('Physics').SetVariantSelection(variant)
        report['variants'][variant]=audit_surface_stage(stage,stage.GetDefaultPrim())
    if args.realize:
        import newton
        from newton.solvers import SolverMuJoCo
        stage.GetDefaultPrim().GetVariantSet('Physics').SetVariantSelection('mujoco')
        UsdPhysics.Joint(stage.GetPrimAtPath('/g1_29dof/Physics/root_joint')).GetJointEnabledAttr().Set(False)
        builder=newton.ModelBuilder()
        # An external collider is necessary: a robot with all self-collisions
        # filtered and no terrain has empty collision masks in the solver.
        builder.add_ground_plane();builder.begin_world()
        builder.add_usd(stage,floating=True,load_visual_shapes=False,collapse_fixed_joints=False)
        builder.end_world()
        solver=SolverMuJoCo(builder.finalize('cpu'),use_mujoco_cpu=True)
        report['realized_solver']=dict(nq=solver.mj_model.nq,colliders=audit_surface_mujoco(solver.mj_model))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
