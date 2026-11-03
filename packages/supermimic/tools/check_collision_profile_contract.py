"""CPU checks for physical checkpoint isolation and explicit weight transfer."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy_model',type=Path,help='Optional old MJB: ensure surface-v2 rejects its missing/mismatched sidecar')
    args=parser.parse_args()
    from somaforge_core.robot_assets import canonical_g1_asset_metadata
    from supermimic.lowstep import checkpoint_metadata,validate_checkpoint
    base=canonical_g1_asset_metadata()
    new=canonical_g1_asset_metadata(collision_profile='surface_v2')
    spec=dict(_sha256='fixture-task',robot_asset=base,schema='supermimic_somaforge_lowstep_v1',
              terrain_layout={},_contact_model='force6')
    legacy=checkpoint_metadata(spec)
    modern=checkpoint_metadata(dict(spec,_execution_robot_asset=new,_collision_profile='surface_v2'))
    assert validate_checkpoint(legacy,legacy) is None
    assert validate_checkpoint(modern,modern) is None
    def rejected(source,target,**kw):
        try:validate_checkpoint(source,target,**kw)
        except ValueError:return
        raise AssertionError('Incompatible physical contract was accepted')
    rejected(legacy,modern);rejected(modern,legacy)
    transfer=validate_checkpoint(legacy,modern,allow_collision_transfer=True)
    assert transfer['role']=='initialization_only' and not transfer['actual_execution_validated']
    bad=deepcopy(legacy);bad['task_manifest_sha256']='other-task'
    rejected(bad,modern,allow_collision_transfer=True)
    bad=deepcopy(legacy);bad['robot_asset_json']='{}'
    rejected(bad,modern,allow_collision_transfer=True)
    if args.legacy_model:
        from supermimic.contact_surface import CollisionSurface
        try:CollisionSurface(args.legacy_model,None,collision_profile='surface_v2')
        except ValueError:pass
        else:raise AssertionError('Legacy geometry was accepted as a surface-v2 atlas')
    print('PASS physical checkpoint isolation, explicit initialization transfer and unchanged task checks')


if __name__=='__main__':main()
