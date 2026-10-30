"""Physical collision migration and evidence compatibility, on the composed USD."""
import pytest

from somaforge_core.collision_profiles import (
    LEGACY, SURFACE, audit_surface_stage, contact_body_names, profile_metadata, require_profile,
)
from somaforge_core.robot_assets import canonical_g1_asset_metadata, canonical_g1_usd_path, validate_g1_asset_metadata


def test_collision_revision_keeps_source_identity_and_versions_physics():
    old=canonical_g1_asset_metadata()
    new=canonical_g1_asset_metadata(collision_profile=SURFACE)
    for field in ('asset_id','urdf_sha256','asset_bundle_sha256'):
        assert new[field]==old[field]
    assert new['base_usd_bundle_sha256']==old['usd_bundle_sha256']
    assert new['usd_bundle_sha256']!=old['usd_bundle_sha256']
    validate_g1_asset_metadata(new,context='surface evidence')
    with pytest.raises(ValueError,match='requires new execution'):
        require_profile(old,SURFACE,context='reference')
    with pytest.raises(ValueError,match='requires new execution'):
        require_profile(new,LEGACY,context='checkpoint')
    require_profile(old,LEGACY,context='old checkpoint')
    for bad in ({'collision_profile':SURFACE},dict(profile_metadata(SURFACE),collision_profile_sha256='bad')):
        with pytest.raises(ValueError,match='fingerprint'):
            require_profile(bad,SURFACE,context='bad evidence')


def test_surface_bindings_exclude_helper_bodies():
    names=contact_body_names(SURFACE)
    for side in ('left','right'):
        assert names[side+'_foot']==(side+'_ankle_roll_link',)
        assert names[side+'_hand']==(side+'_sphere_hand_link',)
        assert names[side+'_heel']==names[side+'_toe']==()
    assert len(contact_body_names(LEGACY)['left_foot'])>1


@pytest.mark.parametrize('variant',['physx','mujoco','physics'])
def test_surface_composition_has_zero_spheres_and_preserves_body_joint_inertia(variant):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdPhysics,UsdGeom
    old=Usd.Stage.Open(str(canonical_g1_usd_path()))
    new=Usd.Stage.Open(str(canonical_g1_usd_path(collision_profile=SURFACE)))
    # All changes stay in anonymous session layers; no asset is rewritten.
    for stage in (old,new):
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetDefaultPrim().GetVariantSet('Physics').SetVariantSelection(variant)
    assert audit_surface_stage(new,new.GetDefaultPrim())==dict(mesh=29,sphere=0,cylinder=4,other=0,rigid_bodies=53)
    def physical_signature(stage):
        result={}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
                # Geometry is the only intentional difference. Compare authored
                # body transforms, mass tensors and every joint attribute.
                result[str(prim.GetPath())]={a.GetName():repr(a.Get()) for a in prim.GetAttributes()
                    if a.GetName().startswith(('physics:','physxRigidBody:','xformOp:'))}
                result[str(prim.GetPath())]['relations']={r.GetName():tuple(map(str,r.GetTargets())) for r in prim.GetRelationships()}
        return result
    assert physical_signature(new)==physical_signature(old)
    assert sum(p.IsA(UsdGeom.Sphere) and p.HasAPI(UsdPhysics.CollisionAPI) for p in old.Traverse())==14
