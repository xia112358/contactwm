"""Moving a checkout preserves identity; changing physics/assets does not."""
import json
import pytest
from somaforge_core.robot_assets import canonical_g1_asset_metadata
from supermimic.lowstep import validate_checkpoint
from supermimic.contact_response import metadata as response_metadata


@pytest.mark.parametrize('profile', ['legacy_v1', 'surface_v2'])
def test_checkpoint_relocation_preserves_canonical_identity(profile):
    asset = canonical_g1_asset_metadata(collision_profile=profile)
    expected = dict(robot_asset_json=json.dumps(asset), task_manifest_sha256='unchanged')
    moved = {**asset, 'urdf_path': '/old-checkout/canonical.urdf', 'usd_path': '/old-checkout/canonical.usda'}
    assert validate_checkpoint({**expected, 'robot_asset_json': json.dumps(moved)}, expected) is None


def test_relocation_does_not_accept_changed_asset_bytes():
    asset = canonical_g1_asset_metadata()
    expected = dict(robot_asset_json=json.dumps(asset))
    changed = {**asset, 'usd_path': '/old-checkout/robot.usda', 'usd_bundle_sha256': '0'*64}
    with pytest.raises(ValueError, match='incompatible G1 USD bundle'):
        validate_checkpoint(dict(robot_asset_json=json.dumps(changed)), expected)


def test_relocation_does_not_accept_changed_task_or_contact_physics():
    asset = canonical_g1_asset_metadata()
    expected = dict(robot_asset_json=json.dumps(asset), task_manifest_sha256='same')
    with pytest.raises(ValueError, match='task_manifest_sha256'):
        validate_checkpoint({**expected, 'task_manifest_sha256': 'different'}, expected)
    with pytest.raises(ValueError, match='Contact-response physics changed'):
        validate_checkpoint(expected, {**expected, 'contact_response_contract': response_metadata('manifold_v2')})


def test_relocation_does_not_implicitly_transfer_collision_profiles():
    source = dict(robot_asset_json=json.dumps(canonical_g1_asset_metadata()))
    target = dict(robot_asset_json=json.dumps(canonical_g1_asset_metadata(collision_profile='surface_v2')))
    with pytest.raises(ValueError, match='collision contract mismatch'):
        validate_checkpoint(source, target)
