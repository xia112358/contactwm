import pytest

from holosoma.config_types.simulator import DepthRayCameraConfig
from holosoma.simulator.isaacsim.quaternion_utils import config_wxyz_to_xyzw, depth_rotation_to_xyzw


@pytest.mark.parametrize("wxyz,xyzw", [
    ((1., 0., 0., 0.), (0., 0., 0., 1.)),
    ((.5, .5, -.5, .5), (.5, -.5, .5, .5)),
    ((0., 1., 0., 0.), (1., 0., 0., 0.)),
])
def test_asset_and_explicit_camera_wxyz(wxyz, xyzw):
    assert config_wxyz_to_xyzw(wxyz) == xyzw
    assert depth_rotation_to_xyzw(wxyz, "wxyz") == xyzw


def test_existing_checkpoint_camera_keeps_physical_frame():
    cfg = DepthRayCameraConfig()
    assert cfg.rotation_convention == "legacy_xyzw"
    assert depth_rotation_to_xyzw(cfg.offset_rot_wxyz, cfg.rotation_convention) == cfg.offset_rot_wxyz
    # This old value means Rx(pi), not identity; silently converting it changes depth observations.
    assert depth_rotation_to_xyzw((1., 0., 0., 0.), cfg.rotation_convention) == (1., 0., 0., 0.)


@pytest.mark.parametrize("convention", ["wxyz", "legacy_xyzw", "unknown"])
def test_invalid_camera_rotation_fails(convention):
    with pytest.raises(ValueError):
        depth_rotation_to_xyzw((1., 0., 0.), convention)
