from __future__ import annotations

from copy import deepcopy

import pytest
from somaforge_core.robot_assets import (
    G1_SPHEREHAND_BUNDLE_SHA256,
    G1_SPHEREHAND_SHA256,
    G1_SPHEREHAND_USD_BUNDLE_SHA256,
    canonical_g1_asset_metadata,
    canonical_g1_source_metadata,
    decode_robot_asset_json,
    encode_robot_asset_json,
    validate_g1_asset_metadata,
)


def test_canonical_g1_asset_bundle() -> None:
    metadata = canonical_g1_asset_metadata()
    assert metadata["urdf_sha256"] == G1_SPHEREHAND_SHA256
    assert metadata["asset_bundle_sha256"] == G1_SPHEREHAND_BUNDLE_SHA256
    assert metadata["usd_bundle_sha256"] == G1_SPHEREHAND_USD_BUNDLE_SHA256


def test_legacy_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy wrong-URDF"):
        validate_g1_asset_metadata(None, context="test artifact")


def test_motion_asset_metadata_requires_only_authoritative_urdf() -> None:
    metadata = decode_robot_asset_json(encode_robot_asset_json(), context="motion artifact")

    assert metadata == canonical_g1_source_metadata()
    assert "usd_bundle_sha256" not in metadata


def test_wrong_fingerprint_is_rejected() -> None:
    metadata = deepcopy(canonical_g1_source_metadata())
    metadata["urdf_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="incompatible robot asset"):
        validate_g1_asset_metadata(metadata, context="test artifact")


def test_present_usd_fingerprint_is_still_validated() -> None:
    metadata = deepcopy(canonical_g1_source_metadata())
    metadata["usd_bundle_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="incompatible G1 USD bundle"):
        validate_g1_asset_metadata(metadata, context="simulator artifact")
