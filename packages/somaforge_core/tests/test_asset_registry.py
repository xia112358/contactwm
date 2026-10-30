from __future__ import annotations

import json
from pathlib import Path

import pytest

from somaforge_core import AssetManifest
from somaforge_core.asset_registry import sha256_file


@pytest.fixture
def external_manifest(tmp_path):
    checkout = tmp_path / "code"
    data = tmp_path / "data"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "runtime").mkdir()
    data.mkdir()
    motion = data / "motion.npz"
    motion.write_bytes(b"unchanged motion fixture")
    (checkout / "runtime/current").symlink_to(data, target_is_directory=True)
    path = checkout / "configs/assets_manifest.json"
    payload = {
        "schema": "somaforge_asset_manifest_v1", "root": "..",
        "external_roots": ["runtime/current"],
        "assets": [{"id": "motion", "kind": "motion",
                    "files": {"default": "runtime/current/motion.npz"},
                    "sha256": {"default": sha256_file(motion)}}],
    }
    path.write_text(json.dumps(payload))
    return path, payload, checkout, data


def test_declared_external_asset_preserves_hash_verification(external_manifest):
    path, payload, checkout, data = external_manifest
    manifest = AssetManifest.load(path)
    assert manifest.resolve("motion") == data / "motion.npz"
    assert manifest.logical_path(data / "motion.npz") == Path("runtime/current/motion.npz")
    assert manifest.logical_path(checkout / "runtime/current/motion.npz") == Path("runtime/current/motion.npz")
    (data / "motion.npz").write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest.resolve("motion")


def test_undeclared_external_root_is_rejected(external_manifest):
    path, payload, checkout, data = external_manifest
    payload.pop("external_roots")
    with pytest.raises(ValueError, match="escapes manifest root"):
        AssetManifest(path, payload).resolve("motion")


@pytest.mark.parametrize("case", ["absolute", "unregistered_alias", "traversal", "escaping_symlink"])
def test_declaring_data_root_does_not_allow_other_escapes(external_manifest, case):
    path, payload, checkout, data = external_manifest
    outside = data.parent / "secret.txt"
    outside.write_bytes(b"outside declared roots")
    if case == "absolute":
        value = str(data / "motion.npz")
    elif case == "unregistered_alias":
        (checkout / "unregistered").symlink_to(data, target_is_directory=True)
        value = "unregistered/motion.npz"
    elif case == "traversal":
        value = "runtime/current/../secret.txt"
    else:
        (data / "escape.txt").symlink_to(outside)
        value = "runtime/current/escape.txt"
    payload["assets"][0]["files"]["default"] = value
    with pytest.raises(ValueError, match="escapes manifest root"):
        AssetManifest(path, payload).resolve("motion")
    with pytest.raises(ValueError, match="outside declared roots"):
        AssetManifest(path, payload).logical_path(outside)


@pytest.mark.parametrize("roots", ["runtime/current", ["../data"], ["/data"], ["."], [""], [None],
                                    ["runtime/current", "runtime/current"]])
def test_external_root_declaration_requires_explicit_relative_directories(external_manifest, roots):
    path, payload, checkout, data = external_manifest
    payload["external_roots"] = roots
    with pytest.raises(ValueError):
        AssetManifest(path, payload)


def test_manifest_resolves_verified_robot_asset() -> None:
    manifest = AssetManifest.load()
    path = manifest.resolve("robot.g1.spherehand", "urdf")
    assert path.name == "g1_29dof_spherehand.urdf"


def test_manifest_does_not_register_removed_legacy_asset() -> None:
    manifest = AssetManifest.load()
    assert "legacy.wrong-urdf" not in manifest.ids()
    with pytest.raises(KeyError, match="not registered"):
        manifest.resolve("legacy.wrong-urdf", "root")


def test_manifest_detects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "somaforge_asset_manifest_v1",
                "root": ".",
                "assets": [
                    {"id": "duplicate", "kind": "motion", "files": {}},
                    {"id": "duplicate", "kind": "motion", "files": {}},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate asset id"):
        AssetManifest.load(path)
