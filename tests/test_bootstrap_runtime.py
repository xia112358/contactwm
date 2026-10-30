"""Bootstrap a different checkout using only committed assets and stdlib Python."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
URDF_RELATIVE = Path("src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf")
USD_RELATIVE = Path("src/holosoma/holosoma/data/robots/converted_rank0/g1_29dof_spherehand")
CONTRACT_RELATIVE = Path("packages/somaforge_core/somaforge_core/robot_assets.py")
EXPECTED_USD_HASH = "2df7b7c6ea4a906c81f5bbee91ea49f7df77622089b676516856fb0cea2c010a"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "relocated_checkout"
    (root / "packages/motion_edit").mkdir(parents=True)
    # Copy only the authoritative URDF's actual mesh closure, not local
    # converted assets, runtime links, packages or machine-specific sidecars.
    source = REPOSITORY / URDF_RELATIVE
    paths = {source, *(source.parent / node.attrib["filename"]
                       for node in ET.parse(source).getroot().iter("mesh"))}
    paths.update({REPOSITORY / CONTRACT_RELATIVE, REPOSITORY / "scripts/bootstrap_runtime.py"})
    for path in paths:
        target = root / path.relative_to(REPOSITORY)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copytree(REPOSITORY / "assets/g1_spherehand_usd", root / "assets/g1_spherehand_usd")
    return root


def run_bootstrap(root, *args, success=True):
    # -S excludes site-packages, including NumPy: installation must not be
    # required before first-run directory and canonical asset preparation.
    result = subprocess.run([sys.executable, "-S", str(root / "scripts/bootstrap_runtime.py"), *map(str, args)],
                            cwd=root.parent, capture_output=True, text=True)
    if success:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0, result.stdout
    return result.stderr


def test_fresh_checkout_bootstraps_without_installed_packages_and_is_idempotent(checkout):
    result = run_bootstrap(checkout)
    assert Path(result["root"]) == checkout
    assert Path(result["data_root"]) == checkout / "runtime/current"
    assert not (checkout / "runtime/current").is_symlink()
    assert len(result["restored_usd_files"]) == 9
    assert result["robot_asset"]["usd_bundle_sha256"] == EXPECTED_USD_HASH
    assert Path(result["robot_asset"]["urdf_path"]) == checkout / URDF_RELATIVE
    assert Path(result["robot_asset"]["usd_path"]).is_relative_to(checkout)
    assert (checkout / "data").readlink() == Path("runtime/current/holosoma/data")
    assert (checkout / "packages/motion_edit/data").readlink() == Path("../../runtime/current/motion_edit/data")
    assert (checkout / "runtime/current/motions").is_dir()
    assert (checkout / "runtime/current/models").is_dir()
    assets = sorted(path for path in (checkout / USD_RELATIVE).rglob("*") if path.is_file())
    original = {path: (file_hash(path), path.stat().st_mtime_ns) for path in assets}
    repeated = run_bootstrap(checkout)
    assert repeated["restored_usd_files"] == []
    assert {path: (file_hash(path), path.stat().st_mtime_ns) for path in assets} == original
    for original_path in (checkout / "assets/g1_spherehand_usd").rglob("*"):
        if original_path.is_file():
            restored = checkout / USD_RELATIVE / original_path.relative_to(checkout / "assets/g1_spherehand_usd")
            assert file_hash(original_path) == file_hash(restored)


def test_explicit_external_root_creates_links_and_cannot_replace_them(checkout):
    external = checkout.parent / "separate_disk"
    result = run_bootstrap(checkout, "--root", checkout, "--data-root", external)
    assert result["data_root"] == str(external)
    assert (checkout / "runtime/current").is_symlink()
    assert (checkout / "runtime/current").resolve() == external
    assert (external / "holosoma/logs").is_dir()
    assert (external / "motion_edit/data").is_dir()
    assert run_bootstrap(checkout, "--data-root", external)["restored_usd_files"] == []
    other = checkout.parent / "another_disk"
    error = run_bootstrap(checkout, "--data-root", other, success=False)
    assert "Refusing to replace" in error
    assert not other.exists()
    assert (checkout / "runtime/current").resolve() == external


def test_existing_output_directory_is_never_replaced(checkout):
    existing = checkout / "logs"
    existing.mkdir()
    note = existing / "keep.txt"
    note.write_text("user output")
    error = run_bootstrap(checkout, success=False)
    assert "Refusing to replace" in error
    assert not existing.is_symlink()
    assert note.read_text() == "user output"
    assert not (checkout / "runtime/current").exists()
    assert not (checkout / USD_RELATIVE).exists()


def test_conflicting_asset_and_corrupt_source_are_rejected_without_overwrite(checkout):
    bad = checkout / USD_RELATIVE / "g1_29dof_spherehand.usda"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"different existing asset")
    error = run_bootstrap(checkout, success=False)
    assert "Existing canonical G1 USD differs" in error
    assert bad.read_bytes() == b"different existing asset"
    assert not (checkout / "runtime/current").exists()
    # Independent bad-input check: even an existing destination conflict may
    # not cause the pinned source bundle to be accepted without validation.
    source = checkout / "assets/g1_spherehand_usd/g1_29dof_spherehand.usda"
    source.write_bytes(b"incomplete source download")
    error = run_bootstrap(checkout, success=False)
    assert "Pinned G1 USD bundle fingerprint mismatch" in error
    assert bad.read_bytes() == b"different existing asset"


def test_preexisting_matching_asset_is_not_rewritten_and_wrong_sidecar_is_rejected(checkout):
    shutil.copytree(checkout / "assets/g1_spherehand_usd", checkout / USD_RELATIVE)
    asset = checkout / USD_RELATIVE / "g1_29dof_spherehand.usda"
    before = (file_hash(asset), asset.stat().st_mtime_ns)
    assert run_bootstrap(checkout)["restored_usd_files"] == []
    assert (file_hash(asset), asset.stat().st_mtime_ns) == before
    sidecar = checkout / USD_RELATIVE / "somaforge_robot_asset.json"
    payload = json.loads(sidecar.read_text())
    payload["usd_bundle_sha256"] = "wrong"
    sidecar.write_text(json.dumps(payload))
    error = run_bootstrap(checkout, success=False)
    assert "incompatible usd_bundle_sha256" in error
    assert json.loads(sidecar.read_text())["usd_bundle_sha256"] == "wrong"
    assert (file_hash(asset), asset.stat().st_mtime_ns) == before
