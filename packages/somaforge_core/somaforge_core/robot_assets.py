from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

G1_SPHEREHAND_ASSET_ID = "g1_29dof_spherehand_v1"
G1_SPHEREHAND_SHA256 = "f869ea6547fd18f40558d91dfc5ad76347f73ba88351c5e7c81e98868ab60937"
G1_SPHEREHAND_BUNDLE_SHA256 = "30cb9ed0b83795ed8861a92e2b800357fdd1067b0986aa997e078de720233f20"
G1_SPHEREHAND_USD_BUNDLE_SHA256 = "2df7b7c6ea4a906c81f5bbee91ea49f7df77622089b676516856fb0cea2c010a"
G1_SPHEREHAND_URDF_RELATIVE = Path("src/holosoma/holosoma/data/robots/g1/g1_29dof_spherehand.urdf")
G1_SPHEREHAND_USD_RELATIVE = Path(
    "src/holosoma/holosoma/data/robots/converted_rank0/g1_29dof_spherehand/g1_29dof_spherehand.usda"
)


def somaforge_root() -> Path:
    configured = os.environ.get("SOMAFORGE_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _is_somaforge_root(root):
            raise FileNotFoundError(f"SOMAFORGE_ROOT is not a SomaForge checkout: {root}")
        return root

    for parent in Path(__file__).resolve().parents:
        if _is_somaforge_root(parent):
            return parent
    raise FileNotFoundError("could not locate the SomaForge checkout; set SOMAFORGE_ROOT")


def canonical_g1_urdf_path() -> Path:
    return somaforge_root() / G1_SPHEREHAND_URDF_RELATIVE


def canonical_g1_usd_path(*, collision_profile: str = "legacy_v1") -> Path:
    from .collision_profiles import LEGACY, SURFACE_USD_RELATIVE, profile_metadata
    profile_metadata(collision_profile)
    if collision_profile != LEGACY:
        return somaforge_root() / SURFACE_USD_RELATIVE
    return somaforge_root() / G1_SPHEREHAND_USD_RELATIVE


def canonical_g1_source_metadata() -> dict[str, str]:
    urdf = canonical_g1_urdf_path()
    if not urdf.is_file():
        raise FileNotFoundError(f"canonical G1 sphere-hand URDF is missing: {urdf}")
    actual_sha256 = _sha256(urdf)
    if actual_sha256 != G1_SPHEREHAND_SHA256:
        raise ValueError(
            "canonical G1 sphere-hand URDF fingerprint mismatch: "
            f"expected {G1_SPHEREHAND_SHA256}, got {actual_sha256} at {urdf}"
        )
    actual_bundle_sha256 = _urdf_bundle_sha256(urdf)
    if G1_SPHEREHAND_BUNDLE_SHA256 and actual_bundle_sha256 != G1_SPHEREHAND_BUNDLE_SHA256:
        raise ValueError(
            "canonical G1 sphere-hand asset bundle fingerprint mismatch: "
            f"expected {G1_SPHEREHAND_BUNDLE_SHA256}, got {actual_bundle_sha256}"
        )
    _validate_foot_collision_topology(urdf)
    return {
        "asset_id": G1_SPHEREHAND_ASSET_ID,
        "urdf_path": str(urdf),
        "urdf_sha256": actual_sha256,
        "asset_bundle_sha256": actual_bundle_sha256,
    }


def build_g1_asset_metadata(usd_path: str | Path) -> dict[str, str]:
    usd = Path(usd_path).expanduser().resolve()
    if not usd.is_file():
        raise FileNotFoundError(f"canonical G1 sphere-hand USD is missing: {usd}")
    return {
        **canonical_g1_source_metadata(),
        "usd_path": str(usd),
        "usd_bundle_sha256": _usd_bundle_sha256(usd),
    }


def canonical_g1_asset_metadata(*, collision_profile: str = "legacy_v1") -> dict[str, str]:
    from .collision_profiles import LEGACY, SURFACE_USD_SHA256, profile_metadata
    profile = profile_metadata(collision_profile)
    if collision_profile != LEGACY:
        base = canonical_g1_asset_metadata()
        path = canonical_g1_usd_path(collision_profile=collision_profile)
        if _sha256(path) != SURFACE_USD_SHA256:
            raise ValueError("Canonical surface-v2 USD fingerprint mismatch: " + str(path))
        return {**base, **profile, "usd_path": str(path),
                "base_usd_bundle_sha256": base["usd_bundle_sha256"],
                "usd_bundle_sha256": _usd_bundle_sha256(path)}
    metadata = build_g1_asset_metadata(canonical_g1_usd_path())
    actual = metadata["usd_bundle_sha256"]
    if G1_SPHEREHAND_USD_BUNDLE_SHA256 and actual != G1_SPHEREHAND_USD_BUNDLE_SHA256:
        raise ValueError(
            "canonical G1 sphere-hand USD bundle fingerprint mismatch: "
            f"expected {G1_SPHEREHAND_USD_BUNDLE_SHA256}, got {actual}"
        )
    return metadata


def validate_g1_asset_metadata(metadata: Mapping[str, Any] | None, *, context: str) -> None:
    if metadata is None:
        raise ValueError(
            f"{context} has no robot asset fingerprint and is treated as legacy wrong-URDF data; regenerate it"
        )
    expected = canonical_g1_source_metadata()
    actual_id = str(metadata.get("asset_id") or "")
    actual_sha256 = str(metadata.get("urdf_sha256") or "")
    actual_bundle_sha256 = str(metadata.get("asset_bundle_sha256") or "")
    if (
        actual_id != expected["asset_id"]
        or actual_sha256 != expected["urdf_sha256"]
        or actual_bundle_sha256 != expected["asset_bundle_sha256"]
    ):
        raise ValueError(
            f"{context} uses an incompatible robot asset "
            f"(asset_id={actual_id!r}, urdf_sha256={actual_sha256!r}, "
            f"asset_bundle_sha256={actual_bundle_sha256!r}); expected {expected['asset_id']} / "
            f"{expected['urdf_sha256']} / {expected['asset_bundle_sha256']}"
        )
    actual_usd_bundle_sha256 = str(metadata.get("usd_bundle_sha256") or "")
    from .collision_profiles import LEGACY, require_profile
    collision_profile = str(metadata.get("collision_profile", LEGACY))
    require_profile(metadata, collision_profile, context=context)
    if collision_profile != LEGACY:
        if not actual_usd_bundle_sha256 or metadata.get("base_usd_bundle_sha256") != G1_SPHEREHAND_USD_BUNDLE_SHA256:
            raise ValueError(f"{context}: missing/incompatible surface collision asset bundle fingerprint")
    if actual_usd_bundle_sha256:
        expected_usd_bundle_sha256 = canonical_g1_asset_metadata(collision_profile=collision_profile)["usd_bundle_sha256"]
        if actual_usd_bundle_sha256 != expected_usd_bundle_sha256:
            raise ValueError(
                f"{context} uses an incompatible G1 USD bundle "
                f"(usd_bundle_sha256={actual_usd_bundle_sha256!r}); "
                f"expected {expected_usd_bundle_sha256!r}"
            )


def encode_robot_asset_json(metadata: Mapping[str, Any] | None = None) -> str:
    return json.dumps(dict(metadata or canonical_g1_source_metadata()), sort_keys=True)


def decode_robot_asset_json(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        validate_g1_asset_metadata(None, context=context)
    raw = value.item() if hasattr(value, "item") else value
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid robot_asset_json") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} robot_asset_json must decode to an object")
    validate_g1_asset_metadata(metadata, context=context)
    return metadata


def validate_canonical_g1_urdf(path: str | Path) -> dict[str, str]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"G1 URDF is missing: {candidate}")
    actual = _sha256(candidate)
    if actual != G1_SPHEREHAND_SHA256:
        raise ValueError(
            f"G1 tool uses the wrong URDF: expected SHA256 {G1_SPHEREHAND_SHA256}, got {actual} at {candidate}"
        )
    bundle = _urdf_bundle_sha256(candidate)
    if bundle != G1_SPHEREHAND_BUNDLE_SHA256:
        raise ValueError(
            f"G1 tool uses an incompatible mesh bundle: expected {G1_SPHEREHAND_BUNDLE_SHA256}, "
            f"got {bundle} at {candidate.parent}"
        )
    _validate_foot_collision_topology(candidate)
    return canonical_g1_source_metadata()


def validate_g1_robot_config(robot_config: Any) -> dict[str, str] | None:
    asset = robot_config.asset
    robot_type = str(asset.robot_type)
    urdf_file = str(asset.urdf_file)
    if not (robot_type.startswith("g1_") or urdf_file.startswith("g1/")):
        return None
    expected_urdf = "g1/g1_29dof_spherehand.urdf"
    if urdf_file != expected_urdf:
        raise ValueError(f"G1 runs must use the canonical sphere-hand URDF; got urdf={urdf_file!r}")
    return canonical_g1_asset_metadata(collision_profile=getattr(asset, "collision_profile", "legacy_v1"))


def _is_somaforge_root(path: Path) -> bool:
    return (
        (path / "src/holosoma/holosoma").is_dir()
        and (path / "packages/motion_edit").is_dir()
        and (path / "packages/somaforge_core").is_dir()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _urdf_bundle_sha256(urdf: Path) -> str:
    asset_root = urdf.parent.resolve()
    mesh_paths: set[Path] = set()
    for mesh in ET.parse(urdf).getroot().iter("mesh"):  # noqa: S314 - trusted canonical local asset
        filename = mesh.get("filename")
        if not filename:
            continue
        resolved = (asset_root / filename).resolve()
        if not resolved.is_relative_to(asset_root):
            raise ValueError(f"URDF mesh escapes the canonical asset root: {filename}")
        if not resolved.is_file():
            raise FileNotFoundError(f"URDF mesh is missing: {resolved}")
        mesh_paths.add(resolved)

    digest = hashlib.sha256()
    for path in [urdf.resolve(), *sorted(mesh_paths)]:
        relative = path.relative_to(asset_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _usd_bundle_sha256(usd: Path) -> str:
    root = usd.parent.resolve()
    files = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )
    if usd.resolve() not in files:
        raise FileNotFoundError(f"USD entry point is not part of its bundle: {usd}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_foot_collision_topology(urdf: Path) -> None:
    root = ET.parse(urdf).getroot()  # noqa: S314 - trusted canonical local asset
    links = {link.get("name", ""): link for link in root.findall("link")}
    collision_owners: dict[str, list[str]] = {}
    for link_name, link in links.items():
        for collision in link.findall("collision"):
            name = collision.get("name", "")
            if name:
                collision_owners.setdefault(name, []).append(link_name)
    duplicates = {name: owners for name, owners in collision_owners.items() if len(owners) > 1}
    if duplicates:
        raise ValueError(f"canonical G1 URDF has duplicate collision names: {duplicates}")

    for side in ("left", "right"):
        ankle = links[f"{side}_ankle_roll_link"]
        direct_spheres = [
            collision.get("name", "")
            for collision in ankle.findall("collision")
            if collision.find("geometry/sphere") is not None
        ]
        if direct_spheres:
            raise ValueError(f"{side} ankle has duplicate direct sphere collisions: {direct_spheres}")
        for index in range(1, 6):
            name = f"{side}_ankle_roll_sphere_{index}"
            sphere_link = links.get(f"{name}_link")
            collisions = [] if sphere_link is None else sphere_link.findall("collision")
            if len(collisions) != 1 or collisions[0].find("geometry/sphere") is None:
                raise ValueError(f"canonical G1 URDF requires exactly one collision sphere for {name}")
