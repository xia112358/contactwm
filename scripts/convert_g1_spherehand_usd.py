#!/usr/bin/env python3
"""Convert the canonical SomaForge G1 sphere-hand URDF for Isaac Lab/Newton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOMAFORGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOMAFORGE_ROOT / "packages" / "somaforge_core"))

from isaaclab.app import AppLauncher  # noqa: E402
from somaforge_core import (  # noqa: E402
    build_g1_asset_metadata,
    canonical_g1_source_metadata,
    canonical_g1_urdf_path,
)

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402


def main() -> None:
    canonical_g1_source_metadata()
    urdf_path = canonical_g1_urdf_path()
    robot_root = urdf_path.parents[1]
    output_root = robot_root / "converted_rank0"
    output_root.mkdir(parents=True, exist_ok=True)
    generated_path = URDFImporter(
        URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=str(output_root),
            collision_from_visuals=False,
            collision_type="Convex Hull",
            allow_self_collision=False,
        )
    ).import_urdf()
    if not generated_path:
        raise RuntimeError("Isaac Sim URDF importer returned no generated USD path")

    usd_path = Path(generated_path).resolve()
    if usd_path.suffix not in {".usd", ".usda"}:
        raise RuntimeError(f"URDF conversion returned an unexpected path: {usd_path}")
    if usd_path.parent.name != urdf_path.stem:
        expected = output_root / urdf_path.stem / usd_path.name
        expected.parent.mkdir(parents=True, exist_ok=True)
        usd_path.rename(expected)
        usd_path = expected.resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"URDF conversion did not produce the expected USD: {usd_path}")
    metadata = build_g1_asset_metadata(usd_path)
    sidecar = usd_path.parent / "somaforge_robot_asset.json"
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated canonical G1 sphere-hand USD: {usd_path}")
    print(f"Asset fingerprint: {sidecar}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
