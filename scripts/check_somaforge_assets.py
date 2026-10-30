#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from somaforge_core import AssetManifest
from somaforge_core.robot_assets import canonical_g1_asset_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify canonical robot identity and registered assets.")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Allow absent external datasets/models; canonical robot assets remain required")
    args = parser.parse_args()
    robot_metadata = canonical_g1_asset_metadata()
    print(json.dumps(robot_metadata, indent=2, sort_keys=True))
    manifest = AssetManifest.load()
    errors = manifest.validate(require_exists=not args.allow_missing)
    if errors:
        raise SystemExit("\n".join(errors))
    robot_record = manifest.get("robot.g1.spherehand")
    expected_fields = {
        "asset_bundle_sha256": robot_metadata["asset_bundle_sha256"],
        "usd_bundle_sha256": robot_metadata["usd_bundle_sha256"],
    }
    for key, expected in expected_fields.items():
        if robot_record.metadata.get(key) != expected:
            raise SystemExit(
                f"robot asset manifest metadata mismatch: {key}={robot_record.metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    print(f"asset manifest: {len(manifest.ids())} entries validated")


if __name__ == "__main__":
    main()
