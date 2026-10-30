#!/usr/bin/env python3
"""Validate a canonical motion/terrain manifest and all referenced hashes."""

from __future__ import annotations

import argparse

from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="runtime/current/manifests/omniretarget_baseline_29.json",
    )
    args = parser.parse_args()
    payload = load_motion_terrain_manifest(args.manifest)
    print(
        f"Motion manifest valid: {len(payload['motion_files'])} motions, "
        f"{len(payload['terrains'])} terrains, backend={payload.get('kinematics_backend')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
