#!/usr/bin/env python3
from __future__ import annotations

import argparse

from somaforge_core import AssetManifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the SomaForge asset manifest.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--include-legacy", action="store_true")
    args = parser.parse_args()
    manifest = AssetManifest.load(args.manifest)
    errors = manifest.validate(
        require_exists=not args.allow_missing,
        include_legacy=args.include_legacy,
    )
    if errors:
        print("Asset manifest validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Asset manifest valid: {len(manifest.ids())} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
