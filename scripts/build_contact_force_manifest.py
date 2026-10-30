#!/usr/bin/env python3
"""Build a motion-matched manifest using contact-force demo motion files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CLIMB_RE = re.compile(r"climb[_-](\d+)")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _climb_id(path: str) -> int:
    match = CLIMB_RE.search(path)
    if match is None:
        raise ValueError(f"Could not infer climb id from path: {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument("--demo-glob", default="climb_*_force_demo*.npz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--description", default="29-motion force-demo manifest.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--relative-paths", action="store_true")
    args = parser.parse_args()

    base = _read_json(args.base_manifest)
    demos = sorted(args.demo_dir.glob(args.demo_glob))
    demo_by_id = {_climb_id(str(path)): path for path in demos}

    motion_files = []
    missing = []
    for entry in base.get("motion_files", []):
        cid = _climb_id(str(entry["motion_file"]))
        demo = demo_by_id.get(cid)
        if demo is None:
            missing.append(cid)
            if args.allow_missing:
                continue
            continue
        motion_path = str(demo if args.relative_paths else demo.resolve())
        motion_files.append(
            {
                "motion_file": motion_path,
                "terrain_id": entry["terrain_id"],
                "weight": entry.get("weight", 1.0),
            }
        )

    if missing and not args.allow_missing:
        raise ValueError(f"Missing force demos for climb ids: {sorted(missing)}")
    if not motion_files:
        raise ValueError("No motion files were added to the force manifest.")

    out = {
        "schema_version": base.get("schema_version", 1),
        "description": args.description,
        "terrains": base["terrains"],
        "motion_files": motion_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.output} motions={len(motion_files)} terrains={len(out['terrains'])}")


if __name__ == "__main__":
    main()
