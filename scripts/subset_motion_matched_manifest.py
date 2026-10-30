#!/usr/bin/env python3
"""Write a motion-matched manifest containing selected motion entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_ids(values: list[str]) -> list[int]:
    ids: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                ids.append(int(part))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--motion-id", nargs="+", required=True, help="Motion ids, comma-separated or repeated.")
    parser.add_argument("--description", default=None)
    args = parser.parse_args()

    manifest = _read_json(args.input)
    ids = _parse_ids(args.motion_id)
    motion_files = manifest.get("motion_files", [])
    if not motion_files:
        raise ValueError(f"{args.input} has no motion_files.")
    bad = [idx for idx in ids if idx < 0 or idx >= len(motion_files)]
    if bad:
        raise ValueError(f"Motion ids out of range for {len(motion_files)} motions: {bad}")

    selected = [motion_files[idx] for idx in ids]
    terrain_ids = {entry["terrain_id"] for entry in selected}
    terrains = [entry for entry in manifest["terrains"] if entry["terrain_id"] in terrain_ids]
    out = {
        "schema_version": manifest.get("schema_version", 1),
        "description": args.description or f"Subset of {args.input} for motion ids {ids}.",
        "terrains": terrains,
        "motion_files": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.output} motion_ids={ids} motions={len(selected)} terrains={len(terrains)}")


if __name__ == "__main__":
    main()
