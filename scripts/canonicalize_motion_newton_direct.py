#!/usr/bin/env python3
"""Canonicalize a Holosoma q trajectory with direct Newton FK.

This command imports Newton/Warp only. It does not launch Isaac Lab, Kit, or a
SimulationContext and does not step a solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from somaforge_core.newton_direct_fk import canonicalize_motion_with_direct_newton_fk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robot-urdf", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = canonicalize_motion_with_direct_newton_fk(
        args.input,
        args.output,
        robot_urdf=args.robot_urdf,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "frames": result.frame_count,
                "body_count": len(result.body_names),
                "joint_count": len(result.joint_names),
                "backend_metadata": result.backend_metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
