#!/usr/bin/env python3
from __future__ import annotations

import argparse

from somaforge_core import load_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the shared SomaForge training pipeline manifest.")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    manifest = load_training_manifest(args.manifest)
    print(f"Training manifest valid: {', '.join(manifest['stages'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
