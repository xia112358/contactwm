#!/usr/bin/env bash
# Compatibility entrypoint: all SomaForge components share main.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/setup_isaaclab3_newton.sh" "$@"
