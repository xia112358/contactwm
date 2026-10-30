#!/usr/bin/env bash
# Compatibility installer: all components share the main profile.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 "$SCRIPT_DIR/manage_environment.py" create --profile main "$@"
