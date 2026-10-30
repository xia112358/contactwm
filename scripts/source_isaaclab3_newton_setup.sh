#!/usr/bin/env bash
# Compatibility activation entrypoint for the single managed main environment.
SOMAFORGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SOMAFORGE_ROOT
_somaforge_activation="$(/usr/bin/python3 "$SOMAFORGE_ROOT/scripts/manage_environment.py" shell --profile main)" || return 1
eval "$_somaforge_activation"
unset _somaforge_activation
