# Move the managed main environment to the system trash so setup can be rerun.
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

ENV_ROOT="$ROOT_DIR/runtime/environments/conda/somaforge"

if ! command -v gio >/dev/null 2>&1; then
    echo "gio is required to move environment files to the system trash." >&2
    exit 1
fi

echo "Moving these targets to the system trash:"
echo "  $ENV_ROOT"
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

[[ ! -e $ENV_ROOT ]] || gio trash "$ENV_ROOT"
