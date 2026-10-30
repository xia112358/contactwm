#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SOMAFORGE_SUPERMIMIC_PYTHON:-python}"
RUNTIME_ROOT="$("$PYTHON_BIN" -c 'from supermimic.paths import runtime_root; print(runtime_root())')"
if [[ "${1:-}" == tools/* || "${1:-}" == supermimic/*.py ]]; then
    tool="$PACKAGE_ROOT/$1"
    shift
    set -- "$tool" "$@"
fi
export TMPDIR="${TMPDIR:-$RUNTIME_ROOT/tmp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$RUNTIME_ROOT/.cache/warp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUNTIME_ROOT/.cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUNTIME_ROOT/.cache/triton}"
export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$RUNTIME_ROOT/.cache/torch}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$RUNTIME_ROOT/.cache/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_ROOT/.cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_ROOT/.cache/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-$RUNTIME_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUNTIME_ROOT/.cache/wandb}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$RUNTIME_ROOT"
exec "$PYTHON_BIN" "$@"
