#!/usr/bin/env bash

SOMAFORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SOMAFORGE_ROOT
export SOMAFORGE_ASSET_MANIFEST="${SOMAFORGE_ROOT}/configs/assets_manifest.json"
export SOMAFORGE_TRAINING_MANIFEST="${SOMAFORGE_ROOT}/configs/training_pipeline_manifest.json"
export PYTHONPATH="${SOMAFORGE_ROOT}/packages/somaforge_core:${SOMAFORGE_ROOT}/packages/motion_edit:${SOMAFORGE_ROOT}/packages/generator:${SOMAFORGE_ROOT}/packages/supermimic:${SOMAFORGE_ROOT}/packages/contact_solver:${SOMAFORGE_ROOT}/src/holosoma:${SOMAFORGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
