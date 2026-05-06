#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/xiangmu/ViLMIL/ViLa-MIL-main"
exec bash "${ROOT_DIR}/scripts/experiments/run_stage7_1_residual_saf_peps.sh" "$@"
