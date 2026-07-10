#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-}"
DATA_ROOT="${2:-}"
OUTPUT_DIR="${3:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${PROJECT_ROOT}" || -z "${DATA_ROOT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 <project-root> <data-root> <output-dir>" >&2
  exit 1
fi

TOOLS_DIR="${PROJECT_ROOT}/ViLa-MIL-main/tools"
mkdir -p "${OUTPUT_DIR}"

echo "[Manifest] project_root=${PROJECT_ROOT}"
echo "[Manifest] data_root=${DATA_ROOT}"
echo "[Manifest] output_dir=${OUTPUT_DIR}"
echo "[Manifest] python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${TOOLS_DIR}/build_stage64J_repro_manifest.py" \
  --project-root "${PROJECT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${TOOLS_DIR}/build_stage64J_asset_manifest.py" \
  --project-root "${PROJECT_ROOT}/ViLa-MIL-main" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode full

"${PYTHON_BIN}" "${TOOLS_DIR}/build_stage64J_biomedclip_cache_manifest.py" \
  --project-root "${PROJECT_ROOT}/ViLa-MIL-main" \
  --output-dir "${OUTPUT_DIR}"
