#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE39_DIR="${STAGE39_DIR:-results_stage39/final_evidence_package}"
OUTPUT_DIR="${OUTPUT_DIR:-results_stage40/paper_ready_assets}"

CMD=(
    "${PYTHON_BIN}" "scripts/analysis/build_stage40_paper_ready_assets.py"
    "--stage39_dir" "${STAGE39_DIR}"
    "--output_dir" "${OUTPUT_DIR}"
)

printf 'Running Step40 paper-ready asset builder:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
