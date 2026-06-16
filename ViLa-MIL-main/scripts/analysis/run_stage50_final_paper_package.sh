#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
STAGE39_DIR="${STAGE39_DIR:-${ROOT_DIR}/results_stage39/final_evidence_package}"
STAGE40_DIR="${STAGE40_DIR:-${ROOT_DIR}/results_stage40/paper_ready_assets}"
STAGE44_DIR="${STAGE44_DIR:-${ROOT_DIR}/results_stage44/stage44_hcrc_light_summary}"
STAGE45_DIR="${STAGE45_DIR:-${ROOT_DIR}/results_stage45/prarc_reliability_audit}"
STAGE47_DIR="${STAGE47_DIR:-${ROOT_DIR}/results_stage47/stage47_prarc_gate_summary}"
STAGE47_GATE_DIAG_DIR="${STAGE47_GATE_DIAG_DIR:-${ROOT_DIR}/results_stage47/stage47_prarc_gate_diagnostics}"
STAGE48B_DIR="${STAGE48B_DIR:-${ROOT_DIR}/results_stage48/stage48b_prarc_v2_variant_sweep_summary}"
STAGE49_DIR="${STAGE49_DIR:-${ROOT_DIR}/results_stage49/final_consolidation}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results_stage50/final_paper_package}"
PAPER_TITLE="${PAPER_TITLE:-Evidence-Guided Cross-Scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification}"

cd "${ROOT_DIR}"

cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/build_stage50_final_paper_package.py"
  --stage39_dir "${STAGE39_DIR}"
  --stage40_dir "${STAGE40_DIR}"
  --stage44_dir "${STAGE44_DIR}"
  --stage45_dir "${STAGE45_DIR}"
  --stage47_dir "${STAGE47_DIR}"
  --stage47_gate_diag_dir "${STAGE47_GATE_DIAG_DIR}"
  --stage48b_dir "${STAGE48B_DIR}"
  --stage49_dir "${STAGE49_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --paper_title "${PAPER_TITLE}"
)

printf "[Run] "
printf "%q " "${cmd[@]}"
printf "\n"

"${cmd[@]}" "$@"
