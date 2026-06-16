#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
STAGE39_DIR="${STAGE39_DIR:-results_stage39/final_evidence_package}"
STAGE44_DIR="${STAGE44_DIR:-results_stage44/stage44_hcrc_light_summary}"
STAGE45_DIR="${STAGE45_DIR:-results_stage45/prarc_reliability_audit}"
STAGE47_DIR="${STAGE47_DIR:-results_stage47/stage47_prarc_gate_summary}"
STAGE47_GATE_DIAG_DIR="${STAGE47_GATE_DIAG_DIR:-results_stage47/stage47_prarc_gate_diagnostics}"
STAGE48B_DIR="${STAGE48B_DIR:-results_stage48/stage48b_prarc_v2_variant_sweep_summary}"
OUTPUT_DIR="${OUTPUT_DIR:-results_stage49/final_consolidation}"
BASELINE_NAME="${BASELINE_NAME:-RCE-v4-CSG-a01-rq16 / DEG skeleton}"

cd "${ROOT_DIR}"

cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/build_stage49_final_consolidation.py"
  --stage39_dir "${STAGE39_DIR}"
  --stage44_dir "${STAGE44_DIR}"
  --stage45_dir "${STAGE45_DIR}"
  --stage47_dir "${STAGE47_DIR}"
  --stage47_gate_diag_dir "${STAGE47_GATE_DIAG_DIR}"
  --stage48b_dir "${STAGE48B_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --baseline_name "${BASELINE_NAME}"
)

printf "[Run] "
printf "%q " "${cmd[@]}"
printf "\n"

"${cmd[@]}" "$@"
