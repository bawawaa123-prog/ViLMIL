#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE32_DIR="${STAGE32_DIR:-results_stage32/stage32_rce_v4_csg_evidence_export}"
STAGE33_DIR="${STAGE33_DIR:-results_stage33/stage33_evidence_failure_analysis}"
STAGE39_DIR="${STAGE39_DIR:-results_stage39/final_evidence_package}"
STAGE44_DIR="${STAGE44_DIR:-results_stage44/stage44_hcrc_light_summary}"
OUTPUT_DIR="${OUTPUT_DIR:-results_stage45/prarc_reliability_audit}"
MIN_PROMPT_OCCURRENCES="${MIN_PROMPT_OCCURRENCES:-3}"
VISUAL_OVERRIDE_THRESHOLD="${VISUAL_OVERRIDE_THRESHOLD:-0.5}"
LOW_MARGIN_THRESHOLD="${LOW_MARGIN_THRESHOLD:-0.05}"
SEED="${SEED:-1}"

CMD=(
  "${PYTHON_BIN}" "scripts/analysis/build_stage45_prarc_reliability_audit.py"
  "--stage32_dir" "${STAGE32_DIR}"
  "--stage33_dir" "${STAGE33_DIR}"
  "--stage39_dir" "${STAGE39_DIR}"
  "--stage44_dir" "${STAGE44_DIR}"
  "--output_dir" "${OUTPUT_DIR}"
  "--min_prompt_occurrences" "${MIN_PROMPT_OCCURRENCES}"
  "--visual_override_threshold" "${VISUAL_OVERRIDE_THRESHOLD}"
  "--low_margin_threshold" "${LOW_MARGIN_THRESHOLD}"
  "--seed" "${SEED}"
)

printf 'Running Step45 PRARC reliability audit:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
