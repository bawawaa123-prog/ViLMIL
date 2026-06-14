#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/results_stage32/stage32_rce_v4_csg_evidence_export}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results_stage33/stage33_evidence_failure_analysis}"
NEAR_ZERO_EPS="${NEAR_ZERO_EPS:-1e-6}"
DOMINANCE_RATIO="${DOMINANCE_RATIO:-0.5}"
LOW_MARGIN_QUANTILE="${LOW_MARGIN_QUANTILE:-0.25}"
TOP_ERROR_CASES="${TOP_ERROR_CASES:-10}"
TOP_CONCEPTS="${TOP_CONCEPTS:-10}"

cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/analysis/build_stage33_evidence_failure_analysis.py"
  --input_dir "$INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --near_zero_eps "$NEAR_ZERO_EPS"
  --dominance_ratio "$DOMINANCE_RATIO"
  --low_margin_quantile "$LOW_MARGIN_QUANTILE"
  --top_error_cases "$TOP_ERROR_CASES"
  --top_concepts "$TOP_CONCEPTS"
)

"${cmd[@]}"
