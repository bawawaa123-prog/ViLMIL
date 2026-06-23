#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
MODE="${MODE:-dry_run}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-test}"
CONCEPT12_PATH="${CONCEPT12_PATH:-$ROOT_DIR/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
SPLIT_DIR="${SPLIT_DIR:-$ROOT_DIR/splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"

FULL_RESULTS_DIR="${FULL_RESULTS_DIR:-$ROOT_DIR/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1}"
WO_CSG_RESULTS_DIR="${WO_CSG_RESULTS_DIR:-$ROOT_DIR/results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1}"

FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-$ROOT_DIR/results_stage54_rce_evidence_interpretability/full}"
WO_CSG_OUTPUT_DIR="${WO_CSG_OUTPUT_DIR:-$ROOT_DIR/results_stage54_rce_evidence_interpretability/wo_csg}"

mkdir -p "$FULL_OUTPUT_DIR" "$WO_CSG_OUTPUT_DIR"

build_cmd() {
  local results_dir="$1"
  local output_dir="$2"
  local -a cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/analysis/export_stage32_rce_v4_csg_evidence.py"
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$results_dir" \
    --output_dir "$output_dir" \
    --fold "$FOLD" \
    --split "$SPLIT" \
    --model_type RCE_MIL_BiomedCLIP \
    --concept_prompt_path "$CONCEPT12_PATH" \
    --split_dir "$SPLIT_DIR"
  )
  printf '%s\0' "${cmd[@]}"
}

run_or_print() {
  local label="$1"
  shift
  local -a cmd=("$@")
  echo "[Step54] $label"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  if [[ "$MODE" != "dry_run" ]]; then
    "${cmd[@]}"
  fi
}

case "$MODE" in
  dry_run)
    mapfile -d '' -t full_cmd < <(build_cmd "$FULL_RESULTS_DIR" "$FULL_OUTPUT_DIR")
    mapfile -d '' -t wo_csg_cmd < <(build_cmd "$WO_CSG_RESULTS_DIR" "$WO_CSG_OUTPUT_DIR")
    run_or_print "full export command" "${full_cmd[@]}"
    run_or_print "wo_csg export command" "${wo_csg_cmd[@]}"
    cat <<'EOF'
[Step54] dry_run only. No inference was executed.
Expected outputs after real export:
  - results_stage54_rce_evidence_interpretability/full/stage32_slide_evidence_summary.csv
  - results_stage54_rce_evidence_interpretability/wo_csg/stage32_slide_evidence_summary.csv

If some fields are still absent after export, inspect export_stage32_rce_v4_csg_evidence.py outputs first.
Do not modify model forward logic inside Step54.
EOF
    ;;
  full)
    mapfile -d '' -t full_cmd < <(build_cmd "$FULL_RESULTS_DIR" "$FULL_OUTPUT_DIR")
    "${full_cmd[@]}"
    ;;
  wo_csg)
    mapfile -d '' -t wo_csg_cmd < <(build_cmd "$WO_CSG_RESULTS_DIR" "$WO_CSG_OUTPUT_DIR")
    "${wo_csg_cmd[@]}"
    ;;
  both)
    mapfile -d '' -t full_cmd < <(build_cmd "$FULL_RESULTS_DIR" "$FULL_OUTPUT_DIR")
    mapfile -d '' -t wo_csg_cmd < <(build_cmd "$WO_CSG_RESULTS_DIR" "$WO_CSG_OUTPUT_DIR")
    "${full_cmd[@]}"
    "${wo_csg_cmd[@]}"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use one of: dry_run, full, wo_csg, both." >&2
    exit 1
    ;;
esac
