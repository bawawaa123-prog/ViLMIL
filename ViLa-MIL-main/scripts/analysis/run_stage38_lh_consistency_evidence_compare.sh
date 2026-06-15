#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage38}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-test}"
TOPK_CONCEPTS="${TOPK_CONCEPTS:-5}"
SKELETON_RESULTS_DIR="${SKELETON_RESULTS_DIR:-results_stage37/lh_consistency_skeleton_5fold_e20_s1}"
LH_RESULTS_DIR="${LH_RESULTS_DIR:-results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1}"

SKELETON_NAME="skeleton"
LH_NAME="lh_l001_m0"

SKELETON_EVIDENCE_DIR="${RESULTS_DIR}/evidence_export_${SKELETON_NAME}_fold${FOLD}_${SPLIT}"
LH_EVIDENCE_DIR="${RESULTS_DIR}/evidence_export_${LH_NAME}_fold${FOLD}_${SPLIT}"
SKELETON_FAILURE_DIR="${RESULTS_DIR}/failure_analysis_${SKELETON_NAME}_fold${FOLD}_${SPLIT}"
LH_FAILURE_DIR="${RESULTS_DIR}/failure_analysis_${LH_NAME}_fold${FOLD}_${SPLIT}"
COMPARE_DIR="${RESULTS_DIR}/stage38_lh_consistency_failure_comparison"

cd "${ROOT_DIR}"

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

run_export() {
  local variant_name="$1"
  local input_results_dir="$2"
  local output_dir="$3"
  local -a cmd=(
    "${PYTHON_BIN}" "scripts/analysis/export_stage32_rce_v4_csg_evidence.py"
    --data_root_dir "${DATA_ROOT_DIR}"
    --results_dir "${input_results_dir}"
    --output_dir "${output_dir}"
    --fold "${FOLD}"
    --split "${SPLIT}"
    --topk_concepts "${TOPK_CONCEPTS}"
    --model_type DEG_MIL_BiomedCLIP
  )
  echo "[Run] Evidence export for ${variant_name}"
  print_command "${cmd[@]}"
  "${cmd[@]}"
}

run_failure_analysis() {
  local variant_name="$1"
  local input_dir="$2"
  local output_dir="$3"
  local -a cmd=(
    "${PYTHON_BIN}" "scripts/analysis/build_stage33_evidence_failure_analysis.py"
    --input_dir "${input_dir}"
    --output_dir "${output_dir}"
    --variant_name "${variant_name}"
    --top_concepts "${TOPK_CONCEPTS}"
  )
  echo "[Run] Failure analysis for ${variant_name}"
  print_command "${cmd[@]}"
  "${cmd[@]}"
}

run_comparison() {
  local -a cmd=(
    "${PYTHON_BIN}" "scripts/analysis/build_stage38_lh_consistency_failure_comparison.py"
    --skeleton_evidence_dir "${SKELETON_EVIDENCE_DIR}"
    --lh_evidence_dir "${LH_EVIDENCE_DIR}"
    --skeleton_failure_dir "${SKELETON_FAILURE_DIR}"
    --lh_failure_dir "${LH_FAILURE_DIR}"
    --output_dir "${COMPARE_DIR}"
    --skeleton_name "${SKELETON_NAME}"
    --lh_name "${LH_NAME}"
  )
  echo "[Run] Stage38 comparison"
  print_command "${cmd[@]}"
  "${cmd[@]}"
}

run_export "${SKELETON_NAME}" "${SKELETON_RESULTS_DIR}" "${SKELETON_EVIDENCE_DIR}"
run_export "${LH_NAME}" "${LH_RESULTS_DIR}" "${LH_EVIDENCE_DIR}"

run_failure_analysis "${SKELETON_NAME}" "${SKELETON_EVIDENCE_DIR}" "${SKELETON_FAILURE_DIR}"
run_failure_analysis "${LH_NAME}" "${LH_EVIDENCE_DIR}" "${LH_FAILURE_DIR}"

run_comparison

echo "[Done] Step38 outputs:"
echo "[Done] ${SKELETON_EVIDENCE_DIR}"
echo "[Done] ${LH_EVIDENCE_DIR}"
echo "[Done] ${SKELETON_FAILURE_DIR}"
echo "[Done] ${LH_FAILURE_DIR}"
echo "[Done] ${COMPARE_DIR}"
