#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-results_stage51b_equivalence_smoke}"
MODE="${MODE:-audit}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}"

run_audit() {
  echo "[Run] Stage51B audit mode"
  PYTHONPATH="${ROOT_DIR}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE_FLAG}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE_FLAG}" \
  DATA_ROOT_DIR="${DATA_ROOT_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/audit_stage51b_deg_skeleton_equivalence.py"
}

run_smoke() {
  echo "[Run] Stage51B smoke mode"
  PYTHONPATH="${ROOT_DIR}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE_FLAG}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE_FLAG}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/main.py" \
    --task task_adenocarcinoma \
    --model_type DEG_MIL_BiomedCLIP \
    --mode transformer \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --data_folder_s features_biomedclip_5x \
    --data_folder_l features_biomedclip_20x \
    --text_prompt_path "${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv" \
    --use_concept_prompt_pool \
    --concept_prompt_path "${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json" \
    --prompt_ensemble_mode embedding_mean \
    --scale_mode dual \
    --k 5 \
    --k_start 0 \
    --k_end 0 \
    --max_epochs 1 \
    --seed 1 \
    --prototype_number 16 \
    --split_dir splits/adenocarcinoma/task_adenocarcinoma_strictcv_100 \
    --results_dir "${RESULTS_DIR}" \
    --exp_code stage51b_deg_skeleton_smoke \
    --rce_use_logit_calibration \
    --rce_use_concept_prior \
    --rce_use_visual_residual \
    --rce_logit_scale_init 10.0 \
    --rce_concept_prior_strength 1.0 \
    --rce_visual_residual_init 0.05 \
    --rce_use_cross_scale_graph \
    --rce_cross_scale_graph_init 0.1 \
    --rce_cross_scale_graph_norm sqrt \
    --testing
}

case "${MODE}" in
  audit)
    run_audit
    ;;
  smoke)
    run_audit
    run_smoke
    ;;
  *)
    echo "[Error] Unsupported MODE: ${MODE}" >&2
    echo "[Error] Supported MODE values: audit, smoke" >&2
    exit 1
    ;;
esac
