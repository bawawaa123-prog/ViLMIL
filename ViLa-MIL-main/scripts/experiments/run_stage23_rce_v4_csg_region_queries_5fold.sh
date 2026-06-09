#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage23}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
VARIANT="${VARIANT:-rq16}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

cd "${ROOT_DIR}"

# main.py treats --k_end as inclusive, so a true 5-fold run uses folds 0..4.
K_FOLDS=5
K_START=0
K_END=4

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

run_variant() {
  local variant="$1"
  local exp_code=""
  local prototype_number=""

  case "${variant}" in
    rq8)
      exp_code="rce_v4_csg_a01_rq8_5fold_e20"
      prototype_number="8"
      ;;
    rq16)
      exp_code="rce_v4_csg_a01_rq16_5fold_e20"
      prototype_number="16"
      ;;
    rq32)
      exp_code="rce_v4_csg_a01_rq32_5fold_e20"
      prototype_number="32"
      ;;
    *)
      echo "[Error] Unsupported variant: ${variant}" >&2
      echo "[Error] Supported VARIANT values: rq8, rq16, rq32, all" >&2
      exit 1
      ;;
  esac

  local -a cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    --task task_adenocarcinoma
    --model_type RCE_MIL_BiomedCLIP
    --mode transformer
    --data_root_dir "${DATA_ROOT_DIR}"
    --data_folder_s features_biomedclip_5x
    --data_folder_l features_biomedclip_20x
    --text_prompt_path "${TEXT_PROMPT_PATH}"
    --use_concept_prompt_pool
    --concept_prompt_path "${CONCEPT12_PATH}"
    --prompt_ensemble_mode embedding_mean
    --scale_mode dual
    --k "${K_FOLDS}"
    --k_start "${K_START}"
    --k_end "${K_END}"
    --max_epochs "${MAX_EPOCHS}"
    --seed "${SEED}"
    --prototype_number "${prototype_number}"
    --split_dir "${SPLIT_DIR}"
    --results_dir "${RESULTS_DIR}"
    --exp_code "${exp_code}"
    --rce_use_logit_calibration
    --rce_use_concept_prior
    --rce_use_visual_residual
    --rce_logit_scale_init 10.0
    --rce_concept_prior_strength 1.0
    --rce_visual_residual_init 0.05
    --rce_use_cross_scale_graph
    --rce_cross_scale_graph_init 0.1
    --rce_cross_scale_graph_norm sqrt
  )

  local -a env_vars=(
    "PYTHONPATH=${ROOT_DIR}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
  )

  echo "[Run] Stage23 RCE-v4-CSG region query sensitivity 5-fold"
  echo "[Run] VARIANT=${variant}"
  echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
  echo "[Run] EXP_CODE=${exp_code}"
  echo "[Run] PROTOTYPE_NUMBER=${prototype_number}"
  echo "[Run] CSG_INIT=0.1"
  echo "[Run] K=${K_FOLDS}"
  echo "[Run] K_START=${K_START}"
  echo "[Run] K_END=${K_END}"
  echo "[Run] MAX_EPOCHS=${MAX_EPOCHS}"
  echo "[Run] SEED=${SEED}"
  print_command env "${env_vars[@]}" "${cmd[@]}"

  env "${env_vars[@]}" "${cmd[@]}"

  echo "[Done] variant finished: ${variant}"
  echo "[Done] Results directory: ${RESULTS_DIR}/${exp_code}_s${SEED}"
}

case "${VARIANT}" in
  rq8|rq16|rq32)
    run_variant "${VARIANT}"
    ;;
  all)
    run_variant rq8
    run_variant rq16
    run_variant rq32
    ;;
  *)
    echo "[Error] Unsupported VARIANT: ${VARIANT}" >&2
    echo "[Error] Supported VARIANT values: rq8, rq16, rq32, all" >&2
    exit 1
    ;;
esac
