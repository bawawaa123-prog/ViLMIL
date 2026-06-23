#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage52_rce_core_ablation}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
VARIANT="${VARIANT:-all}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

cd "${ROOT_DIR}"

K_FOLDS=5
K_START=0
K_END=4
PROTOTYPE_NUMBER=16
CSG_INIT=0.1

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

run_variant() {
  local variant="$1"
  local exp_code=""
  local -a ablation_args=()

  case "${variant}" in
    full)
      exp_code="full_rce_v4_csg_rq16_5fold_e20"
      ablation_args=(
        --rce_use_logit_calibration
        --rce_use_concept_prior
        --rce_use_visual_residual
        --rce_use_cross_scale_graph
      )
      ;;
    wo_csg)
      exp_code="wo_csg_5fold_e20"
      ablation_args=(
        --rce_use_logit_calibration
        --rce_use_concept_prior
        --rce_use_visual_residual
      )
      ;;
    wo_concept_prior)
      exp_code="wo_concept_prior_5fold_e20"
      ablation_args=(
        --rce_use_logit_calibration
        --rce_use_visual_residual
        --rce_use_cross_scale_graph
      )
      ;;
    wo_visual_residual)
      exp_code="wo_visual_residual_5fold_e20"
      ablation_args=(
        --rce_use_logit_calibration
        --rce_use_concept_prior
        --rce_use_cross_scale_graph
      )
      ;;
    wo_logit_calibration)
      exp_code="wo_logit_calibration_5fold_e20"
      ablation_args=(
        --rce_use_concept_prior
        --rce_use_visual_residual
        --rce_use_cross_scale_graph
      )
      ;;
    *)
      echo "[Error] Unsupported variant: ${variant}" >&2
      echo "[Error] Supported VARIANT values: full, wo_csg, wo_concept_prior, wo_visual_residual, wo_logit_calibration, all" >&2
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
    --prototype_number "${PROTOTYPE_NUMBER}"
    --split_dir "${SPLIT_DIR}"
    --results_dir "${RESULTS_DIR}"
    --exp_code "${exp_code}"
    --rce_cross_scale_graph_init "${CSG_INIT}"
    --rce_cross_scale_graph_norm sqrt
    --rce_logit_scale_init 10.0
    --rce_concept_prior_strength 1.0
    --rce_visual_residual_init 0.05
    "${ablation_args[@]}"
  )

  local -a env_vars=(
    "PYTHONPATH=${ROOT_DIR}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
  )

  echo "[Run] Stage52 RCE core ablation 5-fold"
  echo "[Run] VARIANT=${variant}"
  echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
  echo "[Run] EXP_CODE=${exp_code}"
  echo "[Run] MODEL_TYPE=RCE_MIL_BiomedCLIP"
  echo "[Run] PROTOTYPE_NUMBER=${PROTOTYPE_NUMBER}"
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
  full|wo_csg|wo_concept_prior|wo_visual_residual|wo_logit_calibration)
    run_variant "${VARIANT}"
    ;;
  all)
    run_variant full
    run_variant wo_csg
    run_variant wo_concept_prior
    run_variant wo_visual_residual
    run_variant wo_logit_calibration
    ;;
  *)
    echo "[Error] Unsupported VARIANT: ${VARIANT}" >&2
    echo "[Error] Supported VARIANT values: full, wo_csg, wo_concept_prior, wo_visual_residual, wo_logit_calibration, all" >&2
    exit 1
    ;;
esac
