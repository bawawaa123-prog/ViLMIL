#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage27}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
VARIANT="${VARIANT:-skeleton}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

K_FOLDS=5
K_START=0
K_END=4
PROTOTYPE_NUMBER=16
CSG_INIT=0.1
REGION_GRAPH_ALPHA=0.1

cd "${ROOT_DIR}"

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

run_variant() {
  local variant="$1"
  local exp_code=""
  local region_graph_k="off"
  local -a variant_args=()

  case "${variant}" in
    skeleton)
      exp_code="deg_skeleton_5fold_e20"
      ;;
    rg_k2)
      exp_code="deg_region_graph_k2_a01_5fold_e20"
      region_graph_k="2"
      variant_args=(
        --deg_use_region_graph
        --deg_region_graph_k 2
        --deg_region_graph_alpha "${REGION_GRAPH_ALPHA}"
      )
      ;;
    rg_k4)
      exp_code="deg_region_graph_k4_a01_5fold_e20"
      region_graph_k="4"
      variant_args=(
        --deg_use_region_graph
        --deg_region_graph_k 4
        --deg_region_graph_alpha "${REGION_GRAPH_ALPHA}"
      )
      ;;
    rg_k8)
      exp_code="deg_region_graph_k8_a01_5fold_e20"
      region_graph_k="8"
      variant_args=(
        --deg_use_region_graph
        --deg_region_graph_k 8
        --deg_region_graph_alpha "${REGION_GRAPH_ALPHA}"
      )
      ;;
    *)
      echo "[Error] Unsupported variant: ${variant}" >&2
      echo "[Error] Supported VARIANT values: skeleton, rg_k2, rg_k4, rg_k8, all" >&2
      exit 1
      ;;
  esac

  local -a cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    --task task_adenocarcinoma
    --model_type DEG_MIL_BiomedCLIP
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
    --rce_use_logit_calibration
    --rce_use_concept_prior
    --rce_use_visual_residual
    --rce_logit_scale_init 10.0
    --rce_concept_prior_strength 1.0
    --rce_visual_residual_init 0.05
    --rce_use_cross_scale_graph
    --rce_cross_scale_graph_init "${CSG_INIT}"
    --rce_cross_scale_graph_norm sqrt
    "${variant_args[@]}"
  )

  local -a env_vars=(
    "PYTHONPATH=${ROOT_DIR}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
  )

  echo "[Run] Stage27 DEG Spatial Region Graph 5-fold"
  echo "[Run] VARIANT=${variant}"
  echo "[Run] PROTOTYPE_NUMBER=${PROTOTYPE_NUMBER}"
  echo "[Run] CSG_INIT=${CSG_INIT}"
  echo "[Run] REGION_GRAPH_K=${region_graph_k}"
  echo "[Run] REGION_GRAPH_ALPHA=${REGION_GRAPH_ALPHA}"
  echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
  echo "[Run] EXP_CODE=${exp_code}"
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
  skeleton|rg_k2|rg_k4|rg_k8)
    run_variant "${VARIANT}"
    ;;
  all)
    run_variant skeleton
    run_variant rg_k2
    run_variant rg_k4
    run_variant rg_k8
    ;;
  *)
    echo "[Error] Unsupported VARIANT: ${VARIANT}" >&2
    echo "[Error] Supported VARIANT values: skeleton, rg_k2, rg_k4, rg_k8, all" >&2
    exit 1
    ;;
esac
