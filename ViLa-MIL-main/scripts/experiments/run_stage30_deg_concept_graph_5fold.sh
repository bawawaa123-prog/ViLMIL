#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage30}"
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
CONCEPT_GRAPH_ALPHA=0.05

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
  local concept_graph_enabled="false"
  local concept_graph_topk="off"
  local concept_graph_alpha="off"
  local -a variant_args=()

  case "${variant}" in
    skeleton)
      exp_code="deg_skeleton_cg_ablation_5fold_e20"
      ;;
    cg_k2_a005)
      exp_code="deg_concept_graph_k2_a005_5fold_e20"
      concept_graph_enabled="true"
      concept_graph_topk="2"
      concept_graph_alpha="${CONCEPT_GRAPH_ALPHA}"
      variant_args=(
        --deg_use_concept_graph
        --deg_concept_graph_topk 2
        --deg_concept_graph_alpha "${CONCEPT_GRAPH_ALPHA}"
      )
      ;;
    cg_k4_a005)
      exp_code="deg_concept_graph_k4_a005_5fold_e20"
      concept_graph_enabled="true"
      concept_graph_topk="4"
      concept_graph_alpha="${CONCEPT_GRAPH_ALPHA}"
      variant_args=(
        --deg_use_concept_graph
        --deg_concept_graph_topk 4
        --deg_concept_graph_alpha "${CONCEPT_GRAPH_ALPHA}"
      )
      ;;
    cg_k8_a005)
      exp_code="deg_concept_graph_k8_a005_5fold_e20"
      concept_graph_enabled="true"
      concept_graph_topk="8"
      concept_graph_alpha="${CONCEPT_GRAPH_ALPHA}"
      variant_args=(
        --deg_use_concept_graph
        --deg_concept_graph_topk 8
        --deg_concept_graph_alpha "${CONCEPT_GRAPH_ALPHA}"
      )
      ;;
    *)
      echo "[Error] Unsupported variant: ${variant}" >&2
      echo "[Error] Supported VARIANT values: skeleton, cg_k2_a005, cg_k4_a005, cg_k8_a005, all" >&2
      exit 1
      ;;
  esac

  local -a common_args=(
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
    --rce_visual_residual_init 0.05
    --rce_use_cross_scale_graph
    --rce_cross_scale_graph_init "${CSG_INIT}"
    --rce_cross_scale_graph_norm sqrt
  )

  local -a cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    "${common_args[@]}"
    "${variant_args[@]}"
  )

  local -a env_vars=(
    "PYTHONPATH=${ROOT_DIR}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
  )

  echo "[Run] Stage30 DEG Concept Prompt Graph 5-fold"
  echo "[Run] VARIANT=${variant}"
  echo "[Run] CONCEPT_GRAPH_ENABLED=${concept_graph_enabled}"
  echo "[Run] CONCEPT_GRAPH_TOPK=${concept_graph_topk}"
  echo "[Run] CONCEPT_GRAPH_ALPHA=${concept_graph_alpha}"
  echo "[Run] PROTOTYPE_NUMBER=${PROTOTYPE_NUMBER}"
  echo "[Run] CSG_INIT=${CSG_INIT}"
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
  skeleton|cg_k2_a005|cg_k4_a005|cg_k8_a005)
    run_variant "${VARIANT}"
    ;;
  all)
    run_variant skeleton
    run_variant cg_k2_a005
    run_variant cg_k4_a005
    run_variant cg_k8_a005
    ;;
  *)
    echo "[Error] Unsupported VARIANT: ${VARIANT}" >&2
    echo "[Error] Supported VARIANT values: skeleton, cg_k2_a005, cg_k4_a005, cg_k8_a005, all" >&2
    exit 1
    ;;
esac
