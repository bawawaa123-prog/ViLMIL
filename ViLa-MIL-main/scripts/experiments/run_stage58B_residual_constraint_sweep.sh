#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage58B_residual_constraint_sweep}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
CONFIGS="${CONFIGS:-all}"
RUN_TRAIN="${RUN_TRAIN:-0}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

cd "${ROOT_DIR}"

K_FOLDS=5
K_START=0
K_END=0

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

run_config() {
  local config_id="$1"
  local exp_code=""
  local lambda_residual=""
  local ratio_target=""
  local concept_aux_weight=""

  case "${config_id}" in
    A)
      exp_code="rce_v2_rc_config_A_fold0"
      lambda_residual="0.01"
      ratio_target="0.60"
      concept_aux_weight="0.10"
      ;;
    B)
      exp_code="rce_v2_rc_config_B_fold0"
      lambda_residual="0.03"
      ratio_target="0.60"
      concept_aux_weight="0.10"
      ;;
    C)
      exp_code="rce_v2_rc_config_C_fold0"
      lambda_residual="0.05"
      ratio_target="0.60"
      concept_aux_weight="0.20"
      ;;
    D)
      exp_code="rce_v2_rc_config_D_fold0"
      lambda_residual="0.03"
      ratio_target="0.50"
      concept_aux_weight="0.20"
      ;;
    E)
      exp_code="rce_v2_rc_config_E_fold0"
      lambda_residual="0.05"
      ratio_target="0.50"
      concept_aux_weight="0.20"
      ;;
    *)
      echo "[Error] Unsupported config_id: ${config_id}" >&2
      exit 1
      ;;
  esac

  local -a cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    --task task_adenocarcinoma
    --model_type RCE_MIL_BiomedCLIP_v2
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
    --prototype_number 16
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
    --rce_use_residual_constraint
    --rce_residual_constraint_lambda "${lambda_residual}"
    --rce_residual_ratio_target "${ratio_target}"
    --rce_use_concept_aux_loss
    --rce_concept_aux_loss_weight "${concept_aux_weight}"
  )

  local -a env_vars=(
    "PYTHONPATH=${ROOT_DIR}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
  )

  echo "[Run] Stage58B residual constraint sweep"
  echo "[Run] CONFIG_ID=${config_id}"
  echo "[Run] EXP_CODE=${exp_code}"
  echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
  echo "[Run] LAMBDA_RESIDUAL=${lambda_residual}"
  echo "[Run] RATIO_TARGET=${ratio_target}"
  echo "[Run] CONCEPT_AUX_WEIGHT=${concept_aux_weight}"
  echo "[Run] K_START=${K_START}"
  echo "[Run] K_END=${K_END}"
  echo "[Run] MAX_EPOCHS=${MAX_EPOCHS}"
  echo "[Run] SEED=${SEED}"
  print_command env "${env_vars[@]}" "${cmd[@]}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    env "${env_vars[@]}" "${cmd[@]}"
    echo "[Done] config finished: ${config_id}"
    echo "[Done] Results directory: ${RESULTS_DIR}/${exp_code}_s${SEED}"
  else
    echo "[Skip] RUN_TRAIN=${RUN_TRAIN}; command printed only."
  fi
}

case "${CONFIGS}" in
  A|B|C|D|E)
    run_config "${CONFIGS}"
    ;;
  all)
    run_config A
    run_config B
    run_config C
    run_config D
    run_config E
    ;;
  *)
    echo "[Error] Unsupported CONFIGS=${CONFIGS}; use A, B, C, D, E, or all" >&2
    exit 1
    ;;
esac
