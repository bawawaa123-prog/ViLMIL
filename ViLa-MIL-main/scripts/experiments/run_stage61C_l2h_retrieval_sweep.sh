#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISPLAY_ROOT="${DISPLAY_ROOT:-/xiangmu/ViLMIL/ViLa-MIL-main}"
if [[ ! -f "${DISPLAY_ROOT}/main.py" ]]; then
  DISPLAY_ROOT="${ROOT_DIR}"
fi

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage61C_l2h_retrieval_sweep}"
RUN_TRAIN="${RUN_TRAIN:-0}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
CONFIGS="${CONFIGS:-default}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

OUTPUT_DIR="${ROOT_DIR}/${RESULTS_DIR}"
RUN_COMMANDS_FILE="${OUTPUT_DIR}/stage61C_run_commands.txt"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

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
  local alpha_init=""
  local low_topk=""
  local high_max_per_low=""
  local patch_footprint_ratio=""

  case "${config_id}" in
    A)
      exp_code="rce_v2_rcD_l2h_A_fold0"
      alpha_init="0.001"
      low_topk="8"
      high_max_per_low="16"
      patch_footprint_ratio="4.0"
      ;;
    B)
      exp_code="rce_v2_rcD_l2h_B_fold0"
      alpha_init="0.01"
      low_topk="8"
      high_max_per_low="16"
      patch_footprint_ratio="4.0"
      ;;
    C)
      exp_code="rce_v2_rcD_l2h_C_fold0"
      alpha_init="0.05"
      low_topk="8"
      high_max_per_low="16"
      patch_footprint_ratio="4.0"
      ;;
    D)
      exp_code="rce_v2_rcD_l2h_D_fold0"
      alpha_init="0.01"
      low_topk="16"
      high_max_per_low="16"
      patch_footprint_ratio="4.0"
      ;;
    E)
      exp_code="rce_v2_rcD_l2h_E_fold0"
      alpha_init="0.01"
      low_topk="8"
      high_max_per_low="32"
      patch_footprint_ratio="4.0"
      ;;
    F)
      exp_code="rce_v2_rcD_l2h_F_fold0"
      alpha_init="0.01"
      low_topk="8"
      high_max_per_low="16"
      patch_footprint_ratio="6.0"
      ;;
    G)
      exp_code="rce_v2_rcD_l2h_G_fold0"
      alpha_init="0.01"
      low_topk="16"
      high_max_per_low="32"
      patch_footprint_ratio="6.0"
      ;;
    H)
      exp_code="rce_v2_rcD_l2h_H_fold0"
      alpha_init="0.01"
      low_topk="8"
      high_max_per_low="32"
      patch_footprint_ratio="8.0"
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
    --rce_residual_constraint_lambda 0.03
    --rce_residual_ratio_target 0.50
    --rce_use_concept_aux_loss
    --rce_concept_aux_loss_weight 0.20
    --rce_use_l2h_retrieval
    --rce_l2h_mode low_topk_coord_window
    --rce_l2h_fusion high_region_residual
    --rce_l2h_aggregate mean
    --rce_l2h_score_mode low_prompt_max
    --rce_l2h_scale_ratio 1.0
    --rce_l2h_scale 1.0
    --rce_l2h_clip 5.0
    --rce_l2h_alpha_init "${alpha_init}"
    --rce_l2h_low_topk "${low_topk}"
    --rce_l2h_high_max_per_low "${high_max_per_low}"
    --rce_l2h_patch_footprint_ratio "${patch_footprint_ratio}"
  )

  local -a env_vars=(
    "PYTHONPATH=${ROOT_DIR}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
  )

  echo "[Run] Stage61C L2H Retrieval sweep"
  echo "[Run] CONFIG_ID=${config_id}"
  echo "[Run] EXP_CODE=${exp_code}"
  echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
  echo "[Run] L2H_ALPHA_INIT=${alpha_init}"
  echo "[Run] L2H_LOW_TOPK=${low_topk}"
  echo "[Run] L2H_HIGH_MAX_PER_LOW=${high_max_per_low}"
  echo "[Run] L2H_PATCH_FOOTPRINT_RATIO=${patch_footprint_ratio}"
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

{
  echo "cd ${DISPLAY_ROOT}"
  echo "RUN_TRAIN=1 bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=extended bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo
  echo "# Single config examples"
  echo "RUN_TRAIN=1 CONFIGS=A bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=B bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=C bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=D bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=E bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=F bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=G bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo "RUN_TRAIN=1 CONFIGS=H bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh"
  echo
  echo "# Refresh Step61C summary"
  echo "PYTHONPATH=${DISPLAY_ROOT} HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG} ${PYTHON_BIN} scripts/analysis/build_stage61C_l2h_retrieval_sweep_summary.py"
} > "${RUN_COMMANDS_FILE}"

case "${CONFIGS}" in
  A|B|C|D|E|F|G|H)
    run_config "${CONFIGS}"
    ;;
  default)
    run_config A
    run_config B
    run_config C
    run_config D
    run_config E
    run_config F
    run_config G
    ;;
  extended)
    run_config A
    run_config B
    run_config C
    run_config D
    run_config E
    run_config F
    run_config G
    run_config H
    ;;
  *)
    echo "[Error] Unsupported CONFIGS=${CONFIGS}; use A-H, default, or extended" >&2
    exit 1
    ;;
esac
