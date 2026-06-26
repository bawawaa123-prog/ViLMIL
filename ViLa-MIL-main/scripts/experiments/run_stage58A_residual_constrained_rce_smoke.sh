#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage58A_residual_constrained_rce}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
EXP_CODE="${EXP_CODE:-rce_v2_residual_constrained_smoke}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

cd "${ROOT_DIR}"

K_FOLDS=1
K_START=0
K_END=0

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

cmd=(
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
  --exp_code "${EXP_CODE}"
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
  --rce_residual_constraint_lambda 0.05
  --rce_residual_ratio_target 0.5
  --rce_use_concept_aux_loss
  --rce_concept_aux_loss_weight 0.2
)

env_vars=(
  "PYTHONPATH=${ROOT_DIR}"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
)

echo "[Run] Stage58A residual-constrained RCE smoke"
echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
echo "[Run] EXP_CODE=${EXP_CODE}"
echo "[Run] K=${K_FOLDS}"
echo "[Run] K_START=${K_START}"
echo "[Run] K_END=${K_END}"
echo "[Run] MAX_EPOCHS=${MAX_EPOCHS}"
echo "[Run] SEED=${SEED}"
print_command env "${env_vars[@]}" "${cmd[@]}"

env "${env_vars[@]}" "${cmd[@]}"
