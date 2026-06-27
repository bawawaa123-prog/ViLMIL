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
RESULTS_DIR="${RESULTS_DIR:-results_stage60C_ccra_configD_5fold}"
RUN_TRAIN="${RUN_TRAIN:-0}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"
TARGET_CONFIG="${TARGET_CONFIG:-D}"

OUTPUT_DIR="${ROOT_DIR}/${RESULTS_DIR}"
RUN_COMMANDS_FILE="${OUTPUT_DIR}/stage60C_run_commands.txt"

K_FOLDS=5
K_START=0
K_END=4

CCRA_MODE="concept_query_residual"
CCRA_QUERY_SOURCE="prompt_mean"
CCRA_ALPHA_INIT="0.05"
CCRA_SCALE="1.0"
CCRA_NORM="layernorm"
CCRA_DROPOUT="0.0"
CCRA_CLIP="5.0"
EXP_CODE="rce_v2_rcD_ccraD_5fold_e20"
CONFIG_LABEL="D"

if [[ "${TARGET_CONFIG}" == "C" ]]; then
  CCRA_ALPHA_INIT="0.01"
  CCRA_SCALE="1.0"
  CCRA_NORM="layernorm"
  CCRA_DROPOUT="0.0"
  CCRA_CLIP="5.0"
  EXP_CODE="rce_v2_rcD_ccraC_5fold_e20"
  CONFIG_LABEL="C"
elif [[ "${TARGET_CONFIG}" != "D" ]]; then
  echo "[Error] Unsupported TARGET_CONFIG=${TARGET_CONFIG}. Use D or C." >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

local_cmd=(
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
  --rce_residual_constraint_lambda 0.03
  --rce_residual_ratio_target 0.50
  --rce_use_concept_aux_loss
  --rce_concept_aux_loss_weight 0.20
  --rce_use_ccra
  --rce_ccra_mode "${CCRA_MODE}"
  --rce_ccra_query_source "${CCRA_QUERY_SOURCE}"
  --rce_ccra_alpha_init "${CCRA_ALPHA_INIT}"
  --rce_ccra_scale "${CCRA_SCALE}"
  --rce_ccra_norm "${CCRA_NORM}"
  --rce_ccra_dropout "${CCRA_DROPOUT}"
  --rce_ccra_clip "${CCRA_CLIP}"
)

env_vars=(
  "PYTHONPATH=${ROOT_DIR}"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
)

{
  echo "cd ${DISPLAY_ROOT}"
  echo "RUN_TRAIN=1 bash scripts/experiments/run_stage60C_ccra_configD_5fold.sh"
  echo
  echo "# Optional: run secondary config C only if config D shows trade-off"
  echo "TARGET_CONFIG=C RUN_TRAIN=1 bash scripts/experiments/run_stage60C_ccra_configD_5fold.sh"
  echo
  echo "# Refresh Step60C summary after training"
  echo "PYTHONPATH=${DISPLAY_ROOT} HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG} ${PYTHON_BIN} scripts/analysis/build_stage60C_ccra_5fold_summary.py"
} > "${RUN_COMMANDS_FILE}"

echo "[Run] Step60C CCRA config ${CONFIG_LABEL} formal 5-fold validation"
echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
echo "[Run] EXP_CODE=${EXP_CODE}"
echo "[Run] TARGET_CONFIG=${TARGET_CONFIG}"
echo "[Run] K=${K_FOLDS}"
echo "[Run] K_START=${K_START}"
echo "[Run] K_END=${K_END}"
echo "[Run] MAX_EPOCHS=${MAX_EPOCHS}"
echo "[Run] SEED=${SEED}"
echo "[Run] CCRA_MODE=${CCRA_MODE}"
echo "[Run] CCRA_QUERY_SOURCE=${CCRA_QUERY_SOURCE}"
echo "[Run] CCRA_ALPHA_INIT=${CCRA_ALPHA_INIT}"
echo "[Run] CCRA_SCALE=${CCRA_SCALE}"
echo "[Run] CCRA_NORM=${CCRA_NORM}"
echo "[Run] CCRA_DROPOUT=${CCRA_DROPOUT}"
echo "[Run] CCRA_CLIP=${CCRA_CLIP}"
echo "[Run] RUN_COMMANDS_FILE=${RUN_COMMANDS_FILE}"
print_command env "${env_vars[@]}" "${local_cmd[@]}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  env "${env_vars[@]}" "${local_cmd[@]}"
  echo "[Done] Step60C 5-fold finished"
  echo "[Done] Results directory: ${RESULTS_DIR}/${EXP_CODE}_s${SEED}"
else
  echo "[Skip] RUN_TRAIN=${RUN_TRAIN}; command printed only."
fi
