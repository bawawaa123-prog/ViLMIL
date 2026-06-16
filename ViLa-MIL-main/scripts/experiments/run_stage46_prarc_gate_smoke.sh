#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
DATA_FOLDER_S="${DATA_FOLDER_S:-features_biomedclip_5x}"
DATA_FOLDER_L="${DATA_FOLDER_L:-features_biomedclip_20x}"
TASK="${TASK:-task_adenocarcinoma}"
MODEL_TYPE="${MODEL_TYPE:-deg_biomedclip}"
RESULTS_DIR="${RESULTS_DIR:-results_stage46}"
EXP_CODE="${EXP_CODE:-stage46_prarc_gate_smoke_v1_g08_s1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
K_START="${K_START:-0}"
K_END="${K_END:-0}"
SEED="${SEED:-1}"
PRARC_GATE_INIT="${PRARC_GATE_INIT:-0.8}"
PRARC_GATE_HIDDEN_DIM="${PRARC_GATE_HIDDEN_DIM:-16}"
PRARC_FEATURE_SET="${PRARC_FEATURE_SET:-v1}"
PRARC_FEATURE_CLIP="${PRARC_FEATURE_CLIP:-10.0}"
VARIANT="${VARIANT:-prarc_v1_g08}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

K_FOLDS=5
PROTOTYPE_NUMBER=16
VISUAL_RESIDUAL_INIT=0.05
CSG_INIT=0.1

cd "${ROOT_DIR}"

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

case "${VARIANT}" in
  prarc_v1_g08)
    PRARC_GATE_INIT="${PRARC_GATE_INIT:-0.8}"
    PRARC_GATE_HIDDEN_DIM="${PRARC_GATE_HIDDEN_DIM:-16}"
    DEFAULT_EXP_CODE="stage46_prarc_gate_smoke_v1_g08"
    ;;
  prarc_v1_g05)
    PRARC_GATE_INIT="0.5"
    PRARC_GATE_HIDDEN_DIM="${PRARC_GATE_HIDDEN_DIM:-16}"
    DEFAULT_EXP_CODE="stage46_prarc_gate_smoke_v1_g05"
    ;;
  prarc_v1_g10)
    PRARC_GATE_INIT="0.99"
    PRARC_GATE_HIDDEN_DIM="${PRARC_GATE_HIDDEN_DIM:-16}"
    DEFAULT_EXP_CODE="stage46_prarc_gate_smoke_v1_g10"
    ;;
  prarc_v1_h32_g08)
    PRARC_GATE_INIT="0.8"
    PRARC_GATE_HIDDEN_DIM="32"
    DEFAULT_EXP_CODE="stage46_prarc_gate_smoke_v1_h32_g08"
    ;;
  *)
    echo "[Error] Unsupported VARIANT: ${VARIANT}" >&2
    echo "[Error] Supported VARIANT values: prarc_v1_g08 prarc_v1_g05 prarc_v1_g10 prarc_v1_h32_g08" >&2
    exit 1
    ;;
esac

if [[ -z "${EXP_CODE:-}" ]]; then
  EXP_CODE="${DEFAULT_EXP_CODE}"
fi

LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_CODE}_s${SEED}.log"

if [[ "${MODEL_TYPE}" != "deg_biomedclip" && "${MODEL_TYPE}" != "DEG_MIL_BiomedCLIP" ]]; then
  echo "[Error] MODEL_TYPE must be deg_biomedclip or DEG_MIL_BiomedCLIP for Step46 smoke." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
  --task "${TASK}"
  --model_type DEG_MIL_BiomedCLIP
  --mode transformer
  --data_root_dir "${DATA_ROOT_DIR}"
  --data_folder_s "${DATA_FOLDER_S}"
  --data_folder_l "${DATA_FOLDER_L}"
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
  --exp_code "${EXP_CODE}"
  --rce_use_logit_calibration
  --rce_use_concept_prior
  --rce_use_visual_residual
  --rce_visual_residual_init "${VISUAL_RESIDUAL_INIT}"
  --rce_use_cross_scale_graph
  --rce_cross_scale_graph_init "${CSG_INIT}"
  --rce_cross_scale_graph_norm sqrt
  --rce_use_prarc_gate
  --rce_prarc_gate_init "${PRARC_GATE_INIT}"
  --rce_prarc_gate_hidden_dim "${PRARC_GATE_HIDDEN_DIM}"
  --rce_prarc_gate_feature_set "${PRARC_FEATURE_SET}"
  --rce_prarc_feature_clip "${PRARC_FEATURE_CLIP}"
  --rce_prarc_detach_features
  --rce_prarc_export_debug
)

ENV_VARS=(
  "PYTHONPATH=${ROOT_DIR}"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
)

echo "[Run] Stage46 PRARC gate smoke"
echo "[Run] VARIANT=${VARIANT}"
echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
echo "[Run] EXP_CODE=${EXP_CODE}"
echo "[Run] LOG_PATH=${LOG_PATH}"
echo "[Run] MAX_EPOCHS=${MAX_EPOCHS}"
echo "[Run] K_START=${K_START}"
echo "[Run] K_END=${K_END}"
echo "[Run] PRARC_GATE_INIT=${PRARC_GATE_INIT}"
echo "[Run] PRARC_GATE_HIDDEN_DIM=${PRARC_GATE_HIDDEN_DIM}"
echo "[Run] PRARC_FEATURE_SET=${PRARC_FEATURE_SET}"
echo "[Run] PRARC_FEATURE_CLIP=${PRARC_FEATURE_CLIP}"
print_command env "${ENV_VARS[@]}" "${CMD[@]}"

env "${ENV_VARS[@]}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"

echo "[Done] Stage46 smoke finished."
echo "[Done] Results directory: ${RESULTS_DIR}/${EXP_CODE}_s${SEED}"
echo "[Done] Log path: ${LOG_PATH}"
