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
RESULTS_DIR="${RESULTS_DIR:-results_stage43}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
K_START="${K_START:-0}"
K_END="${K_END:-0}"
SEED="${SEED:-1}"
VARIANT="${VARIANT:-hcrc_a005}"
HCRC_PROPOSAL_RADIUS="${HCRC_PROPOSAL_RADIUS:-4096}"
HCRC_NMS_RADIUS="${HCRC_NMS_RADIUS:-512}"
HCRC_BBOX_EXPAND="${HCRC_BBOX_EXPAND:-8}"
HCRC_NUM_ANCHORS="${HCRC_NUM_ANCHORS:-16}"
HCRC_NUM_HIGH_CHILDREN="${HCRC_NUM_HIGH_CHILDREN:-16}"
HCRC_CHILD_STRATEGY="${HCRC_CHILD_STRATEGY:-bbox_containment}"
HCRC_PROMPT_SCALE="${HCRC_PROMPT_SCALE:-high}"
HCRC_COORD_MODE="${HCRC_COORD_MODE:-top_left}"
HCRC_SCALE_RATIO="${HCRC_SCALE_RATIO:-1.0}"
HCRC_CANDIDATE_TOP_L="${HCRC_CANDIDATE_TOP_L:-64}"
HCRC_TOP_G_CONCEPTS="${HCRC_TOP_G_CONCEPTS:-8}"
HCRC_PER_CONCEPT_TOP_M="${HCRC_PER_CONCEPT_TOP_M:-4}"
HCRC_PROMPT_TOPK="${HCRC_PROMPT_TOPK:-3}"
HCRC_MARGIN_WEIGHT="${HCRC_MARGIN_WEIGHT:-0.5}"
HCRC_MIN_CHILD_COUNT="${HCRC_MIN_CHILD_COUNT:-1}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"

K_FOLDS=5
PROTOTYPE_NUMBER=16
VISUAL_RESIDUAL_INIT=0.05
CSG_INIT=0.1
HCRC_ALPHA_INIT="${HCRC_ALPHA_INIT:-0.05}"

cd "${ROOT_DIR}"

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

case "${VARIANT}" in
  hcrc_a005)
    HCRC_ALPHA_INIT="0.05"
    DEFAULT_EXP_CODE="stage43_hcrc_light_smoke_a005"
    ;;
  hcrc_a01)
    HCRC_ALPHA_INIT="0.1"
    DEFAULT_EXP_CODE="stage43_hcrc_light_smoke_a01"
    ;;
  hcrc_a002)
    HCRC_ALPHA_INIT="0.02"
    DEFAULT_EXP_CODE="stage43_hcrc_light_smoke_a002"
    ;;
  *)
    echo "[Error] Unsupported VARIANT: ${VARIANT}" >&2
    echo "[Error] Supported VARIANT values: hcrc_a005 hcrc_a01 hcrc_a002" >&2
    exit 1
    ;;
esac

EXP_CODE="${EXP_CODE:-${DEFAULT_EXP_CODE}}"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_CODE}_s${SEED}.log"

if [[ "${MODEL_TYPE}" != "deg_biomedclip" && "${MODEL_TYPE}" != "DEG_MIL_BiomedCLIP" ]]; then
  echo "[Error] MODEL_TYPE must be deg_biomedclip or DEG_MIL_BiomedCLIP for Step43 smoke." >&2
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
  --rce_use_hcrc
  --rce_hcrc_alpha_init "${HCRC_ALPHA_INIT}"
  --rce_hcrc_num_anchors "${HCRC_NUM_ANCHORS}"
  --rce_hcrc_num_high_children "${HCRC_NUM_HIGH_CHILDREN}"
  --rce_hcrc_proposal_radius "${HCRC_PROPOSAL_RADIUS}"
  --rce_hcrc_nms_radius "${HCRC_NMS_RADIUS}"
  --rce_hcrc_bbox_expand "${HCRC_BBOX_EXPAND}"
  --rce_hcrc_coord_mode "${HCRC_COORD_MODE}"
  --rce_hcrc_scale_ratio "${HCRC_SCALE_RATIO}"
  --rce_hcrc_child_strategy "${HCRC_CHILD_STRATEGY}"
  --rce_hcrc_candidate_top_l "${HCRC_CANDIDATE_TOP_L}"
  --rce_hcrc_top_g_concepts "${HCRC_TOP_G_CONCEPTS}"
  --rce_hcrc_per_concept_top_m "${HCRC_PER_CONCEPT_TOP_M}"
  --rce_hcrc_prompt_topk "${HCRC_PROMPT_TOPK}"
  --rce_hcrc_margin_weight "${HCRC_MARGIN_WEIGHT}"
  --rce_hcrc_prompt_scale "${HCRC_PROMPT_SCALE}"
  --rce_hcrc_min_child_count "${HCRC_MIN_CHILD_COUNT}"
  --rce_hcrc_export_debug
)

ENV_VARS=(
  "PYTHONPATH=${ROOT_DIR}"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
)

echo "[Run] Stage43 HCRC-Light smoke"
echo "[Run] VARIANT=${VARIANT}"
echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
echo "[Run] EXP_CODE=${EXP_CODE}"
echo "[Run] LOG_PATH=${LOG_PATH}"
echo "[Run] MAX_EPOCHS=${MAX_EPOCHS}"
echo "[Run] K_START=${K_START}"
echo "[Run] K_END=${K_END}"
echo "[Run] HCRC_ALPHA_INIT=${HCRC_ALPHA_INIT}"
echo "[Run] HCRC_PROPOSAL_RADIUS=${HCRC_PROPOSAL_RADIUS}"
echo "[Run] HCRC_NMS_RADIUS=${HCRC_NMS_RADIUS}"
echo "[Run] HCRC_BBOX_EXPAND=${HCRC_BBOX_EXPAND}"
echo "[Run] HCRC_NUM_ANCHORS=${HCRC_NUM_ANCHORS}"
echo "[Run] HCRC_NUM_HIGH_CHILDREN=${HCRC_NUM_HIGH_CHILDREN}"
echo "[Run] HCRC_CHILD_STRATEGY=${HCRC_CHILD_STRATEGY}"
echo "[Run] HCRC_PROMPT_SCALE=${HCRC_PROMPT_SCALE}"
print_command env "${ENV_VARS[@]}" "${CMD[@]}"

env "${ENV_VARS[@]}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"

echo "[Done] Stage43 smoke finished."
echo "[Done] Results directory: ${RESULTS_DIR}/${EXP_CODE}_s${SEED}"
echo "[Done] Log path: ${LOG_PATH}"
