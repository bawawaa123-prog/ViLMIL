#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results_stage19}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
RCE_CSG_INIT="${RCE_CSG_INIT:-0.05}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

cd "${ROOT_DIR}"

CMD=(
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
  --k 5
  --k_start 0
  --k_end 0
  --max_epochs 1
  --seed 1
  --prototype_number 16
  --testing
  --split_dir "${SPLIT_DIR}"
  --results_dir "${RESULTS_DIR}"
  --exp_code rce_v4_csg_smoke
  --rce_use_logit_calibration
  --rce_use_concept_prior
  --rce_use_visual_residual
  --rce_logit_scale_init 10.0
  --rce_concept_prior_strength 1.0
  --rce_visual_residual_init 0.05
  --rce_use_cross_scale_graph
  --rce_cross_scale_graph_init "${RCE_CSG_INIT}"
)

ENV_VARS=(
  "PYTHONPATH=${ROOT_DIR}"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
)

env "${ENV_VARS[@]}" "${CMD[@]}"

echo "[Done] RCE-MIL v4 CSG smoke test finished."
echo "[Done] Results directory: ${RESULTS_DIR}/rce_v4_csg_smoke_s1"
