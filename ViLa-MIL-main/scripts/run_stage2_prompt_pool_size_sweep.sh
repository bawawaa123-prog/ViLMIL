#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/xiangmu/ViLMIL/ViLa-MIL-main"
PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/trained_models}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"

STATIC_BASELINE_DIR="${STATIC_BASELINE_DIR:-${ROOT_DIR}/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1}"
CONCEPT6_RESULT_DIR="${CONCEPT6_RESULT_DIR:-${ROOT_DIR}/trained_models/adeno_stage2_concept_mean_s1}"
CONCEPT10_RESULT_DIR="${CONCEPT10_RESULT_DIR:-${ROOT_DIR}/trained_models/adeno_concept10_embedding_mean_s1}"

CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
CONCEPT14_PATH="${CONCEPT14_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core14.json}"

MAX_EPOCHS="${MAX_EPOCHS:-40}"
SEED="${SEED:-1}"
K_FOLDS="${K_FOLDS:-5}"
K_START="${K_START:-0}"
K_END="${K_END:-4}"
LR="${LR:-1e-4}"
PROTO_NUM="${PROTO_NUM:-16}"

cd "${ROOT_DIR}"

run_train() {
  local exp_code="$1"
  shift

  PYTHONPATH="${ROOT_DIR}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${PYTHON_BIN}" "${ROOT_DIR}/main.py" \
    --task task_adenocarcinoma \
    --model_type ViLa_MIL_BiomedCLIP \
    --mode transformer \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --data_folder_s features_biomedclip_5x \
    --data_folder_l features_biomedclip_20x \
    --text_prompt_path "${TEXT_PROMPT_PATH}" \
    --split_dir "${SPLIT_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --exp_code "${exp_code}" \
    --k "${K_FOLDS}" \
    --k_start "${K_START}" \
    --k_end "${K_END}" \
    --max_epochs "${MAX_EPOCHS}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --prototype_number "${PROTO_NUM}" \
    --early_stopping \
    "$@"
}

echo "[1/5] BiomedCLIP static prompt baseline"
echo "Skip rerun. Reuse existing result: ${STATIC_BASELINE_DIR}"

echo "[2/5] Concept-6 embedding_mean"
echo "Skip rerun. Reuse existing result: ${CONCEPT6_RESULT_DIR}"

echo "[3/5] Concept-10 embedding_mean"
echo "Skip rerun. Reuse existing result: ${CONCEPT10_RESULT_DIR}"

echo "[4/5] Concept-12 embedding_mean"
run_train "adeno_concept12_embedding_mean" \
  --use_concept_prompt_pool \
  --concept_prompt_path "${CONCEPT12_PATH}" \
  --prompt_ensemble_mode embedding_mean

echo "[5/5] Concept-14 embedding_mean"
run_train "adeno_concept14_embedding_mean" \
  --use_concept_prompt_pool \
  --concept_prompt_path "${CONCEPT14_PATH}" \
  --prompt_ensemble_mode embedding_mean
