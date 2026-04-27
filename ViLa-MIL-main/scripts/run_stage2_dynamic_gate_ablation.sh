#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/xiangmu/ViLMIL/ViLa-MIL-main"
PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/trained_models}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${ROOT_DIR}/eval_results}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"

CONCEPT12_EMBED_DIR="${CONCEPT12_EMBED_DIR:-${ROOT_DIR}/trained_models/adeno_concept12_embedding_mean_s1}"

MAX_EPOCHS="${MAX_EPOCHS:-40}"
SEED="${SEED:-1}"
K_FOLDS="${K_FOLDS:-5}"
K_START="${K_START:-0}"
K_END="${K_END:-4}"
LR="${LR:-1e-4}"
PROTO_NUM="${PROTO_NUM:-16}"
GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-256}"
RUN_DROPOUT="${RUN_DROPOUT:-0}"

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
    --use_concept_prompt_pool \
    --concept_prompt_path "${CONCEPT12_PATH}" \
    --prompt_ensemble_mode dynamic_gate \
    --use_dynamic_prompt_gate \
    --dynamic_gate_hidden_dim "${GATE_HIDDEN_DIM}" \
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

run_eval() {
  local models_exp_code="$1"
  local save_exp_code="$2"
  shift 2

  PYTHONPATH="${ROOT_DIR}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${PYTHON_BIN}" "${ROOT_DIR}/eval.py" \
    --task task_adenocarcinoma \
    --model_type ViLa_MIL_BiomedCLIP \
    --mode transformer \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --data_folder_s features_biomedclip_5x \
    --data_folder_l features_biomedclip_20x \
    --results_dir "${RESULTS_DIR}" \
    --models_exp_code "${models_exp_code}" \
    --save_exp_code "${save_exp_code}" \
    --split test \
    --k "${K_FOLDS}" \
    --k_start "${K_START}" \
    --k_end "$((K_END + 1))" \
    --text_prompt_path "${TEXT_PROMPT_PATH}" \
    --use_concept_prompt_pool \
    --concept_prompt_path "${CONCEPT12_PATH}" \
    --prompt_ensemble_mode dynamic_gate \
    --use_dynamic_prompt_gate \
    --dynamic_gate_hidden_dim "${GATE_HIDDEN_DIM}" \
    --prototype_number "${PROTO_NUM}" \
    "$@"
}

echo "[0/4] Reuse existing Concept-12 embedding_mean"
echo "Reuse result: ${CONCEPT12_EMBED_DIR}"

echo "[1/4] Train Concept-12 dynamic_gate"
run_train "adeno_concept12_dynamic_gate"

echo "[2/4] Eval Concept-12 dynamic_gate (export prompt weights)"
run_eval "adeno_concept12_dynamic_gate_s${SEED}" "adeno_concept12_dynamic_gate_eval"

echo "[3/4] Train Concept-12 dynamic_gate + residual mean"
run_train "adeno_concept12_dynamic_gate_resmean" \
  --dynamic_gate_residual_mean

echo "[4/4] Eval Concept-12 dynamic_gate + residual mean (export prompt weights)"
run_eval "adeno_concept12_dynamic_gate_resmean_s${SEED}" "adeno_concept12_dynamic_gate_resmean_eval" \
  --dynamic_gate_residual_mean

if [[ "${RUN_DROPOUT}" == "1" ]]; then
  echo "[Optional] Train Concept-12 dynamic_gate + residual mean + prompt_dropout=0.1"
  run_train "adeno_concept12_dynamic_gate_resmean_drop01" \
    --dynamic_gate_residual_mean \
    --prompt_dropout 0.1

  echo "[Optional] Eval Concept-12 dynamic_gate + residual mean + prompt_dropout=0.1"
  run_eval "adeno_concept12_dynamic_gate_resmean_drop01_s${SEED}" "adeno_concept12_dynamic_gate_resmean_drop01_eval" \
    --dynamic_gate_residual_mean \
    --prompt_dropout 0.1
fi
