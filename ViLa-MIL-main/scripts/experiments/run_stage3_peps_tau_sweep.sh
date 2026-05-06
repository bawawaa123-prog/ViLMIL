#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/xiangmu/ViLMIL/ViLa-MIL-main"
PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/trained_models}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"

CONCEPT12_EMBED_DIR="${CONCEPT12_EMBED_DIR:-${ROOT_DIR}/trained_models/adeno_concept12_embedding_mean_s1}"
CONCEPT12_GATE_DIR="${CONCEPT12_GATE_DIR:-${ROOT_DIR}/trained_models/adeno_concept12_dynamic_gate_s1}"
PEPS_TOPK3_TAU01_DIR="${PEPS_TOPK3_TAU01_DIR:-${ROOT_DIR}/trained_models/adeno_concept12_peps_topk3_tau0.1_s1}"
PEPS_TOPK5_TAU01_DIR="${PEPS_TOPK5_TAU01_DIR:-${ROOT_DIR}/trained_models/adeno_concept12_peps_topk5_tau0.1_s1}"

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
  local peps_topk="$2"
  local peps_tau="$3"

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
    --prompt_ensemble_mode peps \
    --peps_topk "${peps_topk}" \
    --peps_tau "${peps_tau}" \
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
    --early_stopping
}

run_eval() {
  local models_exp_code="$1"
  local save_exp_code="$2"
  local peps_topk="$3"
  local peps_tau="$4"

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
    --prompt_ensemble_mode peps \
    --peps_topk "${peps_topk}" \
    --peps_tau "${peps_tau}" \
    --save_peps_weights \
    --prototype_number "${PROTO_NUM}"
}

echo "[Reuse] Concept-12 embedding_mean: ${CONCEPT12_EMBED_DIR}"
echo "[Reuse] Concept-12 MLP dynamic_gate: ${CONCEPT12_GATE_DIR}"
echo "[Reuse] PEPS topk=3 tau=0.1: ${PEPS_TOPK3_TAU01_DIR}"
echo "[Reuse] PEPS topk=5 tau=0.1: ${PEPS_TOPK5_TAU01_DIR}"

for SPEC in "3 0.07" "3 0.15" "5 0.07" "5 0.15"; do
  read -r K TAU <<<"${SPEC}"
  EXP_CODE="adeno_concept12_peps_topk${K}_tau${TAU}"
  SAVE_CODE="adeno_concept12_peps_topk${K}_tau${TAU}"

  echo "[Train] ${EXP_CODE}"
  run_train "${EXP_CODE}" "${K}" "${TAU}"

  echo "[Eval] ${EXP_CODE}"
  run_eval "${EXP_CODE}_s${SEED}" "${SAVE_CODE}" "${K}" "${TAU}"
done

echo "[Aggregate] Extended PEPS summary"
PYTHONPATH="${ROOT_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/aggregate_stage3_peps_extended_results.py"

echo "[Analyze] Fold-level deltas"
PYTHONPATH="${ROOT_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/analyze_stage3_peps_fold_deltas.py"

echo "[Analyze] Prompt usage diversity"
PYTHONPATH="${ROOT_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/analyze_stage3_peps_prompt_usage.py"

echo "[Report] Build markdown report"
PYTHONPATH="${ROOT_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/build_stage3_peps_extended_report.py"

echo "[Done] Outputs are under ${ROOT_DIR}/trained_models/stage3_peps_comparison"
