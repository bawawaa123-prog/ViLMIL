#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/xiangmu/ViLMIL/ViLa-MIL-main"
PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/trained_models}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"

MAX_EPOCHS="${MAX_EPOCHS:-40}"
SEED="${SEED:-1}"
K_FOLDS="${K_FOLDS:-5}"
K_START="${K_START:-0}"
K_END="${K_END:-4}"
LR="${LR:-1e-4}"
PROTO_NUM="${PROTO_NUM:-16}"
PEPS_TOPK="${PEPS_TOPK:-5}"
PEPS_TAU="${PEPS_TAU:-0.07}"

cd "${ROOT_DIR}"

run_train() {
  local exp_code="$1"
  local scale_mode="$2"

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
    --peps_topk "${PEPS_TOPK}" \
    --peps_tau "${PEPS_TAU}" \
    --scale_mode "${scale_mode}" \
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
  local scale_mode="$3"

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
    --peps_topk "${PEPS_TOPK}" \
    --peps_tau "${PEPS_TAU}" \
    --scale_mode "${scale_mode}" \
    --save_peps_weights \
    --prototype_number "${PROTO_NUM}"
}

echo "[Reuse] Primary model: ${ROOT_DIR}/trained_models/adeno_concept12_peps_topk5_tau0.07_s1"
echo "[Reuse] AUC-first model: ${ROOT_DIR}/trained_models/adeno_concept12_peps_topk5_tau0.1_s1"
echo "[Reuse] Embedding baseline: ${ROOT_DIR}/trained_models/adeno_concept12_embedding_mean_s1"
echo "[Reuse] Dynamic gate: ${ROOT_DIR}/trained_models/adeno_concept12_dynamic_gate_s1"
echo "[Reuse] Static baseline: ${ROOT_DIR}/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1"

echo "[Train] PEPS low-only"
run_train "adeno_concept12_peps_topk5_tau0.07_lowonly" "low_only"
echo "[Eval] PEPS low-only"
run_eval "adeno_concept12_peps_topk5_tau0.07_lowonly_s${SEED}" "adeno_concept12_peps_topk5_tau0.07_lowonly" "low_only"

echo "[Train] PEPS high-only"
run_train "adeno_concept12_peps_topk5_tau0.07_highonly" "high_only"
echo "[Eval] PEPS high-only"
run_eval "adeno_concept12_peps_topk5_tau0.07_highonly_s${SEED}" "adeno_concept12_peps_topk5_tau0.07_highonly" "high_only"

echo "[Analyze] Build final DCP-ViLa-MIL tables and report"
PYTHONPATH="${ROOT_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_final_dcp_vila_analysis.py"

echo "[Done] Final outputs are under ${ROOT_DIR}/trained_models/final_dcp_vila_analysis"
