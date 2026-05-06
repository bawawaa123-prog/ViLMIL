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
LR="${LR:-1e-4}"
PROTO_NUM="${PROTO_NUM:-16}"
PEPS_TOPK="${PEPS_TOPK:-5}"
PEPS_TAU="${PEPS_TAU:-0.07}"
SCALE_GATE_HIDDEN_DIM="${SCALE_GATE_HIDDEN_DIM:-128}"
SCALE_GATE_DROPOUT="${SCALE_GATE_DROPOUT:-0.25}"
SCALE_RESIDUAL_GAMMA="${SCALE_RESIDUAL_GAMMA:-0.25}"

cd "${ROOT_DIR}"

echo "[1/4] Smoke train: residual SAF-PEPS, fold0, 1 epoch"
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
  --prototype_number "${PROTO_NUM}" \
  --max_epochs 1 \
  --lr "${LR}" \
  --seed "${SEED}" \
  --early_stopping \
  --testing \
  --k "${K_FOLDS}" \
  --k_start 0 \
  --k_end 0 \
  --split_dir "${SPLIT_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --exp_code smoke_residual_saf_peps \
  --scale_fusion_mode residual_gate \
  --scale_gate_hidden_dim "${SCALE_GATE_HIDDEN_DIM}" \
  --scale_gate_dropout "${SCALE_GATE_DROPOUT}" \
  --scale_residual_gamma "${SCALE_RESIDUAL_GAMMA}"

echo "[2/4] Full 5-fold train: residual SAF-PEPS"
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
  --prototype_number "${PROTO_NUM}" \
  --max_epochs "${MAX_EPOCHS}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --early_stopping \
  --k "${K_FOLDS}" \
  --k_start 0 \
  --k_end 4 \
  --split_dir "${SPLIT_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --exp_code adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25 \
  --scale_fusion_mode residual_gate \
  --scale_gate_hidden_dim "${SCALE_GATE_HIDDEN_DIM}" \
  --scale_gate_dropout "${SCALE_GATE_DROPOUT}" \
  --scale_residual_gamma "${SCALE_RESIDUAL_GAMMA}"

echo "[3/4] Full test evaluation"
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
  --models_exp_code adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25_s1 \
  --save_exp_code adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25 \
  --split test \
  --k "${K_FOLDS}" \
  --k_start 0 \
  --k_end 5 \
  --text_prompt_path "${TEXT_PROMPT_PATH}" \
  --use_concept_prompt_pool \
  --concept_prompt_path "${CONCEPT12_PATH}" \
  --prompt_ensemble_mode peps \
  --peps_topk "${PEPS_TOPK}" \
  --peps_tau "${PEPS_TAU}" \
  --save_peps_weights \
  --prototype_number "${PROTO_NUM}" \
  --scale_fusion_mode residual_gate \
  --scale_gate_hidden_dim "${SCALE_GATE_HIDDEN_DIM}" \
  --scale_gate_dropout "${SCALE_GATE_DROPOUT}" \
  --scale_residual_gamma "${SCALE_RESIDUAL_GAMMA}"

echo "[4/4] Build residual SAF comparison summary"
PYTHONPATH="${ROOT_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/build_stage7_1_residual_saf_analysis.py"

echo "[Done] Outputs:"
echo "  Train dir: ${ROOT_DIR}/trained_models/adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25_s1"
echo "  Eval dir : ${ROOT_DIR}/eval_results/EVAL_adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25"
echo "  Summary  : ${ROOT_DIR}/trained_models/stage7_1_residual_saf_comparison/residual_saf_comparison_summary.csv"
