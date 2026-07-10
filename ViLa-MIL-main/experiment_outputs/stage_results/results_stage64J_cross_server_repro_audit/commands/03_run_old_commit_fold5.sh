#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/private/ljh-data/shared/ViLMIL/hf_cache}"
DATA_ROOT="/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/yiyuan"
PROMPT_PATH="/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv"
CONCEPT_PATH="/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json"
SPLIT_DIR="/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main/splits/adenocarcinoma/task_adenocarcinoma_strictcv_100"
RESULTS_DIR="/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/experiment_outputs/stage_results/results_stage64J_oldcode_fold5_206200"
EXP_CODE="stage64J_oldcode_fold5_206200"

export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

cmd=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/main.py"
  --task task_adenocarcinoma
  --model_type RCE_MIL_BiomedCLIP_v2
  --mode transformer
  --data_root_dir "${DATA_ROOT}"
  --data_folder_s features_biomedclip_5x
  --data_folder_l features_biomedclip_20x
  --text_prompt_path "${PROMPT_PATH}"
  --use_concept_prompt_pool
  --concept_prompt_path "${CONCEPT_PATH}"
  --prompt_ensemble_mode embedding_mean
  --scale_mode dual
  --k 5
  --k_start 4
  --k_end 4
  --max_epochs 20
  --seed 1
  --lr 1e-4
  --label_frac 1.0
  --bag_loss ce
  --opt adam
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
)

echo "Git commit: $(git -C "${PROJECT_ROOT}/.." rev-parse HEAD)"
echo "Python: ${PYTHON_BIN}"
echo "HF cache: ${HF_CACHE_DIR}"
echo "data root: ${DATA_ROOT}"
echo "split: ${SPLIT_DIR}"
echo "output dir: ${RESULTS_DIR}"
echo "CUDA device: 0"
echo "cwd: ${PROJECT_ROOT}"
printf 'Command: CUDA_VISIBLE_DEVICES=0 HF_HOME=%q HUGGINGFACE_HUB_CACHE=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=%q ' "${HF_CACHE_DIR}" "${HF_CACHE_DIR}" "${PROJECT_ROOT}"
printf '%q ' "${cmd[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PROJECT_ROOT}" "${cmd[@]}"
