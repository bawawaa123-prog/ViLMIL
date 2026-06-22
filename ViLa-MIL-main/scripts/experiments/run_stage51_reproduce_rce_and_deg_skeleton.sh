#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-results_stage51_repro}"
MODE="${MODE:-both}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE_FLAG:-1}"
TRANSFORMERS_OFFLINE_FLAG="${TRANSFORMERS_OFFLINE_FLAG:-1}"

K_FOLDS=5
K_START=0
K_END=4
PROTOTYPE_NUMBER=16
CSG_INIT=0.1
RCE_EXP_CODE="rce_step23_rq16_5fold_e20"
DEG_EXP_CODE="deg_skeleton_rq16_5fold_e20"

cd "${ROOT_DIR}"

print_command() {
  local -a cmd=("$@")
  printf "[Cmd] "
  printf "%q " "${cmd[@]}"
  printf "\n"
}

ensure_safe_results_dir() {
  case "${RESULTS_DIR}" in
    results_stage23|results_stage23/*|results_stage27|results_stage27/*)
      echo "[Error] Step51 must not write into historical result directories: ${RESULTS_DIR}" >&2
      exit 1
      ;;
  esac
}

run_reproduction() {
  local mode_name="$1"
  local model_type="$2"
  local exp_code="$3"
  shift 3
  local -a extra_args=("$@")

  local -a common_args=(
    --task task_adenocarcinoma
    --model_type "${model_type}"
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
    --prototype_number "${PROTOTYPE_NUMBER}"
    --split_dir "${SPLIT_DIR}"
    --results_dir "${RESULTS_DIR}"
    --exp_code "${exp_code}"
    --rce_use_logit_calibration
    --rce_use_concept_prior
    --rce_use_visual_residual
    --rce_logit_scale_init 10.0
    --rce_concept_prior_strength 1.0
    --rce_visual_residual_init 0.05
    --rce_use_cross_scale_graph
    --rce_cross_scale_graph_init "${CSG_INIT}"
    --rce_cross_scale_graph_norm sqrt
  )

  local -a cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    "${common_args[@]}"
    "${extra_args[@]}"
  )

  local -a env_vars=(
    "PYTHONPATH=${ROOT_DIR}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_FLAG}"
  )

  echo "[Run] Step51 reproduction path=${mode_name}"
  echo "[Run] MODEL_TYPE=${model_type}"
  echo "[Run] RESULTS_DIR=${RESULTS_DIR}"
  echo "[Run] EXP_CODE=${exp_code}"
  echo "[Run] PROTOTYPE_NUMBER=${PROTOTYPE_NUMBER}"
  echo "[Run] CSG_INIT=${CSG_INIT}"
  echo "[Run] K=${K_FOLDS}"
  echo "[Run] K_START=${K_START}"
  echo "[Run] K_END=${K_END}"
  echo "[Run] MAX_EPOCHS=${MAX_EPOCHS}"
  echo "[Run] SEED=${SEED}"
  if [[ "${model_type}" == "DEG_MIL_BiomedCLIP" ]]; then
    echo "[Run] DEG skeleton extras remain disabled:"
    echo "[Run]   deg_use_region_graph=off"
    echo "[Run]   deg_use_concept_graph=off"
    echo "[Run]   rce_use_visual_evidence_gate=off"
    echo "[Run]   rce_use_prarc_gate=off"
    echo "[Run]   rce_use_hcrc=off"
    echo "[Run]   rce_use_low_high_consistency_loss=off"
  fi
  print_command env "${env_vars[@]}" "${cmd[@]}"

  env "${env_vars[@]}" "${cmd[@]}"

  echo "[Done] path finished: ${mode_name}"
  echo "[Done] Results directory: ${RESULTS_DIR}/${exp_code}_s${SEED}"
}

ensure_safe_results_dir

case "${MODE}" in
  rce)
    run_reproduction "rce" "RCE_MIL_BiomedCLIP" "${RCE_EXP_CODE}"
    ;;
  deg)
    run_reproduction "deg" "DEG_MIL_BiomedCLIP" "${DEG_EXP_CODE}"
    ;;
  both)
    run_reproduction "rce" "RCE_MIL_BiomedCLIP" "${RCE_EXP_CODE}"
    run_reproduction "deg" "DEG_MIL_BiomedCLIP" "${DEG_EXP_CODE}"
    ;;
  *)
    echo "[Error] Unsupported MODE: ${MODE}" >&2
    echo "[Error] Supported MODE values: rce, deg, both" >&2
    exit 1
    ;;
esac
