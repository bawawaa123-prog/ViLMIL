#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results_stage9}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
TEXT_PROMPT_PATH="${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
CONCEPT12_PATH="${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
VARIANT="${VARIANT:-prior_calib}"

cd "${ROOT_DIR}"

run_variant() {
  local variant="$1"
  local exp_code=""
  local extra_args=()

  case "${variant}" in
    prior_calib)
      exp_code="rce_mil_v2_prior_calib_5fold_e20"
      extra_args=(
        --rce_use_logit_calibration
        --rce_use_concept_prior
        --rce_logit_scale_init 10.0
        --rce_concept_prior_strength 1.0
      )
      ;;
    prior_only)
      exp_code="rce_mil_v2_prior_5fold_e20"
      extra_args=(
        --rce_use_concept_prior
        --rce_concept_prior_strength 1.0
      )
      ;;
    *)
      echo "[Error] Unsupported variant: ${variant}" >&2
      echo "[Error] Supported VARIANT values: prior_calib, prior_only, all" >&2
      exit 1
      ;;
  esac

  echo "[Run] RCE-MIL v2 5-fold variant: ${variant}"

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
    --k_end 4
    --max_epochs 20
    --seed 1
    --prototype_number 16
    --split_dir "${SPLIT_DIR}"
    --results_dir "${RESULTS_DIR}"
    --exp_code "${exp_code}"
    "${extra_args[@]}"
  )

  PYTHONPATH="${ROOT_DIR}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${CMD[@]}"

  echo "[Done] variant finished: ${variant}"
  echo "[Done] Results directory: ${RESULTS_DIR}/${exp_code}_s1"
}

case "${VARIANT}" in
  prior_calib|prior_only)
    run_variant "${VARIANT}"
    ;;
  all)
    run_variant prior_calib
    run_variant prior_only
    ;;
  *)
    echo "[Error] Unsupported VARIANT: ${VARIANT}" >&2
    echo "[Error] Supported VARIANT values: prior_calib, prior_only, all" >&2
    exit 1
    ;;
esac
