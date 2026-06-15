#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
DATA_FOLDER_S="${DATA_FOLDER_S:-features_biomedclip_5x}"
DATA_FOLDER_L="${DATA_FOLDER_L:-features_biomedclip_20x}"
CSV_PATH="${CSV_PATH:-${ROOT_DIR}/dataset_csv/all_data.csv}"
SPLIT_DIR="${SPLIT_DIR:-${ROOT_DIR}/splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results_stage41/low_high_coordinate_audit}"
MAX_SLIDES="${MAX_SLIDES:-0}"
LOW_PATCH_SIZE="${LOW_PATCH_SIZE:-256}"
HIGH_PATCH_SIZE="${HIGH_PATCH_SIZE:-256}"
BBOX_EXPAND="${BBOX_EXPAND:-1.0}"
TRY_SCALE_RATIOS="${TRY_SCALE_RATIOS:-1,2,4,0.5,0.25}"
COORD_MODE="${COORD_MODE:-auto}"
NUM_EXAMPLES="${NUM_EXAMPLES:-20}"
SEED="${SEED:-1}"

CMD=(
    "${PYTHON_BIN}" "scripts/analysis/build_stage41_low_high_coordinate_audit.py"
    "--data_root_dir" "${DATA_ROOT_DIR}"
    "--data_folder_s" "${DATA_FOLDER_S}"
    "--data_folder_l" "${DATA_FOLDER_L}"
    "--csv_path" "${CSV_PATH}"
    "--split_dir" "${SPLIT_DIR}"
    "--fold" "${FOLD}"
    "--split" "${SPLIT}"
    "--output_dir" "${OUTPUT_DIR}"
    "--max_slides" "${MAX_SLIDES}"
    "--low_patch_size" "${LOW_PATCH_SIZE}"
    "--high_patch_size" "${HIGH_PATCH_SIZE}"
    "--bbox_expand" "${BBOX_EXPAND}"
    "--try_scale_ratios" "${TRY_SCALE_RATIOS}"
    "--coord_mode" "${COORD_MODE}"
    "--num_examples" "${NUM_EXAMPLES}"
    "--seed" "${SEED}"
)

printf 'Running Step41 low-high coordinate audit:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
