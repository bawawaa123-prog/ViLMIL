#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ljh/anaconda3/envs/vila_mil/bin/python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results_stage32/stage32_rce_v4_csg_evidence_export}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-test}"
CKPT_PATH="${CKPT_PATH:-}"
CONCEPT12_PATH="${CONCEPT12_PATH:-$ROOT_DIR/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json}"
SPLIT_DIR="${SPLIT_DIR:-$ROOT_DIR/splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"

cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/analysis/export_stage32_rce_v4_csg_evidence.py"
  --data_root_dir "$DATA_ROOT_DIR"
  --results_dir "$RESULTS_DIR"
  --output_dir "$OUTPUT_DIR"
  --fold "$FOLD"
  --split "$SPLIT"
  --concept_prompt_path "$CONCEPT12_PATH"
  --split_dir "$SPLIT_DIR"
)

if [[ -n "$CKPT_PATH" ]]; then
  cmd+=(--ckpt_path "$CKPT_PATH")
fi

"${cmd[@]}"
