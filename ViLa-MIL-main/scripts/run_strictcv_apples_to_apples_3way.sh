#!/usr/bin/env bash
set -euo pipefail

# Fair 3-way rerun on the strict 5-fold adenocarcinoma split:
# 1) baseline (static prompts only)
# 2) p1+p3 (dynamic prompt + offline RAG)
# 3) p1+p2+p3 (dynamic prompt + offline RAG + VCP)
#
# This script performs strict preflight checks before training:
# - split integrity
# - exact 5-fold membership pattern
# - offline RAG cache coverage
#
# Usage:
#   cd /xiangmu/ViLMIL/ViLa-MIL-main
#   bash scripts/run_strictcv_apples_to_apples_3way.sh
#
# Optional overrides:
#   ONLY=static bash scripts/run_strictcv_apples_to_apples_3way.sh
#   MAX_EPOCHS=50 RUN_TAG=myrerun bash scripts/run_strictcv_apples_to_apples_3way.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ONLY="${ONLY:-all}"

DATA_CSV="${DATA_CSV:-$REPO_ROOT/dataset_csv/all_data.csv}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/xiangmu/data/VILMIL}"
DATA_FOLDER_S="${DATA_FOLDER_S:-features_biomedclip_5x}"
DATA_FOLDER_L="${DATA_FOLDER_L:-features_biomedclip_20x}"
SPLIT_DIR="${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/trained_models}"

STATIC_PROMPT="${STATIC_PROMPT:-$REPO_ROOT/text_prompt/adenocarcinoma_dual_scale_prompt.csv}"
PROMPT_POOL="${PROMPT_POOL:-$REPO_ROOT/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv}"
RAG_CACHE="${RAG_CACHE:-$REPO_ROOT/results/rag_cache_qwen14b_fold0_prefill.jsonl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
LR="${LR:-1e-4}"
PROTOTYPE_NUMBER="${PROTOTYPE_NUMBER:-16}"

DYN_WARMUP_EPOCHS="${DYN_WARMUP_EPOCHS:-4}"
RAG_START_EPOCH="${RAG_START_EPOCH:-16}"
VCP_START_EPOCH="${VCP_START_EPOCH:-16}"

RETRIEVAL_TOPK="${RETRIEVAL_TOPK:-2}"
RETRIEVAL_TEMP="${RETRIEVAL_TEMP:-0.15}"
DYNAMIC_PROMPT_MIX="${DYNAMIC_PROMPT_MIX:-0.8}"

VCP_BETA="${VCP_BETA:-0.03}"
VCP_DROPOUT="${VCP_DROPOUT:-0.1}"

RAG_TOPK="${RAG_TOPK:-3}"
RAG_MODEL="${RAG_MODEL:-qwen2.5:14b-instruct}"
RAG_URL="${RAG_URL:-http://localhost:11434/api/generate}"
RAG_TEMPERATURE="${RAG_TEMPERATURE:-0.2}"
RAG_MAX_TOKENS="${RAG_MAX_TOKENS:-256}"
RAG_TIMEOUT_SEC="${RAG_TIMEOUT_SEC:-60}"
RAG_MAX_RETRIES="${RAG_MAX_RETRIES:-2}"
RAG_RETRY_DELAY_SEC="${RAG_RETRY_DELAY_SEC:-0.5}"
RAG_FALLBACK="${RAG_FALLBACK:-dynamic}"

RUN_TAG="${RUN_TAG:-strictcv_a2a_e${MAX_EPOCHS}_s${SEED}}"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
}

echo "== Preflight =="
require_path "$DATA_CSV"
require_path "$DATA_ROOT_DIR/$DATA_FOLDER_S"
require_path "$DATA_ROOT_DIR/$DATA_FOLDER_L"
require_path "$REPO_ROOT/$SPLIT_DIR"
require_path "$STATIC_PROMPT"
require_path "$PROMPT_POOL"
require_path "$RAG_CACHE"

"$PYTHON_BIN" - <<'PY' "$DATA_CSV" "$REPO_ROOT/$SPLIT_DIR" "$RAG_CACHE"
import csv
import json
import os
import sys
from collections import Counter, defaultdict

data_csv, split_dir, rag_cache = sys.argv[1:4]

rows = list(csv.DictReader(open(data_csv, newline="")))
slide_ids = [r["slide_id"].strip() for r in rows]
case_ids = [r["case_id"].strip() for r in rows]
slide_to_case = {r["slide_id"].strip(): r["case_id"].strip() for r in rows}

if len(slide_ids) != len(set(slide_ids)):
    raise SystemExit("[ERROR] all_data.csv contains duplicated slide_id values.")

if len(case_ids) != len(set(case_ids)):
    raise SystemExit("[ERROR] all_data.csv contains duplicated case_id values; this script assumes one slide per case.")

split_files = [os.path.join(split_dir, f"splits_{i}.csv") for i in range(5)]
missing = [p for p in split_files if not os.path.isfile(p)]
if missing:
    raise SystemExit(f"[ERROR] Missing split files: {missing}")

assign_path = os.path.join(split_dir, "strict_fold_assignments.csv")
if not os.path.isfile(assign_path):
    raise SystemExit("[ERROR] strict_fold_assignments.csv is missing.")

case_fold_ids = defaultdict(set)
for r in csv.DictReader(open(assign_path, newline="")):
    case_fold_ids[r["case_id"].strip()].add(int(r["fold_id"]))

multi_fold_cases = [cid for cid, folds in case_fold_ids.items() if len(folds) > 1]
if multi_fold_cases:
    raise SystemExit(f"[ERROR] Some cases map to multiple fold_id values: {multi_fold_cases[:10]}")

membership = defaultdict(Counter)
for fold_idx, split_csv in enumerate(split_files):
    split_sets = {k: set() for k in ("train", "val", "test")}
    split_case_sets = {k: set() for k in ("train", "val", "test")}
    for r in csv.DictReader(open(split_csv, newline="")):
        for split_name in ("train", "val", "test"):
            sid = (r.get(split_name) or "").strip()
            if not sid:
                continue
            split_sets[split_name].add(sid)
            split_case_sets[split_name].add(slide_to_case[sid])
            membership[sid][split_name] += 1

    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise SystemExit(f"[ERROR] Slide leakage detected inside fold {fold_idx}.")

    if split_case_sets["train"] & split_case_sets["val"] or split_case_sets["train"] & split_case_sets["test"] or split_case_sets["val"] & split_case_sets["test"]:
        raise SystemExit(f"[ERROR] Case leakage detected inside fold {fold_idx}.")

patterns = Counter((c["train"], c["val"], c["test"]) for c in membership.values())
if patterns != Counter({(3, 1, 1): len(slide_ids)}):
    raise SystemExit(f"[ERROR] Unexpected 5-fold membership pattern: {dict(patterns)}")

cache_ids = set()
with open(rag_cache, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        sid = str(obj.get("slide_id", "")).strip()
        if sid:
            cache_ids.add(sid)

missing_cache = sorted(set(slide_ids) - cache_ids)
if missing_cache:
    raise SystemExit(
        f"[ERROR] Offline RAG cache is incomplete: missing {len(missing_cache)} slides. "
        f"Sample: {missing_cache[:10]}"
    )

print("[OK] strictcv split passed integrity checks.")
print(f"[OK] samples={len(slide_ids)} | unique_cases={len(case_ids)} | cache_coverage={len(cache_ids)}")
PY

mkdir -p "$RESULTS_DIR"

common_args=(
  --seed "$SEED"
  --drop_out
  --early_stopping
  --early_stopping_gate after_rag_start
  --lr "$LR"
  --max_epochs "$MAX_EPOCHS"
  --k 5
  --k_start 0
  --k_end 4
  --bag_loss ce
  --task task_adenocarcinoma
  --model_type ViLa_MIL_BiomedCLIP
  --mode transformer
  --data_root_dir "$DATA_ROOT_DIR"
  --data_folder_s "$DATA_FOLDER_S"
  --data_folder_l "$DATA_FOLDER_L"
  --split_dir "$SPLIT_DIR"
  --results_dir "$RESULTS_DIR"
  --text_prompt_path "$STATIC_PROMPT"
  --prototype_number "$PROTOTYPE_NUMBER"
  --rag_start_epoch "$RAG_START_EPOCH"
)

run_train() {
  local exp_code="$1"
  shift

  echo
  echo "============================================================"
  echo "Running experiment: $exp_code"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$PYTHON_BIN" main.py \
    "${common_args[@]}" \
    --exp_code "$exp_code" \
    "$@"
}

if [[ "$ONLY" == "all" || "$ONLY" == "static" ]]; then
  run_train "${RUN_TAG}_static"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "p1p3" ]]; then
  run_train "${RUN_TAG}_p1p3_offline" \
    --enable_dynamic_prompt \
    --prompt_pool_path "$PROMPT_POOL" \
    --retrieval_topk "$RETRIEVAL_TOPK" \
    --retrieval_temp "$RETRIEVAL_TEMP" \
    --dynamic_prompt_mix "$DYNAMIC_PROMPT_MIX" \
    --dynamic_prompt_warmup_epochs "$DYN_WARMUP_EPOCHS" \
    --enable_rag_rewrite \
    --rag_mode offline \
    --rag_cache_path "$RAG_CACHE" \
    --rag_topk "$RAG_TOPK" \
    --rag_ollama_model "$RAG_MODEL" \
    --rag_ollama_url "$RAG_URL" \
    --rag_temperature "$RAG_TEMPERATURE" \
    --rag_max_tokens "$RAG_MAX_TOKENS" \
    --rag_timeout_sec "$RAG_TIMEOUT_SEC" \
    --rag_max_retries "$RAG_MAX_RETRIES" \
    --rag_retry_delay_sec "$RAG_RETRY_DELAY_SEC" \
    --rag_fallback "$RAG_FALLBACK"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "p1p2p3" ]]; then
  run_train "${RUN_TAG}_p1p2p3_offline" \
    --enable_dynamic_prompt \
    --prompt_pool_path "$PROMPT_POOL" \
    --retrieval_topk "$RETRIEVAL_TOPK" \
    --retrieval_temp "$RETRIEVAL_TEMP" \
    --dynamic_prompt_mix "$DYNAMIC_PROMPT_MIX" \
    --dynamic_prompt_warmup_epochs "$DYN_WARMUP_EPOCHS" \
    --enable_rag_rewrite \
    --rag_mode offline \
    --rag_cache_path "$RAG_CACHE" \
    --rag_topk "$RAG_TOPK" \
    --rag_ollama_model "$RAG_MODEL" \
    --rag_ollama_url "$RAG_URL" \
    --rag_temperature "$RAG_TEMPERATURE" \
    --rag_max_tokens "$RAG_MAX_TOKENS" \
    --rag_timeout_sec "$RAG_TIMEOUT_SEC" \
    --rag_max_retries "$RAG_MAX_RETRIES" \
    --rag_retry_delay_sec "$RAG_RETRY_DELAY_SEC" \
    --rag_fallback "$RAG_FALLBACK" \
    --enable_vcp \
    --vcp_beta "$VCP_BETA" \
    --vcp_dropout "$VCP_DROPOUT" \
    --vcp_start_epoch "$VCP_START_EPOCH"
fi

echo
echo "All requested runs finished."
echo "Result root: $RESULTS_DIR"
