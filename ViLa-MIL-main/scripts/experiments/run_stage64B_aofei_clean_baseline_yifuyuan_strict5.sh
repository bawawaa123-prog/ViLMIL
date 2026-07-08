#!/usr/bin/env bash
set -euo pipefail

cd /xiangmu/ViLMIL/ViLa-MIL-main

RUN_TRAINING="${RUN_TRAINING:-0}"

required_paths=(
  "data/Yiyuan/features_biomedclip_5x"
  "data/Yiyuan/features_biomedclip_20x"
  "dataset_csv/all_data.csv"
  "splits/Yifuyuan_strict/splits_0.csv"
  "splits/Yifuyuan_strict/splits_1.csv"
  "splits/Yifuyuan_strict/splits_2.csv"
  "splits/Yifuyuan_strict/splits_3.csv"
  "splits/Yifuyuan_strict/splits_4.csv"
  "text_prompt/adenocarcinoma_dual_scale_prompt.csv"
  "models/model_ViLa_MIL_BiomedCLIP_AofeiClean.py"
)

echo "[Stage64B] Prechecking main-project paths only"
missing=0
for path in "${required_paths[@]}"; do
  if [[ -e "$path" ]]; then
    echo "[OK] $path"
  else
    echo "[MISSING] $path"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "[Stage64B] Precheck failed because one or more required main-project paths are missing."
  exit 1
fi

cmd=(
  python main.py
  --data_root_dir data/Yiyuan
  --data_folder_s features_biomedclip_5x
  --data_folder_l features_biomedclip_20x
  --model_type ViLa_MIL_BiomedCLIP_AofeiClean
  --mode transformer
  --task task_adenocarcinoma
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv
  --prototype_number 16
  --split_dir splits/Yifuyuan_strict
  --results_dir experiment_outputs/stage_results/results_stage64B_aofei_clean_biomedclip_yifuyuan_strict5
  --exp_code adenocarcinoma_biomedclip_aofei_clean_yifuyuan_strict5
  --k 5
  --k_start 0
  --k_end 4
  --seed 1
  --lr 1e-4
  --label_frac 1.0
  --bag_loss ce
  --early_stopping
  --drop_out
)

echo
echo "[Stage64B] Suggested training command:"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$RUN_TRAINING" != "1" ]]; then
  echo "[Stage64B] RUN_TRAINING=${RUN_TRAINING}; not launching training."
  echo "[Stage64B] To run training manually: RUN_TRAINING=1 bash scripts/experiments/run_stage64B_aofei_clean_baseline_yifuyuan_strict5.sh"
  exit 0
fi

echo "[Stage64B] RUN_TRAINING=1; launching training command."
"${cmd[@]}"
