# Step64A Precheck Report

- check time: 2026-07-07 20:16:21 CST
- git branch: `dev-rce-innovation`
- git status short: clean
- final decision: runnable

## Data paths

- `data/Yiyuan/`: exists
- `data/Yiyuan/features_biomedclip_5x/`: exists
- `data/Yiyuan/features_biomedclip_20x/`: exists
- `dataset_csv/all_data.csv`: exists
- `splits/Yifuyuan_strict/`: exists
- `text_prompt/adenocarcinoma_dual_scale_prompt.csv`: exists

## Split files status

- `splits/Yifuyuan_strict/splits_0.csv`: exists
- `splits/Yifuyuan_strict/splits_1.csv`: exists
- `splits/Yifuyuan_strict/splits_2.csv`: exists
- `splits/Yifuyuan_strict/splits_3.csv`: exists
- `splits/Yifuyuan_strict/splits_4.csv`: exists

## Prompt path status

- prompt csv exists
- prompt csv columns: `class_name`, `low_resolution_description`, `high_resolution_description`
- prompt rows: 2
- loaded prompt count should be 4, matching `2 x n_classes` for `task_adenocarcinoma`

## Feature directory file count

- `data/Yiyuan/features_biomedclip_5x/`: 968 files
- `data/Yiyuan/features_biomedclip_20x/`: 968 files
- low feature sample: `2460239-B2.h5`, `2460242-B2.h5`, `2460399-B2.h5`, `2460404-B3.h5`, `2460405-B2.h5`
- high feature sample: `2460239-B2.h5`, `2460242-B2.h5`, `2460399-B2.h5`, `2460404-B3.h5`, `2460405-B2.h5`

## Dataset csv columns

- columns: `case_id`, `slide_id`, `label`
- label distribution:
  - `Adenocarcinoma`: 637
  - `NonAdenocarcinoma`: 331

## Split csv columns

- `splits_0.csv`: `train`, `val`, `test`
- `splits_1.csv`: `train`, `val`, `test`
- `splits_2.csv`: `train`, `val`, `test`
- `splits_3.csv`: `train`, `val`, `test`
- `splits_4.csv`: `train`, `val`, `test`

## Model type support status

- `main.py` supports `--model_type ViLa_MIL_BiomedCLIP`
- `utils/core_utils.py` contains the `ViLa_MIL_BiomedCLIP` initialization branch
- `models/model_ViLa_MIL_BiomedCLIP.py` exists
- `task_adenocarcinoma` reads `dataset_csv/all_data.csv`
- `data_root_dir + data_folder_s / data_folder_l` resolves to:
  - `data/Yiyuan/features_biomedclip_5x`
  - `data/Yiyuan/features_biomedclip_20x`
- `--k_start 0 --k_end 4 --k 5` covers all 5 folds because `k_end` is inclusive
- `--max_epochs` does not need to be added because `main.py` defaults to 80 and clamps larger values to 80

## BiomedCLIP cache note

- `models/model_ViLa_MIL_BiomedCLIP.py` loads BiomedCLIP from `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- if the model is not already cached locally, first run may require network access
- if it is already cached and you want offline-only execution, you may export:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

## Final training command

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main

python main.py \
  --data_root_dir data/Yiyuan \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --task task_adenocarcinoma \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16 \
  --split_dir splits/Yifuyuan_strict \
  --results_dir experiment_outputs/stage_results/results_stage64A_vila_biomedclip_yifuyuan_strict5 \
  --exp_code adenocarcinoma_biomedclip_dual_yifuyuan_strict5 \
  --k 5 \
  --k_start 0 \
  --k_end 4 \
  --seed 1 \
  --lr 1e-4 \
  --label_frac 1.0 \
  --bag_loss ce \
  --early_stopping \
  --drop_out
```
