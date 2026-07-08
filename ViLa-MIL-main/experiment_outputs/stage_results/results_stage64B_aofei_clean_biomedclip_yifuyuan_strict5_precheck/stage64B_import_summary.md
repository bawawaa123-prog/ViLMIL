# Step64B Import Summary

- Purpose: import the Aofei clean ViLa-MIL BiomedCLIP baseline into the main ViLMIL project as a new independent model type.
- Aofei repository usage: only as the clean baseline code source for `models/model_ViLa_MIL_BiomedCLIP_AofeiClean.py`.
- Data policy: do not use Aofei `data`, `splits`, `dataset_csv`, or `results`.
- Added model type: `ViLa_MIL_BiomedCLIP_AofeiClean`.
- New model origin: migrated from the Aofei clean baseline implementation.
- Existing main-project baseline safety: `models/model_ViLa_MIL_BiomedCLIP.py` was not overwritten.
- RCE safety: no RCE model file was modified.
- Training status: no training was run in Step64B.
- Precheck script: `scripts/experiments/run_stage64B_aofei_clean_baseline_yifuyuan_strict5.sh`.

Recommended command:

```bash
python main.py \
  --data_root_dir data/Yiyuan \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --model_type ViLa_MIL_BiomedCLIP_AofeiClean \
  --mode transformer \
  --task task_adenocarcinoma \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16 \
  --split_dir splits/Yifuyuan_strict \
  --results_dir experiment_outputs/stage_results/results_stage64B_aofei_clean_biomedclip_yifuyuan_strict5 \
  --exp_code adenocarcinoma_biomedclip_aofei_clean_yifuyuan_strict5 \
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

Follow-up comparison targets:

1. Aofei repo baseline result: AUC `0.971236` / ACC `0.928716` / F1 `0.920241`
2. Current ViLMIL extended BiomedCLIP baseline: AUC `0.9659` / ACC `0.9256` / F1 `0.9150`
3. Step58C RCE primary: AUC `0.964842` / ACC `0.927691` / F1 `0.919625`
