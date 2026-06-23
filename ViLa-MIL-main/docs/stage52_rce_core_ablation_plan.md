# Step52 RCE Core Ablation Plan

## Purpose

Step52 prepares the core ablation entrypoints for the confirmed paper main model:

- main model: `RCE-v4-CSG-a01-rq16`
- code file: `models/model_RCE_MIL_BiomedCLIP.py`
- reference training script: `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`

The goal is to isolate the contribution of the main RCE components without changing model logic.

## Why Only RCE In Step52

Step52 is intentionally limited to the RCE main model because the paper source-of-truth has already been pinned to the RCE path.

This step is not about:

- DEG shell verification
- DEG repair
- new module development
- unified multi-branch experimentation

It is only about measuring what happens when the confirmed RCE main model loses one core component at a time.

## Variants

All variants use:

- `--model_type RCE_MIL_BiomedCLIP`
- `--prototype_number 16`
- `--concept_prompt_path dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
- `--split_dir splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- `--text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- `--data_folder_s features_biomedclip_5x`
- `--data_folder_l features_biomedclip_20x`
- `--scale_mode dual`
- `--prompt_ensemble_mode embedding_mean`
- `--k 5`
- `--k_start 0`
- `--k_end 4`
- `--max_epochs 20`
- `--seed 1`

### `full`

Keeps the full Stage23 `rq16` configuration:

- `--rce_use_cross_scale_graph`
- `--rce_use_concept_prior`
- `--rce_use_visual_residual`
- `--rce_use_logit_calibration`

This is the direct reference point for the Step52 ablation table.

### `wo_csg`

Disables:

- `--rce_use_cross_scale_graph`

This tests the claim that the cross-scale graph contributes useful low/high evidence interaction beyond plain low+high logits fusion.

### `wo_concept_prior`

Disables:

- `--rce_use_concept_prior`

This tests the claim that class-aware concept priors improve concept prompt weighting and final discrimination.

### `wo_visual_residual`

Disables:

- `--rce_use_visual_residual`

This tests the claim that direct visual residual logits add complementary evidence on top of concept-based logits.

### `wo_logit_calibration`

Disables:

- `--rce_use_logit_calibration`

This tests the claim that calibrated final logits improve downstream classification quality and confidence shaping.

## Training Script

Script:

`scripts/experiments/run_stage52_rce_core_ablation_5fold.sh`

Outputs:

- `results_stage52_rce_core_ablation/full_rce_v4_csg_rq16_5fold_e20_s1/`
- `results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1/`
- `results_stage52_rce_core_ablation/wo_concept_prior_5fold_e20_s1/`
- `results_stage52_rce_core_ablation/wo_visual_residual_5fold_e20_s1/`
- `results_stage52_rce_core_ablation/wo_logit_calibration_5fold_e20_s1/`

## Run Commands

Run full RCE:

```bash
VARIANT=full bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh
```

Run w/o CSG:

```bash
VARIANT=wo_csg bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh
```

Run w/o concept prior:

```bash
VARIANT=wo_concept_prior bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh
```

Run w/o visual residual:

```bash
VARIANT=wo_visual_residual bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh
```

Run w/o logit calibration:

```bash
VARIANT=wo_logit_calibration bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh
```

Run all variants:

```bash
VARIANT=all bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh
```

## Comparison After Training

Comparison script:

`scripts/analysis/compare_stage52_rce_core_ablation.py`

Run it after the training outputs are available:

```bash
python scripts/analysis/compare_stage52_rce_core_ablation.py
```

Outputs:

- `docs/stage52_rce_core_ablation_comparison.md`
- `docs/stage52_rce_core_ablation_comparison.csv`

The comparison reads:

- historical Stage23 full RCE result
- Step52 full result
- Step52 `wo_csg`
- Step52 `wo_concept_prior`
- Step52 `wo_visual_residual`
- Step52 `wo_logit_calibration`

Metrics:

- `test_auc`
- `test_acc`
- `test_f1`
- `balanced_acc`
- `pr_auc`

## Scope Boundary

Step52 does not:

- use `DEG_MIL_BiomedCLIP`
- repair DEG
- modify `models/model_RCE_MIL_BiomedCLIP.py`
- modify `models/model_DEG_MIL_BiomedCLIP.py`
- modify `main.py`
- modify `utils/core_utils.py`
- introduce any new modeling module

It only adds the training entry script, comparison script, and planning document for RCE core ablation.
