# Main Model Reproduction And Code Lineage

## 1. Final Main-Model Conclusion

- Main model name: `RCE-v4-CSG-a01-rq16`
- Main model file: `models/model_RCE_MIL_BiomedCLIP.py`
- Main training script: `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
- Main result directory: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Runtime `model_type`: `RCE_MIL_BiomedCLIP`

This conclusion is based on three facts checked directly in code and results:

1. `main.py` accepts `--model_type RCE_MIL_BiomedCLIP` and passes it into the shared training entry.
2. `utils/core_utils.py` dispatches `RCE_MIL_BiomedCLIP` to `models/model_RCE_MIL_BiomedCLIP.py`, while `DEG_MIL_BiomedCLIP` is dispatched separately to `models/model_DEG_MIL_BiomedCLIP.py`.
3. `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv` exactly matches the target paper metrics:
   - `test_auc=0.9702127532300915`
   - `test_acc=0.9225094813311255`
   - `test_f1=0.9144563517525356`
   - `pr_auc=0.9443991016694445`

## 2. Code Call Chain

Verified call chain:

`main.py -> utils/core_utils.py -> models/model_RCE_MIL_BiomedCLIP.py`

Evidence:

- `main.py` defines `--model_type` with `RCE_MIL_BiomedCLIP` and `DEG_MIL_BiomedCLIP` as separate choices.
- `main.py` imports `train` from `utils/core_utils.py`.
- `utils/core_utils.py` checks `args.model_type`.
  - If `args.model_type == 'RCE_MIL_BiomedCLIP'`, it imports `RCE_MIL_BiomedCLIP` from `models/model_RCE_MIL_BiomedCLIP.py`.
  - If `args.model_type == 'DEG_MIL_BiomedCLIP'`, it imports `DEG_MIL_BiomedCLIP` from `models/model_DEG_MIL_BiomedCLIP.py`.

So the Stage23 main run does not pass through the DEG skeleton file. It directly uses the RCE file.

## 3. Stage23 `VARIANT=rq16` Parameters

From `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`, `VARIANT=rq16` resolves to:

- `--model_type RCE_MIL_BiomedCLIP`
- `--prototype_number 16`
- `--rce_use_logit_calibration`
- `--rce_use_concept_prior`
- `--rce_use_visual_residual`
- `--rce_use_cross_scale_graph`
- `--rce_cross_scale_graph_init 0.1`
- `--rce_cross_scale_graph_norm sqrt`
- `--concept_prompt_path ${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
- `--split_dir splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- `--text_prompt_path ${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv`

The same script also fixes:

- `--task task_adenocarcinoma`
- `--data_root_dir /xiangmu/data/VILMIL` by default
- `--data_folder_s features_biomedclip_5x`
- `--data_folder_l features_biomedclip_20x`
- `--prompt_ensemble_mode embedding_mean`
- `--scale_mode dual`
- `--k 5`
- `--k_start 0`
- `--k_end 4`
- `--max_epochs 20`
- `--seed 1`
- `--rce_logit_scale_init 10.0`
- `--rce_concept_prior_strength 1.0`
- `--rce_visual_residual_init 0.05`

## 4. Historical Result Verification

### Stage23 main result

`results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`

- `test_auc=0.9702127532300915`
- `test_acc=0.9225094813311255`
- `test_f1=0.9144563517525356`
- `val_auc=0.9711256597292707`
- `balanced_acc=0.9171426221612053`
- `sensitivity=0.9001809136137494`
- `specificity=0.9341043307086615`
- `pr_auc=0.9443991016694445`

### `epoch_details.csv`

`results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/epoch_details.csv` confirms:

- 5 folds were run.
- Each fold was trained for 20 epochs.
- The per-epoch curves are consistent with a normal full 5-fold training run rather than a smoke test or partial export.

## 5. Why Multiple Stages Show `acc≈0.9225`

This happens for two different reasons.

### 5.1 Exact same main-model config, different experimental packaging

The following result files are bitwise identical at the `result.csv` level:

- `results_stage22/rce_v4_csg_a01_5fold_e20_s1/result.csv`
- `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`
- `results_stage27/deg_skeleton_5fold_e20_s1/result.csv`
- `results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1/result.csv`
- `results_stage35/visual_gate_skeleton_5fold_e20_s1/result.csv`
- `results_stage37/lh_consistency_skeleton_5fold_e20_s1/result.csv`

All six files have the same MD5:

`6fa1a1822a32791665301bfac8936ab3`

So these are not merely "close". They are exact replays of the same metric table.

### 5.2 Later modules sometimes produce nearby but non-identical values

Later stages such as Stage30 concept graph, Stage35 visual gate, Stage37 low-high consistency, Stage44 HCRC, and Stage47 PRARC can produce values numerically near `0.9225`, but they are not exact matches and they use extra modules. Those runs should not be relabeled as the original paper main model.

Examples:

- `results_stage30/deg_concept_graph_k4_a005_5fold_e20_s1/result.csv`
  - `test_acc=0.9132097644356605`
- `results_stage35/visual_gate_gate001_5fold_e20_s1/result.csv`
  - `test_acc=0.9163399391058169`
- `results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1/result.csv`
  - `test_acc=0.9276854868863843`
- `results_stage44/stage44_hcrc_a01_b8_s1/result.csv`
  - `test_acc=0.9194166978259709`
- `results_stage47/stage47_prarc_v1_g05_s1/result.csv`
  - `test_acc=0.9132151060306608`

These are later branches or ablations, not the original source run.

## 6. Stage Relationship Summary

### Stage9

- `results_stage9/rce_mil_5fold_e20_s1/result.csv`
  - Base RCE-MIL without the later full RCE-v4 bundle.
- `results_stage9/rce_mil_v3_prior_calib_vr_a01_5fold_e20_s1/result.csv`
  - Adds prior calibration and visual residual, but still no cross-scale graph.
- Stage9 is an earlier ancestor, not the final main model.

### Stage22

- `run_stage22_rce_v4_csg_5fold.sh` introduces the RCE-v4-CSG package with `prototype_number=16`.
- `VARIANT=csg_a01` matches the final Stage23 main result exactly.
- Stage22 should be treated as the immediate pre-Stage23 equivalent of the main model.

### Stage23

- `run_stage23_rce_v4_csg_region_queries_5fold.sh` sweeps region query count (`rq8`, `rq16`, `rq32`).
- `rq16` is the explicit region-query-labeled main-model run and matches the target paper metrics.
- This is the cleanest current source-of-truth because the naming makes the `prototype_number=16` choice explicit.

### Stage27

- `run_stage27_deg_region_graph_5fold.sh`
- `VARIANT=skeleton` uses `DEG_MIL_BiomedCLIP` but does not enable region graph.
- This stage is a DEG wrapper baseline, not the original source, even though its metrics exactly replay Stage23.

### Stage30

- `run_stage30_deg_concept_graph_5fold.sh`
- `VARIANT=skeleton` is another DEG wrapper replay baseline.
- `cg_k2_a005`, `cg_k4_a005`, `cg_k8_a005` are concept-graph ablations and are not the original main model.

### Stage35

- `run_stage35_visual_gate_5fold.sh`
- `VARIANT=skeleton` replays the Stage23 config.
- `gate*` variants turn on `rce_use_visual_evidence_gate` and are later ablations.

### Stage37

- `run_stage37_lh_consistency_5fold.sh`
- `VARIANT=skeleton` replays the Stage23 config.
- `lh_*` variants turn on `rce_use_low_high_consistency_loss` and are later ablations.

### Stage44

- `run_stage44_hcrc_light_5fold.sh`
- Turns on `rce_use_hcrc`.
- This is a later DEG-based extension, not the main-model origin.

### Stage47

- `run_stage47_prarc_gate_5fold.sh`
- Turns on `rce_use_prarc_gate`.
- This is another later DEG-based extension, not the main-model origin.

## 7. Which Result Counts As The Paper Main Model

### Can be used as paper main model

- `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`

This is the preferred current citation target because:

1. It uses the direct RCE file via `model_type=RCE_MIL_BiomedCLIP`.
2. It matches the target paper metrics exactly.
3. Its script explicitly documents the region-query setting as `rq16`.

### Equivalent but should be treated as lineage neighbors, not the preferred citation target

- `results_stage22/rce_v4_csg_a01_5fold_e20_s1`
- `results_stage27/deg_skeleton_5fold_e20_s1`
- `results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1`
- `results_stage35/visual_gate_skeleton_5fold_e20_s1`
- `results_stage37/lh_consistency_skeleton_5fold_e20_s1`

Interpretation:

- Stage22 `csg_a01` is an earlier direct RCE equivalent.
- Stage27/30/35/37 `skeleton` runs are DEG-shell replays of the same logits path.

### Should only be treated as skeleton baselines or negative ablations

- Stage27 `rg_k2`, `rg_k4`, `rg_k8`
- Stage30 `cg_k2_a005`, `cg_k4_a005`, `cg_k8_a005`
- Stage35 `gate*`
- Stage37 `lh_*`
- Stage44 `hcrc_*`
- Stage47 `prarc_*`

## 8. DEG Skeleton Equivalence Check

`models/model_DEG_MIL_BiomedCLIP.py` is implemented as:

- `class DEG_MIL_BiomedCLIP(RCE_MIL_BiomedCLIP)`

So DEG is built on top of the current RCE implementation rather than replacing it.

In its `forward`, the Stage23-equivalent path is preserved when these extra switches stay off:

- `deg_use_region_graph=False`
- `deg_use_concept_graph=False`
- `rce_use_visual_evidence_gate=False`
- `rce_use_prarc_gate=False`
- `rce_use_hcrc=False`
- `rce_use_low_high_consistency_loss=False`

Under that setting, the DEG forward still does:

1. region aggregation
2. concept-prompt evidence logits
3. low + high logits sum
4. optional visual residual
5. optional cross-scale graph
6. optional logit calibration

That is the same effective Stage23 logits path.

Therefore the correct conclusion is:

`DEG skeleton can serve as the later unified ablation shell, but the original main-model source remains RCE-v4-CSG-a01-rq16.`

## 9. Reproduction

### Historical source command

```bash
bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh VARIANT=rq16
```

### This reproduction run

To avoid overwriting historical artifacts in `results_stage23/...`, this reproduction was launched with a new output root:

```bash
RESULTS_DIR=results_stage23_repro bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh
```

Expected output directory:

`results_stage23_repro/rce_v4_csg_a01_rq16_5fold_e20_s1`

Verified environment paths:

- data root: `/xiangmu/data/VILMIL`
- low-scale features: `/xiangmu/data/VILMIL/features_biomedclip_5x`
- high-scale features: `/xiangmu/data/VILMIL/features_biomedclip_20x`
- split dir: `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- prompt csv: `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- concept pool json: `dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`

Final reproduction metrics were read from:

`results_stage23_repro/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`

Reproduced metrics:

- `test_auc=0.9702127532300915`
- `test_acc=0.9225094813311255`
- `test_f1=0.9144563517525356`
- `pr_auc=0.9443991016694445`

Comparison against historical `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`:

- mean-row delta for `test_auc`: `0.0`
- mean-row delta for `test_acc`: `0.0`
- mean-row delta for `test_f1`: `0.0`
- mean-row delta for `pr_auc`: `0.0`
- mean-row delta for `val_auc`: `0.0`

Conclusion:

- `acc≈0.9225` was successfully reproduced.
- The reproduction is not just close; the saved `result.csv` is numerically identical to the historical one for both `mean` and `std` rows.

## 10. Recommended Source-Of-Truth Policy

- Use Stage23 `RCE-v4-CSG-a01-rq16` as the source-of-truth main model.
- Use DEG skeleton as the later unified ablation shell.
- Do not rewrite the main-model identity around HCRC, PRARC, visual gate, region graph, concept graph, or low-high consistency branches.
- Do not promote a later module into the final main model unless it truly surpasses the Stage23 source-of-truth and is intentionally adopted as the new paper main model.

## 11. Added Helper Script

Added:

- `scripts/analysis/check_main_model_lineage.py`

What it does:

1. Reads the key `result.csv` files.
2. Prints a markdown comparison table.
3. Computes deltas relative to Stage23.
4. Labels rows as `reference`, `exact_match`, `near_match`, or `different`.
5. Saves:
   - `docs/main_model_lineage_comparison.md`
   - `docs/main_model_lineage_comparison.csv`
