# Step47 PRARC Gate 5-Fold Report

## Step47 Purpose
- Run formal 5-fold validation for Step46 PRARC gate variants.
- Compare PRARC variants against the current baseline `RCE-v4-CSG-a01-rq16 / DEG skeleton`.
- Check whether PRARC behaves as a real sample-adaptive gate or collapses toward a near-scalar gate.

## Step46 Smoke Inheritance
- test_auc: `0.970999053030303`
- test_acc: `0.922680412371134`
- test_f1: `0.9164441381686622`
- balanced_acc: `0.9303977272727272`
- sensitivity: `0.9545454545454546`
- specificity: `0.90625`
- pr_auc: `0.9372102134155632`
- gate_mean: `0.79714035987854`
- gate_min: `0.7971360087394714`
- gate_max: `0.7971456050872803`
- gate_nonconstant: `True`
- visual_residual_adjusted: `True`

## Variants Run
- requested_variants: `prarc_v1_g05, prarc_v1_g08, prarc_v1_g10`
- successful_variants: `prarc_v1_g05, prarc_v1_g08, prarc_v1_g10`
- failed_or_missing_variants: `none`

## 5-Fold Metrics
| variant | status | test_auc_mean | test_auc_std | test_acc_mean | test_acc_std | test_f1_mean | test_f1_std | balanced_acc_mean | balanced_acc_std | sensitivity_mean | specificity_mean | pr_auc_mean | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prarc_v1_g05 | ok | 0.9600 | 0.0178 | 0.9132 | 0.0252 | 0.9035 | 0.0277 | 0.9028 | 0.0252 | 0.8700 | 0.9357 | 0.9330 | 0.0224 |
| prarc_v1_g08 | ok | 0.9588 | 0.0175 | 0.8988 | 0.0236 | 0.8880 | 0.0231 | 0.8876 | 0.0110 | 0.8521 | 0.9232 | 0.9210 | 0.0430 |
| prarc_v1_g10 | ok | 0.9570 | 0.0216 | 0.8967 | 0.0172 | 0.8838 | 0.0173 | 0.8787 | 0.0125 | 0.8218 | 0.9356 | 0.9303 | 0.0330 |

## Baseline Comparison
- baseline_model: `RCE-v4-CSG-a01-rq16 / DEG skeleton`
| variant | test_auc_delta | test_acc_delta | test_f1_delta | balanced_acc_delta | sensitivity_delta | specificity_delta | pr_auc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prarc_v1_g05 | -0.0102 | -0.0093 | -0.0109 | -0.0143 | -0.0302 | 0.0016 | -0.0114 |
| prarc_v1_g08 | -0.0114 | -0.0237 | -0.0265 | -0.0295 | -0.0481 | -0.0109 | -0.0234 |
| prarc_v1_g10 | -0.0132 | -0.0259 | -0.0307 | -0.0384 | -0.0783 | 0.0015 | -0.0141 |

## Stability Check
| variant | status | checkpoint_exists | checkpoint_count | missing_checkpoint_folds | missing_folds | log_exists | has_traceback | has_nan_or_inf | run_dir | fold_summary_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prarc_v1_g05 | ok | True | 5 |  |  | True | False | False | /home/ljh/ViLMIL/ViLa-MIL-main/results_stage47/stage47_prarc_v1_g05_s1 | /home/ljh/ViLMIL/ViLa-MIL-main/results_stage47/stage47_prarc_v1_g05_s1/fold_summary.csv |
| prarc_v1_g08 | ok | True | 5 |  |  | True | False | False | /home/ljh/ViLMIL/ViLa-MIL-main/results_stage47/stage47_prarc_v1_g08_s1 | /home/ljh/ViLMIL/ViLa-MIL-main/results_stage47/stage47_prarc_v1_g08_s1/fold_summary.csv |
| prarc_v1_g10 | ok | True | 5 |  |  | True | False | False | /home/ljh/ViLMIL/ViLa-MIL-main/results_stage47/stage47_prarc_v1_g10_s1 | /home/ljh/ViLMIL/ViLa-MIL-main/results_stage47/stage47_prarc_v1_g10_s1/fold_summary.csv |

## Gate Diagnostics Summary
_Gate diagnostics outputs exist, but no valid slide-level diagnostic rows are available yet._

## Recommendation
- recommended_variant: `prarc_v1_g05`
- recommend_enter_step48: `False`
- recommendation_reason: Performance is below baseline or gate still looks near-scalar.
- best_variant_auc: `0.9600`
- best_variant_pr_auc: `0.9330`
- best_variant_auc_delta: `-0.0102`
- best_variant_f1_delta: `-0.0109`
- best_variant_balanced_acc_delta: `-0.0143`
