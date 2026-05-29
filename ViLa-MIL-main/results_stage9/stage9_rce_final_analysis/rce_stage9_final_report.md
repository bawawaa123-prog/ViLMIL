# Stage9 RCE Final Analysis

Step12 aggregates existing Stage9 RCE result files only. It does not modify models, run training, run 5-fold evaluation, or extract features.

## Stage9 RCE Evolution

1. `RCE-MIL base`: initial region-concept evidence baseline.
2. `RCE-MIL v2 prior_calib`: adds concept prior and logit calibration.
3. `RCE-MIL v3 visual residual evidence branch`: keeps the v2 prior/calibration path and adds a visual residual evidence branch.

## Recommendation

Current recommended RCE version: `RCE-MIL v3 prior_calib + visual_residual_init=0.05` (`RCE-v3-VR-a005`).
Key metrics for the recommended variant: `test_auc`=0.965526 ± 0.015368, `test_acc`=0.921473 ± 0.016511, `test_f1`=0.912704 ± 0.019068, `balanced_acc`=0.913436 ± 0.021693, `pr_auc`=0.927760 ± 0.036059.
Compared with `RCE-MIL base`, the recommended variant changes mean metrics by `test_auc`=+0.002231, `test_acc`=+0.039250, `test_f1`=+0.045973, `balanced_acc`=+0.052144, `pr_auc`=+0.000214.
Compared with `RCE-MIL v2 prior_calib`, the recommended variant changes mean metrics by `test_auc`=+0.002681, `test_acc`=+0.005176, `test_f1`=+0.005058, `balanced_acc`=+0.003847, `pr_auc`=+0.011814.

## PEPS Reference

The report uses `Concept-12 PEPS topk=5 tau=0.07` from `trained_models/final_dcp_vila_analysis/final_main_table.csv` as the PEPS reference.
`RCE-v3-VR-a005` should be treated as the current best RCE choice: its AUC is close to PEPS, while ACC/F1 still trail PEPS. The main value of RCE is the region-concept evidence structure itself, not only the headline metrics.
Available metric gaps versus PEPS: `test_auc` gap vs PEPS=-0.002801, `test_acc` gap vs PEPS=-0.017574, `test_f1` gap vs PEPS=-0.019475, `pr_auc` gap vs PEPS=-0.018106.

## Output Files

- `results_stage9/stage9_rce_final_analysis/rce_stage9_main_comparison.csv`
- `results_stage9/stage9_rce_final_analysis/rce_stage9_main_comparison.md`
- `results_stage9/stage9_rce_final_analysis/rce_stage9_metric_deltas.csv`
- `results_stage9/stage9_rce_final_analysis/rce_stage9_final_report.md`

## Input Status

- Available rows: 6
- Missing or invalid rows: 0

## Warnings and Parse Notes

- PEPS reference row is missing metric 'val_acc'.

## Next Step

Step13: RCE region-concept evidence export.
