# Stage35 Visual Gate Summary

Step35 reads existing `results_stage35/*/fold_summary.csv` files and compares DEG skeleton against visual-gate ablations.

## Inputs

- `skeleton` -> `missing`
- `no_visual_residual` -> `missing`
- `gate0` -> `missing`
- `gate001` -> `missing`
- `gate005` -> `missing`
- `gate05` -> `missing`
- `gate1` -> `results_stage35/visual_gate_gate1_5fold_e1_s1/fold_summary.csv`

## Key Answers

1. `test_auc` best variant: `gate1` with `0.970407 ± 0.000000`
2. `balanced_acc` best variant: `gate1` with `0.787642 ± 0.000000`
3. Most balanced `sensitivity/specificity`: `gate1` with gap `0.393466`, `sensitivity`=0.590909 ± 0.000000, `specificity`=0.984375 ± 0.000000
4. `no_visual_residual` vs `skeleton`: insufficient data
5. `gate1` vs `skeleton`: insufficient data
6. Specificity collapse among `gate0/gate001/gate005`: not detected from the loaded runs.
7. Recommendation: do not promote Step36 yet from the currently loaded gate results.
8. Current loaded results do not justify switching to Low-High Evidence Consistency Loss immediately.

## Mean ± Std Summary

- `skeleton`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA
- `no_visual_residual`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA
- `gate0`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA
- `gate001`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA
- `gate005`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA
- `gate05`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA
- `gate1`: `test_auc`=0.970407 ± 0.000000, `test_acc`=0.850515 ± 0.000000, `test_f1`=0.812885 ± 0.000000, `balanced_acc`=0.787642 ± 0.000000, `sensitivity`=0.590909 ± 0.000000, `specificity`=0.984375 ± 0.000000, `pr_auc`=0.942668 ± 0.000000

## Deltas Vs Skeleton

- `no_visual_residual - skeleton`: N/A
- `gate0 - skeleton`: N/A
- `gate001 - skeleton`: N/A
- `gate005 - skeleton`: N/A
- `gate05 - skeleton`: N/A
- `gate1 - skeleton`: N/A

## Output Files

- `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_summary.csv`
- `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_metric_deltas.csv`
- `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_rankings.csv`
- `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_report.md`
- `results_stage35/stage35_visual_gate_summary/stage35_recommendations.json`

## Compact Tables

### Summary

| variant | status | matched_epochs | matched_seed | test_auc_formatted | test_f1_formatted | balanced_acc_formatted | sensitivity_formatted | specificity_formatted | pr_auc_formatted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton | missing | NA | NA | NA | NA | NA | NA | NA | NA |
| no_visual_residual | missing | NA | NA | NA | NA | NA | NA | NA | NA |
| gate0 | missing | NA | NA | NA | NA | NA | NA | NA | NA |
| gate001 | missing | NA | NA | NA | NA | NA | NA | NA | NA |
| gate005 | missing | NA | NA | NA | NA | NA | NA | NA | NA |
| gate05 | missing | NA | NA | NA | NA | NA | NA | NA | NA |
| gate1 | ok | 1.0 | 1.0 | 0.970407 ± 0.000000 | 0.812885 ± 0.000000 | 0.787642 ± 0.000000 | 0.590909 ± 0.000000 | 0.984375 ± 0.000000 | 0.942668 ± 0.000000 |

### Rankings

| variant | overall_rank | test_auc_rank | balanced_acc_rank | sensitivity_rank | specificity_rank | sens_spec_gap_rank |
| --- | --- | --- | --- | --- | --- | --- |
| gate1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| gate0 | NA | NA | NA | NA | NA | NA |
| gate001 | NA | NA | NA | NA | NA | NA |
| gate005 | NA | NA | NA | NA | NA | NA |
| gate05 | NA | NA | NA | NA | NA | NA |
| no_visual_residual | NA | NA | NA | NA | NA | NA |
| skeleton | NA | NA | NA | NA | NA | NA |
