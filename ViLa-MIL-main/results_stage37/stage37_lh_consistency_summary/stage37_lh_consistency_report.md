# Stage37 Low-High Consistency Summary

Step37 compares the DEG skeleton against low-high evidence consistency loss variants using existing `fold_summary.csv` files.

## Inputs

- Results directory: `results_stage37`
- `skeleton` -> `results_stage37/lh_consistency_skeleton_5fold_e20_s1/fold_summary.csv`
- `lh_l0001_m0` -> `results_stage37/lh_consistency_lh_l0001_m0_5fold_e20_s1/fold_summary.csv`
- `lh_l0005_m0` -> `results_stage37/lh_consistency_lh_l0005_m0_5fold_e20_s1/fold_summary.csv`
- `lh_l001_m0` -> `results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1/fold_summary.csv`
- `lh_l0005_m002` -> `results_stage37/lh_consistency_lh_l0005_m002_5fold_e20_s1/fold_summary.csv`
- `lh_l001_m002` -> `results_stage37/lh_consistency_lh_l001_m002_5fold_e20_s1/fold_summary.csv`
- `lh_l001_m005` -> `results_stage37/lh_consistency_lh_l001_m005_5fold_e20_s1/fold_summary.csv`
- `lh_l005_m0` -> `missing`
- `lh_l005_m005` -> `missing`

## Key Answers

1. Best `test_auc`: `skeleton` with `0.970213 +/- 0.013016`
2. Best `balanced_acc`: `lh_l001_m0` with `0.919651 +/- 0.017079`
3. Most balanced `sensitivity/specificity`: `skeleton` with gap `0.033923`, `sensitivity`=0.900181 +/- 0.026688, `specificity`=0.934104 +/- 0.020100
4. No loaded variant is clearly better than skeleton by the current thresholds.
5. Light lambda variants do not look more stable than medium/strong variants from loaded results.
6. No specificity/sensitivity collapse detected from loaded runs.
7. The loaded data do not support marking all consistency variants as negative yet.
8. Recommendation: do not start Step38 from these loaded results unless more complete runs change the ranking.

## Mean +/- Std Summary

- `skeleton`: `test_auc`=0.970213 +/- 0.013016, `test_acc`=0.922509 +/- 0.018779, `test_f1`=0.914456 +/- 0.020681, `balanced_acc`=0.917143 +/- 0.019648, `sensitivity`=0.900181 +/- 0.026688, `specificity`=0.934104 +/- 0.020100, `pr_auc`=0.944399 +/- 0.025168
- `lh_l0001_m0`: `test_auc`=0.963209 +/- 0.016736, `test_acc`=0.916324 +/- 0.019929, `test_f1`=0.906931 +/- 0.022285, `balanced_acc`=0.906646 +/- 0.022083, `sensitivity`=0.876074 +/- 0.035248, `specificity`=0.937217 +/- 0.021615, `pr_auc`=0.927411 +/- 0.033440
- `lh_l0005_m0`: `test_auc`=0.960499 +/- 0.011793, `test_acc`=0.908044 +/- 0.022798, `test_f1`=0.898356 +/- 0.025274, `balanced_acc`=0.901790 +/- 0.027497, `sensitivity`=0.882135 +/- 0.064630, `specificity`=0.921444 +/- 0.037684, `pr_auc`=0.915856 +/- 0.032236
- `lh_l001_m0`: `test_auc`=0.968077 +/- 0.014300, `test_acc`=0.927685 +/- 0.016301, `test_f1`=0.919662 +/- 0.017992, `balanced_acc`=0.919651 +/- 0.017079, `sensitivity`=0.894211 +/- 0.028896, `specificity`=0.945091 +/- 0.021574, `pr_auc`=0.935375 +/- 0.031037
- `lh_l0005_m002`: `test_auc`=0.962198 +/- 0.019207, `test_acc`=0.916319 +/- 0.019953, `test_f1`=0.906712 +/- 0.022634, `balanced_acc`=0.905918 +/- 0.023102, `sensitivity`=0.873044 +/- 0.045707, `specificity`=0.938792 +/- 0.026916, `pr_auc`=0.924852 +/- 0.036919
- `lh_l001_m002`: `test_auc`=0.956144 +/- 0.018358, `test_acc`=0.910133 +/- 0.014386, `test_f1`=0.899470 +/- 0.018163, `balanced_acc`=0.899008 +/- 0.025022, `sensitivity`=0.863998 +/- 0.063654, `specificity`=0.934018 +/- 0.020406, `pr_auc`=0.900170 +/- 0.043612
- `lh_l001_m005`: `test_auc`=0.963618 +/- 0.017142, `test_acc`=0.914214 +/- 0.029476, `test_f1`=0.905710 +/- 0.030386, `balanced_acc`=0.908640 +/- 0.024263, `sensitivity`=0.891135 +/- 0.042388, `specificity`=0.926144 +/- 0.050760, `pr_auc`=0.917010 +/- 0.042279
- `lh_l005_m0`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA
- `lh_l005_m005`: `test_auc`=NA, `test_acc`=NA, `test_f1`=NA, `balanced_acc`=NA, `sensitivity`=NA, `specificity`=NA, `pr_auc`=NA

## Deltas Vs Skeleton

- `lh_l0001_m0 - skeleton`: `test_auc`=-0.007004, `test_acc`=-0.006186, `test_f1`=-0.007526, `balanced_acc`=-0.010497, `sensitivity`=-0.024107, `specificity`=+0.003113, `pr_auc`=-0.016988, `sens_spec_gap`=+0.027219, `sens_spec_min`=-0.024107
- `lh_l0005_m0 - skeleton`: `test_auc`=-0.009714, `test_acc`=-0.014465, `test_f1`=-0.016100, `balanced_acc`=-0.015353, `sensitivity`=-0.018046, `specificity`=-0.012660, `pr_auc`=-0.028543, `sens_spec_gap`=+0.005386, `sens_spec_min`=-0.018046
- `lh_l001_m0 - skeleton`: `test_auc`=-0.002136, `test_acc`=+0.005176, `test_f1`=+0.005206, `balanced_acc`=+0.002508, `sensitivity`=-0.005970, `specificity`=+0.010987, `pr_auc`=-0.009024, `sens_spec_gap`=+0.016957, `sens_spec_min`=-0.005970
- `lh_l0005_m002 - skeleton`: `test_auc`=-0.008015, `test_acc`=-0.006191, `test_f1`=-0.007744, `balanced_acc`=-0.011225, `sensitivity`=-0.027137, `specificity`=+0.004687, `pr_auc`=-0.019548, `sens_spec_gap`=+0.031825, `sens_spec_min`=-0.027137
- `lh_l001_m002 - skeleton`: `test_auc`=-0.014069, `test_acc`=-0.012376, `test_f1`=-0.014986, `balanced_acc`=-0.018134, `sensitivity`=-0.036183, `specificity`=-0.000086, `pr_auc`=-0.044230, `sens_spec_gap`=+0.036097, `sens_spec_min`=-0.036183
- `lh_l001_m005 - skeleton`: `test_auc`=-0.006595, `test_acc`=-0.008295, `test_f1`=-0.008746, `balanced_acc`=-0.008503, `sensitivity`=-0.009046, `specificity`=-0.007960, `pr_auc`=-0.027389, `sens_spec_gap`=+0.001086, `sens_spec_min`=-0.009046
- `lh_l005_m0 - skeleton`: N/A
- `lh_l005_m005 - skeleton`: N/A

## Output Files

- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_summary.csv`
- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_metric_deltas.csv`
- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_rankings.csv`
- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_report.md`
- `results_stage37/stage37_lh_consistency_summary/stage37_recommendations.json`

## Warnings

- Multiple Stage37 result dirs matched variant=lh_l0005_m0; using /home/ljh/ViLMIL/ViLa-MIL-main/results_stage37/lh_consistency_lh_l0005_m0_5fold_e20_s1

## Compact Tables

### Summary

| variant | status | matched_epochs | matched_seed | test_auc_formatted | test_f1_formatted | balanced_acc_formatted | sensitivity_formatted | specificity_formatted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton | ok | 20.0 | 1.0 | 0.970213 +/- 0.013016 | 0.914456 +/- 0.020681 | 0.917143 +/- 0.019648 | 0.900181 +/- 0.026688 | 0.934104 +/- 0.020100 |
| lh_l0001_m0 | ok | 20.0 | 1.0 | 0.963209 +/- 0.016736 | 0.906931 +/- 0.022285 | 0.906646 +/- 0.022083 | 0.876074 +/- 0.035248 | 0.937217 +/- 0.021615 |
| lh_l0005_m0 | ok | 20.0 | 1.0 | 0.960499 +/- 0.011793 | 0.898356 +/- 0.025274 | 0.901790 +/- 0.027497 | 0.882135 +/- 0.064630 | 0.921444 +/- 0.037684 |
| lh_l001_m0 | ok | 20.0 | 1.0 | 0.968077 +/- 0.014300 | 0.919662 +/- 0.017992 | 0.919651 +/- 0.017079 | 0.894211 +/- 0.028896 | 0.945091 +/- 0.021574 |
| lh_l0005_m002 | ok | 20.0 | 1.0 | 0.962198 +/- 0.019207 | 0.906712 +/- 0.022634 | 0.905918 +/- 0.023102 | 0.873044 +/- 0.045707 | 0.938792 +/- 0.026916 |
| lh_l001_m002 | ok | 20.0 | 1.0 | 0.956144 +/- 0.018358 | 0.899470 +/- 0.018163 | 0.899008 +/- 0.025022 | 0.863998 +/- 0.063654 | 0.934018 +/- 0.020406 |
| lh_l001_m005 | ok | 20.0 | 1.0 | 0.963618 +/- 0.017142 | 0.905710 +/- 0.030386 | 0.908640 +/- 0.024263 | 0.891135 +/- 0.042388 | 0.926144 +/- 0.050760 |
| lh_l005_m0 | missing | NA | NA | NA | NA | NA | NA | NA |
| lh_l005_m005 | missing | NA | NA | NA | NA | NA | NA | NA |

### Rankings

| variant | overall_rank | test_auc_rank | balanced_acc_rank | sens_spec_gap_rank | sens_spec_min_rank |
| --- | --- | --- | --- | --- | --- |
| skeleton | 1.0 | 1.0 | 2.0 | 1.0 | 1.0 |
| lh_l001_m0 | 2.0 | 2.0 | 1.0 | 4.0 | 2.0 |
| lh_l001_m005 | 3.0 | 3.0 | 3.0 | 2.0 | 3.0 |
| lh_l0001_m0 | 4.0 | 4.0 | 4.0 | 5.0 | 5.0 |
| lh_l0005_m002 | 5.0 | 5.0 | 5.0 | 6.0 | 6.0 |
| lh_l0005_m0 | 6.0 | 6.0 | 6.0 | 3.0 | 4.0 |
| lh_l001_m002 | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
| lh_l005_m0 | NA | NA | NA | NA | NA |
| lh_l005_m005 | NA | NA | NA | NA | NA |
