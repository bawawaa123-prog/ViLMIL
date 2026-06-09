# Stage24 RCE-v4-CSG Summary

Step24 reads existing Stage22 and Stage23 result CSV files only. It does not run training, extract features, or modify any model code.

## Stage22 CSG Init Comparison

- `csg_a005`: `test_auc`=0.961928 ± 0.018122, `test_acc`=0.897719 ± 0.018903, `test_f1`=0.885097 ± 0.022322, `balanced_acc`=0.883203 ± 0.028104, `pr_auc`=0.921331 ± 0.038047
- `csg_a01`: `test_auc`=0.970213 ± 0.013016, `test_acc`=0.922509 ± 0.018779, `test_f1`=0.914456 ± 0.020681, `balanced_acc`=0.917143 ± 0.019648, `pr_auc`=0.944399 ± 0.025168

## Stage23 Region Query Number Comparison

- `rq8`: `test_auc`=0.963998 ± 0.017688, `test_acc`=0.910122 ± 0.010627, `test_f1`=0.897900 ± 0.011859, `balanced_acc`=0.891196 ± 0.019233, `pr_auc`=0.935604 ± 0.033146
- `rq16`: `test_auc`=0.970213 ± 0.013016, `test_acc`=0.922509 ± 0.018779, `test_f1`=0.914456 ± 0.020681, `balanced_acc`=0.917143 ± 0.019648, `pr_auc`=0.944399 ± 0.025168
- `rq32`: `test_auc`=0.957726 ± 0.014726, `test_acc`=0.907003 ± 0.014766, `test_f1`=0.895646 ± 0.015013, `balanced_acc`=0.891710 ± 0.013509, `pr_auc`=0.921822 ± 0.031662

## Key Deltas

- `csg_a01 - csg_a005`: `test_auc`=+0.008285, `test_acc`=+0.024790, `test_f1`=+0.029359, `balanced_acc`=+0.033940, `pr_auc`=+0.023068
- `rq16 - rq8`: `test_auc`=+0.006215, `test_acc`=+0.012387, `test_f1`=+0.016556, `balanced_acc`=+0.025946, `pr_auc`=+0.008795
- `rq16 - rq32`: `test_auc`=+0.012487, `test_acc`=+0.015507, `test_f1`=+0.018810, `balanced_acc`=+0.025433, `pr_auc`=+0.022577
- `RCE-v4-CSG-a01-rq16 - RCE-MIL base`: `test_auc`=+0.006918, `test_acc`=+0.040286, `test_f1`=+0.047725, `balanced_acc`=+0.055851, `pr_auc`=+0.016854
- `RCE-v4-CSG-a01-rq16 - RCE-v3-VR-a005`: `test_auc`=+0.004687, `test_acc`=+0.001036, `test_f1`=+0.001752, `balanced_acc`=+0.003707, `pr_auc`=+0.016639
- `RCE-v4-CSG-a01-rq16 - Concept-12 PEPS reference`: `test_auc`=+0.001886, `test_acc`=-0.016538, `test_f1`=-0.017723, `balanced_acc`=-0.014132, `pr_auc`=-0.001467

## Stage9 Reference Deltas

- `RCE-v4-CSG-a01-rq16 - RCE-MIL base`: `test_auc`=+0.006918, `test_acc`=+0.040286, `test_f1`=+0.047725, `balanced_acc`=+0.055851, `pr_auc`=+0.016854
- `RCE-v4-CSG-a01-rq16 - RCE-v3-VR-a005`: `test_auc`=+0.004687, `test_acc`=+0.001036, `test_f1`=+0.001752, `balanced_acc`=+0.003707, `pr_auc`=+0.016639
- `RCE-v4-CSG-a01-rq16 - Concept-12 PEPS reference`: `test_auc`=+0.001886, `test_acc`=-0.016538, `test_f1`=-0.017723, `balanced_acc`=-0.014132, `pr_auc`=-0.001467

## Recommendation

Current recommended main configuration: `RCE-v4-CSG-a01-rq16`.
Stage22 best CSG init by mean `test_auc`: `csg_a01`.
Stage23 best region query setting by mean `test_auc`: `rq16`.
`prototype_number=16` remains the default suggestion for the next DEG-MIL stage.
`csg_a01` outperforms `csg_a005` on the main Stage22 5-fold metrics.
`rq16` is the strongest Stage23 candidate across the main 5-fold summary metrics.

## Output Files

- `results_stage24/stage24_rce_v4_csg_summary/stage24_stage22_csg_init_summary.csv`
- `results_stage24/stage24_rce_v4_csg_summary/stage24_stage23_region_query_summary.csv`
- `results_stage24/stage24_rce_v4_csg_summary/stage24_metric_deltas.csv`
- `results_stage24/stage24_rce_v4_csg_summary/stage24_rce_v4_csg_summary_report.md`

## Summary Tables

### Stage22

| variant | status | test_auc_formatted | test_acc_formatted | test_f1_formatted | balanced_acc_formatted | pr_auc_formatted |
| --- | --- | --- | --- | --- | --- | --- |
| csg_a005 | ok | 0.961928 ± 0.018122 | 0.897719 ± 0.018903 | 0.885097 ± 0.022322 | 0.883203 ± 0.028104 | 0.921331 ± 0.038047 |
| csg_a01 | ok | 0.970213 ± 0.013016 | 0.922509 ± 0.018779 | 0.914456 ± 0.020681 | 0.917143 ± 0.019648 | 0.944399 ± 0.025168 |

### Stage23

| variant | prototype_number | status | test_auc_formatted | test_acc_formatted | test_f1_formatted | balanced_acc_formatted | pr_auc_formatted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq8 | 8 | ok | 0.963998 ± 0.017688 | 0.910122 ± 0.010627 | 0.897900 ± 0.011859 | 0.891196 ± 0.019233 | 0.935604 ± 0.033146 |
| rq16 | 16 | ok | 0.970213 ± 0.013016 | 0.922509 ± 0.018779 | 0.914456 ± 0.020681 | 0.917143 ± 0.019648 | 0.944399 ± 0.025168 |
| rq32 | 32 | ok | 0.957726 ± 0.014726 | 0.907003 ± 0.014766 | 0.895646 ± 0.015013 | 0.891710 ± 0.013509 | 0.921822 ± 0.031662 |
