# Stage 8.1-A L2H-SDA-PEPS Small Lambda Re-check

## Metric Summary

| exp_id | experiment | status | test_auc_mean | test_acc_mean | test_f1_mean | val_auc_mean | balanced_acc_mean | sensitivity_mean | specificity_mean | pr_auc_mean | test_auc_std | test_acc_std | test_f1_std | val_auc_std | balanced_acc_std | sensitivity_std | specificity_std | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_peps | Baseline PEPS topk=5 tau=0.07 | ok | 0.968327 | 0.939047 | 0.932179 | 0.972999 | 0.931275 | 0.906422 | 0.956127 | 0.945866 | 0.006489 | 0.007595 | 0.007740 | 0.007620 | 0.005445 | 0.029051 | 0.024452 | 0.014115 |
| l2h_lambda010 | L2H-SDA-PEPS lambda=0.10 | ok | 0.955674 | 0.901875 | 0.887241 | 0.957837 | 0.881789 | 0.818498 | 0.945079 | 0.908406 | 0.026535 | 0.049981 | 0.061428 | 0.015957 | 0.065229 | 0.115996 | 0.026754 | 0.063168 |
| l2h_lambda005 | L2H-SDA-PEPS lambda=0.05 | ok | 0.955419 | 0.916303 | 0.906609 | 0.967388 | 0.906629 | 0.876029 | 0.937229 | 0.921202 | 0.024824 | 0.027813 | 0.032071 | 0.015189 | 0.035161 | 0.061806 | 0.019834 | 0.046190 |
| delta_l005_minus_base | lambda=0.05 - baseline | delta | -0.012908 | -0.022745 | -0.025570 | -0.005612 | -0.024646 | -0.030393 | -0.018898 | -0.024665 | 0.018335 | 0.020218 | 0.024331 | 0.007569 | 0.029715 | 0.032755 | -0.004618 | 0.032075 |
| delta_l010_minus_base | lambda=0.10 - baseline | delta | -0.012653 | -0.037172 | -0.044938 | -0.015162 | -0.049486 | -0.087924 | -0.011048 | -0.037460 | 0.020047 | 0.042386 | 0.053688 | 0.008337 | 0.059783 | 0.086945 | 0.002302 | 0.049053 |
| delta_l005_minus_l010 | lambda=0.05 - lambda=0.10 | delta | -0.000255 | 0.014428 | 0.019368 | 0.009550 | 0.024841 | 0.057531 | -0.007849 | 0.012795 | -0.001711 | -0.022168 | -0.029357 | -0.000768 | -0.030068 | -0.054190 | -0.006920 | -0.016978 |

## Fold Delta

| fold | test_auc_delta_l005_minus_base | test_auc_delta_l010_minus_base | test_auc_delta_l005_minus_l010 | test_acc_delta_l005_minus_base | test_acc_delta_l010_minus_base | test_acc_delta_l005_minus_l010 | test_f1_delta_l005_minus_base | test_f1_delta_l010_minus_base | test_f1_delta_l005_minus_l010 | val_auc_delta_l005_minus_base | val_auc_delta_l010_minus_base | val_auc_delta_l005_minus_l010 | balanced_acc_delta_l005_minus_base | balanced_acc_delta_l010_minus_base | balanced_acc_delta_l005_minus_l010 | sensitivity_delta_l005_minus_base | sensitivity_delta_l010_minus_base | sensitivity_delta_l005_minus_l010 | specificity_delta_l005_minus_base | specificity_delta_l010_minus_base | specificity_delta_l005_minus_l010 | pr_auc_delta_l005_minus_base | pr_auc_delta_l010_minus_base | pr_auc_delta_l005_minus_l010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.002604 | 0.000414 | 0.002190 | 0.005155 | -0.010309 | 0.015464 | 0.006362 | -0.012595 | 0.018957 | -0.020005 | -0.021839 | 0.001835 | 0.011245 | -0.018821 | 0.030066 | 0.030303 | -0.045455 | 0.075758 | -0.007812 | 0.007812 | -0.015625 | 0.009742 | -0.000696 | 0.010438 |
| 2 | -0.019058 | -0.048828 | 0.029770 | -0.056701 | -0.118557 | 0.061856 | -0.068173 | -0.150721 | 0.082549 | 0.008814 | -0.012927 | 0.021742 | -0.087003 | -0.177912 | 0.090909 | -0.181818 | -0.363636 | 0.181818 | 0.007812 | 0.007812 | 0.000000 | -0.036049 | -0.137982 | 0.101934 |
| 3 | 0.012222 | 0.007639 | 0.004583 | 0.000000 | 0.000000 | 0.000000 | 0.000814 | 0.000416 | 0.000397 | -0.011095 | -0.016941 | 0.005846 | 0.007051 | 0.003526 | 0.003526 | 0.029851 | 0.014925 | 0.014925 | -0.015748 | -0.007874 | -0.007874 | 0.004912 | 0.004065 | 0.000847 |
| 4 | -0.047244 | -0.023383 | -0.023861 | -0.041451 | -0.056995 | 0.015544 | -0.045536 | -0.062329 | 0.016793 | -0.004056 | -0.024517 | 0.020461 | -0.042412 | -0.057862 | 0.015450 | -0.045455 | -0.060606 | 0.015152 | -0.039370 | -0.055118 | 0.015748 | -0.086838 | -0.045062 | -0.041776 |
| 5 | -0.013064 | 0.000895 | -0.013958 | -0.020725 | 0.000000 | -0.020725 | -0.021317 | 0.000539 | -0.021856 | -0.001716 | 0.000414 | -0.002131 | -0.012109 | 0.003639 | -0.015748 | 0.015152 | 0.015152 | 0.000000 | -0.039370 | -0.007874 | -0.031496 | -0.015092 | -0.007624 | -0.007467 |

## Key Findings

- Compared with lambda=0.10, lambda=0.05 is better on: test_acc, test_f1, val_auc, balanced_acc, sensitivity, pr_auc.
- Compared with baseline, lambda=0.05 is close to or above baseline on: none.
- Fold 2: test_auc delta (0.05 - baseline)=-0.019058, test_acc delta=-0.056701, test_f1 delta=-0.068173, sensitivity delta=-0.181818.
- Fold 4: test_auc delta (0.05 - baseline)=-0.047244, test_acc delta=-0.041451, test_f1 delta=-0.045536, sensitivity delta=-0.045455.
- Sensitivity mean: baseline=0.906422, lambda=0.10=0.818498, lambda=0.05=0.876029.
- Test AUC std: baseline=0.006489, lambda=0.10=0.026535, lambda=0.05=0.024824.
- Test F1 std: baseline=0.007740, lambda=0.10=0.061428, lambda=0.05=0.032071.