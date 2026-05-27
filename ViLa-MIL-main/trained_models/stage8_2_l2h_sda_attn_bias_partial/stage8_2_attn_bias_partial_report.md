# Stage 8.2 L2H-SDA Attention-Bias PEPS Partial Check

## Partial Metric Summary

| exp_id | experiment | status | test_auc_mean | test_acc_mean | test_f1_mean | val_auc_mean | balanced_acc_mean | sensitivity_mean | specificity_mean | pr_auc_mean | test_auc_std | test_acc_std | test_f1_std | val_auc_std | balanced_acc_std | sensitivity_std | specificity_std | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_peps | Baseline PEPS topk=5 tau=0.07 | partial_3folds | 0.966918 | 0.934601 | 0.927901 | 0.971287 | 0.932122 | 0.924242 | 0.940002 | 0.940651 | 0.007971 | 0.002279 | 0.001817 | 0.003952 | 0.004547 | 0.024742 | 0.015920 | 0.009610 |
| l2h_lambda005 | Stage 8.1-A L2H-SDA-PEPS lambda=0.05 | partial_3folds | 0.945686 | 0.903602 | 0.892118 | 0.966205 | 0.892732 | 0.858586 | 0.926878 | 0.902937 | 0.026621 | 0.025765 | 0.030465 | 0.015058 | 0.037527 | 0.074570 | 0.003973 | 0.047639 |
| l2h_attn_bias_b005 | Stage 8.2 L2H-SDA Attention-Bias beta=0.05 | partial_3folds | 0.956530 | 0.910502 | 0.900327 | 0.963223 | 0.900417 | 0.868687 | 0.932148 | 0.918451 | 0.010597 | 0.006392 | 0.007738 | 0.011523 | 0.011095 | 0.028570 | 0.009552 | 0.001832 |
| delta_attn_bias_minus_base | attn_bias - baseline | delta | -0.010389 | -0.024099 | -0.027574 | -0.008065 | -0.031705 | -0.055556 | -0.007854 | -0.022200 | 0.002626 | 0.004113 | 0.005921 | 0.007572 | 0.006548 | 0.003828 | -0.006368 | -0.007778 |
| delta_attn_bias_minus_l005 | attn_bias - lambda=0.05 | delta | 0.010844 | 0.006900 | 0.008209 | -0.002982 | 0.007685 | 0.010101 | 0.005270 | 0.015515 | -0.016024 | -0.019373 | -0.022727 | -0.003534 | -0.026432 | -0.046000 | 0.005579 | -0.045807 |

## Fold Delta

| fold | test_auc_delta_attn_bias_minus_base | test_auc_delta_attn_bias_minus_l005 | test_acc_delta_attn_bias_minus_base | test_acc_delta_attn_bias_minus_l005 | test_f1_delta_attn_bias_minus_base | test_f1_delta_attn_bias_minus_l005 | val_auc_delta_attn_bias_minus_base | val_auc_delta_attn_bias_minus_l005 | balanced_acc_delta_attn_bias_minus_base | balanced_acc_delta_attn_bias_minus_l005 | sensitivity_delta_attn_bias_minus_base | sensitivity_delta_attn_bias_minus_l005 | specificity_delta_attn_bias_minus_base | specificity_delta_attn_bias_minus_l005 | pr_auc_delta_attn_bias_minus_base | pr_auc_delta_attn_bias_minus_l005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -0.008286 | -0.010890 | -0.015464 | -0.020619 | -0.016730 | -0.023091 | -0.019650 | 0.000355 | -0.015388 | -0.026634 | -0.015152 | -0.045455 | -0.015625 | -0.007812 | -0.037866 | -0.047609 |
| 2 | -0.003374 | 0.015684 | -0.030928 | 0.025773 | -0.036617 | 0.031556 | 0.000705 | -0.008109 | -0.049124 | 0.037879 | -0.106061 | 0.075758 | 0.007812 | 0.000000 | -0.012376 | 0.023673 |
| 4 | -0.019506 | 0.027738 | -0.025907 | 0.015544 | -0.029374 | 0.016162 | -0.005249 | -0.001193 | -0.030601 | 0.011811 | -0.045455 | 0.000000 | -0.015748 | 0.023622 | -0.016358 | 0.070480 |

## Key Findings

- Fold 1 preservation check: test_auc delta vs baseline=-0.008286, test_acc delta=-0.015464, test_f1 delta=-0.016730.
- Fold 2 sensitivity check: delta vs baseline=-0.106061, delta vs lambda=0.05=0.075758.
- Fold 4 test_auc check: delta vs baseline=-0.019506, delta vs lambda=0.05=0.027738.
- Partial sensitivity mean: baseline=0.924242, lambda=0.05=0.858586, attn_bias=0.868687.
- Partial test_auc mean: baseline=0.966918, lambda=0.05=0.945686, attn_bias=0.956530.
- Partial test_f1 mean: baseline=0.927901, lambda=0.05=0.892118, attn_bias=0.900327.