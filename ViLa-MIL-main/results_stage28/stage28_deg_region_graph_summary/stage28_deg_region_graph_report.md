# Stage28 DEG Spatial Region Graph Summary

Step28 reads existing Stage27 result CSV files only. It does not run training, modify model computation, or extract features.

## Mean ± Std Summary

- `skeleton`: `test_auc`=0.970213 ± 0.013016, `test_acc`=0.922509 ± 0.018779, `test_f1`=0.914456 ± 0.020681, `balanced_acc`=0.917143 ± 0.019648, `pr_auc`=0.944399 ± 0.025168
- `rg_k2`: `test_auc`=0.957680 ± 0.018029, `test_acc`=0.910080 ± 0.022236, `test_f1`=0.900262 ± 0.024120, `balanced_acc`=0.900466 ± 0.021540, `pr_auc`=0.922397 ± 0.031184
- `rg_k4`: `test_auc`=0.962692 ± 0.018895, `test_acc`=0.917328 ± 0.023440, `test_f1`=0.908046 ± 0.024718, `balanced_acc`=0.905994 ± 0.018577, `pr_auc`=0.940036 ± 0.028010
- `rg_k8`: `test_auc`=0.965456 ± 0.011463, `test_acc`=0.917323 ± 0.021029, `test_f1`=0.907427 ± 0.023867, `balanced_acc`=0.905217 ± 0.025630, `pr_auc`=0.934208 ± 0.022867

## Deltas Vs Skeleton

- `rg_k2 - skeleton`: `test_auc`=-0.012533, `test_acc`=-0.012430, `test_f1`=-0.014195, `balanced_acc`=-0.016677, `pr_auc`=-0.022002
- `rg_k4 - skeleton`: `test_auc`=-0.007521, `test_acc`=-0.005181, `test_f1`=-0.006410, `balanced_acc`=-0.011149, `pr_auc`=-0.004363
- `rg_k8 - skeleton`: `test_auc`=-0.004756, `test_acc`=-0.005187, `test_f1`=-0.007029, `balanced_acc`=-0.011926, `pr_auc`=-0.010191

## Recommendation

Current recommended main configuration: `RCE-v4-CSG-a01-rq16 / DEG skeleton`.
`DEG skeleton` remains the strongest Stage27 configuration across the main summary metrics.
All current Spatial Region Graph variants (`k=2`, `k=4`, `k=8`) remain below the DEG skeleton and should not be treated as the main performance path.
`k=8` is the closest Region Graph variant on mean `test_auc`, but it still stays below the DEG skeleton and is not consistently best across the other main metrics.
The current Spatial Region Graph should remain an optional exploration module rather than the main DEG-MIL performance module.
A more stable follow-up would be a gated or zero-init Region Graph, or a higher-priority Concept Prompt Graph line, before revisiting cross-scale region graph ideas.

## Output Files

- `results_stage28/stage28_deg_region_graph_summary/stage28_deg_region_graph_summary.csv`
- `results_stage28/stage28_deg_region_graph_summary/stage28_deg_region_graph_metric_deltas.csv`
- `results_stage28/stage28_deg_region_graph_summary/stage28_deg_region_graph_report.md`

## Summary Table

| variant | region_graph | status | test_auc_formatted | test_acc_formatted | test_f1_formatted | balanced_acc_formatted | pr_auc_formatted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton | off | ok | 0.970213 ± 0.013016 | 0.922509 ± 0.018779 | 0.914456 ± 0.020681 | 0.917143 ± 0.019648 | 0.944399 ± 0.025168 |
| rg_k2 | k=2 | ok | 0.957680 ± 0.018029 | 0.910080 ± 0.022236 | 0.900262 ± 0.024120 | 0.900466 ± 0.021540 | 0.922397 ± 0.031184 |
| rg_k4 | k=4 | ok | 0.962692 ± 0.018895 | 0.917328 ± 0.023440 | 0.908046 ± 0.024718 | 0.905994 ± 0.018577 | 0.940036 ± 0.028010 |
| rg_k8 | k=8 | ok | 0.965456 ± 0.011463 | 0.917323 ± 0.021029 | 0.907427 ± 0.023867 | 0.905217 ± 0.025630 | 0.934208 ± 0.022867 |
