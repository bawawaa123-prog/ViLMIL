# Step52B RCE Ablation Paper Table

## Main Table

| Variant | AUC | ACC | F1 | BACC | PR-AUC | ΔAUC vs full | ΔACC vs full | ΔF1 vs full | ΔBACC vs full | ΔPR-AUC vs full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full RCE-v4-CSG-rq16 | 0.9702 ± 0.0130 | 0.9225 ± 0.0188 | 0.9145 ± 0.0207 | 0.9171 ± 0.0196 | 0.9444 ± 0.0252 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| w/o CSG | 0.9655 ± 0.0154 | 0.9215 ± 0.0165 | 0.9127 ± 0.0191 | 0.9134 ± 0.0217 | 0.9278 ± 0.0361 | -0.0047 | -0.0010 | -0.0018 | -0.0037 | -0.0166 |
| w/o concept prior | 0.9592 ± 0.0221 | 0.9019 ± 0.0390 | 0.8903 ± 0.0445 | 0.8891 ± 0.0447 | 0.9078 ± 0.0495 | -0.0110 | -0.0206 | -0.0241 | -0.0280 | -0.0366 |
| w/o visual residual | 0.9645 ± 0.0192 | 0.9049 ± 0.0167 | 0.8946 ± 0.0160 | 0.8960 ± 0.0154 | 0.9239 ± 0.0467 | -0.0057 | -0.0176 | -0.0198 | -0.0211 | -0.0205 |
| w/o logit calibration | 0.9686 ± 0.0169 | 0.9215 ± 0.0277 | 0.9111 ± 0.0338 | 0.9076 ± 0.0394 | 0.9486 ± 0.0226 | -0.0017 | -0.0010 | -0.0034 | -0.0095 | +0.0042 |

The largest overall degradation comes from `w/o concept prior`, which indicates that concept prior is the most critical single component among the tested RCE modules.

## Statistical Notes

- Fold-level test metrics were extracted from `fold_summary.csv` when available.
- `epoch_details.csv` was not used for fold-level test deltas because it only stores train/val trajectories.
- Paired t-tests are reported only when matched fold-level test metrics are available.

## Paired T-Test Table

| Variant | p(AUC) | p(ACC) | p(F1) | p(BACC) | p(PR-AUC) | Fold Data Status |
| --- | --- | --- | --- | --- | --- | --- |
| Full RCE-v4-CSG-rq16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | self-reference |
| w/o CSG | 0.2282 | 0.8655 | 0.7938 | 0.5728 | 0.0878 | ok |
| w/o concept prior | 0.1383 | 0.2344 | 0.2254 | 0.1742 | 0.0606 | ok |
| w/o visual residual | 0.1743 | 0.2320 | 0.2271 | 0.2523 | 0.2008 | ok |
| w/o logit calibration | 0.7506 | 0.9353 | 0.8246 | 0.5864 | 0.6665 | ok |

## Fold-Level Delta Preview

| variant | paper_label | fold | status | test_auc | delta_test_auc_vs_full | test_acc | delta_test_acc_vs_full | test_f1 | delta_test_f1_vs_full | balanced_acc | delta_balanced_acc_vs_full | pr_auc | delta_pr_auc_vs_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage23_history | Stage23 reference | 1 | ready | 0.9757339015151516 | 0.0 | 0.9072164948453608 | 0.0 | 0.8987826086956522 | 0.0 | 0.9076704545454546 | 0.0 | 0.9547979037555256 | 0.0 |
| stage23_history | Stage23 reference | 2 | ready | 0.9545454545454546 | 0.0 | 0.9123711340206184 | 0.0 | 0.9020405738557044 | 0.0 | 0.9005681818181819 | 0.0 | 0.9209130872748004 | 0.0 |
| stage23_history | Stage23 reference | 3 | ready | 0.9920084616288636 | 0.0 | 0.9587628865979382 | 0.0 | 0.9544012222352802 | 0.0 | 0.9544012222352802 | 0.0 | 0.986152388544612 | 0.0 |
| stage23_history | Stage23 reference | 4 | ready | 0.9681460272011454 | 0.0 | 0.911917098445596 | 0.0 | 0.9024933876192458 | 0.0 | 0.9039608685277976 | 0.0 | 0.9433541685814426 | 0.0 |
| stage23_history | Stage23 reference | 5 | ready | 0.9606299212598424 | 0.0 | 0.922279792746114 | 0.0 | 0.914563966356795 | 0.0 | 0.9191123836793128 | 0.0 | 0.9167779601908418 | 0.0 |
| full | Full RCE-v4-CSG-rq16 | 1 | ready | 0.9757339015151516 | 0.0 | 0.9072164948453608 | 0.0 | 0.8987826086956522 | 0.0 | 0.9076704545454546 | 0.0 | 0.9547979037555256 | 0.0 |
| full | Full RCE-v4-CSG-rq16 | 2 | ready | 0.9545454545454546 | 0.0 | 0.9123711340206184 | 0.0 | 0.9020405738557044 | 0.0 | 0.9005681818181819 | 0.0 | 0.9209130872748004 | 0.0 |
| full | Full RCE-v4-CSG-rq16 | 3 | ready | 0.9920084616288636 | 0.0 | 0.9587628865979382 | 0.0 | 0.9544012222352802 | 0.0 | 0.9544012222352802 | 0.0 | 0.986152388544612 | 0.0 |
| full | Full RCE-v4-CSG-rq16 | 4 | ready | 0.9681460272011454 | 0.0 | 0.911917098445596 | 0.0 | 0.9024933876192458 | 0.0 | 0.9039608685277976 | 0.0 | 0.9433541685814426 | 0.0 |
| full | Full RCE-v4-CSG-rq16 | 5 | ready | 0.9606299212598424 | 0.0 | 0.922279792746114 | 0.0 | 0.914563966356795 | 0.0 | 0.9191123836793128 | 0.0 | 0.9167779601908418 | 0.0 |

