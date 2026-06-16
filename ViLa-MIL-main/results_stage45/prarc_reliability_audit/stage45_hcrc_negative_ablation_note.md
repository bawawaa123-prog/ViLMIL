# Stage45 HCRC Negative Ablation Note

## Step44 Results
### hcrc_a002_b8
- test_auc: `0.9613`
- test_acc: `0.9246`
- test_f1: `0.9167`
- balanced_acc: `0.9194`
- sensitivity: `0.9032`
- specificity: `0.9357`
- pr_auc: `0.9209`
### hcrc_a005_b8
- test_auc: `0.9614`
- test_acc: `0.9008`
- test_f1: `0.8856`
- balanced_acc: `0.8846`
- sensitivity: `0.8336`
- specificity: `0.9356`
- pr_auc: `0.9181`
### hcrc_a01_b8
- test_auc: `0.9683`
- test_acc: `0.9194`
- test_f1: `0.9110`
- balanced_acc: `0.9127`
- sensitivity: `0.8912`
- specificity: `0.9341`
- pr_auc: `0.9351`

## Why HCRC Is Not The Main Model
- Step44 completed cleanly for `hcrc_a002_b8`, `hcrc_a005_b8`, and `hcrc_a01_b8`, but none beat the baseline by the Step44 decision rules.
- `hcrc_a002_b8` improves `ACC / F1 / Balanced ACC`, but `AUC` drops from `0.9702` to `0.9613` and `PR-AUC` drops from `0.9444` to `0.9209`.
- `hcrc_a01_b8` is the closest variant on `AUC` (`0.9683`), but its classification metrics and `PR-AUC` still remain below baseline.
- `hcrc_a005_b8` is weaker than baseline on both ranking metrics and classification metrics.
- The current evidence therefore supports keeping HCRC as a negative ablation or future-work branch rather than the default model direction.

## Why Step46 Moves To PRARC
- Step33 showed that visual residual evidence is often strong, but its failures are sample-dependent rather than globally suppressible.
- Step39 already concluded that a scalar visual gate is a negative ablation, which points toward sample-adaptive reliability calibration instead of another fixed residual design.
- Step45 therefore audits reliability features that can drive a sample-adaptive residual gate in Step46 without changing the current forward path yet.
