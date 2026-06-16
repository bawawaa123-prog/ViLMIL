# Step48b PRARC-v2 Variant Sweep Report

## Step48b Purpose
- Run fold0 / 1 epoch smoke variants on top of the existing Step48 PRARC-v2 path.
- Do not enter 5-fold or evidence export unless a variant shows materially stronger gate dynamics without obvious metric collapse.

## Known Step48 Issue
- `v2_gain2_g08` was engineering-stable but its gate stayed near-constant.
- Step48 reference gate mean/std/range: `0.9318` / `0.0017` / `0.0050`.
- Step48 conflict vs non-conflict gate gap: `-0.0020`; direction was correct but too small.

## Variants In Scope
- Requested variants: `v2_gain2_g08, v2_gain4_g08, v2_confprior_g08, v2_varreg_g08`

## v2_gain2_g08
- smoke_completed: `True`
- checkpoint_exists: `True`
- test_auc: `0.9732`
- test_acc: `0.8918`
- test_f1: `0.8702`
- balanced_acc: `0.8482`
- sensitivity: `0.7121`
- specificity: `0.9844`
- pr_auc: `0.9396`
- gate_mean/std/range: `0.9318` / `0.0017` / `0.0050`
- gate_min/max: `0.9281` / `0.9330`
- conflict_gate_mean: `0.9301`
- non_conflict_gate_mean: `0.9321`
- conflict_minus_nonconflict: `-0.0020`
- log_has_traceback: `False`
- log_has_nan_or_inf: `False`
- recommend_for_step49: `False`

## v2_gain4_g08
- smoke_completed: `True`
- checkpoint_exists: `True`
- test_auc: `0.9731`
- test_acc: `0.8918`
- test_f1: `0.8702`
- balanced_acc: `0.8482`
- sensitivity: `0.7121`
- specificity: `0.9844`
- pr_auc: `0.9396`
- gate_mean/std/range: `0.9946` / `0.0003` / `0.0009`
- gate_min/max: `0.9939` / `0.9948`
- conflict_gate_mean: `0.9943`
- non_conflict_gate_mean: `0.9946`
- conflict_minus_nonconflict: `-0.0003`
- log_has_traceback: `False`
- log_has_nan_or_inf: `False`
- recommend_for_step49: `False`

## v2_confprior_g08
- smoke_completed: `True`
- checkpoint_exists: `True`
- test_auc: `0.9732`
- test_acc: `0.8918`
- test_f1: `0.8702`
- balanced_acc: `0.8482`
- sensitivity: `0.7121`
- specificity: `0.9844`
- pr_auc: `0.9396`
- gate_mean/std/range: `0.9318` / `0.0017` / `0.0055`
- gate_min/max: `0.9275` / `0.9330`
- conflict_gate_mean: `0.9294`
- non_conflict_gate_mean: `0.9322`
- conflict_minus_nonconflict: `-0.0028`
- log_has_traceback: `False`
- log_has_nan_or_inf: `False`
- recommend_for_step49: `False`

## v2_varreg_g08
- smoke_completed: `True`
- checkpoint_exists: `True`
- test_auc: `0.9732`
- test_acc: `0.8918`
- test_f1: `0.8702`
- balanced_acc: `0.8482`
- sensitivity: `0.7121`
- specificity: `0.9844`
- pr_auc: `0.9396`
- gate_mean/std/range: `0.9318` / `0.0017` / `0.0050`
- gate_min/max: `0.9281` / `0.9330`
- conflict_gate_mean: `0.9301`
- non_conflict_gate_mean: `0.9321`
- conflict_minus_nonconflict: `-0.0020`
- log_has_traceback: `False`
- log_has_nan_or_inf: `False`
- recommend_for_step49: `False`

## Overall Judgment
- best_gate_dynamics_variant: `v2_confprior_g08`
- best_metric_retention_variant: `v2_gain2_g08`
- recommend_enter_step49: `False`
- recommended_variant: `None`
- conclusion: `No Step48b variant met the combined stability + dynamics bar. PRARC should currently be treated as a negative ablation.`
- step49_decision: `Do not enter Step49 PRARC-v2 5-fold.`

