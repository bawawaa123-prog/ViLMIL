# Stage46 PRARC Gate Smoke Report

## Smoke Status
- run_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage46/stage46_prarc_gate_smoke_v1_g08_s1_s1`
- smoke_completed: `True`
- checkpoint_exists: `True`
- split_0_results_exists: `True`
- fold_summary_exists: `True`

## Fold0 / 1 Epoch Metrics
- test_auc: `0.970999053030303`
- test_acc: `0.922680412371134`
- test_f1: `0.9164441381686622`
- balanced_acc: `0.9303977272727272`
- sensitivity: `0.9545454545454546`
- specificity: `0.90625`
- pr_auc: `0.9372102134155632`

## PRARC Debug Probe
- probe_success: `True`
- prarc_enabled: `True`
- prarc_gate_init: `0.8`
- gate_mean: `0.79714035987854`
- gate_min: `0.7971360087394714`
- gate_max: `0.7971456050872803`
- gate_nonconstant: `True`
- gate_features_present: `True`
- gate_feature_names: `['concept_pred_margin_abs', 'low_high_margin_agreement', 'visual_concept_conflict', 'dominant_source_ratio', 'prediction_confidence_margin', 'low_high_sign_agreement']`
- visual_residual_adjusted: `True`
- skip_reason: `None`

## Runtime Safety
- log_has_traceback: `False`
- log_has_nan_or_inf: `False`
- checkpoint_params_finite: `True`

## Recommendation
- recommend_enter_step47_prarc_5fold: `True`
