# Stage51B DEG Skeleton Equivalence Audit

## Summary

- Static Stage51 vs Stage23 flag audit completed for run_stage51_reproduce_rce_and_deg_skeleton.sh and run_stage23_rce_v4_csg_region_queries_5fold.sh.
- Real batch source: split=0 train sample | slide_id=2460242-B2.
- DEG skeleton passthrough active: True.
- Extra DEG trainable parameters in skeleton mode: 0.
- Total failed checks: 0.

## Status Counts

- PASS: 47
- FAIL: 0
- N/A: 8

## Detailed Checks

| Category | Check | Status | Details |
| --- | --- | --- | --- |
| static_script_flag | prototype_number | PASS | stage51=16 | stage23=16 | expected_ref=16 |
| static_script_flag | rce_use_logit_calibration | PASS | stage51=true | stage23=true | expected_ref=true |
| static_script_flag | rce_use_concept_prior | PASS | stage51=true | stage23=true | expected_ref=true |
| static_script_flag | rce_use_visual_residual | PASS | stage51=true | stage23=true | expected_ref=true |
| static_script_flag | rce_use_cross_scale_graph | PASS | stage51=true | stage23=true | expected_ref=true |
| static_script_flag | rce_cross_scale_graph_init | PASS | stage51=0.1 | stage23=0.1 | expected_ref=0.1 |
| static_script_flag | rce_cross_scale_graph_norm | PASS | stage51=sqrt | stage23=sqrt | expected_ref=sqrt |
| static_script_flag | rce_logit_scale_init | PASS | stage51=10.0 | stage23=10.0 | expected_ref=10.0 |
| static_script_flag | rce_concept_prior_strength | PASS | stage51=1.0 | stage23=1.0 | expected_ref=1.0 |
| static_script_flag | rce_visual_residual_init | PASS | stage51=0.05 | stage23=0.05 | expected_ref=0.05 |
| static_script_flag | concept_prompt_path | PASS | stage51=${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json} | stage23=${CONCEPT12_PATH:-${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json} | expected_ref=${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json |
| static_script_flag | split_dir | PASS | stage51=${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100} | stage23=${SPLIT_DIR:-splits/adenocarcinoma/task_adenocarcinoma_strictcv_100} | expected_ref=splits/adenocarcinoma/task_adenocarcinoma_strictcv_100 |
| static_script_flag | text_prompt_path | PASS | stage51=${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv} | stage23=${TEXT_PROMPT_PATH:-${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv} | expected_ref=${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv |
| static_script_flag | data_folder_s | PASS | stage51=features_biomedclip_5x | stage23=features_biomedclip_5x | expected_ref=features_biomedclip_5x |
| static_script_flag | data_folder_l | PASS | stage51=features_biomedclip_20x | stage23=features_biomedclip_20x | expected_ref=features_biomedclip_20x |
| static_script_flag | scale_mode | PASS | stage51=dual | stage23=dual | expected_ref=dual |
| static_script_flag | prompt_ensemble_mode | PASS | stage51=embedding_mean | stage23=embedding_mean | expected_ref=embedding_mean |
| static_script_flag | k | PASS | stage51=5 | stage23=5 | expected_ref=${K_FOLDS} |
| static_script_flag | k_start | PASS | stage51=0 | stage23=0 | expected_ref=${K_START} |
| static_script_flag | k_end | PASS | stage51=4 | stage23=4 | expected_ref=${K_END} |
| static_script_flag | max_epochs | PASS | stage51=${MAX_EPOCHS:-20} | stage23=${MAX_EPOCHS:-20} | expected_ref=${MAX_EPOCHS} |
| static_script_flag | seed | PASS | stage51=${SEED:-1} | stage23=${SEED:-1} | expected_ref=${SEED} |
| static_disabled_module | deg_use_region_graph | PASS | stage51_flag_present=False | cli_default_false=True |
| static_disabled_module | deg_use_concept_graph | PASS | stage51_flag_present=False | cli_default_false=True |
| static_disabled_module | rce_use_visual_evidence_gate | PASS | stage51_flag_present=False | cli_default_false=True |
| static_disabled_module | rce_use_prarc_gate | PASS | stage51_flag_present=False | cli_default_false=True |
| static_disabled_module | rce_use_hcrc | PASS | stage51_flag_present=False | cli_default_false=True |
| static_disabled_module | rce_use_low_high_consistency_loss | PASS | stage51_flag_present=False | cli_default_false=True |
| dynamic_native_params | common_parameter_names | PASS | common_count=376 |
| dynamic_native_params | common_parameter_shapes | PASS | none |
| dynamic_native_params | common_parameter_initial_values | PASS | none |
| dynamic_native_params | deg_extra_trainable_parameters | PASS | none |
| dynamic_output_native_init | y_prob | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_native_init | y_hat | PASS | shape=(1, 1) | max_abs_diff=0 |
| dynamic_output_native_init | loss | PASS | shape=() | max_abs_diff=0 |
| dynamic_output_native_init | final_logits | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_native_init | visual_logits | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_native_init | cross_scale_logits | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_native_init | low_scale_logits | N/A | both unavailable |
| dynamic_output_native_init | high_scale_logits | N/A | both unavailable |
| dynamic_output_native_init | lh_consistency_loss | N/A | both unavailable |
| dynamic_output_native_init | total_loss | N/A | both unavailable |
| dynamic_native_grads | deg_extra_parameter_grad_flow | PASS | none |
| dynamic_shared_weights | load_state_dict_missing_keys | PASS | none |
| dynamic_shared_weights | load_state_dict_unexpected_keys | PASS | none |
| dynamic_output_shared_weights | y_prob | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_shared_weights | y_hat | PASS | shape=(1, 1) | max_abs_diff=0 |
| dynamic_output_shared_weights | loss | PASS | shape=() | max_abs_diff=0 |
| dynamic_output_shared_weights | final_logits | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_shared_weights | visual_logits | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_shared_weights | cross_scale_logits | PASS | shape=(1, 2) | max_abs_diff=0 |
| dynamic_output_shared_weights | low_scale_logits | N/A | both unavailable |
| dynamic_output_shared_weights | high_scale_logits | N/A | both unavailable |
| dynamic_output_shared_weights | lh_consistency_loss | N/A | both unavailable |
| dynamic_output_shared_weights | total_loss | N/A | both unavailable |
