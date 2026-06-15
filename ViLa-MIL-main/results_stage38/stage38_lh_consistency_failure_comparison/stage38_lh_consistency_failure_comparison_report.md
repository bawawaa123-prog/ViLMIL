# Stage38 LH-Consistency Failure Comparison

## Scope
- This step does not train any model.
- This step re-exports evidence and compares Step33-style failure analysis between two existing variants.

## Compared Variants
- Skeleton: `skeleton`
- LH consistency: `lh_l001_m0`

## Key Findings
- Fixed cases: `6`
- Regressed cases: `2`
- Persistent errors: `12`
- Low-high conflict delta (`lh_l001_m0 - skeleton`): `-2`
- Visual residual override delta (`lh_l001_m0 - skeleton`): `1`
- AUC delta: `-0.0039`
- PR-AUC delta: `-0.0034`
- Recommendation: `keep_skeleton_as_final_main_model`

## Rationale
- visual_residual_override increases

## Metric Comparison

| variant | test_auc | test_acc | test_f1 | balanced_acc | sensitivity | specificity | pr_auc | num_correct | num_error | num_slides | manifest_test_auc | manifest_test_acc | manifest_test_f1 | manifest_balanced_acc | manifest_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton | 0.9757339015151515 | 0.9072164948453608 | 0.8987826086956522 | 0.9076704545454546 | 0.9090909090909091 | 0.90625 | 0.9547979037555255 | 176.0 | 18.0 | 194.0 | 0.9757339015151515 | 0.9072164948453608 | 0.8987826086956522 | 0.9076704545454546 | 0.9547979037555255 |
| lh_l001_m0 | 0.9718276515151515 | 0.9278350515463918 | 0.919625946969697 | 0.919625946969697 | 0.8939393939393939 | 0.9453125 | 0.9514222173693676 | 180.0 | 14.0 | 194.0 | 0.9718276515151515 | 0.9278350515463918 | 0.919625946969697 | 0.919625946969697 | 0.9514222173693676 |
| lh_l001_m0 - skeleton | -0.00390625 | 0.020618556701030966 | 0.02084333827404483 | 0.011955492424242431 | -0.015151515151515138 | 0.0390625 | -0.003375686386157928 | 4.0 | -4.0 | NA | NA | NA | NA | NA | NA |

## Failure Type Comparison

| failure_type | skeleton_count_any_label | skeleton_count_as_primary | lh_l001_m0_count_any_label | lh_l001_m0_count_as_primary | lh_l001_m0_minus_skeleton_any_label | lh_l001_m0_minus_skeleton_primary |
| --- | --- | --- | --- | --- | --- | --- |
| uncertain_low_margin | 14 | 0 | 11 | 0 | -3 | 0 |
| concept_wrong_class_drift | 7 | 1 | 5 | 0 | -2 | -1 |
| csg_misleading | 2 | 0 | 0 | 0 | -2 | 0 |
| high_scale_dominant_wrong | 11 | 0 | 9 | 0 | -2 | 0 |
| low_high_conflict | 10 | 4 | 8 | 0 | -2 | -4 |
| low_scale_dominant_wrong | 3 | 0 | 2 | 0 | -1 | 0 |
| prompt_confusion | 4 | 0 | 5 | 0 | 1 | 0 |
| visual_residual_override | 13 | 13 | 14 | 14 | 1 | 1 |

## Error Overlap Summary

- Total overlap rows: `194`
- Fixed rows: `6`
- Regressed rows: `2`
- Persistent rows: `12`

## Low / High Conflict Comparison

| variant | error_low_high_conflict_count | error_both_support_wrong_count | error_margin_gap_mean | error_margin_gap_median | fixed_cases_with_conflict_repaired |
| --- | --- | --- | --- | --- | --- |
| skeleton | 10 | 7 | 0.07268947185043033 | 0.060074711218476295 | NA |
| lh_l001_m0 | 8 | 5 | 0.08300534637445312 | 0.0748761661816387 | NA |
| lh_l001_m0 - skeleton | -2 | -2 | 0.010315874524022786 | 0.014801454963162403 | 4.0 |

## Visual Override Comparison

| variant | visual_residual_override_error_count | error_visual_source_ratio_mean | all_visual_source_ratio_mean | error_wrong_class_visual_support_ratio |
| --- | --- | --- | --- | --- |
| skeleton | 13 | 0.6286197268637571 | 0.8286127242077914 | NA |
| lh_l001_m0 | 14 | 0.7949460684331477 | 0.7938890261301206 | NA |
| lh_l001_m0 - skeleton | 1 | 0.16632634156939063 | -0.03472369807767084 | NA |

## Fixed Cases Preview

| slide_id | label | skeleton_pred | lh_pred | skeleton_correct | lh_correct | status | skeleton_failure_labels | lh_failure_labels | skeleton_primary_failure_type | lh_primary_failure_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2462598-B | 0 | 1 | 0 | False | True | fixed | low_high_conflict|uncertain_low_margin | NA | low_high_conflict | correct_prediction |
| 2468673-B2 | 0 | 1 | 0 | False | True | fixed | low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | NA | low_high_conflict | correct_prediction |
| 2471959-B2 | 0 | 1 | 0 | False | True | fixed | low_high_conflict|uncertain_low_margin | NA | low_high_conflict | correct_prediction |
| 2487558-B2 | 0 | 1 | 0 | False | True | fixed | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong|uncertain_low_margin | NA | visual_residual_override | correct_prediction |
| 25021640B3 | 0 | 1 | 0 | False | True | fixed | visual_residual_override|concept_wrong_class_drift|low_scale_dominant_wrong|csg_misleading|uncertain_low_margin | NA | visual_residual_override | correct_prediction |
| 25023690B3 | 0 | 1 | 0 | False | True | fixed | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | NA | visual_residual_override | correct_prediction |

## Regressed Cases Preview

| slide_id | label | skeleton_pred | lh_pred | skeleton_correct | lh_correct | status | skeleton_failure_labels | lh_failure_labels | skeleton_primary_failure_type | lh_primary_failure_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25015422B | 0 | 0 | 1 | True | False | regressed | NA | visual_residual_override|low_high_conflict|uncertain_low_margin | correct_prediction | visual_residual_override |
| 25023961B2 | 1 | 1 | 0 | True | False | regressed | NA | visual_residual_override|low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | correct_prediction | visual_residual_override |

## Persistent Error Preview

| slide_id | label | skeleton_pred | lh_pred | skeleton_correct | lh_correct | status | skeleton_failure_labels | lh_failure_labels | skeleton_primary_failure_type | lh_primary_failure_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2471884-B2 | 0 | 1 | 1 | False | False | persistent_error | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong | visual_residual_override | visual_residual_override |
| 2485275-B2 | 0 | 1 | 1 | False | False | persistent_error | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override | visual_residual_override |
| 25001929B3 | 0 | 1 | 1 | False | False | persistent_error | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong | visual_residual_override | visual_residual_override |
| 25015087B2 | 0 | 1 | 1 | False | False | persistent_error | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override | visual_residual_override |
| 25019273B | 1 | 0 | 0 | False | False | persistent_error | visual_residual_override|low_high_conflict|uncertain_low_margin | visual_residual_override|low_high_conflict|uncertain_low_margin|prompt_confusion | visual_residual_override | visual_residual_override |
| 25023687B2 | 0 | 1 | 1 | False | False | persistent_error | low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | visual_residual_override|uncertain_low_margin | low_high_conflict | visual_residual_override |
| 25023972B3 | 1 | 0 | 0 | False | False | persistent_error | visual_residual_override|uncertain_low_margin|prompt_confusion | visual_residual_override|concept_wrong_class_drift|low_scale_dominant_wrong|prompt_confusion | visual_residual_override | visual_residual_override |
| 25023977B2 | 0 | 1 | 1 | False | False | persistent_error | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override | visual_residual_override |
| 25025076B | 1 | 0 | 0 | False | False | persistent_error | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override | visual_residual_override |
| 25027723B4 | 1 | 0 | 0 | False | False | persistent_error | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | visual_residual_override | visual_residual_override |
| 25033447B4 | 1 | 0 | 0 | False | False | persistent_error | concept_wrong_class_drift|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong|csg_inactive_or_zero|uncertain_low_margin | concept_wrong_class_drift | visual_residual_override |
| 25057865B1 | 1 | 0 | 0 | False | False | persistent_error | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|csg_misleading|uncertain_low_margin | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | visual_residual_override | visual_residual_override |

## Output Files

- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_variant_metric_comparison.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_failure_type_comparison.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_error_overlap.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_fixed_cases.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_regressed_cases.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_persistent_error_cases.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_low_high_conflict_comparison.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_visual_override_comparison.csv`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_recommendations.json`
- `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage38/stage38_lh_consistency_failure_comparison/stage38_lh_consistency_failure_comparison_report.md`
