# Stage17 Cross-Scale Evidence Conflict Analysis

Step17 is a post-hoc cross-scale evidence analysis over the fold0 test full export. It does not modify the model and does not run training.

## Inputs

- root: `/home/ljh/ViLMIL/ViLa-MIL-main`
- evidence_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage13_rce_evidence_export_fold0_test_full`
- graph_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage14_concept_class_graph_fold0`
- stage16_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage16_failure_case_narratives_fold0`
- fold0_slide_count: `194`

## Scale-Evidence Definition

For each slide, scale, and class, Step17 aggregates the top concept paths using:

`aggregated_component = evidence_score * prompt_weight * edge_strength * (1 / rank)`

The slide-level low/high margins below use the signed convention `class0_score - class1_score`.

## Overall Agreement

- low_high_agreement_rate: `0.4381`
- correct_slide_count: `180`
- incorrect_slide_count: `14`

## Correct vs Incorrect Conflict Distribution

| group_value | conflict_type | count | rate |
| --- | --- | --- | --- |
| incorrect | consistent_correct_support | 0 | 0.0000 |
| incorrect | consistent_wrong_class_drift | 8 | 0.5714 |
| incorrect | high_scale_dominant_correct | 0 | 0.0000 |
| incorrect | high_scale_dominant_wrong | 5 | 0.3571 |
| incorrect | low_scale_dominant_correct | 0 | 0.0000 |
| incorrect | low_scale_dominant_wrong | 0 | 0.0000 |
| incorrect | weak_or_mixed_conflict | 1 | 0.0714 |
| correct | consistent_correct_support | 50 | 0.2778 |
| correct | consistent_wrong_class_drift | 0 | 0.0000 |
| correct | high_scale_dominant_correct | 103 | 0.5722 |
| correct | high_scale_dominant_wrong | 0 | 0.0000 |
| correct | low_scale_dominant_correct | 1 | 0.0056 |
| correct | low_scale_dominant_wrong | 0 | 0.0000 |
| correct | weak_or_mixed_conflict | 26 | 0.1444 |

## True Label Pattern Distribution

| group_value | conflict_type | count | rate |
| --- | --- | --- | --- |
| Adenocarcinoma | consistent_correct_support | 2 | 0.0156 |
| Adenocarcinoma | consistent_wrong_class_drift | 8 | 0.0625 |
| Adenocarcinoma | high_scale_dominant_correct | 93 | 0.7266 |
| Adenocarcinoma | high_scale_dominant_wrong | 0 | 0.0000 |
| Adenocarcinoma | low_scale_dominant_correct | 1 | 0.0078 |
| Adenocarcinoma | low_scale_dominant_wrong | 0 | 0.0000 |
| Adenocarcinoma | weak_or_mixed_conflict | 24 | 0.1875 |
| NonAdenocarcinoma | consistent_correct_support | 48 | 0.7273 |
| NonAdenocarcinoma | consistent_wrong_class_drift | 0 | 0.0000 |
| NonAdenocarcinoma | high_scale_dominant_correct | 10 | 0.1515 |
| NonAdenocarcinoma | high_scale_dominant_wrong | 5 | 0.0758 |
| NonAdenocarcinoma | low_scale_dominant_correct | 0 | 0.0000 |
| NonAdenocarcinoma | low_scale_dominant_wrong | 0 | 0.0000 |
| NonAdenocarcinoma | weak_or_mixed_conflict | 3 | 0.0455 |

## Predicted Label Pattern Distribution

| group_value | conflict_type | count | rate |
| --- | --- | --- | --- |
| Adenocarcinoma | consistent_correct_support | 2 | 0.0159 |
| Adenocarcinoma | consistent_wrong_class_drift | 0 | 0.0000 |
| Adenocarcinoma | high_scale_dominant_correct | 93 | 0.7381 |
| Adenocarcinoma | high_scale_dominant_wrong | 5 | 0.0397 |
| Adenocarcinoma | low_scale_dominant_correct | 1 | 0.0079 |
| Adenocarcinoma | low_scale_dominant_wrong | 0 | 0.0000 |
| Adenocarcinoma | weak_or_mixed_conflict | 25 | 0.1984 |
| NonAdenocarcinoma | consistent_correct_support | 48 | 0.7059 |
| NonAdenocarcinoma | consistent_wrong_class_drift | 8 | 0.1176 |
| NonAdenocarcinoma | high_scale_dominant_correct | 10 | 0.1471 |
| NonAdenocarcinoma | high_scale_dominant_wrong | 0 | 0.0000 |
| NonAdenocarcinoma | low_scale_dominant_correct | 0 | 0.0000 |
| NonAdenocarcinoma | low_scale_dominant_wrong | 0 | 0.0000 |
| NonAdenocarcinoma | weak_or_mixed_conflict | 2 | 0.0294 |

## Step16 Failure-Case Mapping

| slide_id | error_direction | low_support_class_name | high_support_class_name | predicted_scale_relation | conflict_type | step16_expected_conflict_type | step16_pattern_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25001929B3 | Adenocarcinoma -> NonAdenocarcinoma | NonAdenocarcinoma | NonAdenocarcinoma | both_scales_support_predicted | consistent_wrong_class_drift | consistent_wrong_class_drift | 1.0 |
| 25023972B3 | NonAdenocarcinoma -> Adenocarcinoma | NonAdenocarcinoma | Adenocarcinoma | high_only_supports_predicted | high_scale_dominant_wrong | high_scale_dominant_wrong | 1.0 |

### 25001929B3

- predicted_scale_relation: `both_scales_support_predicted`
- conflict_type: `consistent_wrong_class_drift`
- step16_expected_conflict_type: `consistent_wrong_class_drift`
- pattern_match: `1`

### 25023972B3

- predicted_scale_relation: `high_only_supports_predicted`
- conflict_type: `high_scale_dominant_wrong`
- step16_expected_conflict_type: `high_scale_dominant_wrong`
- pattern_match: `1`

## Failure Pattern Lists

- high_scale_dominant_wrong_slides: `["25019273B", "25023972B3", "25027723B4", "25033447B4", "25057865B1"]`
- consistent_wrong_class_drift_slides: `["2471884-B2", "2485275-B2", "2487558-B2", "25015087B2", "25015422B", "25023687B2", "25023977B2", "25001929B3"]`

High-scale override errors appear when high scale supports the predicted class while low scale does not.
Consistent wrong-class drift appears when both scales move toward the same wrong class.

## Interpretation Boundary

This is a post-hoc cross-scale evidence analysis only. It does not retrain the model, does not alter Step13/14/15/16 artifacts, and should be treated as model-behavior analysis rather than pathology validation.

## Warnings

- None

## Next Suggested Step

- Step18 cross-scale evidence graph prototype
- Step18 learnable concept-class graph prototype
