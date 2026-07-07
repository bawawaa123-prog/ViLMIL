# Step45 PRARC Reliability Audit

## Purpose
- Step45 is an audit stage. It does not train a model and does not modify the existing forward path.
- The goal is to identify reliability features that separate correct vs error slides and visual-override vs non-override slides, so that Step46 can build a sample-adaptive residual gate on stable inputs.

## Inputs Read
- Loaded files: `21`
- results_stage32/stage32_rce_v4_csg_evidence_export/stage32_slide_evidence_summary.csv
- results_stage32/stage32_rce_v4_csg_evidence_export/stage32_top_concepts_long.csv
- results_stage32/stage32_rce_v4_csg_evidence_export/stage32_error_cases.csv
- results_stage32/stage32_rce_v4_csg_evidence_export/stage32_top_csg_pairs.csv
- results_stage32/stage32_rce_v4_csg_evidence_export/stage32_manifest.json
- results_stage33/stage33_evidence_failure_analysis/stage33_slide_failure_labels.csv
- results_stage33/stage33_evidence_failure_analysis/stage33_error_failure_cases.csv
- results_stage33/stage33_evidence_failure_analysis/stage33_visual_residual_diagnostics.csv
- results_stage33/stage33_evidence_failure_analysis/stage33_evidence_source_stats.csv
- results_stage33/stage33_evidence_failure_analysis/stage33_prompt_confusion_summary.csv
- results_stage33/stage33_evidence_failure_analysis/stage33_prompt_reliability_preview.csv
- results_stage33/stage33_evidence_failure_analysis/stage33_failure_type_counts.csv
- results_stage33/stage33_evidence_failure_analysis/stage33_recommendations.json
- results_stage39/final_evidence_package/stage39_evidence_calibration_summary.csv
- results_stage39/final_evidence_package/stage39_final_performance_summary.csv
- results_stage39/final_evidence_package/stage39_negative_ablation_summary.csv
- results_stage39/final_evidence_package/stage39_final_model_recommendation.json
- results_stage44/stage44_hcrc_light_summary/stage44_hcrc_5fold_summary.csv
- results_stage44/stage44_hcrc_light_summary/stage44_hcrc_vs_baseline.csv
- results_stage44/stage44_hcrc_light_summary/stage44_manifest.json
- results_stage44/stage44_hcrc_light_summary/stage44_hcrc_light_report.md

## Missing Or Warning Inputs
- No requested input file was missing.

## Missing Feature Notes
- No required Step45 feature was fully missing in the current exports.

## Slide-Level Reliability Summary
- Slides audited: `194`
- Errors: `18`
- Visual residual override flags: `13`
- Dominant visual source ratio mean: `0.8286`
- Error slides with visual supporting the wrong class: `1.0000`
- Error slides with low-high conflict: `0.5556`

Top failure types:
| failure_type | count |
| --- | --- |
| visual_residual_override | 13.0000 |
| low_high_conflict | 4.0000 |
| concept_wrong_class_drift | 1.0000 |

## Prompt Reliability Summary
- Prompt reliability is audited from existing test-export evidence only.
- Step45 must not be used to directly freeze prompt weights from this test export. Any Step46 use must recompute prompt-derived statistics inside training folds to avoid leakage.
| scale | class_name | concept_text | occurrences | error_rate | visual_override_count | reliability_label | recommended_action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high | NonAdenocarcinoma | salt-and-pepper chromatin | 3.0000 | 1.0000 | 1.0000 | error_prone | downweight |
| high | Adenocarcinoma | mucin-producing tumor cells | 3.0000 | 0.3333 | 1.0000 | error_prone | downweight |
| high | NonAdenocarcinoma | dense eosinophilic cytoplasm | 17.0000 | 0.2941 | 3.0000 | mixed | monitor |
| high | NonAdenocarcinoma | inflammatory cells and granulomas | 14.0000 | 0.2857 | 2.0000 | mixed | monitor |
| low | NonAdenocarcinoma | solid non-gland-forming tumor nests | 32.0000 | 0.2500 | 7.0000 | mixed | downweight |
| high | NonAdenocarcinoma | spindle-shaped malignant cells | 28.0000 | 0.2500 | 6.0000 | mixed | downweight |
| low | NonAdenocarcinoma | metastatic tumor in lung | 20.0000 | 0.2500 | 5.0000 | mixed | downweight |
| high | Adenocarcinoma | solid sheets with nuclear atypia | 9.0000 | 0.2222 | 2.0000 | mixed | downweight |
| low | NonAdenocarcinoma | pulmonary hamartoma | 60.0000 | 0.2000 | 8.0000 | mixed | monitor |
| high | Adenocarcinoma | glandular lumen formation | 5.0000 | 0.2000 | 1.0000 | mixed | downweight |

## Visual Residual Override Summary
- Visual residual is most useful when it agrees with concept evidence and increases confidence on already-correct slides.
- Visual residual is most misleading when it dominates source ratio, supports the wrong class, and conflicts with the concept margin sign.
- This supports an adaptive gate that sees concept margin, low-high agreement, visual-concept conflict, dominant-source ratio, and prediction margin together.
| group_family | group_value | n_slides | visual_source_ratio_mean | visual_effective_margin_mean | visual_supports_wrong_ratio | visual_concept_conflict_mean |
| --- | --- | --- | --- | --- | --- | --- |
| correctness | correct | 176.0000 | 0.8491 | 0.8550 | 0.0057 | 0.0284 |
| correctness | error | 18.0000 | 0.6286 | -0.3123 | 1.0000 | 0.2222 |
| failure_type | concept_wrong_class_drift | 1.0000 | 0.1381 | -0.0120 | 1.0000 | 0.0000 |
| failure_type | correct_prediction | 176.0000 | 0.8491 | 0.8550 | 0.0057 | 0.0284 |
| failure_type | low_high_conflict | 4.0000 | 0.2031 | -0.0353 | 1.0000 | 0.5000 |
| failure_type | visual_residual_override | 13.0000 | 0.7973 | -0.4206 | 1.0000 | 0.1538 |
| overall | all | 194.0000 | 0.8286 | 0.7467 | 0.0979 | 0.0464 |
| override | non_visual_override | 181.0000 | 0.8309 | 0.8305 | 0.0331 | 0.0387 |
| override | visual_override | 13.0000 | 0.7973 | -0.4206 | 1.0000 | 0.1538 |

## Feature Separability Ranking
| rank | feature_name | auc_error | auc_visual_override | missing_ratio | threshold_direction |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | visual_margin_abs | 0.8908 | 0.8270 | 0.0000 | lower=>error |
| 2.0000 | prediction_confidence_margin | 0.8851 | 0.8219 | 0.0000 | lower=>error |
| 3.0000 | prompt_error_risk_score | 0.8273 | 0.8079 | 0.0000 | higher=>error |
| 4.0000 | prompt_reliability_score | 0.8273 | 0.8079 | 0.0000 | lower=>error |
| 5.0000 | concept_margin_abs | 0.8280 | 0.7722 | 0.0000 | lower=>error |
| 6.0000 | dominant_source_ratio | 0.7509 | 0.6515 | 0.0000 | lower=>error |
| 7.0000 | high_margin_abs | 0.7260 | 0.6630 | 0.0000 | lower=>error |
| 8.0000 | visual_over_concept_ratio | 0.7446 | 0.6354 | 0.0000 | lower=>error |
| 9.0000 | low_margin_abs | 0.6553 | 0.7170 | 0.0000 | lower=>error |
| 10.0000 | low_high_margin_agreement | 0.7030 | 0.6188 | 0.0000 | lower=>error |

## PRARC Gate Feature Proposal
- safe_features_v1: `concept_margin_abs, low_high_margin_agreement, visual_concept_conflict, dominant_source_ratio, prediction_confidence_margin, low_high_sign_agreement`
- optional_features_v2: `visual_margin_abs, high_margin_abs, low_margin_abs, visual_over_concept_ratio, prompt_error_risk_score, prompt_reliability_score`
- exclude_features: `csg_abs_margin`

## Stage39 / Stage44 Context
- Stage39 evidence calibration summary: fixed `6`, regressed `2`, persistent `12`.
- Low-high conflicts moved from `10` to `8`, but visual override increased from `13` to `14`.
- Stage39 negative ablation summary:
  - attention-centroid region graph: semantic region token does not equal a true spatial region
  - concept prompt graph: plain feature-level prompt smoothing weakens evidence discrimination
  - scalar visual gate: visual residual cannot be safely suppressed by one global scalar
  - low-high consistency: reduces low-high conflict but introduces visual residual override trade-off
- Stage44 enter_step45 from manifest: `False`
- Stage44 reason: `All valid HCRC variants remain below baseline by the Step44 decision rules.`
