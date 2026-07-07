# Step33 Evidence Failure / Conflict Analysis - lh_l001_m0

## Scope
- This step does not train the model.
- This step does not modify the model.
- This step only analyzes Step32 evidence export.

## Inputs / Outputs
- Input directory: `results_stage38/evidence_export_lh_l001_m0_fold0_test`
- Output directory: `results_stage38/failure_analysis_lh_l001_m0_fold0_test`
- Variant name: `lh_l001_m0`
- Read files:
  - `stage32_slide_evidence_summary.csv`
  - `stage32_top_concepts_long.csv`
  - `stage32_top_csg_pairs.csv`
  - `stage32_error_cases.csv`
  - `stage32_manifest.json`
  - `stage32_evidence_export_report.md` (optional)

## Exported Slide Counts
- Slides: `194`
- Correct: `180`
- Error: `14`

## Repeated Metrics
- test AUC: `0.9718`
- test ACC: `0.9278`
- test F1: `0.9196`
- Balanced ACC: `0.9196`
- PR-AUC: `0.9514`

## Evidence Source Magnitude Comparison
- `low` mean abs margin: `0.0524`
- `high` mean abs margin: `0.1043`
- `visual` mean abs margin: `0.7093`
- `csg` mean abs margin: `0.0000`

## Visual Residual Diagnosis
- Conclusion: visual residual appears strong enough to justify an explicit gate in Step34.
- Ratios and dominance below use `visual_alpha * visual_margin`, not the raw visual logits alone.
- Mean visual source ratio: `0.7939`
- Visual supports true ratio: `0.9227`
- Visual supports predicted ratio: `0.9948`
- Visual supports wrong ratio on errors: `1.0000`
- Suggested Step34 visual gate init: `0.00 or 0.01`
- Suggested Step34 strategy: start with a scalar visual gate, then compare a slide-adaptive gate only if scalar gating helps.

## CSG Diagnosis
- Conclusion: CSG is present, but it remains much weaker than the visual and concept branches on exported evidence.
- CSG margin below is the effective contribution `csg_alpha * csg_margin`; raw branch magnitude is listed separately.
- Mean abs effective CSG margin: `0.000019`
- Max abs effective CSG margin: `0.000080`
- Mean abs raw CSG margin: `0.000205`
- Max abs raw CSG margin: `0.000842`
- Mean top CSG pair score: `0.000008`
- Max top CSG pair score: `0.000025`
- Pair export mismatch ratio: `0.0103`
- Inactive ratio: `0.0103`

## Low / High Conflict Diagnosis
- All slides conflict count: `49` / `194`
- Correct slides conflict count: `41` / `180`
- Error slides conflict count: `8` / `14`
- Error slides both-support-wrong count: `5`
- Error slides both-support-true count: `1`
- Recommendation: high-scale or concept-level margin control is worth testing because wrong-class concept agreement exists in errors.

## Prompt Confusion Diagnosis
- Error-prone concept preview:
| scale | class_name | concept_text | occurrences | error_count | error_rate_among_occurrences | reliability_hint |
| --- | --- | --- | --- | --- | --- | --- |
| high | Adenocarcinoma | columnar tumor cells with nucleoli | 5 | 2 | 0.4000 | mixed |
| high | Adenocarcinoma | solid sheets with nuclear atypia | 9 | 3 | 0.3333 | mixed |
| low | Adenocarcinoma | acinar growth pattern | 9 | 3 | 0.3333 | mixed |
| low | Adenocarcinoma | mixed acinar papillary architecture | 9 | 3 | 0.3333 | mixed |
| high | Adenocarcinoma | glandular lumen formation | 3 | 1 | 0.3333 | mixed |
| low | Adenocarcinoma | papillary architecture | 9 | 2 | 0.2222 | mixed |
| low | NonAdenocarcinoma | pulmonary hamartoma | 43 | 7 | 0.1628 | mixed |
| low | NonAdenocarcinoma | metastatic tumor in lung | 31 | 5 | 0.1613 | mixed |
| low | NonAdenocarcinoma | benign reactive lung tissue | 45 | 6 | 0.1333 | mixed |
| high | NonAdenocarcinoma | dense eosinophilic cytoplasm | 30 | 4 | 0.1333 | mixed |

## Failure Type Counts
| failure_type | count_as_primary | count_any_label | proportion_among_errors |
| --- | --- | --- | --- |
| visual_residual_override | 14 | 14 | 1.0000 |
| uncertain_low_margin | 0 | 11 | 0.0000 |
| high_scale_dominant_wrong | 0 | 9 | 0.0000 |
| low_high_conflict | 0 | 8 | 0.0000 |
| concept_wrong_class_drift | 0 | 5 | 0.0000 |
| prompt_confusion | 0 | 5 | 0.0000 |
| low_scale_dominant_wrong | 0 | 2 | 0.0000 |
| csg_inactive_or_zero | 0 | 1 | 0.0000 |

## Top Error Cases
| slide_id | label | label_name | pred | pred_name | pred_margin | dominant_source | dominant_source_ratio | low_true_vs_wrong_margin | high_true_vs_wrong_margin | visual_effective_margin | csg_effective_margin | evidence_agreement | primary_failure_type | failure_labels | supporting_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25023961B2 | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 0.8033 | visual | 0.8487 | -0.0090 | 0.0047 | -0.0770 | 0.0000 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. low-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. error-prone top concepts are present: solid sheets with nuclear atypia | columnar tumor cells with nucleoli |
| 25015422B | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 2.7611 | visual | 0.8206 | 0.0472 | -0.0199 | -0.3073 | -0.0000 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. final prediction margin is in the low-confidence region. |
| 25023687B2 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 3.1790 | visual | 0.9547 | 0.0129 | 0.0031 | -0.3384 | 0.0000 | source_conflict | visual_residual_override | visual_residual_override|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. final prediction margin is in the low-confidence region. |
| 25057865B1 | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 3.2713 | visual | 0.6126 | 0.0626 | -0.1143 | -0.2797 | 0.0000 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. error-prone top concepts are present: mixed acinar papillary architecture | acinar growth pattern |
| 25023977B2 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 3.4149 | visual | 0.6672 | 0.0375 | -0.1027 | -0.2812 | 0.0000 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. |
| 25033447B4 | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 3.5088 | visual | 0.5814 | -0.0072 | -0.1416 | -0.2066 | -0.0000 | all_support_pred | visual_residual_override | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong|csg_inactive_or_zero|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. concept evidence mainly supports the predicted wrong class. high-scale concept evidence is the stronger wrong-class driver. CSG appears inactive or very weak on this slide. final prediction margin is in the low-confidence region. |
| 25027723B4 | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 3.6501 | visual | 0.8989 | 0.0094 | -0.0299 | -0.3493 | 0.0001 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. error-prone top concepts are present: acinar growth pattern | mixed acinar papillary architecture | glandular lumen formation | columnar tumor cells with nucleoli |
| 25015087B2 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 4.7069 | visual | 0.7235 | 0.0571 | -0.1064 | -0.4279 | -0.0000 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. |
| 25025076B | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 5.3884 | visual | 0.7722 | 0.0177 | -0.1147 | -0.4489 | -0.0000 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. |
| 25019273B | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 5.5082 | visual | 0.8181 | 0.0713 | -0.0561 | -0.5733 | 0.0001 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|uncertain_low_margin|prompt_confusion | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. final prediction margin is in the low-confidence region. error-prone top concepts are present: acinar growth pattern | mixed acinar papillary architecture | solid sheets with nuclear atypia |

## Step34 Recommendation
- Evidence-level gated fusion: yes
- Start with: start with a scalar visual gate, then compare a slide-adaptive gate only if scalar gating helps.
- Suggested ablations:
  - scalar visual gate init `0.00` vs `0.01` vs current-equivalent `0.05`
  - keep / remove CSG residual path while visual gate is active
  - scalar-only gate vs slide-adaptive gate if scalar gating helps

## Step35 / Step36 Recommendation
- Evidence consistency / margin loss: yes, if Step34 shows that source conflict remains a major error pattern.
- Most worth constraining first: the high-scale concept branch when it dominates wrong-class errors, then low-high consistency, then CSG only after export fidelity is clarified.

## Warnings
- none
