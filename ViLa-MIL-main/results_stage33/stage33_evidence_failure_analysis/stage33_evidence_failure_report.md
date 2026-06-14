# Step33 Evidence Failure / Conflict Analysis

## Scope
- This step does not train the model.
- This step does not modify the model.
- This step only analyzes Step32 evidence export.

## Inputs / Outputs
- Input directory: `results_stage32/stage32_rce_v4_csg_evidence_export`
- Output directory: `results_stage33/stage33_evidence_failure_analysis`
- Read files:
  - `stage32_slide_evidence_summary.csv`
  - `stage32_top_concepts_long.csv`
  - `stage32_top_csg_pairs.csv`
  - `stage32_error_cases.csv`
  - `stage32_manifest.json`
  - `stage32_evidence_export_report.md` (optional)

## Exported Slide Counts
- Slides: `194`
- Correct: `176`
- Error: `18`

## Repeated Metrics
- test AUC: `0.9757`
- test ACC: `0.9072`
- test F1: `0.8988`
- Balanced ACC: `0.9077`
- PR-AUC: `0.9548`

## Evidence Source Magnitude Comparison
- `low` mean abs margin: `0.0466`
- `high` mean abs margin: `0.0827`
- `visual` mean abs margin: `0.8055`
- `csg` mean abs margin: `0.0000`

## Visual Residual Diagnosis
- Conclusion: visual residual appears strong enough to justify an explicit gate in Step34.
- Ratios and dominance below use `visual_alpha * visual_margin`, not the raw visual logits alone.
- Mean visual source ratio: `0.8286`
- Visual supports true ratio: `0.9021`
- Visual supports predicted ratio: `0.9948`
- Visual supports wrong ratio on errors: `1.0000`
- Suggested Step34 visual gate init: `0.00 or 0.01`
- Suggested Step34 strategy: start with a scalar visual gate, then compare a slide-adaptive gate only if scalar gating helps.

## CSG Diagnosis
- Conclusion: CSG is present, but it remains much weaker than the visual and concept branches on exported evidence.
- CSG margin below is the effective contribution `csg_alpha * csg_margin`; raw branch magnitude is listed separately.
- Mean abs effective CSG margin: `0.000037`
- Max abs effective CSG margin: `0.000136`
- Mean abs raw CSG margin: `0.000377`
- Max abs raw CSG margin: `0.001402`
- Mean top CSG pair score: `0.000009`
- Max top CSG pair score: `0.000046`
- Pair export mismatch ratio: `0.0052`
- Inactive ratio: `0.0000`

## Low / High Conflict Diagnosis
- All slides conflict count: `58` / `194`
- Correct slides conflict count: `48` / `176`
- Error slides conflict count: `10` / `18`
- Error slides both-support-wrong count: `7`
- Error slides both-support-true count: `1`
- Recommendation: high-scale or concept-level margin control is worth testing because wrong-class concept agreement exists in errors.

## Prompt Confusion Diagnosis
- Error-prone concept preview:
| scale | class_name | concept_text | occurrences | error_count | error_rate_among_occurrences | reliability_hint |
| --- | --- | --- | --- | --- | --- | --- |
| high | NonAdenocarcinoma | salt-and-pepper chromatin | 3 | 3 | 1.0000 | error_prone |
| high | Adenocarcinoma | intracytoplasmic mucin vacuoles | 2 | 1 | 0.5000 | rare |
| high | Adenocarcinoma | lepidic tumor cells along alveolar septa | 2 | 1 | 0.5000 | rare |
| high | Adenocarcinoma | mucin-producing tumor cells | 3 | 1 | 0.3333 | mixed |
| high | NonAdenocarcinoma | dense eosinophilic cytoplasm | 17 | 5 | 0.2941 | mixed |
| high | NonAdenocarcinoma | inflammatory cells and granulomas | 14 | 4 | 0.2857 | mixed |
| low | NonAdenocarcinoma | solid non-gland-forming tumor nests | 32 | 8 | 0.2500 | mixed |
| high | NonAdenocarcinoma | spindle-shaped malignant cells | 28 | 7 | 0.2500 | mixed |
| low | NonAdenocarcinoma | metastatic tumor in lung | 20 | 5 | 0.2500 | mixed |
| high | Adenocarcinoma | solid sheets with nuclear atypia | 9 | 2 | 0.2222 | mixed |

## Failure Type Counts
| failure_type | count_as_primary | count_any_label | proportion_among_errors |
| --- | --- | --- | --- |
| visual_residual_override | 13 | 13 | 0.7222 |
| low_high_conflict | 4 | 10 | 0.2222 |
| concept_wrong_class_drift | 1 | 7 | 0.0556 |
| uncertain_low_margin | 0 | 14 | 0.0000 |
| high_scale_dominant_wrong | 0 | 11 | 0.0000 |
| prompt_confusion | 0 | 4 | 0.0000 |
| low_scale_dominant_wrong | 0 | 3 | 0.0000 |
| csg_misleading | 0 | 2 | 0.0000 |

## Top Error Cases
| slide_id | label | label_name | pred | pred_name | pred_margin | dominant_source | dominant_source_ratio | low_true_vs_wrong_margin | high_true_vs_wrong_margin | visual_effective_margin | csg_effective_margin | evidence_agreement | primary_failure_type | failure_labels | supporting_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2462598-B | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 0.0087 | high | 0.4956 | -0.0571 | 0.0581 | -0.0020 | 0.0000 | source_conflict | low_high_conflict | low_high_conflict|uncertain_low_margin | low and high evidence support different classes. final prediction margin is in the low-confidence region. |
| 25023972B3 | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 0.0164 | visual | 0.5103 | 0.0117 | 0.0255 | -0.0388 | 0.0000 | source_conflict | visual_residual_override | visual_residual_override|uncertain_low_margin|prompt_confusion | visual residual supports the wrong class and dominates available source margin. final prediction margin is in the low-confidence region. error-prone top concepts are present: mucin-producing tumor cells |
| 2471959-B2 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 0.2000 | high | 0.4382 | -0.0444 | 0.0724 | -0.0484 | 0.0000 | source_conflict | low_high_conflict | low_high_conflict|uncertain_low_margin | low and high evidence support different classes. final prediction margin is in the low-confidence region. |
| 25023687B2 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 0.4283 | low | 0.6906 | -0.0487 | 0.0135 | -0.0083 | 0.0000 | source_conflict | low_high_conflict | low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | low and high evidence support different classes. low-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. error-prone top concepts are present: salt-and-pepper chromatin |
| 25033447B4 | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 0.8568 | high | 0.7633 | -0.0086 | -0.0663 | -0.0120 | 0.0000 | source_conflict | concept_wrong_class_drift | concept_wrong_class_drift|high_scale_dominant_wrong|uncertain_low_margin | concept evidence mainly supports the predicted wrong class. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. |
| 2468673-B2 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 0.9738 | visual | 0.3843 | -0.0742 | 0.0578 | -0.0825 | 0.0000 | source_conflict | low_high_conflict | low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | low and high evidence support different classes. low-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. error-prone top concepts are present: salt-and-pepper chromatin |
| 25027723B4 | 1 | NonAdenocarcinoma | 0 | Adenocarcinoma | 1.2353 | visual | 0.7346 | 0.0097 | -0.0286 | -0.1064 | 0.0001 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. |
| 25023690B3 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 1.6328 | visual | 0.7141 | 0.0188 | -0.0393 | -0.1451 | -0.0000 | source_conflict | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | visual residual supports the wrong class and dominates available source margin. low and high evidence support different classes. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. error-prone top concepts are present: salt-and-pepper chromatin |
| 25021640B3 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 2.6668 | visual | 0.7918 | -0.0494 | -0.0069 | -0.2142 | -0.0000 | all_support_pred | visual_residual_override | visual_residual_override|concept_wrong_class_drift|low_scale_dominant_wrong|csg_misleading|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. concept evidence mainly supports the predicted wrong class. low-scale concept evidence is the stronger wrong-class driver. CSG margin supports the predicted wrong class with non-trivial magnitude. final prediction margin is in the low-confidence region. |
| 2487558-B2 | 0 | Adenocarcinoma | 1 | NonAdenocarcinoma | 3.2116 | visual | 0.9631 | -0.0058 | -0.0062 | -0.3138 | -0.0000 | all_support_pred | visual_residual_override | visual_residual_override|concept_wrong_class_drift|high_scale_dominant_wrong|uncertain_low_margin | visual residual supports the wrong class and dominates available source margin. concept evidence mainly supports the predicted wrong class. high-scale concept evidence is the stronger wrong-class driver. final prediction margin is in the low-confidence region. |

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
