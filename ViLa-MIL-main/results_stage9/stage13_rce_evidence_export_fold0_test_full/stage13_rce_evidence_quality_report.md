# Stage13 RCE Evidence Quality Check

This is a lightweight small-sample evidence quality check, not a training-result evaluation.

## Input Status

- evidence_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage13_rce_evidence_export_fold0_test_full`
- exported_slides_from_report: `194`

| check | status | row_count | slide_id_unique | missing_required_columns | correct_matches_pred_eq_label | prob_sum_close_to_one_ratio | alpha_in_range_ratio | accuracy | aligned_slide_ratio | scale_values | class_values | rank_min | rank_max | concept_text_nonempty_ratio | entry_count | has_required_keys_ratio | low_shape_summary | high_shape_summary | has_nan | has_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slide_prediction_evidence | ok | 194.0 | True |  | True | 1.0 | 1.0 | 0.9278350515463918 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| slide_top_concepts | ok | 3880.0 | NA | NA | NA | NA | NA | NA | 1.0 | high,low | 0,1 | 1.0 | 5.0 | 1.0 | NA | NA | NA | NA | NA | NA |
| region_concept_evidence | ok | NA | NA | NA | NA | NA | NA | NA | 1.0 | NA | NA | NA | NA | NA | 194.0 | 1.0 | (2, 16, 12):194 | (2, 16, 12):194 | False | False |

## Quality Summary

- prediction accuracy on exported slides: `0.9278`
- warnings_found: `0`

## Top Concept Frequency

| scale | class_id | concept_id | concept_text | count |
| --- | --- | --- | --- | --- |
| high | 0 | adeno_papillary_high | A high magnification histopathology image showing adenocarcinoma cells lining fibrovascular cores. | 194 |
| low | 0 | adeno_mucinous | A low magnification histopathology image showing mucinous lung adenocarcinoma with extracellular mucin pools. | 194 |
| low | 1 | nonadeno_hamartoma_low | A low magnification histopathology image showing pulmonary hamartoma with cartilage and benign mesenchymal components. | 194 |
| low | 1 | nonadeno_fibroinflammatory_low | A low magnification histopathology image showing a fibroinflammatory lung lesion with scarring and no malignant gland formation. | 191 |
| low | 0 | adeno_alveolar_replacement_low | A low magnification histopathology image showing atypical adenocarcinoma glands replacing pre-existing alveolar structures. | 159 |
| high | 0 | adeno_fused_glands_high | A high magnification histopathology image showing fused back-to-back adenocarcinoma glands with little intervening stroma. | 142 |
| high | 0 | adeno_hobnail_high | A high magnification histopathology image showing hobnail-like adenocarcinoma cells lining alveolar spaces. | 142 |
| low | 1 | nonadeno_solid_nests_low | A low magnification histopathology image showing solid non-gland-forming tumor nests without adenocarcinoma-type acinar structures. | 139 |
| low | 0 | adeno_lepidic | A low magnification histopathology image showing lung adenocarcinoma with lepidic growth pattern. | 134 |
| high | 1 | nonadeno_dense_eosinophilic_cytoplasm_high | A high magnification histopathology image showing squamous carcinoma cells with dense eosinophilic cytoplasm and distinct cell borders. | 133 |

## Evidence Shape Preview

| slide_id | low_region_concept_shape | high_region_concept_shape | final_logits_shape | visual_logits_shape |
| --- | --- | --- | --- | --- |
| 2460239-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2460628-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2460641-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2460960-B | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2461671-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2461677-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2461700-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2462478-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2462598-B | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |
| 2462637-B2 | (2, 16, 12) | (2, 16, 12) | (2,) | (2,) |

## Warnings

- None

## Recommendation

If file alignment, probability checks, tensor shapes, and NaN/Inf checks all look healthy, the evidence export is ready for larger fold0 export or Step14 visualization.

Step14: concept-class graph or evidence visualization.
