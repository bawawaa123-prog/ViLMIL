# Stage16 Failure-Case Evidence Narrative Summary

Step16 is a post-hoc narrative summary built from Step15 selected slides and upstream Step13d/Step14 evidence.

- It does not modify the model.
- It does not run training.
- It does not claim pathology-doctor validation; the wording below stays at the level of model evidence suggests / model appears to rely on.

## Inputs

- root: `/home/ljh/ViLMIL/ViLa-MIL-main`
- stage15_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage15_rce_evidence_visualization_fold0`
- evidence_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage13_rce_evidence_export_fold0_test_full`
- graph_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage14_concept_class_graph_fold0`
- selected_slide_count: `8`
- correct_selected_slides: `6`
- incorrect_selected_slides: `2`
- top_k_paths: `6`

## Correct vs Incorrect Summary

- Incorrect slides show mean predicted-vs-true best evidence gap `0.1449`.
- Correct slides with both scales supporting the predicted class: `6` / `6`.

## Step15 Context

Step15 summary file was available and used as upstream context.

## Failure Cases

| slide_id | error_direction | evidence_score_gap | edge_strength_gap | low_scale_relation | high_scale_relation | scale_relation |
| --- | --- | --- | --- | --- | --- | --- |
| 25001929B3 | Adenocarcinoma -> NonAdenocarcinoma | 0.2247 | 0.0716 | predicted_class_stronger | predicted_class_stronger | consistent_predicted_class_stronger |
| 25023972B3 | NonAdenocarcinoma -> Adenocarcinoma | 0.0650 | -0.0215 | mixed_or_close | predicted_class_stronger | cross_scale_conflict |

### 25001929B3

- error_direction: `Adenocarcinoma -> NonAdenocarcinoma`
- predicted_class_top_concepts: nonadeno_intercellular_bridges_high (ev=0.2374, edge=0.1075, rank=1) | nonadeno_hamartoma_low (ev=0.1376, edge=0.0988, rank=1) | nonadeno_keratinization_high (ev=0.2132, edge=0.0537, rank=2) | nonadeno_fibroinflammatory_low (ev=0.0615, edge=0.0128, rank=2) | nonadeno_dense_eosinophilic_cytoplasm_high (ev=0.1893, edge=0.0460, rank=3) | nonadeno_inflammatory_low (ev=0.0358, edge=0.0046, rank=3)
- true_class_top_concepts: adeno_mucinous (ev=0.0127, edge=0.0358, rank=1) | adeno_hobnail_high (ev=-0.0251, edge=0.0259, rank=1) | adeno_solid (ev=-0.0021, edge=-0.0150, rank=2) | adeno_prominent_nucleoli_high (ev=-0.0283, edge=0.0072, rank=2) | adeno_multifocal_glandular_low (ev=-0.0065, edge=-0.0085, rank=3) | adeno_acinar_high (ev=-0.0333, edge=0.0070, rank=3)
- low/high relation: `predicted_class_stronger` / `predicted_class_stronger`
- hypothesis: Model evidence suggests a cross-scale drift toward NonAdenocarcinoma, with both low and high scale relying on nonadeno_intercellular_bridges_high, nonadeno_hamartoma_low, nonadeno_keratinization_high more strongly than the true-class concepts adeno_mucinous, adeno_hobnail_high, adeno_solid.

### 25023972B3

- error_direction: `NonAdenocarcinoma -> Adenocarcinoma`
- predicted_class_top_concepts: adeno_papillary_high (ev=0.2102, edge=0.0860, rank=1) | adeno_mucinous (ev=0.0816, edge=0.0358, rank=1) | adeno_atypical_glands_high (ev=0.1987, edge=0.0621, rank=2) | adeno_alveolar_replacement_low (ev=0.0742, edge=0.0341, rank=2) | adeno_micropapillary_high (ev=0.1866, edge=0.0476, rank=3) | adeno_lepidic (ev=0.0729, edge=0.0139, rank=3)
- true_class_top_concepts: nonadeno_intercellular_bridges_high (ev=0.1451, edge=0.1075, rank=1) | nonadeno_hamartoma_low (ev=0.0988, edge=0.0988, rank=1) | nonadeno_dense_eosinophilic_cytoplasm_high (ev=0.1375, edge=0.0460, rank=2) | nonadeno_fibroinflammatory_low (ev=0.0241, edge=0.0128, rank=2) | nonadeno_keratinization_high (ev=0.1262, edge=0.0537, rank=3) | nonadeno_solid_nests_low (ev=0.0202, edge=0.0143, rank=3)
- low/high relation: `mixed_or_close` / `predicted_class_stronger`
- hypothesis: Model appears to rely mainly on high-scale Adenocarcinoma cues such as adeno_papillary_high, adeno_mucinous, adeno_atypical_glands_high, while low-scale evidence is weaker or conflicted against the true class.

## Representative Success Cases

| slide_id | predicted_class_name | scale_support_summary | matches_step14_top_edge | strongest_supporting_concepts |
| --- | --- | --- | --- | --- |
| 25024688B2 | Adenocarcinoma | both_scales_support_prediction | 1 | adeno_papillary_high (ev=0.2421, edge=0.0860, rank=1) | adeno_alveolar_replacement_low (ev=0.2054, edge=0.0341, rank=1) | adeno_atypical_glands_high (ev=0.2417, edge=0.0621, rank=2) | adeno_lepidic (ev=0.1984, edge=0.0139, rank=2) | adeno_micropapillary_high (ev=0.2374, edge=0.0476, rank=3) | adeno_mucinous (ev=0.1966, edge=0.0358, rank=3) |
| 2475022-B | Adenocarcinoma | both_scales_support_prediction | 1 | adeno_alveolar_replacement_low (ev=0.2181, edge=0.0341, rank=1) | adeno_atypical_glands_high (ev=0.2151, edge=0.0621, rank=1) | adeno_papillary_high (ev=0.2117, edge=0.0860, rank=2) | adeno_lepidic (ev=0.2113, edge=0.0139, rank=2) | adeno_mucinous (ev=0.2084, edge=0.0358, rank=3) | adeno_fused_glands_high (ev=0.2070, edge=0.0387, rank=3) |
| 25024310B3 | Adenocarcinoma | both_scales_support_prediction | 1 | adeno_alveolar_replacement_low (ev=0.2383, edge=0.0341, rank=1) | adeno_micropapillary_high (ev=0.1948, edge=0.0476, rank=1) | adeno_lepidic (ev=0.2337, edge=0.0139, rank=2) | adeno_atypical_glands_high (ev=0.1929, edge=0.0621, rank=2) | adeno_mucinous (ev=0.2330, edge=0.0358, rank=3) | adeno_papillary_high (ev=0.1926, edge=0.0860, rank=3) |
| 25028514B | NonAdenocarcinoma | both_scales_support_prediction | 1 | nonadeno_intercellular_bridges_high (ev=0.2047, edge=0.1075, rank=1) | nonadeno_hamartoma_low (ev=0.1239, edge=0.0988, rank=1) | nonadeno_keratinization_high (ev=0.1830, edge=0.0537, rank=2) | nonadeno_fibroinflammatory_low (ev=0.0677, edge=0.0128, rank=2) | nonadeno_dense_eosinophilic_cytoplasm_high (ev=0.1578, edge=0.0460, rank=3) | nonadeno_sarcomatoid_low (ev=0.0535, edge=-0.0017, rank=3) |

### 25024688B2

- support_summary: `both_scales_support_prediction`
- supporting_concepts: adeno_papillary_high (ev=0.2421, edge=0.0860, rank=1) | adeno_alveolar_replacement_low (ev=0.2054, edge=0.0341, rank=1) | adeno_atypical_glands_high (ev=0.2417, edge=0.0621, rank=2) | adeno_lepidic (ev=0.1984, edge=0.0139, rank=2) | adeno_micropapillary_high (ev=0.2374, edge=0.0476, rank=3) | adeno_mucinous (ev=0.1966, edge=0.0358, rank=3)
- narrative: Model evidence suggests both scales consistently support Adenocarcinoma, especially through adeno_papillary_high, adeno_alveolar_replacement_low, adeno_atypical_glands_high.

### 2475022-B

- support_summary: `both_scales_support_prediction`
- supporting_concepts: adeno_alveolar_replacement_low (ev=0.2181, edge=0.0341, rank=1) | adeno_atypical_glands_high (ev=0.2151, edge=0.0621, rank=1) | adeno_papillary_high (ev=0.2117, edge=0.0860, rank=2) | adeno_lepidic (ev=0.2113, edge=0.0139, rank=2) | adeno_mucinous (ev=0.2084, edge=0.0358, rank=3) | adeno_fused_glands_high (ev=0.2070, edge=0.0387, rank=3)
- narrative: Model evidence suggests both scales consistently support Adenocarcinoma, especially through adeno_alveolar_replacement_low, adeno_atypical_glands_high, adeno_papillary_high.

### 25024310B3

- support_summary: `both_scales_support_prediction`
- supporting_concepts: adeno_alveolar_replacement_low (ev=0.2383, edge=0.0341, rank=1) | adeno_micropapillary_high (ev=0.1948, edge=0.0476, rank=1) | adeno_lepidic (ev=0.2337, edge=0.0139, rank=2) | adeno_atypical_glands_high (ev=0.1929, edge=0.0621, rank=2) | adeno_mucinous (ev=0.2330, edge=0.0358, rank=3) | adeno_papillary_high (ev=0.1926, edge=0.0860, rank=3)
- narrative: Model evidence suggests both scales consistently support Adenocarcinoma, especially through adeno_alveolar_replacement_low, adeno_micropapillary_high, adeno_lepidic.

## Possible Misleading Or High-Coverage Concepts

- failure-case predicted concept `nonadeno_intercellular_bridges_high` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `nonadeno_hamartoma_low` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `nonadeno_keratinization_high` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `nonadeno_fibroinflammatory_low` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `nonadeno_dense_eosinophilic_cytoplasm_high` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `nonadeno_inflammatory_low` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `adeno_papillary_high` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `adeno_mucinous` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `adeno_atypical_glands_high` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `adeno_alveolar_replacement_low` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `adeno_micropapillary_high` appeared `1` times across the incorrect-slide top concepts.
- failure-case predicted concept `adeno_lepidic` appeared `1` times across the incorrect-slide top concepts.
- Step14 did not flag any concept-class edge as high-frequency-low-distinction under its saved thresholds.

## Interpretation Boundary

This step is a post-hoc narrative summary only. It does not retrain the model, does not alter Step13/14/15 artifacts, and should be treated as qualitative interpretation rather than validated pathology evidence.

## Warnings

- None

## Next Suggested Step

- Step17 learnable concept-class graph prototype
- Step17 cross-scale evidence conflict analysis
