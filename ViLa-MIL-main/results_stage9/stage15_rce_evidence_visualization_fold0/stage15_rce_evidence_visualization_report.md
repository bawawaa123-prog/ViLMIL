# Stage15 RCE Evidence Visualization

Step15 is a visualization and interpretation layer built on top of existing Step13d evidence export and Step14 concept-class graph outputs.

- It does not modify the model.
- It does not run training.
- It does not represent pathology-doctor validation or clinical decision guidance.

## Inputs

- root: `/home/ljh/ViLMIL/ViLa-MIL-main`
- evidence_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage13_rce_evidence_export_fold0_test_full`
- graph_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage14_concept_class_graph_fold0`
- out_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage15_rce_evidence_visualization_fold0`
- selected_slide_count: `8`
- evidence_path_rows: `54`
- graph_edge_rows: `45`

## Selection Logic

- For each true label, select high-confidence correct slides first.
- If available, also include a small number of high-confidence incorrect slides for failure-case interpretation.
- Slide-level heatmaps visualize predicted-class region-concept similarity from Stage13d pickle arrays with expected shape `(2, 16, 12)`.

## Selected Slides

| slide_id | label | pred | correct | class_name_true | class_name_pred | true_prob | pred_prob | confidence_margin | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25024688B2 | 0 | 0 | 1 | Adenocarcinoma | Adenocarcinoma | 1.0000 | 1.0000 | 1.0000 | correct_high_confidence |
| 2475022-B | 0 | 0 | 1 | Adenocarcinoma | Adenocarcinoma | 1.0000 | 1.0000 | 1.0000 | correct_high_confidence |
| 25024310B3 | 0 | 0 | 1 | Adenocarcinoma | Adenocarcinoma | 1.0000 | 1.0000 | 1.0000 | correct_high_confidence |
| 25001929B3 | 0 | 1 | 0 | Adenocarcinoma | NonAdenocarcinoma | 0.0002 | 0.9998 | 0.9997 | incorrect_high_confidence |
| 25028514B | 1 | 1 | 1 | NonAdenocarcinoma | NonAdenocarcinoma | 1.0000 | 1.0000 | 1.0000 | correct_high_confidence |
| 25025087B | 1 | 1 | 1 | NonAdenocarcinoma | NonAdenocarcinoma | 1.0000 | 1.0000 | 1.0000 | correct_high_confidence |
| 25045625B1 | 1 | 1 | 1 | NonAdenocarcinoma | NonAdenocarcinoma | 1.0000 | 1.0000 | 1.0000 | correct_high_confidence |
| 25023972B3 | 1 | 0 | 0 | NonAdenocarcinoma | Adenocarcinoma | 0.0008 | 0.9992 | 0.9984 | incorrect_high_confidence |

## Overall Top Concept-Class Edges

| scale | class_name | concept_id | concept_text | edge_strength | mean_evidence_score | slide_coverage | mean_rank | n_topk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | NonAdenocarcinoma | nonadeno_intercellular_bridges_high | A high magnification histopathology image showing squamous cell carcinoma with intercellular bridges. | 0.1075 | 0.2270 | 0.6701 | 1.4154 | 130 |
| low | NonAdenocarcinoma | nonadeno_hamartoma_low | A low magnification histopathology image showing pulmonary hamartoma with cartilage and benign mesenchymal components. | 0.0988 | 0.1028 | 1.0000 | 1.0412 | 194 |
| high | Adenocarcinoma | adeno_papillary_high | A high magnification histopathology image showing adenocarcinoma cells lining fibrovascular cores. | 0.0860 | 0.1596 | 1.0000 | 1.8557 | 194 |
| high | Adenocarcinoma | adeno_atypical_glands_high | A high magnification histopathology image showing atypical gland-forming tumor cells with pseudostratified nuclei. | 0.0621 | 0.2181 | 0.6082 | 2.1356 | 118 |
| high | NonAdenocarcinoma | nonadeno_keratinization_high | A high magnification histopathology image showing squamous cell carcinoma with keratinization and keratin pearls. | 0.0537 | 0.2115 | 0.6340 | 2.4959 | 123 |
| high | Adenocarcinoma | adeno_micropapillary_high | A high magnification histopathology image showing small floating clusters of adenocarcinoma cells without fibrovascular cores. | 0.0476 | 0.2147 | 0.6598 | 2.9766 | 128 |
| high | NonAdenocarcinoma | nonadeno_dense_eosinophilic_cytoplasm_high | A high magnification histopathology image showing squamous carcinoma cells with dense eosinophilic cytoplasm and distinct cell borders. | 0.0460 | 0.1903 | 0.6856 | 2.8346 | 133 |
| high | Adenocarcinoma | adeno_fused_glands_high | A high magnification histopathology image showing fused back-to-back adenocarcinoma glands with little intervening stroma. | 0.0387 | 0.1949 | 0.7320 | 3.6831 | 142 |

## Evidence Path Preview

The CSV keeps the full `slide -> region -> concept -> class` path details. A short preview is shown below.

| slide_id | label | pred | correct | scale | class_id | concept_id | concept_text | rank | evidence_score | prompt_weight | edge_strength | prompt_id | region_id | region_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2475022-B | 0 | 0 | 1 | high | 0 | adeno_atypical_glands_high | A high magnification histopathology image showing atypical gland-forming tumor cells with pseudostratified nuclei. | 1 | 0.2151 | 0.1072 | 0.0621 | 9 | 5.0 | 0.2151 |
| 2475022-B | 0 | 0 | 1 | high | 0 | adeno_papillary_high | A high magnification histopathology image showing adenocarcinoma cells lining fibrovascular cores. | 2 | 0.2117 | 0.1043 | 0.0860 | 2 | 5.0 | 0.2117 |
| 2475022-B | 0 | 0 | 1 | high | 0 | adeno_fused_glands_high | A high magnification histopathology image showing fused back-to-back adenocarcinoma glands with little intervening stroma. | 3 | 0.2070 | 0.0996 | 0.0387 | 10 | 5.0 | 0.2070 |
| 2475022-B | 0 | 0 | 1 | high | 0 | adeno_micropapillary_high | A high magnification histopathology image showing small floating clusters of adenocarcinoma cells without fibrovascular cores. | 4 | 0.2062 | 0.0990 | 0.0476 | 3 | 5.0 | 0.2062 |
| 2475022-B | 0 | 0 | 1 | low | 0 | adeno_alveolar_replacement_low | A low magnification histopathology image showing atypical adenocarcinoma glands replacing pre-existing alveolar structures. | 1 | 0.2181 | 0.1041 | 0.0341 | 11 | 0.0 | 0.2181 |
| 2475022-B | 0 | 0 | 1 | low | 0 | adeno_lepidic | A low magnification histopathology image showing lung adenocarcinoma with lepidic growth pattern. | 2 | 0.2113 | 0.0972 | 0.0139 | 0 | 0.0 | 0.2113 |
| 2475022-B | 0 | 0 | 1 | low | 0 | adeno_mucinous | A low magnification histopathology image showing mucinous lung adenocarcinoma with extracellular mucin pools. | 3 | 0.2084 | 0.0942 | 0.0358 | 5 | 0.0 | 0.2084 |
| 2475022-B | 0 | 0 | 1 | low | 0 | adeno_cribriform_low | A low magnification histopathology image showing lung adenocarcinoma with cribriform glandular architecture. | 4 | 0.2053 | 0.0932 | 0.0181 | 10 | 0.0 | 0.2053 |
| 25001929B3 | 0 | 1 | 0 | high | 1 | nonadeno_intercellular_bridges_high | A high magnification histopathology image showing squamous cell carcinoma with intercellular bridges. | 1 | 0.2374 | 0.3083 | 0.1075 | 1 | 5.0 | 0.2374 |
| 25001929B3 | 0 | 1 | 0 | high | 1 | nonadeno_keratinization_high | A high magnification histopathology image showing squamous cell carcinoma with keratinization and keratin pearls. | 2 | 0.2132 | 0.2420 | 0.0537 | 0 | 5.0 | 0.2132 |
| 25001929B3 | 0 | 1 | 0 | high | 1 | nonadeno_dense_eosinophilic_cytoplasm_high | A high magnification histopathology image showing squamous carcinoma cells with dense eosinophilic cytoplasm and distinct cell borders. | 3 | 0.1893 | 0.1898 | 0.0460 | 6 | 5.0 | 0.1893 |
| 25001929B3 | 0 | 1 | 0 | low | 1 | nonadeno_hamartoma_low | A low magnification histopathology image showing pulmonary hamartoma with cartilage and benign mesenchymal components. | 1 | 0.1376 | 0.2299 | 0.0988 | 2 | 15.0 | 0.1376 |
| 25001929B3 | 0 | 1 | 0 | low | 1 | nonadeno_fibroinflammatory_low | A low magnification histopathology image showing a fibroinflammatory lung lesion with scarring and no malignant gland formation. | 2 | 0.0615 | 0.1077 | 0.0128 | 8 | 15.0 | 0.0615 |
| 25001929B3 | 0 | 1 | 0 | low | 1 | nonadeno_solid_nests_low | A low magnification histopathology image showing solid non-gland-forming tumor nests without adenocarcinoma-type acinar structures. | 5 | 0.0216 | 0.0712 | 0.0143 | 10 | 15.0 | 0.0216 |
| 25023972B3 | 1 | 0 | 0 | high | 0 | adeno_papillary_high | A high magnification histopathology image showing adenocarcinoma cells lining fibrovascular cores. | 1 | 0.2102 | 0.1283 | 0.0860 | 2 | 5.0 | 0.2102 |
| 25023972B3 | 1 | 0 | 0 | high | 0 | adeno_atypical_glands_high | A high magnification histopathology image showing atypical gland-forming tumor cells with pseudostratified nuclei. | 2 | 0.1987 | 0.1136 | 0.0621 | 9 | 5.0 | 0.1987 |
| 25023972B3 | 1 | 0 | 0 | high | 0 | adeno_micropapillary_high | A high magnification histopathology image showing small floating clusters of adenocarcinoma cells without fibrovascular cores. | 3 | 0.1866 | 0.1017 | 0.0476 | 3 | 5.0 | 0.1866 |
| 25023972B3 | 1 | 0 | 0 | high | 0 | adeno_fused_glands_high | A high magnification histopathology image showing fused back-to-back adenocarcinoma glands with little intervening stroma. | 4 | 0.1823 | 0.0971 | 0.0387 | 10 | 5.0 | 0.1823 |
| 25023972B3 | 1 | 0 | 0 | low | 0 | adeno_mucinous | A low magnification histopathology image showing mucinous lung adenocarcinoma with extracellular mucin pools. | 1 | 0.0816 | 0.1054 | 0.0358 | 5 | 0.0 | 0.0816 |
| 25023972B3 | 1 | 0 | 0 | low | 0 | adeno_alveolar_replacement_low | A low magnification histopathology image showing atypical adenocarcinoma glands replacing pre-existing alveolar structures. | 2 | 0.0742 | 0.0983 | 0.0341 | 11 | 0.0 | 0.0742 |
| 25023972B3 | 1 | 0 | 0 | low | 0 | adeno_lepidic | A low magnification histopathology image showing lung adenocarcinoma with lepidic growth pattern. | 3 | 0.0729 | 0.0969 | 0.0139 | 0 | 0.0 | 0.0729 |
| 25023972B3 | 1 | 0 | 0 | low | 0 | adeno_cribriform_low | A low magnification histopathology image showing lung adenocarcinoma with cribriform glandular architecture. | 4 | 0.0591 | 0.0859 | 0.0181 | 10 | 0.0 | 0.0591 |
| 25024310B3 | 0 | 0 | 1 | high | 0 | adeno_micropapillary_high | A high magnification histopathology image showing small floating clusters of adenocarcinoma cells without fibrovascular cores. | 1 | 0.1948 | 0.1071 | 0.0476 | 3 | 5.0 | 0.1948 |
| 25024310B3 | 0 | 0 | 1 | high | 0 | adeno_atypical_glands_high | A high magnification histopathology image showing atypical gland-forming tumor cells with pseudostratified nuclei. | 2 | 0.1929 | 0.1040 | 0.0621 | 9 | 5.0 | 0.1929 |

## Graph JSON Snapshot

- class_nodes: `2`
- concept_nodes: `45`
- concept_to_class_edges: `45`

## Step13 Quality Note

Stage13d quality report was available and used as upstream context for visualization readiness.

## Generated Outputs

- `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_selected_slides.csv`
- `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_slide_evidence_paths.csv`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/top_concept_class_edges_by_class.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/top_concept_class_edges_by_scale.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/selected_slide_probability_overview.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_25024688B2_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_2475022-B_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_25024310B3_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_25001929B3_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_25028514B_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_25025087B_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_25045625B1_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_25023972B3_predclass_region_concept_heatmap.png`
- `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_visualization_summary.csv`

## Warnings

- None

## Interpretation Boundary

These figures and tables are post-hoc evidence summaries intended for qualitative interpretation only. They should be treated as model-behavior visualization rather than validated biomedical evidence.
