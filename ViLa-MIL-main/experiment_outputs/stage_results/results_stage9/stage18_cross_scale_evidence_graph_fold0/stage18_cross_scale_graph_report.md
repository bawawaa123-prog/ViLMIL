# Stage18 Cross-Scale Evidence Graph Prototype

Step18 builds a post-hoc cross-scale evidence graph prototype from the existing Step13d, Step14, and Step17 outputs.

- It is not a learnable model.
- It does not modify the current model.
- It does not run training.

## Inputs

- root: `/home/ljh/ViLMIL/ViLa-MIL-main`
- evidence_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage13_rce_evidence_export_fold0_test_full`
- graph_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage14_concept_class_graph_fold0`
- conflict_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage17_cross_scale_conflict_analysis_fold0`
- fold0_slide_count: `194`

## Graph Structure

- nodes: `low_concept`, `high_concept`, `class`, `conflict_type`
- edges: `concept_to_class`, `low_high_pair`, `conflict_to_class`

## Top Low/High Concept Pairs

| low_concept_id | high_concept_id | co_occurrence_count | co_occurrence_rate | mean_joint_evidence | dominant_conflict_type | high_scale_override_rate | wrong_class_drift_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nonadeno_hamartoma_low | adeno_papillary_high | 98 | 0.5052 | 0.0018 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_hamartoma_low | adeno_atypical_glands_high | 97 | 0.5000 | 0.0016 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_hamartoma_low | adeno_fused_glands_high | 97 | 0.5000 | 0.0012 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_hamartoma_low | adeno_micropapillary_high | 96 | 0.4948 | 0.0013 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_fibroinflammatory_low | adeno_papillary_high | 96 | 0.4948 | 0.0008 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_fibroinflammatory_low | adeno_atypical_glands_high | 95 | 0.4897 | 0.0005 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_fibroinflammatory_low | adeno_fused_glands_high | 95 | 0.4897 | 0.0001 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_fibroinflammatory_low | adeno_micropapillary_high | 94 | 0.4845 | 0.0002 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_solid_nests_low | adeno_papillary_high | 85 | 0.4381 | 0.0008 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_solid_nests_low | adeno_atypical_glands_high | 85 | 0.4381 | 0.0005 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_solid_nests_low | adeno_fused_glands_high | 85 | 0.4381 | 0.0001 | high_scale_dominant_correct | 1.0000 | 0.0000 |
| nonadeno_solid_nests_low | adeno_micropapillary_high | 83 | 0.4278 | 0.0002 | high_scale_dominant_correct | 1.0000 | 0.0000 |

## High-Scale Override Pairs

| low_concept_id | high_concept_id | co_occurrence_count | incorrect_count | high_scale_override_rate | dominant_conflict_type |
| --- | --- | --- | --- | --- | --- |
| nonadeno_hamartoma_low | adeno_papillary_high | 98 | 5 | 1.0000 | high_scale_dominant_correct |
| nonadeno_fibroinflammatory_low | adeno_papillary_high | 96 | 5 | 1.0000 | high_scale_dominant_correct |
| nonadeno_hamartoma_low | adeno_atypical_glands_high | 97 | 4 | 1.0000 | high_scale_dominant_correct |
| nonadeno_hamartoma_low | adeno_fused_glands_high | 97 | 4 | 1.0000 | high_scale_dominant_correct |
| nonadeno_hamartoma_low | adeno_micropapillary_high | 96 | 4 | 1.0000 | high_scale_dominant_correct |
| nonadeno_fibroinflammatory_low | adeno_atypical_glands_high | 95 | 4 | 1.0000 | high_scale_dominant_correct |
| nonadeno_fibroinflammatory_low | adeno_fused_glands_high | 95 | 4 | 1.0000 | high_scale_dominant_correct |
| nonadeno_fibroinflammatory_low | adeno_micropapillary_high | 94 | 4 | 1.0000 | high_scale_dominant_correct |

## Consistent Wrong-Class Drift Pairs

| low_concept_id | high_concept_id | co_occurrence_count | incorrect_count | wrong_class_drift_rate | dominant_conflict_type |
| --- | --- | --- | --- | --- | --- |
| nonadeno_inflammatory_low | nonadeno_hamartoma_high | 7 | 3 | 0.2857 | consistent_correct_support |
| nonadeno_benign_reactive_low | nonadeno_hamartoma_high | 11 | 4 | 0.2727 | weak_or_mixed_conflict |
| nonadeno_fibroinflammatory_low | nonadeno_hamartoma_high | 12 | 4 | 0.2500 | weak_or_mixed_conflict |
| nonadeno_sarcomatoid_low | nonadeno_hamartoma_high | 4 | 2 | 0.2500 | consistent_correct_support |
| nonadeno_solid_nests_low | nonadeno_spindle_atypia_high | 17 | 4 | 0.2353 | weak_or_mixed_conflict |
| nonadeno_hamartoma_low | nonadeno_hamartoma_high | 13 | 4 | 0.2308 | weak_or_mixed_conflict |
| nonadeno_metastatic_low | nonadeno_spindle_atypia_high | 13 | 3 | 0.2308 | weak_or_mixed_conflict |
| nonadeno_benign_reactive_low | nonadeno_spindle_atypia_high | 13 | 3 | 0.2308 | consistent_correct_support |

## Conflict-Type To Class Edges

| conflict_type | class_name | count | incorrect_rate | main_true_label | main_pred_label |
| --- | --- | --- | --- | --- | --- |
| high_scale_dominant_correct | Adenocarcinoma | 93 | 0.0000 | Adenocarcinoma | Adenocarcinoma |
| consistent_correct_support | NonAdenocarcinoma | 48 | 0.0000 | NonAdenocarcinoma | NonAdenocarcinoma |
| weak_or_mixed_conflict | Adenocarcinoma | 25 | 0.0400 | Adenocarcinoma | Adenocarcinoma |
| high_scale_dominant_correct | NonAdenocarcinoma | 10 | 0.0000 | NonAdenocarcinoma | NonAdenocarcinoma |
| consistent_wrong_class_drift | NonAdenocarcinoma | 8 | 1.0000 | Adenocarcinoma | NonAdenocarcinoma |
| high_scale_dominant_wrong | Adenocarcinoma | 5 | 1.0000 | NonAdenocarcinoma | Adenocarcinoma |
| consistent_correct_support | Adenocarcinoma | 2 | 0.0000 | Adenocarcinoma | Adenocarcinoma |
| weak_or_mixed_conflict | NonAdenocarcinoma | 2 | 0.0000 | NonAdenocarcinoma | NonAdenocarcinoma |
| low_scale_dominant_correct | Adenocarcinoma | 1 | 0.0000 | Adenocarcinoma | Adenocarcinoma |

## Step17 Failure Pattern Interpretation

### 25001929B3

- low/high concept pair: `nonadeno_hamartoma_low <-> nonadeno_intercellular_bridges_high`
- co_occurrence_count: `81`
- dominant_conflict_type: `consistent_correct_support`
- high_scale_override_rate: `0.0000`
- wrong_class_drift_rate: `0.0988`

### 25023972B3

- low/high concept pair: `nonadeno_hamartoma_low <-> adeno_papillary_high`
- co_occurrence_count: `98`
- dominant_conflict_type: `high_scale_dominant_correct`
- high_scale_override_rate: `1.0000`
- wrong_class_drift_rate: `0.0000`

## Interpretation Boundary

Step18 is a post-hoc graph prototype only. It structures existing cross-scale evidence into graph-form CSV/JSON outputs, but it is not a learnable graph module and does not change the trained model.

## Warnings

- None

## Next Suggested Step

- Step19 learnable cross-scale graph module prototype
- Step19 learnable concept-class graph module prototype
