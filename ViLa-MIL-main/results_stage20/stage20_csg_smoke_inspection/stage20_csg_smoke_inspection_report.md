# Step20 CSG Smoke Artifact Inspection

## Scope
- Step20 only performs checkpoint artifact inspection.
- Step20 does not retrain the model, does not modify model code, and does not judge final model performance.
- The inspected checkpoint comes from a smoke run with `--testing` and `1 epoch`, so adjacency values may stay very small or unstable.
- If adjacency is near zero after smoke, that is not a failure by itself; a non-testing fold0 pilot is needed to judge learnable patterns.

## Inputs
- Selected checkpoint: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage19/rce_v4_csg_smoke_s1/s_0_checkpoint.pt`
- Tried checkpoint paths: `["/home/ljh/ViLMIL/ViLa-MIL-main/results_stage19/rce_v4_csg_smoke_s1/s_0_checkpoint.pt"]`
- Step18 pair CSV: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_low_high_concept_pairs.csv`
- Step18 summary CSV: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_cross_scale_graph_summary.csv`
- Concept prompt JSON: `/home/ljh/ViLMIL/ViLa-MIL-main/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`

## Parameter Presence
- `rce_cross_scale_graph_adj` found: `True`
- `rce_cross_scale_graph_alpha` found: `True`
- adjacency key: `rce_cross_scale_graph_adj`
- alpha key: `rce_cross_scale_graph_alpha`

## Adjacency Shape And Range
- adjacency shape: `(2, 12, 12)`
- expected shape: `(2, 12, 12)`
- shape matches expected `num_classes x num_low_concepts x num_high_concepts`: `True`
- raw adjacency stats: `min=-0.002617`, `max=0.002605`, `mean=-0.001261`, `std=0.001254`, `abs_max=0.002617`, `l1=0.462218`, `l2=0.030186`
- raw adjacency finite checks: `has_nan=False`, `has_inf=False`, `nonzero_count=288`
- `tanh(adj)` stats: `min=-0.002617`, `max=0.002605`, `mean=-0.001261`, `std=0.001254`, `abs_max=0.002617`, `l1=0.462217`, `l2=0.030186`

## Alpha
- learned alpha value: `0.050335`

## Top Learned Low-High Pairs
### Class 0 - Adenocarcinoma
| rank | low_concept_id | high_concept_id | raw_weight | tanh_weight | abs_tanh_weight |
| --- | --- | --- | --- | --- | --- |
| 1 | adeno_solid | adeno_micropapillary_high | -0.0026166841853410006 | -0.0026166781317442656 | 0.0026166781317442656 |
| 2 | adeno_solid | adeno_fused_glands_high | -0.002613149816170335 | -0.0026131437625736 | 0.0026131437625736 |
| 3 | adeno_solid | adeno_solid_high | -0.002611343516036868 | -0.0026113376952707767 | 0.0026113376952707767 |
| 4 | adeno_solid | adeno_prominent_nucleoli_high | -0.0026087425649166107 | -0.0026087367441505194 | 0.0026087367441505194 |
| 5 | adeno_solid | adeno_acinar_high | -0.002604170236736536 | -0.0026041644159704447 | 0.0026041644159704447 |
| 6 | adeno_solid | adeno_papillary_high | -0.00259502069093287 | -0.0025950148701667786 | 0.0025950148701667786 |
| 7 | adeno_solid | adeno_atypical_glands_high | -0.0025905435904860497 | -0.0025905377697199583 | 0.0025905377697199583 |
| 8 | adeno_solid | adeno_columnar_cells_high | -0.0025870660319924355 | -0.002587060211226344 | 0.002587060211226344 |
| 9 | adeno_solid | adeno_hobnail_high | -0.0025853991974145174 | -0.002585393376648426 | 0.002585393376648426 |
| 10 | adeno_solid | adeno_intracytoplasmic_mucin_high | -0.0025731504429131746 | -0.002573144854977727 | 0.002573144854977727 |
| 11 | adeno_solid | adeno_lepidic_high | -0.0025501141790300608 | -0.002550108591094613 | 0.002550108591094613 |
| 12 | adeno_solid | adeno_mucinous_high | -0.0025208182632923126 | -0.0025208129081875086 | 0.0025208129081875086 |
| 13 | adeno_acinar | adeno_micropapillary_high | -0.00246748561039567 | -0.0024674804881215096 | 0.0024674804881215096 |
| 14 | adeno_acinar | adeno_acinar_high | -0.0024626031517982483 | -0.0024625982623547316 | 0.0024625982623547316 |
| 15 | adeno_acinar | adeno_papillary_high | -0.002452140673995018 | -0.0024521357845515013 | 0.0024521357845515013 |
| 16 | adeno_acinar | adeno_solid_high | -0.002451069885864854 | -0.002451064996421337 | 0.002451064996421337 |
| 17 | adeno_mixed_acinar_papillary_low | adeno_micropapillary_high | -0.002450573490932584 | -0.002450568601489067 | 0.002450568601489067 |
| 18 | adeno_mixed_acinar_papillary_low | adeno_acinar_high | -0.00244709849357605 | -0.002447093604132533 | 0.002447093604132533 |
| 19 | adeno_acinar | adeno_atypical_glands_high | -0.0024426658637821674 | -0.0024426609743386507 | 0.0024426609743386507 |
| 20 | adeno_acinar | adeno_fused_glands_high | -0.0024414551444351673 | -0.0024414502549916506 | 0.0024414502549916506 |

### Class 1 - NonAdenocarcinoma
| rank | low_concept_id | high_concept_id | raw_weight | tanh_weight | abs_tanh_weight |
| --- | --- | --- | --- | --- | --- |
| 1 | nonadeno_hamartoma_low | nonadeno_sarcomatoid_high | 0.002605026587843895 | 0.0026050207670778036 | 0.0026050207670778036 |
| 2 | nonadeno_hamartoma_low | nonadeno_dense_eosinophilic_cytoplasm_high | 0.002582971006631851 | 0.00258296518586576 | 0.00258296518586576 |
| 3 | nonadeno_hamartoma_low | nonadeno_nuclear_molding_high | 0.002580569125711918 | 0.0025805633049458265 | 0.0025805633049458265 |
| 4 | nonadeno_hamartoma_low | nonadeno_neuroendocrine_high | 0.002572218421846628 | 0.0025722128339111805 | 0.0025722128339111805 |
| 5 | nonadeno_hamartoma_low | nonadeno_inflammatory_high | 0.0025629610754549503 | 0.0025629554875195026 | 0.0025629554875195026 |
| 6 | nonadeno_hamartoma_low | nonadeno_spindle_atypia_high | 0.0025626954156905413 | 0.0025626898277550936 | 0.0025626898277550936 |
| 7 | nonadeno_hamartoma_low | nonadeno_keratinization_high | 0.002557713072746992 | 0.0025577074848115444 | 0.0025577074848115444 |
| 8 | nonadeno_hamartoma_low | nonadeno_intercellular_bridges_high | 0.0025576897896826267 | 0.002557684201747179 | 0.002557684201747179 |
| 9 | nonadeno_hamartoma_low | nonadeno_fibrosis_inflammation_high | 0.0025406840723007917 | 0.0025406787171959877 | 0.0025406787171959877 |
| 10 | nonadeno_hamartoma_low | nonadeno_chondroid_matrix_high | 0.0024984939955174923 | 0.002498488873243332 | 0.002498488873243332 |
| 11 | nonadeno_hamartoma_low | nonadeno_hamartoma_high | 0.0024651275016367435 | 0.002465122612193227 | 0.002465122612193227 |
| 12 | nonadeno_hamartoma_low | nonadeno_chondromyxoid_matrix_high | 0.0023247776553034782 | 0.0023247734643518925 | 0.0023247734643518925 |
| 13 | nonadeno_necrotic_squamous_low | nonadeno_inflammatory_high | -0.001232916722074151 | -0.001232916139997542 | 0.001232916139997542 |
| 14 | nonadeno_necrotic_squamous_low | nonadeno_spindle_atypia_high | -0.0012252244632691145 | -0.0012252238811925054 | 0.0012252238811925054 |
| 15 | nonadeno_necrotic_squamous_low | nonadeno_neuroendocrine_high | -0.0012230745051056147 | -0.0012230739230290055 | 0.0012230739230290055 |
| 16 | nonadeno_necrotic_squamous_low | nonadeno_chondroid_matrix_high | -0.0012219317723065615 | -0.0012219311902299523 | 0.0012219311902299523 |
| 17 | nonadeno_necrotic_squamous_low | nonadeno_fibrosis_inflammation_high | -0.0012137931771576405 | -0.0012137925950810313 | 0.0012137925950810313 |
| 18 | nonadeno_necrotic_squamous_low | nonadeno_intercellular_bridges_high | -0.0012040836736559868 | -0.0012040830915793777 | 0.0012040830915793777 |
| 19 | nonadeno_necrotic_squamous_low | nonadeno_keratinization_high | -0.0011822659289464355 | -0.0011822653468698263 | 0.0011822653468698263 |
| 20 | nonadeno_necrotic_squamous_low | nonadeno_nuclear_molding_high | -0.0011624762555584311 | -0.0011624757898971438 | 0.0011624757898971438 |

## Step18 Overlap
| comparison_name | class_id | overlap_count | learned_pair_count | step18_pair_count | overlap_rate_vs_learned | overlap_rate_vs_step18 |
| --- | --- | --- | --- | --- | --- | --- |
| step18_top_mean_joint_evidence | all | 3 | 40 | 20 | 0.075 | 0.15 |
| step18_top_mean_joint_evidence | 0 | 0 | 20 | 20 | 0.0 | 0.0 |
| step18_top_mean_joint_evidence | 1 | 3 | 20 | 20 | 0.15 | 0.15 |
| step18_high_scale_override | all | 0 | 40 | 20 | 0.0 | 0.0 |
| step18_high_scale_override | 0 | 0 | 20 | 20 | 0.0 | 0.0 |
| step18_high_scale_override | 1 | 0 | 20 | 20 | 0.0 | 0.0 |
| step18_wrong_class_drift | all | 2 | 40 | 20 | 0.05 | 0.1 |
| step18_wrong_class_drift | 0 | 0 | 20 | 20 | 0.0 | 0.0 |
| step18_wrong_class_drift | 1 | 2 | 20 | 20 | 0.1 | 0.1 |

- This overlap is a lightweight exact-text comparison against Step18 post-hoc pairs.
- Pay special attention to `step18_high_scale_override` and `step18_wrong_class_drift` rows; any overlap suggests the learnable CSG path is already pointing toward known post-hoc patterns.

## Conclusion
- Step20 only inspects artifacts from the existing Step19 smoke checkpoint.
- Step20 does not re-run smoke, does not train, does not run 5-fold, and does not extract features.
- Next suggested step: `Step21 non-testing fold0 pilot: adjacency exists and basic stats look normal for smoke inspection.`
