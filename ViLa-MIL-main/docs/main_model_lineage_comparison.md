# Main Model Lineage Comparison

Reference row: Stage23 `RCE-v4-CSG-a01-rq16`.

## Comparison Table

| stage | label | kind | match_stage23 | test_auc | test_acc | test_f1 | pr_auc | val_auc | delta_test_auc | delta_test_acc | delta_test_f1 | delta_pr_auc | delta_val_auc | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage23 | RCE-v4-CSG-a01-rq16 | source_of_truth | reference | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9443991016694444 | 0.9711256597292708 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv |
| Stage27 | DEG skeleton | skeleton_replay | exact_match | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9443991016694444 | 0.9711256597292708 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage27/deg_skeleton_5fold_e20_s1/result.csv |
| Stage30 | DEG skeleton cg ablation | skeleton_replay | exact_match | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9443991016694444 | 0.9711256597292708 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1/result.csv |
| Stage35 | visual gate skeleton | skeleton_replay | exact_match | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9443991016694444 | 0.9711256597292708 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage35/visual_gate_skeleton_5fold_e20_s1/result.csv |
| Stage37 | lh consistency skeleton | skeleton_replay | exact_match | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9443991016694444 | 0.9711256597292708 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage37/lh_consistency_skeleton_5fold_e20_s1/result.csv |
| Stage22 | RCE-v4-CSG-a01 | pre_stage23_equivalent | exact_match | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9443991016694444 | 0.9711256597292708 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage22/rce_v4_csg_a01_5fold_e20_s1/result.csv |
| Stage30 | DEG concept graph k=4 | non_source_variant | different | 0.9612413699888176 | 0.9132097644356604 | 0.90274418394331 | 0.9375773806993069 | 0.9540105986264098 | -0.008971383241274 | -0.009299716895465271 | -0.011712167809225571 | -0.006821720970137557 | -0.017115061102860984 | results_stage30/deg_concept_graph_k4_a005_5fold_e20_s1/result.csv |
| Stage35 | visual gate gate001 | non_source_variant | different | 0.9651951882721326 | 0.9163399391058168 | 0.90496721119286 | 0.9334943615176888 | 0.96029687650242 | -0.00501756495795902 | -0.006169542225308855 | -0.009489140559675535 | -0.010904740151755576 | -0.010828783226850769 | results_stage35/visual_gate_gate001_5fold_e20_s1/result.csv |
| Stage37 | lh consistency l001_m0 | non_source_variant | different | 0.9680769687487756 | 0.9276854868863844 | 0.9196623747674412 | 0.9353752102077774 | 0.96589129836234 | -0.0021357844813160165 | 0.005176005555258767 | 0.005206023014905581 | -0.009023891461667044 | -0.005234361366930784 | results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1/result.csv |
| Stage44 | HCRC a01 b8 | non_source_variant | different | 0.968305798428224 | 0.9194166978259708 | 0.9109518803833692 | 0.9350723049787172 | 0.9647907340534264 | -0.001906954801867644 | -0.0030927835051548502 | -0.0035044713691664287 | -0.009326796690727224 | -0.006334925675844372 | results_stage44/stage44_hcrc_a01_b8_s1/result.csv |
| Stage47 | PRARC v1 g05 | non_source_variant | different | 0.9600075187769098 | 0.9132151060306608 | 0.9035291601668944 | 0.9329697362697352 | 0.9616708784504464 | -0.010205234453181777 | -0.009294375300464841 | -0.01092719158564115 | -0.01142936539970918 | -0.009454781278824376 | results_stage47/stage47_prarc_v1_g05_s1/result.csv |

## Match Status

- `reference`: the Stage23 source-of-truth row.
- `exact_match`: every tracked metric matches Stage23 within machine precision.
- `near_match`: all tracked metrics stay within 0.005 of Stage23.
- `different`: at least one tracked metric differs by more than 0.005.

