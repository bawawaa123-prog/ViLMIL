# Stage52 RCE Core Ablation Comparison

Reference file: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`

## Comparison Table

| variant | label | status | test_auc | test_acc | test_f1 | balanced_acc | pr_auc | delta_test_auc_vs_stage23 | delta_test_acc_vs_stage23 | delta_test_f1_vs_stage23 | delta_balanced_acc_vs_stage23 | delta_pr_auc_vs_stage23 | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage23_history | Historical Stage23 full RCE | ready | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9171426221612052 | 0.9443991016694444 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv |
| full | Step52 full RCE | ready | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9171426221612052 | 0.9443991016694444 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage52_rce_core_ablation/full_rce_v4_csg_rq16_5fold_e20_s1/result.csv |
| wo_csg | Step52 w/o CSG | ready | 0.965526151899771 | 0.9214732119010736 | 0.9127044306286952 | 0.9134360771527116 | 0.9277598115400644 | -0.004686601330320617 | -0.0010362694300520836 | -0.001751921123840372 | -0.0037065450084936558 | -0.016639290129380035 | results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1/result.csv |
| wo_concept_prior | Step52 w/o concept prior | ready | 0.9591807538007884 | 0.9018642166550932 | 0.8903255559014646 | 0.8891071469112918 | 0.9077836807363866 | -0.011031999429303219 | -0.02064526467603245 | -0.02413079585107103 | -0.028035475249913455 | -0.03661542093305781 | results_stage52_rce_core_ablation/wo_concept_prior_5fold_e20_s1/result.csv |
| wo_visual_residual | Step52 w/o visual residual | ready | 0.9645064122168328 | 0.9049249505902462 | 0.894629777500904 | 0.8960086212192439 | 0.923915292255666 | -0.005706341013258842 | -0.0175845307408794 | -0.01982657425163159 | -0.021134000941961384 | -0.020483809413778364 | results_stage52_rce_core_ablation/wo_visual_residual_5fold_e20_s1/result.csv |
| wo_logit_calibration | Step52 w/o logit calibration | ready | 0.968550025129364 | 0.9214892366860744 | 0.9110998645369368 | 0.907647002883756 | 0.9485974218751028 | -0.0016627281007276506 | -0.0010202446450512381 | -0.00335648721559878 | -0.009495619277449241 | 0.004198320205658357 | results_stage52_rce_core_ablation/wo_logit_calibration_5fold_e20_s1/result.csv |

Delta columns are computed against the historical Stage23 mean metrics.

