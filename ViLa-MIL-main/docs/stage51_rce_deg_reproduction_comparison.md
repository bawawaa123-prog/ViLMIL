# Stage51 RCE vs DEG Skeleton Reproduction Comparison

Reference file: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`

## Comparison Table

| run | label | status | test_auc | test_acc | test_f1 | balanced_acc | pr_auc | delta_test_auc_vs_stage23 | delta_test_acc_vs_stage23 | delta_test_f1_vs_stage23 | delta_balanced_acc_vs_stage23 | delta_pr_auc_vs_stage23 | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage23_history | Historical Stage23 main model | ready | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9171426221612052 | 0.9443991016694444 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv |
| stage51_rce | Step51 RCE reproduction | ready | 0.9702127532300916 | 0.9225094813311256 | 0.9144563517525356 | 0.9171426221612052 | 0.9443991016694444 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | results_stage51_repro/rce_step23_rq16_5fold_e20_s1/result.csv |
| stage51_deg | Step51 DEG skeleton reproduction | ready | 0.9562685632325132 | 0.908039100475402 | 0.8973019967856812 | 0.8961030956082151 | 0.930843152021546 | -0.013944189997578404 | -0.014470380855723608 | -0.017154354966854357 | -0.02103952655299013 | -0.013555949647898413 | results_stage51_repro/deg_skeleton_rq16_5fold_e20_s1/result.csv |

Delta columns are computed against the historical Stage23 mean metrics.

