# Step60C CCRA Config D 5-fold formal validation

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。 仅复用了 Step60B 已兼容的 CCRA 审计逻辑。
3. config D 5-fold 是否完成：是。
4. Step60C 的 5-fold AUC / ACC / F1 / BACC / PR-AUC： 0.958225 / 0.916292 / 0.906171 / 0.903745 / 0.938132。
5. 相比 Stage57C baseline 的差异： AUC -0.011988, ACC -0.006218, F1 -0.008285, BACC -0.013398, PR-AUC -0.006267。
6. 相比 Step58C residual-constrained baseline 的差异： AUC -0.006617, ACC -0.011399, F1 -0.013454, BACC -0.015955, PR-AUC +0.006581。
7. 相比 Step59C Dynamic CSG reference 的差异： AUC -0.011099, ACC -0.006212, F1 -0.007626, BACC -0.009852, PR-AUC -0.003744。
8. visual_ratio 是否仍保持低水平： 是，0.292764。
9. concept_ratio 是否仍保持高水平： 是，0.707236。
10. learned CCRA alpha 是否在多个 fold 中非零： 是，folds=[1, 2, 3, 4, 5]。
11. low/high CCRA delta 是否在多个 fold 中非零： 是，folds=[1, 2, 3, 4, 5]。
12. full / concept_only / visual_only 的 5-fold branch 表现如何：
   full: ACC 0.916292, AUC 0.958225, F1 0.906171。
   concept_only: ACC 0.916270, AUC 0.959245, F1 0.905446。
   visual_only: ACC 0.908034, AUC 0.952747, F1 0.897748。
13. 是否可以把 CCRA config D 作为新的候选主模型： 否（decision=tradeoff_ccra）。
14. 下一步建议： 补跑 Step60C config C 的 5-fold。

## Status

- Stage57C baseline dir: `results_stage57C_rce_v2_copy_reproduction/rce_v2_copy_csg_a01_rq16_5fold_e20_s1`
- Stage58C baseline dir: `results_stage58C_residual_constrained_configD_5fold/rce_v2_rcD_l003_t050_aux020_5fold_e20_s1`
- Stage59C reference dir: `results_stage59C_dynamic_csg_configA_5fold/rce_v2_rcD_dynCSG_A_5fold_e20_s1`
- Stage60C candidate dir: `results_stage60C_ccra_configD_5fold/rce_v2_rcD_ccraD_5fold_e20_s1`
- training_status: `completed`
- completed_folds_detected: `[0, 1, 2, 3, 4]`

## Decision Basis

- delta_acc=-0.011399
- delta_auc=-0.006617
- delta_f1=-0.013454
- delta_bacc=-0.015955
- delta_pr_auc=+0.006581
- visual_ratio_mean=0.292764 vs step57B=0.719580
- concept_ratio_mean=0.707236 vs step57B=0.280420
- csg_ratio_mean=0.002577
- learned_ccra_alpha_mean=0.076677
- low_ccra_delta_abs_mean=0.492044
- high_ccra_delta_abs_mean=0.685860
- alpha_nonzero_folds=[1, 2, 3, 4, 5]
- delta_nonzero_folds=[1, 2, 3, 4, 5]
- full_minus_concept_acc=0.000021
- full_minus_concept_auc=-0.001020
- anomaly_folds=[]
- collapse_folds=[]
