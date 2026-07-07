# Step59C Dynamic CSG Config A 5-fold formal validation

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。
3. 是否完成 config A 5-fold 训练：是。
4. config A 的 5-fold AUC / ACC / F1 / BACC / PR-AUC： 0.969323 / 0.922504 / 0.913797 / 0.913596 / 0.941877。
5. 相比 Step58C config D 的差异： AUC +0.004482, ACC -0.005187, F1 -0.005828, BACC -0.006104, PR-AUC +0.010326。
6. visual_ratio 是否仍保持较低且未回到 Step57B visual-dominant： 是，0.337585 vs Step57B 0.719580。
7. concept_ratio 是否仍保持较高： 是，0.662415。
8. csg_ratio 或 csg_logits_delta_abs_mean 是否有可观察提升： 是，csg_ratio_mean=0.001467，csg_logits_delta_abs_mean=0.000046。
9. learned_dynamic_csg_alpha 是否在多个 fold 上非零： 是，folds=[0, 1, 2, 3, 4]。
10. full / concept_only / visual_only / csg_only 的 5-fold branch 表现如何：
   full: ACC 0.922504, AUC 0.969323, F1 0.913797。
   concept_only: ACC 0.921473, AUC 0.965428, F1 0.912665。
   visual_only: ACC 0.909081, AUC 0.963852, F1 0.898929。
   csg_only: ACC 0.632108, AUC 0.738785, F1 0.560453。
11. Dynamic CSG 是否应进入新的候选主模型： 是（decision=candidate_primary）。
12. 是否应停止 Dynamic CSG 并转向 Step60： 否。

## Status

- Stage57C baseline dir: `results_stage57C_rce_v2_copy_reproduction/rce_v2_copy_csg_a01_rq16_5fold_e20_s1`
- Stage58C baseline dir: `results_stage58C_residual_constrained_configD_5fold/rce_v2_rcD_l003_t050_aux020_5fold_e20_s1`
- Stage59C candidate dir: `results_stage59C_dynamic_csg_configA_5fold/rce_v2_rcD_dynCSG_A_5fold_e20_s1`
- training_status: `completed`
- completed_folds_detected: `[0, 1, 2, 3, 4]`

## Decision Basis

- delta_acc=-0.005187
- delta_auc=+0.004482
- delta_f1=-0.005828
- delta_bacc=-0.006104
- delta_pr_auc=+0.010326
- visual_ratio_mean=0.337585 vs step57B=0.719580
- concept_ratio_mean=0.662415 vs step57B=0.280420
- csg_ratio_mean=0.001467
- csg_logits_delta_abs_mean=0.000046
- learned_alpha_mean=0.017729
- alpha_nonzero_folds=[0, 1, 2, 3, 4]
- full_minus_concept_acc=0.001031
- full_minus_concept_auc=0.003895
- collapse_folds=[]
