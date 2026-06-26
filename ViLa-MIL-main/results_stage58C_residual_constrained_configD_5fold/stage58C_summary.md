# Step58C Residual-Constrained Config D 5-fold formal validation

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。
3. 是否完成 config D 5-fold 训练：是。
4. config D 的 5-fold AUC / ACC / F1 / BACC / PR-AUC： 0.964842 / 0.927691 / 0.919625 / 0.919700 / 0.931551。
5. 相比 Stage57C baseline 的差异： AUC -0.005371, ACC +0.005181, F1 +0.005168, BACC +0.002558, PR-AUC -0.012848。
6. visual_ratio_mean 是否明显下降： 是，0.320575 vs Step57B 0.7196。
7. concept_ratio_mean 是否明显上升： 是，0.679425 vs Step57B 0.2804。
8. full / concept_only / visual_only 的 5-fold branch 表现如何：
   full: ACC 0.927691, AUC 0.964842, F1 0.919625。
   concept_only: ACC 0.917366, AUC 0.961629, F1 0.907764。
   visual_only: ACC 0.910117, AUC 0.962506, F1 0.900614。
9. 是否可以把 config D 作为 residual-constrained RCE 候选主模型： 是（decision=candidate_primary）。
10. 下一步建议： 进入 Step59A Dynamic CSG all-off implementation。

## Status

- Stage57C baseline dir: `results_stage57C_rce_v2_copy_reproduction/rce_v2_copy_csg_a01_rq16_5fold_e20_s1`
- Stage58C candidate dir: `results_stage58C_residual_constrained_configD_5fold/rce_v2_rcD_l003_t050_aux020_5fold_e20_s1`
- training_status: `completed`
- completed_folds_detected: `[0, 1, 2, 3, 4]`

## Decision Basis

- delta_acc=+0.005181
- delta_auc=-0.005371
- delta_f1=+0.005168
- delta_bacc=+0.002558
- delta_pr_auc=-0.012848
- visual_ratio_mean=0.320575 vs step57B=0.7196
- concept_ratio_mean=0.679425 vs step57B=0.2804
- full_minus_concept_acc=0.010325
- full_minus_concept_auc=0.003213
- collapse_folds=[]
