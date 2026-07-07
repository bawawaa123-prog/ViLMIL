# Step60D CCRA Config C secondary formal audit

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。仅复用现有 Step57B / Step60C 审计逻辑。
3. config C 5-fold 结果是否已存在且完整：是。
4. Step60D config C 的 5-fold AUC / ACC / F1 / BACC / PR-AUC： 0.968601 / 0.922493 / 0.915115 / 0.920087 / 0.937097。
5. 相比 Stage57C baseline 的差异： AUC -0.001612, ACC -0.000016, F1 +0.000659, BACC +0.002944, PR-AUC -0.007302。
6. 相比 Step58C residual-constrained baseline 的差异： AUC +0.003759, ACC -0.005197, F1 -0.004509, BACC +0.000386, PR-AUC +0.005546。
7. 相比 Step59C Dynamic CSG reference 的差异： AUC -0.000723, ACC -0.000011, F1 +0.001318, BACC +0.006490, PR-AUC -0.004779。
8. 相比 Step60C config D 的差异： AUC +0.010376, ACC +0.006202, F1 +0.008944, BACC +0.016342, PR-AUC -0.001035。
9. visual_ratio 是否仍保持低水平： 是，0.357331。
10. concept_ratio 是否仍保持高水平： 是，0.642669。
11. learned CCRA alpha 是否在多个 fold 中非零： 是，folds=[1, 2, 3, 4, 5]。
12. low/high CCRA delta 是否在多个 fold 中非零： 是，folds=[1, 2, 3, 4, 5]。
13. full / concept_only / visual_only 的 5-fold branch 表现如何：
   full: ACC 0.922493, AUC 0.968637, F1 0.915115。
   concept_only: ACC 0.925586, AUC 0.971223, F1 0.918047。
   visual_only: ACC 0.910128, AUC 0.960088, F1 0.902445。
14. Step60D 最终结论： secondary_preferred_over_config_d。
15. 下一步建议： 用 config C 作为 CCRA 更均衡的正式代表。

## Status

- Stage57C baseline dir: `results_stage57C_rce_v2_copy_reproduction/rce_v2_copy_csg_a01_rq16_5fold_e20_s1`
- Stage58C baseline dir: `results_stage58C_residual_constrained_configD_5fold/rce_v2_rcD_l003_t050_aux020_5fold_e20_s1`
- Stage59C reference dir: `results_stage59C_dynamic_csg_configA_5fold/rce_v2_rcD_dynCSG_A_5fold_e20_s1`
- Stage60C config D dir: `results_stage60C_ccra_configD_5fold/rce_v2_rcD_ccraD_5fold_e20_s1`
- Stage60D config C source dir: `results_stage60C_ccra_configD_5fold/rce_v2_rcD_ccraC_5fold_e20_s1`
- source_status: `completed`
- completed_folds_detected: `[0, 1, 2, 3, 4]`
- ccra_norm: `layernorm`
- ccra_query_source: `prompt_mean`

## Decision Basis

- delta_vs_step58c_acc=-0.005197
- delta_vs_step58c_auc=+0.003759
- delta_vs_step58c_f1=-0.004509
- delta_vs_step58c_bacc=+0.000386
- delta_vs_step58c_pr_auc=+0.005546
- delta_vs_step60c_acc=+0.006202
- delta_vs_step60c_auc=+0.010376
- delta_vs_step60c_f1=+0.008944
- delta_vs_step60c_bacc=+0.016342
- delta_vs_step60c_pr_auc=-0.001035
- visual_ratio_mean=0.357331 vs step57B=0.719580
- concept_ratio_mean=0.642669 vs step57B=0.280420
- csg_ratio_mean=0.002062
- learned_ccra_alpha_mean=0.040617
- low_ccra_delta_abs_mean=0.341818
- high_ccra_delta_abs_mean=0.573899
- alpha_nonzero_folds=[1, 2, 3, 4, 5]
- delta_nonzero_folds=[1, 2, 3, 4, 5]
- full_minus_concept_acc=-0.003093
- full_minus_concept_auc=-0.002586
- anomaly_folds=[]
- collapse_folds=[]
