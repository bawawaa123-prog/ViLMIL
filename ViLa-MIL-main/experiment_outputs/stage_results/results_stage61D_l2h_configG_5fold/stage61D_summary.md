# Step61D Low-to-High Retrieval config G 5-fold formal validation

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否
3. preflight 等价检查是否通过：是。
4. config G 5-fold 是否完成：是。
5. Step61D 的 5-fold AUC / ACC / F1 / BACC / PR-AUC：AUC=0.967921 ACC=0.921479 F1=0.911256 BACC=0.904859 PR_AUC=0.938282。
6. 相比 Stage57C baseline 的差异：AUC=-0.002292 ACC=-0.001031 F1=-0.003200 BACC=-0.012283 PR_AUC=-0.006117。
7. 相比 Step58C residual-constrained baseline 的差异：AUC=+0.003079 ACC=-0.006212 F1=-0.008368 BACC=-0.014841 PR_AUC=+0.006731。
8. 相比 Step59C Dynamic CSG reference 的差异：AUC=-0.001403 ACC=-0.001025 F1=-0.002541 BACC=-0.008737 PR_AUC=-0.003595。
9. 相比 Step60D CCRA config C 的差异：AUC=-0.000680 ACC=-0.001015 F1=-0.003859 BACC=-0.015227 PR_AUC=+0.001185。
10. visual_ratio 是否仍保持低水平：是。
11. concept_ratio 是否仍保持高水平：是。
12. learned L2H alpha 是否在多个 fold 中非零：是。
13. l2h_delta_abs_mean 是否在多个 fold 中非零：否。
14. retrieved_high_match_counts_mean 是否合理：是。
15. zero-match 比例是否低：是。
16. full / concept_only / visual_only 的 5-fold branch 表现如何：full(AUC=0.967921, ACC=0.921479) concept_only(AUC=0.965738, ACC=0.919411) visual_only(AUC=0.965463, ACC=0.914262)。
17. 是否可以把 L2H config G 作为新的候选主模型：否。
18. 如果不适合作为主模型，是否保留为 L2H retrieval 解释性变体：否。
19. 下一步建议：stop_l2h_and_enter_final_consolidation。

## Notes

- preflight status: `passed`
- preflight checks: `{'original_rce_unchanged': True, 'no_l2h_params_when_off': True, 'l2h_off_equivalence': True, 'full_all_off_equivalence': True, 'config_g_buildable': True}`
- config G build check: `{'ok': True, 'reason': None, 'missing_keys': ['rce_l2h_alpha'], 'unexpected_keys': [], 'l2h_enabled': True, 'learned_alpha': 0.009999999776482582, 'low_topk': 16, 'high_max_per_low': 32, 'patch_footprint_ratio': 6.0}`
- run_status: `completed`
- decision: `not_selected`
- visual_ratio_mean: `0.281393`
- concept_ratio_mean: `0.718607`
- csg_ratio_mean: `0.001207`
- learned_l2h_alpha_mean: `0.011931`
- l2h_delta_abs_mean: `None`
- retrieved_high_match_counts_mean: `27.823063`
- retrieved_high_zero_match_percent: `6.4e-05`
- Step61C carried minimal model fix into Step61D: 将 L2H patch extent 估计从坐标差分 median 调整为 p95，以更贴近 Step61A 的 patch footprint 审计；原因是原实现会把检索窗口压缩成 stride 级别，导致 retrieved match count 系统性偏低。
