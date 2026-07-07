# Step60B CCRA fold0 sweep

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：是，且仅做最小修复范围外的兼容扩展？否。 仅扩展了 `build_stage57B_logit_contribution_audit.py` 对 CCRA 参数的重建支持，便于读取 CCRA checkpoint；未改动模型逻辑。
3. 本 Step 实际跑了哪些 CCRA config：['A', 'B', 'C', 'D', 'E', 'F']。
4. 哪些 config completed / pending / skipped：completed=['A', 'B', 'C', 'D', 'E', 'F']；pending=['无']；skipped=['G']。
5. 哪个 config 的 ACC/AUC/F1 最好： ACC=B， AUC=E， F1=C。
6. CCRA 的 learned alpha 是否出现非零：是。
7. CCRA 的 low/high delta 是否出现非零：是。
8. visual_ratio 是否仍保持低水平：是。
9. concept_ratio 是否保持较高：是。
10. 是否存在值得进入 Step60C 5-fold 的候选配置： 是。
11. 推荐进入 Step60C 的 selected config 是哪个： `D`。
12. 下一步建议是什么： 进入 Step60C 5-fold。

## Selection Result

- decision: `selected_for_step60C_5fold`
- reason: `performance_close_and_ccra_observable`
- rank D: score=8.380683, acc_delta=-0.005155, auc_delta=-0.004617, f1_delta=-0.006180, visual_ratio_delta=+0.020781, concept_ratio_delta=-0.020781, alpha=0.07902898639440536, low_delta=0.3711886154775767, high_delta=0.8804680100190885
- rank C: score=6.572922, acc_delta=+0.005155, auc_delta=+0.001184, f1_delta=+0.006701, visual_ratio_delta=-0.033302, concept_ratio_delta=+0.033302, alpha=0.04096035659313202, low_delta=0.27063357749396993, high_delta=0.4912639689660564
- rank F: score=5.960524, acc_delta=-0.020619, auc_delta=-0.012547, f1_delta=-0.022503, visual_ratio_delta=-0.061676, concept_ratio_delta=+0.061676, alpha=0.03597203642129898, low_delta=0.2522057865832731, high_delta=0.37716666616729855
- rank E: score=3.278413, acc_delta=+0.005155, auc_delta=+0.005682, f1_delta=+0.006127, visual_ratio_delta=+0.077354, concept_ratio_delta=-0.077354, alpha=0.034364886581897736, low_delta=0.0790940245911095, high_delta=0.24944734991979353
- rank B: score=1.054377, acc_delta=+0.005155, auc_delta=+0.004143, f1_delta=+0.006127, visual_ratio_delta=-0.015591, concept_ratio_delta=+0.015591, alpha=-4.45923291181316e-07, low_delta=1.0144917337467174e-09, high_delta=1.3859704666873335e-09
- rank A: score=0.643798, acc_delta=-0.010309, auc_delta=-0.008404, f1_delta=-0.010245, visual_ratio_delta=+0.012590, concept_ratio_delta=-0.012590, alpha=7.089816289607143e-09, low_delta=1.1206281626376856e-16, high_delta=1.8896010681020918e-16

## Secondary Reference

- Step59C mean ACC/AUC/F1/BACC/PR-AUC: 0.9242346666666666, 0.968126, 0.9159593333333332, 0.9168130666666666, 0.9392755999999999
