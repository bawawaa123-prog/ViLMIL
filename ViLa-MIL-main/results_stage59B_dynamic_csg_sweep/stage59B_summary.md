# Step59B Dynamic CSG fold0 sweep

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。
3. 本 Step 实际跑了哪些 Dynamic CSG config：['A', 'B', 'C', 'D', 'E']。
4. 哪些 config completed / pending / skipped：completed=['A', 'B', 'C', 'D', 'E']；pending=['无']；skipped=['F', 'G']。
5. 哪个 config 的 ACC/AUC/F1 最好： ACC=A， AUC=A， F1=A。
6. 哪个 config 的 csg_ratio 或 csg_logits_delta 提升最明显： csg_ratio=A， csg_logits_delta=A。
7. Dynamic CSG 是否真的产生了非零动态影响：是。
8. visual_ratio 是否仍保持在 Step58C config D 的低水平：是。
9. 推荐进入 Step59C 的 selected config 是哪个： `A`。
10. 如果没有推荐配置，是否建议停止 Dynamic CSG，转向 Step60： 否。
11. 下一步建议是什么： 进入 Step59C 5-fold。

## Selection Result

- decision: `selected_for_5fold`
- reason: `performance_close_and_csg_improved`
- rank A: score=2.114051, acc_delta=+0.015464, auc_delta=+0.010890, f1_delta=+0.018689, csg_ratio_delta=+0.001872, csg_logits_delta_abs_mean=0.00011726535263058151
- rank D: score=1.682290, acc_delta=-0.005155, auc_delta=+0.003551, f1_delta=-0.006906, csg_ratio_delta=-0.000292, csg_logits_delta_abs_mean=3.952962900669816e-05
- rank C: score=1.189512, acc_delta=-0.005155, auc_delta=+0.003906, f1_delta=-0.003589, csg_ratio_delta=-0.000330, csg_logits_delta_abs_mean=5.4426744578243595e-05
- rank B: score=1.050452, acc_delta=+0.000000, auc_delta=+0.005800, f1_delta=+0.001237, csg_ratio_delta=-0.000008, csg_logits_delta_abs_mean=5.382096658875e-05
- rank E: score=1.050452, acc_delta=+0.000000, auc_delta=+0.005800, f1_delta=+0.001237, csg_ratio_delta=-0.000008, csg_logits_delta_abs_mean=5.382096658875e-05
