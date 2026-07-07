# Step58B Residual-Constrained RCE parameter sweep

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。
3. 本 Step 实际跑了哪些 config：完成 ['A', 'B', 'C', 'D', 'E']；待完成 ['无']。
4. 哪些 config 已完成，哪些 pending：completed=['A', 'B', 'C', 'D', 'E']；pending=['无']。
5. 哪个 config 的 ACC/AUC/F1 最好： ACC=C， AUC=B， F1=C。
6. 哪个 config 对 visual_ratio 降低最明显： `D`。
7. 推荐进入 Step58C 的 selected config 是哪个： `D`。
8. 该 selected config 相比 Step57B baseline：
   visual_ratio 是否下降：是。
   concept_ratio 是否上升：是。
   ACC/AUC/F1 是否保持稳定：是。
9. 是否建议进入 Step58C 进行 5-fold 正式验证：是。

## Baseline

- Step57B fold0 full acc=0.907216
- Step57B fold0 visual_ratio_mean=0.719580
- Step57B fold0 concept_ratio_mean=0.280420

## Recommendation Logic

- 选择 `D`：它在性能保持、visual ratio 下降、concept ratio 上升三者之间取得了最好的综合平衡。
- 该配置 score=3.478127，visual_reduction=0.400394，concept_gain=0.400394，full_vs_concept_acc_gap=0.010309。
