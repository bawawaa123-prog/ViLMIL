# Step58A Residual-Constrained RCE all-off equivalence implementation

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 默认行为：否。
3. 新增了哪些 residual constraint 参数： `--rce_use_residual_constraint`、`--rce_residual_constraint_lambda`、 `--rce_residual_ratio_target`、`--rce_residual_constraint_type`、 `--rce_use_concept_aux_loss`、`--rce_concept_aux_loss_weight`、 `--rce_residual_ratio_eps`、`--rce_residual_ratio_detach`。
4. all-off 时是否保持等价：是。
5. residual constraint loss 的计算方式是什么： `visual_ratio = ||visual_residual_logits|| / (||concept_logits|| + ||visual_residual_logits|| + eps)`； `residual_constraint_loss = mean(ReLU(visual_ratio - target)^2)`。
6. concept auxiliary loss 的计算方式是什么： 对 `concept_logits = low_evidence_logits + high_evidence_logits + csg_logits` 单独计算 `CE(concept_logits, label)`，并按权重加到总 loss。
7. 是否完成 py_compile / bash -n：是。
8. 是否完成 smoke test：是。
9. 下一步是否建议进入 Step58B 参数小扫：是，可以进入。

## Checks

- 原始 RCE 文件改动检查：`False`
- 新参数在 `main.py` 中存在：`True`
- 新参数在 `utils/core_utils.py` 中透传：`True`
- RCE-v2 支持 residual constraint 属性：`True`
- all-off 等价通过：`True`
- smoke 通过：`True`
