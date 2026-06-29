# Step61B Low-to-High Concept-guided Retrieval all-off audit

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 默认行为：否，默认关闭路径保持不变。
3. rce_use_l2h_retrieval=False 时是否创建 L2H retrieval 专属参数： 否。
4. full all-off forward 等价检查是否通过： 是。
5. Step58C config D 上的 l2h-off 等价检查是否通过： 是。
6. 开启 L2H 且 alpha_init=0.0 时，forward 输出是否仍与基线一致： 是。
7. 新增了哪些 L2H 参数：rce_use_l2h_retrieval, rce_l2h_mode, rce_l2h_low_topk, rce_l2h_high_max_per_low, rce_l2h_scale_ratio, rce_l2h_patch_footprint_ratio, rce_l2h_alpha_init, rce_l2h_scale, rce_l2h_fusion, rce_l2h_aggregate, rce_l2h_score_mode, rce_l2h_detach_low_scores, rce_l2h_min_high_matches, rce_l2h_clip。
8. L2H 的计算方式是什么： 先用 low patch 与 low prompt 的 concept similarity 做 low_prompt_max 打分并取 top-k；再按 low coords 在 high coords 中做同坐标系窗口检索；对 retrieved high patch features 做 mean aggregation；最后用 fused_high_region = original_high_region + alpha * scale * retrieved_context 做 residual fusion，后续继续复用原有 high evidence / concept evidence 路径。
9. alpha 初始化是多少，以及为什么 Step61B 使用 0.0：0.0；0.0 最安全，便于 all-off / smoke 审计，同时验证 retrieval 接线本身不引入额外 logits 扰动；Step61C 可扫 0.001 / 0.01 / 0.05。
10. 坐标检索 smoke 是否通过： 是。
11. forward smoke 是否通过： 是。
12. residual constraint 机制在 Step58C config D 主线上是否仍保留： 是。
13. 当前 L2H 实现是否安全处理 coords 缺失场景： 是。
14. 是否可以进入 Step61C L2H 参数探索： 是。

## Audit Notes

- forward return format preserved: `{'type': 'tuple', 'length': 3, 'y_prob_shape': [1, 2], 'y_hat_shape': [1, 1], 'loss_is_scalar': True}`
- l2h default enabled flag: `False`
- l2h params when off: `[]`
- l2h breakdown keys present: `{'l2h_enabled': True, 'l2h_mode': True, 'l2h_alpha': True, 'l2h_scale': True, 'l2h_score_mode': True, 'l2h_low_topk': True, 'l2h_high_max_per_low': True, 'l2h_scale_ratio': True, 'l2h_patch_footprint_ratio': True, 'low_patch_concept_scores_shape': True, 'low_patch_features_shape': True, 'high_patch_features_shape': True, 'low_coords_shape': True, 'high_coords_shape': True, 'high_region_features_shape': True, 'retrieved_high_patch_features_shape': True, 'fused_high_region_features_shape': True, 'skipped_reason': True}`
- l2h_enabled in smoke: `1.0`
- skipped_reason in smoke: `None`
- low/high coords shape in smoke: `[187, 2]` / `[2592, 2]`
- retrieved_high_patch_features_shape in smoke: `[1, 8, 16, 512]`
- fused_high_region_features_shape in smoke: `[1, 16, 512]`
- retrieved match count stats in smoke: `{'shape': [1, 8], 'sum': 16, 'max': 4, 'mean': 2.0}`
- smoke training script syntax check: `True`

## Recommendation

- Step61B 只完成 L2H Retrieval 的安全接线、默认关闭、坐标检索 smoke、forward smoke 与 all-off 审计，没有启动完整 5-fold。
- 结论：可以进入 Step61C，对 alpha_init / low_topk / high_max_per_low 做小扫。
