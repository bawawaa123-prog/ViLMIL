# Step60A Concept-conditioned Region Aggregation all-off audit

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否改变 RCE-v2 默认行为：否，默认关闭路径保持不变。
3. rce_use_ccra=False 时是否创建 CCRA 子模块或 CCRA 参数： 否。
4. 如果创建了，是否证明不影响随机初始化顺序、optimizer 参数集合和 forward 输出： 是。
5. 新增了哪些 CCRA 参数：rce_use_ccra, rce_ccra_mode, rce_ccra_alpha_init, rce_ccra_scale, rce_ccra_num_queries, rce_ccra_query_source, rce_ccra_detach_prompt, rce_ccra_norm, rce_ccra_dropout, rce_ccra_clip。
6. CCRA 的计算方式是什么： 先对低/高尺度 prompt 做 prompt_mean，作为 concept query；再对 patch features 做 cross-attention，得到 concept-conditioned region；最后用 fused_region = original_region + alpha * scale * ccra_region 做 residual fusion，然后继续复用原有 region-concept evidence 代码。
7. prompt_mean 为什么只是保守初版： 它会压缩 class/prompt 维度差异，安全但表达力有限；后续 Step60B 可探索 classwise prompt、concept bank、class-specific concept query。
8. alpha 初始化是多少，以及为什么 Step60A 使用 0.0：0.0；0.0 最安全，便于 all-off / smoke 审计，但也意味着初期梯度可能偏弱，Step60B 可扫 0.001 / 0.01 / 0.05。
9. CCRA-off forward 等价检查是否通过： 是。
10. full all-off forward 等价检查是否通过： 是。
11. last_logit_breakdown 是否一致或无异常： 是。
12. CCRA smoke 是否通过： 是。
13. Step58C residual constraint 机制是否仍保留： 是。
14. Step59 Dynamic CSG 机制是否仍保留： 是。
15. 是否可以进入 Step60B CCRA 参数小扫： 是。

## Audit Notes

- forward return format preserved: `{'type': 'tuple', 'length': 3, 'y_prob_shape': [1, 2], 'y_hat_shape': [1, 1], 'loss_is_scalar': True}`
- ccra default enabled flag: `False`
- ccra params when off: `[]`
- ccra breakdown keys present: `{'ccra_enabled': True, 'ccra_mode': True, 'ccra_alpha': True, 'ccra_scale': True, 'ccra_query_source': True, 'ccra_norm': True, 'low_ccra_delta_abs_mean': True, 'high_ccra_delta_abs_mean': True, 'low_original_region_norm': True, 'high_original_region_norm': True, 'low_fused_region_norm': True, 'high_fused_region_norm': True, 'low_ccra_region_norm': True, 'high_ccra_region_norm': True, 'low_original_region_shape': True, 'high_original_region_shape': True, 'low_ccra_region_shape': True, 'high_ccra_region_shape': True, 'low_fused_region_shape': True, 'high_fused_region_shape': True, 'low_prompt_feature_shape': True, 'high_prompt_feature_shape': True, 'low_patch_feature_shape': True, 'high_patch_feature_shape': True}`
- ccra_enabled in smoke: `1.0`
- low/high ccra delta abs mean: `0.0` / `0.0`
- low/high original region shape: `[1, 16, 512]` / `[1, 16, 512]`
- low/high fused region shape: `[1, 16, 512]` / `[1, 16, 512]`
- smoke training script syntax check: `True`

## Recommendation

- Step60A 只完成 CCRA 的安全接线、默认关闭与 all-off 审计，没有启动完整 5-fold。
- 结论：可以进入 Step60B，对 CCRA 做参数小扫。
