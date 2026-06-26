# Step59A Dynamic CSG all-off implementation audit

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 默认行为：否，默认关闭路径保持不变。
3. 新增了哪些 Dynamic CSG 参数：rce_use_dynamic_csg, rce_dynamic_csg_mode, rce_dynamic_csg_alpha_init, rce_dynamic_csg_scale, rce_dynamic_csg_norm, rce_dynamic_csg_detach_evidence, rce_dynamic_csg_clip。
4. Dynamic CSG 的计算方式是什么： 先用 low/high concept evidence 构造 sample-level outer product，再按设定 norm 做稳定归一化，得到 delta_adj(sample)，最后用 dynamic_adj = static_adj + alpha * delta_adj 参与 cross-scale logits 计算。
5. alpha 初始化是多少：0.0。
6. dynamic-off 等价检查是否通过： 是。
7. full all-off 等价检查是否通过： 是。
8. 开启 Dynamic CSG 后 smoke test 是否通过： 是。
9. Step58C config D 的 residual constraint 机制是否仍然保留： 是。
10. 下一步是否建议进入 Step59B Dynamic CSG 参数小扫： 是。

## Audit Notes

- forward return format preserved: `{'type': 'tuple', 'length': 3, 'y_prob_shape': [1, 2], 'y_hat_shape': [1, 1], 'loss_is_scalar': True}`
- dynamic default enabled flag: `False`
- dynamic breakdown keys present: `{'dynamic_csg_enabled': True, 'dynamic_csg_alpha': True, 'dynamic_csg_mode': True, 'dynamic_delta_mean': True, 'dynamic_delta_std': True, 'dynamic_adj_mean': True, 'dynamic_adj_std': True, 'static_csg_logits_mean': True, 'dynamic_csg_logits_mean': True, 'csg_logits_delta_mean': True, 'csg_logits_delta_abs_mean': True}`
- dynamic_csg_enabled in smoke: `1.0`
- dynamic_delta_mean/std in smoke: `0.00694444` / `7.064e-05`
- csg_logits_delta_abs_mean in smoke: `0.0`
- smoke training script syntax check: `True`

## Recommendation

- Step59A 只完成 Dynamic CSG 的安全接线、默认关闭与轻量审计，没有启动完整 5-fold。
- 结论：可以进入 Step59B，对 Dynamic CSG 做参数小扫。
