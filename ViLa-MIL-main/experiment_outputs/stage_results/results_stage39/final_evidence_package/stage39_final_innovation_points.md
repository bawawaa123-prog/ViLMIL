# Step39 Final Innovation Points

## 1. Region-Concept Evidence Learning
- 解决问题：将病理切片中的区域级视觉线索与概念级文本证据对齐，避免仅靠 bag-level 注意力给出黑盒预测。
- 相比 ViLa-MIL 的区别：不只做提示词增强的 MIL 聚合，而是显式组织 region-to-concept 的证据路径，并保留 visual residual / concept support 的分解结果。
- 对应实验/分析支撑：Step24 显示 `CSG a01 > a005`，且 `RCE-v4-CSG-a01-rq16` 成为当前最稳主干。
- 角色定位：最终主模型模块。

## 2. Cross-Scale Concept Evidence Reasoning
- 解决问题：低倍与高倍概念证据经常互补，单尺度证据不足以解释病理类别判断。
- 相比 ViLa-MIL 的区别：不仅保留双尺度输入，还把低倍/高倍概念支持关系纳入统一的证据推理视角，而不是只在最终分类层做简单融合。
- 对应实验/分析支撑：Step24 中 `rq16 > rq8/rq32`，Step38 中 low/high conflict 的变化证明跨尺度证据关系是可分析、可校准的。
- 角色定位：最终主模型模块。

## 3. Evidence Source Decomposition and Failure Diagnosis
- 解决问题：需要知道错误来自 low-scale concept、high-scale concept、visual residual 还是 cross-scale pair，而不是只记录分类是否正确。
- 相比 ViLa-MIL 的区别：增加 evidence source decomposition、failure typing、conflict/override 诊断链路，使解释性分析可量化、可复现。
- 对应实验/分析支撑：Step33/38 输出了 `low_high_conflict`、`visual_residual_override`、`concept_wrong_class_drift` 等失败类型与案例表。
- 角色定位：diagnostic analysis，不改变最终主模型主体逻辑。

## 4. Evidence Calibration Analysis
- 解决问题：在不继续堆新模块的前提下，评估 evidence regularization 是否真正改善证据一致性，还是只是换来新的偏差。
- 相比 ViLa-MIL 的区别：不仅比较主指标，还比较 fixed/regressed/persistent cases、low-high conflict 与 visual residual override 的结构性变化。
- 对应实验/分析支撑：Step37/38 表明 `lh_l001_m0` 可减少 low-high conflict，但带来 visual residual override trade-off。
- 角色定位：secondary trade-off variant + diagnostic ablation，不作为最终默认主模型。
