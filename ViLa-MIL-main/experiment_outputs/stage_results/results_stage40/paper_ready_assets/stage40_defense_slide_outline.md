# Step40 Defense Slide Outline

## 1. 研究背景
- 主要内容：病理切片分类中的 slide-level vision-language MIL 已取得进展，但仍缺少更稳定的 evidence path 建模。
- 应放图/表：问题定义图 + WSI 场景图
- 讲述重点：强调任务不仅是分类，更是可诊断的证据建模。

## 2. ViLa-MIL baseline 与不足
- 主要内容：回顾 ViLa-MIL 的 slide-level 对齐主线，并指出仅做 slide-level 对齐时的解释性和 failure diagnosis 局限。
- 应放图/表：ViLa-MIL baseline 示意图
- 讲述重点：不要否定 baseline，而是明确本工作的切入点。

## 3. 研究目标：从 slide-level 对齐到 diagnostic evidence modeling
- 主要内容：提出从 slide-level alignment 走向 region-concept evidence、cross-scale reasoning、evidence decomposition。
- 应放图/表：目标示意图
- 讲述重点：把问题提升为诊断级证据建模。

## 4. 方法总览
- 主要内容：介绍最终主线：ViLa-MIL alignment → Region-Concept Evidence Learning → Cross-Scale Concept Evidence Reasoning → Evidence Source Decomposition → Evidence Path Interpretability。
- 应放图/表：Step40 main figure mermaid / 方法总图
- 讲述重点：明确默认模型是 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。

## 5. Region-Concept Evidence Learning
- 主要内容：说明区域证据 token 如何与病理概念建立显式对齐。
- 应放图/表：区域-概念证据路径图
- 讲述重点：突出它与纯 bag-level 聚合的区别。

## 6. Cross-Scale Concept Evidence Reasoning
- 主要内容：说明低倍/高倍概念证据如何共同参与 final decision。
- 应放图/表：low/high evidence path 图
- 讲述重点：强调 `rq16` 与 CSG 设计是最终被实验支持的部分。

## 7. 实验设置
- 主要内容：数据划分、BiomedCLIP 特征、strict CV、评价指标和不继续训练新模块的收敛策略。
- 应放图/表：实验设置表
- 讲述重点：讲清楚结果来自完整流程，而不是后验挑选。

## 8. 主性能结果
- 主要内容：给出最终默认模型及主性能对比。
- 应放图/表：Table 1：主性能对比表
- 讲述重点：结论收敛到 skeleton。

## 9. 消融实验
- 主要内容：展示 CSG 强度、region query 数、以及最终主线的正向实验支持。
- 应放图/表：Table 2：RCE/CSG/rq 消融表
- 讲述重点：突出 `CSG a01 > a005`、`rq16 > rq8/rq32`。

## 10. Negative ablation 与为什么不继续堆 graph/gate
- 主要内容：解释 region graph、concept graph、scalar gate 为什么保留为 negative ablation。
- 应放图/表：Table 3：negative ablation 表
- 讲述重点：强调这是有价值的实验发现，而不是失败就删除。

## 11. Evidence decomposition 与 failure analysis
- 主要内容：展示 evidence source decomposition、failure typing 和 evidence export 能力。
- 应放图/表：evidence pipeline mermaid + failure type 表
- 讲述重点：把解释性分析作为核心产物而非附录点缀。

## 12. Skeleton vs low-high consistency trade-off
- 主要内容：说明 `lh_l001_m0` 的 fixed/regressed/persistent case trade-off。
- 应放图/表：Table 4：evidence calibration trade-off
- 讲述重点：强调它是 secondary trade-off variant，不是最终默认模型。

## 13. 可解释案例
- 主要内容：展示 fixed / regressed / persistent 三类案例及对应 concept / CSG pair。
- 应放图/表：Table 5：案例表
- 讲述重点：让观众看到 evidence path 的具体变化。

## 14. 结论与创新点
- 主要内容：总结 4 个最终创新点和最终主模型收束。
- 应放图/表：创新点总结页
- 讲述重点：避免夸大所有模块都有效。

## 15. 局限与未来工作
- 主要内容：说明不能声称什么，以及下一步 Prompt Reliability / Refined Prompt Pool 分支。
- 应放图/表：claims vs limitations 表
- 讲述重点：把未来工作聚焦到 prompt reliability，而不是继续堆 graph/gate。
