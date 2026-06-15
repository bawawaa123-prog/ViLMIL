# Step40 Method Overview

最终方法主线：

`ViLa-MIL slide-level vision-language alignment`
→ `Region-Concept Evidence Learning`
→ `Cross-Scale Concept Evidence Reasoning`
→ `Evidence Source Decomposition`
→ `Evidence Path Interpretability`

## 最终默认模型
- `RCE-v4-CSG-a01-rq16 / DEG skeleton`

## 方法主线说明
- 基线仍然是 ViLa-MIL 的 slide-level vision-language alignment，但最终工作重点已经从单纯的 slide-level 对齐推进到 diagnostic evidence modeling。
- `Region-Concept Evidence Learning` 对应区域证据 token 与病理概念的显式对齐，使模型不仅给出分类，还给出 region-to-concept 的证据路径。
- `Cross-Scale Concept Evidence Reasoning` 保留低倍与高倍两条概念证据链，并通过 cross-scale concept reasoning 解释它们如何共同支持最终判断。
- `Evidence Source Decomposition` 将 final logits 拆分为 low-scale concept、high-scale concept、CSG cross-scale pair、visual residual 等来源，支撑 failure diagnosis。
- `Evidence Path Interpretability` 使 evidence export、failure typing、fixed/regressed/persistent case analysis 能成为最终论文叙事的一部分，而不是训练后的附加观察。

## 结论边界
- `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0` 是 secondary trade-off variant，不是最终默认模型。
- Step40 只整理最终论文/报告/答辩资产，不引入新训练、不修改模型主体逻辑、不新增新模块。
