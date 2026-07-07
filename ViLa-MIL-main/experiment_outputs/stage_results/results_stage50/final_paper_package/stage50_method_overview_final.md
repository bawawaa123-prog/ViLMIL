# Stage50 Final Method Overview

最终主模型保持 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。

## Main Pipeline
- 输入由低倍与高倍 WSI patches 构成，并分别提取 `BiomedCLIP` patch features。
- learnable region queries 对 patch features 做区域级聚合，形成可用于诊断解释的 region evidence tokens。
- region evidence tokens 与 concept prompt pool 进行 region-concept similarity 建模，得到 low-scale 与 high-scale concept evidence logits。
- 低倍与高倍 concept evidence 进一步进入 `CSG` cross-scale concept reasoning，以保留 concept-level cross-scale support path。
- 原始视觉分支通过 visual residual 提供补充判别信息，但最终仍以 calibrated logits 汇总各来源证据。
- 预测输出同时支持 evidence decomposition，可拆成 low evidence、high evidence、CSG evidence、visual residual evidence，用于 failure diagnosis。

## Final Narrative Boundary
- 主模型叙事只包含：BiomedCLIP low/high patch features、learnable region queries、region-concept evidence、concept prompt pool、low/high concept evidence logits、CSG、visual residual、logit calibration、evidence decomposition。
- `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0` 仅作为 secondary trade-off variant 出现在 calibration / ablation 章节。
- `HCRC-Light`、`PRARC-v1`、`PRARC-v2` 只能出现在 negative ablation、limitations 或 future work 中，不能写成最终主模型组件。

## Why This Is The Final Main Path
- Stage39 明确支持 `CSG a01 > CSG a005` 与 `rq16 > rq8/rq32`。
- Stage44、Stage47、Stage48b 没有给出比主干更强的 HCRC/PRARC 替代路径。
- 因此最终论文主线应强调 evidence-guided consolidation，而不是继续扩展 graph/gate 模块。
