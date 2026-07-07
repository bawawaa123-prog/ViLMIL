# Step40 Paper Section Draft

## 摘要草稿
本文围绕病理全切片分类中的 vision-language MIL 展开研究，重点不是继续堆叠新的图模块或门控模块，而是在 ViLa-MIL 基础上构建可诊断的证据建模框架。我们提出 Region-Concept Evidence Learning 与 Cross-Scale Concept Evidence Reasoning，使模型能够在 slide-level 分类之外显式组织区域证据、病理概念证据以及跨尺度概念关系。同时，我们构建了 Evidence Source Decomposition 与 failure diagnosis 分析链路，用于识别 low-scale/high-scale concept conflict、visual residual override 等关键错误模式。实验表明，`RCE-v4-CSG-a01-rq16 / DEG skeleton` 是当前最稳的默认模型；`CSG a01` 与 `rq16` 得到明确实验支持，而 Spatial Region Graph、Concept Prompt Graph 与 Scalar Visual Gate 未超过 skeleton。进一步的 evidence calibration 分析显示，low-high consistency regularization 能减少 low-high conflict，但会引入 visual residual override trade-off，因此更适合作为 secondary variant 而非最终默认模型。

## 引言贡献点
- 提出 Region-Concept Evidence Learning，使区域级视觉证据与病理概念证据形成显式对应关系。
- 提出 Cross-Scale Concept Evidence Reasoning，把低倍/高倍概念证据纳入统一证据推理视角。
- 构建 Evidence Source Decomposition and Failure Diagnosis，使错误分析可以定位到 concept conflict、visual override 等具体来源。
- 给出 Evidence Calibration Analysis，证明 low-high consistency 更像 trade-off variant，而不是无条件更优的新主模型。

## 方法章节结构
1. ViLa-MIL baseline 与问题定义。
2. Region-Concept Evidence Learning。
3. Cross-Scale Concept Evidence Reasoning。
4. Evidence Source Decomposition。
5. Evidence Export / Failure Diagnosis。

## 实验章节结构
1. 数据集、strict CV 设置与评价指标。
2. 主性能比较。
3. RCE/CSG/rq 消融实验。
4. Negative ablation：region graph、concept graph、scalar gate。
5. Evidence calibration：skeleton vs low-high consistency。
6. 可解释案例与 failure analysis。

## 消融实验描述
消融实验表明，最终主模型的有效性主要来自 `CSG a01` 与 `rq16` 的选择，而不是来自额外 graph/gate 模块。具体而言，Stage24 显示 `CSG a01 > CSG a005`，且 `rq16 > rq8/rq32`，因此最终 RCE 主干收敛为 `RCE-v4-CSG-a01-rq16`。相比之下，Stage28 的 Spatial Region Graph、Stage31 的 Concept Prompt Graph 以及 Stage35 的 Scalar Visual Gate 均未超过 `DEG skeleton`。这些结果说明，语义 region token 不能简单等价为真实 spatial region，普通 feature-level prompt smoothing 会削弱 evidence discrimination，而 visual residual 也不能被一个全局 scalar gate 稳定替代。

## 解释性分析描述
在 Step38 的 evidence calibration 分析中，`lh_l001_m0` 相比 skeleton 带来了 `6` 个 fixed cases、`2` 个 regressed cases 和 `12` 个 persistent errors；同时 `low_high_conflict` 从 `10` 降至 `8`，`both_support_wrong` 从 `7` 降至 `5`。然而，`visual_residual_override` 从 `13` 上升到 `14`，并伴随 `AUC=-0.0039`、`PR-AUC=-0.0034` 的轻微下降。因此，我们将 low-high consistency 视为 evidence calibration trade-off，而非最终默认主模型。

## 局限性
- 本工作不能证明所有 graph/gate 模块都有效，反而显示其中若干设计更适合作为 negative ablation。
- low-high consistency 不能被表述为全面超过 skeleton，因为它同时带来了 visual residual override 的代价。
- 当前 example cases 的 top concept / top CSG pair 展示主要来自已有汇总字段，后续若需要更细粒度图示，仍可补充独立可视化素材。

## 结论
综合所有阶段结果，当前最稳默认模型为 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。`RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0` 作为 secondary trade-off variant 具有一定 evidence calibration 价值，但不替代默认主模型。最终论文叙事应聚焦于 region-concept evidence、cross-scale concept reasoning、evidence source decomposition 以及 failure diagnosis / calibration analysis，而不应再扩展为更多 graph/gate 模块的堆叠。
