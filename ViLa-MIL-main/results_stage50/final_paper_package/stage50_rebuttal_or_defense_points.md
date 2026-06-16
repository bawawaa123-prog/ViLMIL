# Stage50 Rebuttal Or Defense Points

## Q: 为什么不把 HCRC 作为主模型？
因为 Step44 的三组 HCRC-Light 5-fold 结果都没有超过基线。最接近的 `hcrc_a01_b8` 仍低于基线的 AUC 和 PR-AUC，所以它可以作为系统性探索结果，但不能被写成最终主模型。

## Q: 为什么不把 PRARC 作为主模型？
因为 Step47 的 PRARC-v1 5-fold 指标整体低于基线，而 Step48b 的 PRARC-v2 虽然工程稳定，但 gate_std 与 gate_range 仍不足以支撑“强样本自适应门控”的结论。

## Q: 为什么 negative ablation 有价值？
因为它说明最终主模型不是凭直觉保留下来的，而是在多个替代 graph/gate 方向被正式验证后，仍由 evidence path 最清晰、指标最稳的方案胜出。

## Q: 为什么最终选择 skeleton？
因为 `RCE-v4-CSG-a01-rq16 / DEG skeleton` 在 Stage39 已经是最稳默认模型，且后续 Stage44/47/48b 没有出现更强替代者。它同时保留了 region-concept evidence、CSG 和 evidence decomposition 的核心叙事。

## Q: CSG 与 HCRC 的区别是什么？
CSG 是 concept-level 的跨尺度关系建模，直接作用于 low/high concept evidence；HCRC 则更接近 spatial correspondence / child routing 路线。当前证据支持前者更稳定，后者仍受弱 high evidence 引入风险影响。

## Q: visual residual override 是否已经解决？
没有。Stage33/39/45 都表明它仍是主要错误来源，low-high consistency 只能部分缓解 conflict，PRARC 也未能稳定解决 residual override。

## Q: 当前方法的主要局限是什么？
主要局限是 residual override 仍强，HCRC 依赖较松的 spatial coverage，PRARC 门控又容易退化为近似 scalar，因此当前工作更像完成了可靠的主模型定稿与负消融收束。

## Q: 后续如何继续提升？
优先做 loss-level residual calibration、uncertainty-aware suppression、train-split-only reliability learning，以及 evidence-margin auxiliary loss，而不是继续堆叠 graph/gate 模块。
