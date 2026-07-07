# Step40 Final Claims and Limitations

## 可以安全声称
- `RCE-v4-CSG-a01-rq16` 是当前最稳主模型。
- `CSG a01` 和 `rq16` 得到实验支持。
- `region graph / concept graph / scalar gate` 是有价值的 negative ablation。
- `low-high consistency` 可减少 `low-high conflict`，但不是最终默认主模型。
- 模型支持 `evidence source decomposition` 和 `failure diagnosis`。

## 不能安全声称
- 不能说完整 `dual graph` 模块全面有效。
- 不能说 `low-high consistency` 全面超过 `skeleton`。
- 不能说 `scalar gate` 解决 `visual residual override`。
- 不能说所有指标都超过 `PEPS / ViLa-MIL`，除非有完整对比表支持。
- 不能把 explored modules 画成最终默认主模型的一部分。

## 口径建议
- 最终默认模型只写 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。
- `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0` 的定位是 secondary trade-off variant。
- 论文和答辩中应把 graph/gate 线作为 negative ablation 结果，而不是未完成的主模型路线。
