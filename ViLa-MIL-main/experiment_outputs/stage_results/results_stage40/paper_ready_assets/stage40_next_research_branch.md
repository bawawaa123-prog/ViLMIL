# Step40 Next Research Branch

## 推荐方向：Prompt Reliability / Refined Prompt Pool

### 为什么它比继续 graph/gate 更合理
- Step28/31/35 已经表明继续堆 region graph、concept graph、scalar gate 的边际回报很低。
- 当前 persistent / regressed cases 中仍然可以看到 prompt confusion、wrong-class concept drift、low-high conflict 等问题，这更像 prompt reliability 问题，而不是图结构容量不足。
- 因此下一轮创新更合理的方向，是提升 prompt pool 的判别性和稳定性，而不是继续增加结构模块。

### 需要分析哪些 prompt 容易出现在错误样本中
- 哪些高频概念在错误样本中持续支持 wrong class。
- 哪些 low-scale 概念与 high-scale 概念在错误样本中形成 conflict。
- 哪些 prompt 在 fixed/regressed/persistent 三类样本里表现出明显不同的可靠性。
- 哪些 prompt 易与 visual residual override 同时出现，提示其语义边界不够稳定。

### 如何构建 Concept-10 / Concept-8 refined prompt pool
- 先基于 Step38/39 的 error cases，统计错误高频 prompt 与稳定支持正确类别的 prompt。
- 删去在 wrong-class 样本中高频出现、且跨 fold 稳定性差的 prompt。
- 对语义相近但区分度不足的 prompt 做合并或改写，保留更病理学明确的概念表述。
- 优先构建 `Concept-10`，再进一步压缩成更保守的 `Concept-8`，比较其与当前 core12 prompt pool 的主性能和 failure profile。

### 如何设计下一轮 Step41 / Step42
- Step41：Prompt Reliability Audit。统计 prompt confusion、wrong-class support、error-prone prompt ranking、prompt stability across folds。
- Step42：Refined Prompt Pool Evaluation。构建 `Concept-10 / Concept-8` refined prompt pool，并复用当前最终主线进行正式比较。

### 边界说明
- 这属于下一轮研究分支，不影响当前最终模型收束。
- 当前默认模型仍然是 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。
