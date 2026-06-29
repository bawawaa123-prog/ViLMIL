# Step62 Claims To Make And Avoid

## 一、可以写的 claims

1. Residual-constrained training shifts final evidence from visual residual toward concept evidence.
2. Step58C improves ACC / F1 / BACC over the reproduced RCE-v2 baseline while keeping AUC within an acceptable range.
3. Dynamic CSG produces nonzero sample-adaptive graph updates, but its logit-level effect is small.
4. CCRA produces stable nonzero concept-conditioned region changes and can be reported as an interpretable variant.
5. L2H retrieval is feasible at the data/coordinate level and produces stable retrieval coverage, but is not selected due to weaker hard classification metrics.

## 二、不能写或需要避免的 claims

1. 不要说 Dynamic CSG 是主要分类贡献来源。
2. 不要说 CCRA 明确超过所有 baseline。
3. 不要说 L2H retrieval 是最终有效主模型。
4. 不要只挑 ACC 或只挑 AUC，而忽略 trade-off。
5. 不要声称所有创新模块组合后形成最终统一最优模型，因为当前没有完成组合模型验证。
6. 不要把 Step61D 的 L2H 写成 positive main result，只能写成 explored but not selected。
