# Step62 论文可用最终结果整理

## 1. 当前最终模型选择结论

最终主模型选择为 `step58C_residual_constrained_primary`，对应结果目录 `results_stage58C_residual_constrained_configD_5fold`。该模型来自 Step58C residual-constrained RCE config D，是当前所有已完成正式 5-fold 创新模型中最适合作为论文主结果的模型。

## 2. 五个正式模型的结果表

| model_id | model_role | decision | AUC | ACC | F1 | BACC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stage57C_rce_v2_baseline | baseline_reference | reproduced_baseline | 0.970213 | 0.922509 | 0.914456 | 0.917143 | 0.944399 |
| step58C_residual_constrained_primary | final_primary_model | candidate_primary | 0.964842 | 0.927691 | 0.919625 | 0.919700 | 0.931551 |
| step59C_dynamic_csg_variant | secondary_variant | candidate_primary | 0.969323 | 0.922504 | 0.913797 | 0.913596 | 0.941877 |
| step60D_ccra_variant | secondary_variant | secondary_preferred_over_config_d | 0.968601 | 0.922493 | 0.915115 | 0.920087 | 0.937097 |
| step61D_l2h_not_selected | not_selected | not_selected | 0.967921 | 0.921479 | 0.911256 | 0.904859 | 0.938282 |

## 3. 为什么 Step58C 是主模型

Step58C 的 5-fold 指标为 AUC=0.964842、ACC=0.927691、F1=0.919625、BACC=0.919700、PR_AUC=0.931551。相对 Stage57C baseline，它提升了 ACC/F1/BACC，同时 visual_ratio 降到 0.320575、concept_ratio 升到 0.679425，且 full 分支仍强于 concept_only（ACC 差值 +0.010325，AUC 差值 +0.003213）。

## 4. 为什么 Step59C / Step60D 是变体

Step59C Dynamic CSG 的 AUC / PR-AUC 相对 Step58C 更高，但 ACC / F1 / BACC 更低；同时 dynamic alpha 多 fold 非零，但 csg logit delta 很小，因此更适合作为“有可观测 sample-adaptive 更新，但主分类贡献有限”的变体。

Step60D CCRA 的整体性能接近主线模型，并且 learned alpha、low delta、high delta 都稳定非零，说明 concept-conditioned region aggregation 机制是成立的；但 full branch 略弱于 concept_only，因此更适合作为可解释 balanced representative，而不是主模型。

## 5. 为什么 Step61D 不选

Step61D L2H retrieval 在数据/坐标层面是可行的：retrieval match count 稳定、zero-match 几乎为零、learned alpha 非零。但它的 ACC/F1/BACC 不足以与 Step58C 竞争，因此只能写成 explored but not selected。

## 6. visual_ratio / concept_ratio 的解释

visual_ratio 表示 final logits 中来自 visual residual 的相对占比，concept_ratio 表示来自 concept evidence 的相对占比。Step57B 的单 fold baseline 审计显示 visual_ratio 约为 0.7196、concept_ratio 约为 0.2804，说明原始 reproduced baseline 更偏视觉残差。Step58C 及之后的几条创新线都把证据结构推向 concept-dominant，这一点是本轮创新的核心可解释结论。

## 7. CSG、CCRA、L2H 三条探索线的最终定位

Dynamic CSG：定位为 secondary variant。可以说它确实引入了 nonzero sample-adaptive graph updates，但不能说它是主要分类来源。

CCRA：定位为 secondary variant。可以说它带来了稳定非零的 concept-conditioned region changes，是更强的解释性变体之一。

L2H retrieval：定位为 rejected exploratory line。可以说检索覆盖稳定、机制可行，但不能说它形成了更好的最终分类模型。

## 8. 论文主结果怎么写

论文主结果应以 Step58C 为主：强调 residual-constrained training 在不引入新的未验证组合模型前提下，将 final evidence 从 visual residual 转向 concept evidence，并提升 ACC/F1/BACC。

## 9. 消融实验怎么写

消融表建议按 baseline RCE-v2、+ residual constraint、+ residual constraint + dynamic CSG、+ residual constraint + CCRA、+ residual constraint + L2H retrieval 组织。重点不是声称所有模块都超过 baseline，而是说明每条创新线分别回答了不同问题：主性能、sample-adaptive graph、concept-conditioned region aggregation、coordinate retrieval feasibility。

## 10. 局限性怎么写

当前没有完成所有创新模块的联合组合验证，因此不能声称存在统一最优组合模型。Dynamic CSG 的 logit 影响偏小，CCRA 的 full branch 没有稳定压过 concept_only，L2H retrieval 的导出 debug 字段仍有缺项（如 l2h_delta_abs_mean 为空）。

## 11. 后续工作怎么写

后续工作建议进入结果冻结与论文资产整理阶段，而不是继续扩展新模块。若未来继续研究，可优先考虑补齐更加标准化的多 fold interpretability export，而不是重新开新的训练线。

## 附：解释性与分支摘要

| model_id | interpretability_source | interpretability_is_formal_5fold | visual_ratio_mean | concept_ratio_mean | csg_ratio_mean | visual_ratio_gt_0_5_percent | full_acc | full_auc | concept_only_acc | concept_only_auc | visual_only_acc | visual_only_auc | full_minus_concept_acc | full_minus_concept_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage57C_rce_v2_baseline | step57B_single_fold_reference | False | 0.719580 | 0.280420 | 0.000040 | 0.938144 | 0.907216 | 0.975734 | 0.907216 | 0.960582 | 0.902062 | 0.975971 | 0.000000 | 0.015152 |
| step58C_residual_constrained_primary | results_stage58C_residual_constrained_configD_5fold | True | 0.320575 | 0.679425 | 0.002286 | 0.157198 | 0.927691 | 0.964842 | 0.917366 | 0.961629 | 0.910117 | 0.962506 | 0.010325 | 0.003213 |
| step59C_dynamic_csg_variant | results_stage59C_dynamic_csg_configA_5fold | True | 0.337585 | 0.662415 | 0.001467 | 0.183778 | 0.922504 | 0.969323 | 0.921473 | 0.965428 | 0.909081 | 0.963852 | 0.001031 | 0.003895 |
| step60D_ccra_variant | results_stage60D_ccra_configC_formal | True | 0.357331 | 0.642669 | 0.002062 | 0.184958 | 0.922493 | 0.968637 | 0.925586 | 0.971223 | 0.910128 | 0.960088 | -0.003093 | -0.002586 |
| step61D_l2h_not_selected | results_stage61D_l2h_configG_5fold | True | 0.281393 | 0.718607 | 0.001207 | 0.107537 | 0.921479 | 0.967921 | 0.919411 | 0.965738 | 0.914262 | 0.965463 | 0.002067 | 0.002182 |

| model_id | module_type | module_enabled | module_alpha_mean | module_delta_abs_mean | module_signal_interpretation | visual_ratio_mean | concept_ratio_mean | evidence_shift_interpretation | claim_strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage57C_rce_v2_baseline | none | False |  |  | Baseline reference only; no extra innovation module is enabled. | 0.719580 | 0.280420 | Evidence remains visually dominant. | reference_only |
| step58C_residual_constrained_primary | residual_constraint | True |  |  | Residual constraint is a training-time regularizer. No standalone module alpha/delta artifact was exported, so its signal is inferred from the stable shift from visual-dominant evidence toward concept-dominant evidence. | 0.320575 | 0.679425 | Evidence is shifted away from visual residual and toward concept evidence. | strong_primary |
| step59C_dynamic_csg_variant | dynamic_csg | True | 0.017729 | 0.000046 | Dynamic CSG produced nonzero sample-adaptive graph updates across folds, but the logit-level delta remained very small. | 0.337585 | 0.662415 | Evidence is shifted away from visual residual and toward concept evidence. | conservative_variant |
| step60D_ccra_variant | ccra | True | 0.040617 | 0.457858 | CCRA produced stable nonzero concept-conditioned low/high region changes across all audited folds. | 0.357331 | 0.642669 | Evidence is shifted away from visual residual and toward concept evidence. | moderate_variant |
| step61D_l2h_not_selected | l2h_retrieval | True | 0.011931 |  | L2H retrieval achieved stable coordinate-level coverage and nonzero learned alpha, but the exported debug package did not contain a populated l2h_delta_abs_mean field. | 0.281393 | 0.718607 | Evidence is shifted away from visual residual and toward concept evidence. | exploratory_not_selected |

## Warnings

- Stage57C baseline branch/contribution fields are populated from Step57B single-fold audit as reference only; formal 5-fold baseline branch/contribution artifacts are not available.
