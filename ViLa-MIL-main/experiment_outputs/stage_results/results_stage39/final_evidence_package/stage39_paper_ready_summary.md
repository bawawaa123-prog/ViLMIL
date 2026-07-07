# Step39 Final Evidence Package and Paper-Ready Summary

## 研究动机
本阶段的目标不是继续训练新模型，而是在现有最优结果上收敛论文叙事：明确最终默认模型、整理哪些模块构成主创新、哪些探索应当作为 negative ablation 保留，并把证据解释与错误分析整理成可直接进入论文/报告的产物。

## 方法概述
最终方法主线为 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。它保留 Region-Concept Evidence Learning 与 Cross-Scale Concept Evidence Reasoning 这两条核心设计，同时通过 low/high concept evidence、visual residual、cross-scale pair evidence 的分解分析，使后续 failure diagnosis 与 calibration analysis 可以在不改动主模型主体逻辑的前提下完成。

## 最终推荐模型
Recommended default model: `RCE-v4-CSG-a01-rq16 / DEG skeleton`

Secondary trade-off variant: `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0`

Reason: skeleton has stronger AUC/PR-AUC and remains the most robust default. lh_l001_m0 reduces fold0/test errors and low-high conflict but increases visual_residual_override and slightly reduces AUC/PR-AUC.

## 主要实验结论
- Step24 说明 `CSG a01 > CSG a005`，且 `rq16 > rq8/rq32`，因此最终 RCE 主干固定为 `RCE-v4-CSG-a01-rq16`。
- Step28 与 Step31 说明无论是 Spatial Region Graph 还是 Concept Prompt Graph，都没有超过 `DEG skeleton`，因此最终主模型不再堆 graph 模块。
- Step35 说明 Scalar Visual Gate 也没有超过 skeleton，说明 visual residual 不能被一个全局 gate 简单替代。
- Step37/38 说明 `lh_l001_m0` 具备 evidence calibration 价值，但还不足以替换 skeleton 成为最终默认模型。

## 消融实验结论
| topic | comparison | source_stage | paper_ready_conclusion | delta_test_auc | delta_pr_auc |
| --- | --- | --- | --- | --- | --- |
| CSG strength | csg_a01 - csg_a005 | stage24 | CSG a01 > CSG a005 | 0.0082850305336239 | 0.0230678935801611 |
| Region query count | rq16 - rq8 | stage24 | rq16 > rq8 | 0.0062150687373795 | 0.0087949059468064 |
| Region query count | rq16 - rq32 | stage24 | rq16 > rq32 | 0.0124866952371286 | 0.0225768383794476 |
| Spatial Region Graph | rg_k8 - skeleton | stage28 | Spatial Region Graph did not outperform skeleton | -0.0047564388152293 | -0.0101909069933815 |
| Concept Prompt Graph | cg_k8_a005 - skeleton | stage31 | Concept Prompt Graph did not outperform skeleton | -0.0056500670413144 | -0.0042570881096622 |
| Scalar Visual Gate | gate1 - skeleton | stage35 | Scalar Visual Gate did not outperform skeleton | -0.00466642438754683 | -0.020869147678342137 |
| Low-High Consistency | lh_l001_m0 - skeleton | stage37+stage38 | Low-High Consistency is a trade-off variant | -0.00390625 | -0.0033756863861579 |

## 解释性与错误分析
- fixed cases = `6`
- regressed cases = `2`
- persistent errors = `12`
- low_high_conflict: `10 -> 8`
- both_support_wrong: `7 -> 5`
- visual_residual_override: `13 -> 14`
- AUC delta = `-0.0039`
- PR-AUC delta = `-0.0034`
- 这说明 low-high consistency 的主要收益在于缓解 low/high evidence conflict，而主要代价在于更容易把错误样本交给 visual residual 接管。

## Current Negative Ablation Narrative
| module | paper_ready_interpretation | implication | role |
| --- | --- | --- | --- |
| attention-centroid region graph | attention-centroid region graph: semantic region token does not equal true spatial region | graph edges built from centroided prompt tokens fail to provide stable spatial inductive bias | diagnostic ablation |
| concept prompt graph | concept prompt graph: ordinary feature-level prompt smoothing weakens evidence discrimination | concept-to-concept diffusion blurs class-critical prompt evidence instead of sharpening it | diagnostic ablation |
| scalar visual gate | scalar visual gate: visual residual cannot be simply suppressed by a global scalar | visual evidence interacts with concept evidence in a sample-dependent manner and resists one-number gating | diagnostic ablation |
| low-high consistency | low-high consistency: reduces low-high conflict but carries a visual residual override trade-off | consistency regularization calibrates evidence alignment but is not clean enough to replace the default model | secondary trade-off variant |

## 当前限制
- 当前证据表明 low-high consistency 更像 calibration trade-off，而不是 clean gain；因此不宜直接替换默认主模型。
- Spatial/Concept graph 与 scalar gate 都没有稳定收益，说明继续盲目堆 graph/gate 的边际回报很低。
- Example evidence cases 已能关联 top concepts / top CSG pairs，但当前使用的是 `stage32_slide_evidence_summary.csv` 里的汇总字段，而不是更细粒度的独立 per-slide 明细表。

## 后续工作
- Step40：根据 `stage39_paper_ready_summary.md` 生成最终论文主图/方法图说明、实验表格说明和答辩汇报材料。
- 如果还要继续模型创新，可以转向 Prompt Reliability / Refined Prompt Pool，但不建议再盲目堆 graph 或 gate。

## Example Cases Included
| case_group | selection_rank | slide_id | label | label_name | skeleton_pred | lh_pred | skeleton_pred_name | lh_pred_name | skeleton_primary_failure_type | lh_primary_failure_type | skeleton_failure_labels | lh_failure_labels | skeleton_final_margin | lh_final_margin | skeleton_low_high_joint_state | lh_low_high_joint_state | skeleton_dominant_source | lh_dominant_source | skeleton_visual_source_ratio | lh_visual_source_ratio | selection_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 1 | 2462598-B | 0 | Adenocarcinoma | 1 | 0 | NonAdenocarcinoma | Adenocarcinoma | low_high_conflict | correct_prediction | low_high_conflict|uncertain_low_margin | N/A | -0.0087442398071284 | 4.739243984222412 | conflict | conflict | high | visual | 0.0170107760841615 | 0.7742628142528889 | represents repaired low-high conflict / visual override behavior |
| fixed | 2 | 2468673-B2 | 0 | Adenocarcinoma | 1 | 0 | NonAdenocarcinoma | Adenocarcinoma | low_high_conflict | correct_prediction | low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | N/A | -0.9737696647644044 | 3.1285730600357056 | conflict | conflict | visual | visual | 0.3843036827087641 | 0.5680807511168112 | represents repaired low-high conflict / visual override behavior |
| fixed | 3 | 2471959-B2 | 0 | Adenocarcinoma | 1 | 0 | NonAdenocarcinoma | Adenocarcinoma | low_high_conflict | correct_prediction | low_high_conflict|uncertain_low_margin | N/A | -0.1999548673629758 | 1.7742162942886353 | conflict | conflict | high | high | 0.2928694650432292 | 0.3854150682328052 | represents repaired low-high conflict / visual override behavior |
| regressed | 1 | 25015422B | 0 | Adenocarcinoma | 0 | 1 | Adenocarcinoma | NonAdenocarcinoma | correct_prediction | visual_residual_override | N/A | visual_residual_override|low_high_conflict|uncertain_low_margin | 2.206884741783142 | -2.761066406965256 | both_support_true | conflict | visual | visual | 0.7829998792505194 | 0.820608645793681 | represents a new failure introduced by consistency regularization |
| regressed | 2 | 25023961B2 | 1 | NonAdenocarcinoma | 1 | 0 | NonAdenocarcinoma | Adenocarcinoma | correct_prediction | visual_residual_override | N/A | visual_residual_override|low_high_conflict|low_scale_dominant_wrong|uncertain_low_margin|prompt_confusion | 6.292264521121979 | -0.8033177852630622 | both_support_true | conflict | visual | visual | 0.8795846546281327 | 0.8486937668959043 | represents a new failure introduced by consistency regularization |
| persistent_error | 1 | 25019273B | 1 | NonAdenocarcinoma | 0 | 0 | Adenocarcinoma | Adenocarcinoma | visual_residual_override | visual_residual_override | visual_residual_override|low_high_conflict|uncertain_low_margin | visual_residual_override|low_high_conflict|uncertain_low_margin|prompt_confusion | -4.017699599266052 | -5.50821590423584 | conflict | conflict | visual | visual | 0.9100514504096284 | 0.818134942672171 | represents a persistent hard case across both variants |
| persistent_error | 2 | 25023977B2 | 0 | Adenocarcinoma | 1 | 1 | NonAdenocarcinoma | NonAdenocarcinoma | visual_residual_override | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | -4.951011657714844 | -3.41491961479187 | conflict | conflict | visual | visual | 0.7880461689920787 | 0.6672225048742627 | represents a persistent hard case across both variants |
| persistent_error | 3 | 25025076B | 1 | NonAdenocarcinoma | 0 | 0 | Adenocarcinoma | Adenocarcinoma | visual_residual_override | visual_residual_override | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | visual_residual_override|low_high_conflict|high_scale_dominant_wrong|uncertain_low_margin | -6.41008734703064 | -5.388396263122559 | conflict | conflict | visual | visual | 0.7988088510868069 | 0.7722197248570467 | represents a persistent hard case across both variants |

## Warnings / Data Notes
- Stage35 summary CSV exists but does not contain the expected full e20 skeleton/gate rows; using raw fold_summary.csv files instead.
- results_stage38/evidence_export_skeleton_fold0_test/stage32_top_concepts_per_slide.csv is missing; Step39 falls back to stage32_slide_evidence_summary.csv top concept fields.
- results_stage38/evidence_export_lh_l001_m0_fold0_test/stage32_top_concepts_per_slide.csv is missing; Step39 falls back to stage32_slide_evidence_summary.csv top concept fields.
- results_stage38/evidence_export_skeleton_fold0_test/stage32_top_csg_pairs_per_slide.csv is missing; Step39 falls back to stage32_slide_evidence_summary.csv top CSG pair fields.
- results_stage38/evidence_export_lh_l001_m0_fold0_test/stage32_top_csg_pairs_per_slide.csv is missing; Step39 falls back to stage32_slide_evidence_summary.csv top CSG pair fields.
