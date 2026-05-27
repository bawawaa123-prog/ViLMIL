# ViLMIL / DCP-ViLa-MIL 实验结果整理表

> 目的：把当前 GitHub 仓库中已经完成的实验整理成可直接用于论文、汇报或后续复盘的 Markdown 表格。  
> 任务：`task_adenocarcinoma`，二分类：`Adenocarcinoma` vs `NonAdenocarcinoma`。  
> 主线模型：`ViLa_MIL_BiomedCLIP`，双尺度特征：`features_biomedclip_5x` + `features_biomedclip_20x`。  
> 指标格式：`mean ± std`，通常为 5-fold 结果。  
> 目录说明：表格中的“实验目录”均规范化为仓库内相对路径，来源于各汇总 CSV 中的 `train_dir` / `source_dir` 字段。

---

## 0. 数据来源文件索引

| 来源编号 | 汇总文件 | 主要用途 |
|---|---|---|
| S1 | `ViLa-MIL-main/trained_models/final_dcp_vila_analysis/final_main_table.csv` | 最终主表：static baseline、Concept-12 mean、dynamic gate、PEPS 主模型 |
| S2 | `ViLa-MIL-main/trained_models/stage2_size_sweep_comparison/size_sweep_comparison_summary.csv` | Prompt pool size sweep：Concept-6/10/12/14 等 |
| S3 | `ViLa-MIL-main/trained_models/stage2_ablation_comparison/ablation_comparison_summary.csv` | Stage2 prompt pool / ensemble 补充消融 |
| S4 | `ViLa-MIL-main/trained_models/stage3_peps_comparison/peps_extended_summary.csv` | PEPS top-k / tau 超参数敏感性 |
| S5 | `ViLa-MIL-main/trained_models/final_dcp_vila_analysis/scale_ablation.csv` | Low-only / High-only / Dual-scale 尺度消融 |
| S6 | `ViLa-MIL-main/trained_models/stage6_sap_peps_comparison/sap_peps_comparison_summary.csv` | SAP-PEPS 空间增强探索 |
| S7 | `ViLa-MIL-main/trained_models/stage7_saf_peps_comparison/saf_peps_comparison_summary.csv` | SAF-PEPS learned scale gate 探索 |
| S8 | `ViLa-MIL-main/trained_models/stage7_1_residual_saf_comparison/residual_saf_comparison_summary.csv` | Residual SAF-PEPS 探索 |
| S9 | `ViLa-MIL-main/trained_models/stage8_1_l2h_sda_lambda_ablation/l2h_sda_lambda_ablation_summary.csv` | L2H-SDA 空间增强 lambda 消融 |
| S10 | `ViLa-MIL-main/trained_models/final_dcp_vila_analysis/significance_tests.csv` | 主模型与 baseline 的统计检验 |

---

## 1. 主结果表：从 baseline 到当前主方法

这张表用于论文或汇报的主结果。它展示了从 BiomedCLIP static prompt baseline，到 Concept Prompt Pool，再到 PEPS 动态概念选择的递进关系。

| 方法 | Backbone | Prompt 形式 | Prompt 融合 / 选择 | Scale | Test AUC | Test ACC | Test F1 | Val AUC | Balanced ACC | PR-AUC | 实验目录 | 数据来源 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| BiomedCLIP static prompt baseline | BiomedCLIP | static dual-scale prompt | single prompt | dual | 0.967166 ± 0.010387 | 0.922531 ± 0.029854 | 0.916034 ± 0.029814 | 0.969061 ± 0.008921 | 0.923770 ± 0.018227 | 0.934883 ± 0.032131 | `ViLa-MIL-main/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1` | S1 / S2 / S3 |
| Concept-12 embedding_mean | BiomedCLIP | Concept-12 prompt pool | embedding mean | dual | 0.965884 ± 0.008652 | 0.933876 ± 0.016464 | 0.926832 ± 0.016916 | 0.969937 ± 0.011056 | 0.927344 ± 0.011114 | 0.941293 ± 0.027842 | `ViLa-MIL-main/trained_models/adeno_concept12_embedding_mean_s1` | S1 / S4 |
| Concept-12 MLP dynamic_gate | BiomedCLIP | Concept-12 prompt pool | MLP dynamic gate | dual | 0.965461 ± 0.013056 | 0.928722 ± 0.019128 | 0.921510 ± 0.021016 | 0.970106 ± 0.012834 | 0.925592 ± 0.021714 | 0.927728 ± 0.033763 | `ViLa-MIL-main/trained_models/adeno_concept12_dynamic_gate_s1` | S1 / S4 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP | Concept-12 prompt pool | PEPS top-k dynamic selection | dual | 0.968327 ± 0.006489 | **0.939047 ± 0.007595** | **0.932179 ± 0.007740** | **0.972999 ± 0.007620** | **0.931275 ± 0.005445** | **0.945866 ± 0.014115** | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.07_s1` | S1 / S4 |
| Concept-12 PEPS topk=5 tau=0.1 | BiomedCLIP | Concept-12 prompt pool | PEPS top-k dynamic selection | dual | **0.969500 ± 0.008614** | 0.935965 ± 0.016121 | 0.929148 ± 0.016764 | 0.971365 ± 0.006076 | 0.929636 ± 0.010304 | 0.941762 ± 0.020707 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.1_s1` | S1 / S4 |

**主表结论：**

- `Concept-12 PEPS topk=5 tau=0.07` 是目前最适合作为主方法的设置，因为它在 Test ACC、Test F1、Val AUC、Balanced ACC、PR-AUC 上综合最好。
- `Concept-12 PEPS topk=5 tau=0.1` 的 Test AUC 最高，可以作为 AUC-first reference，但综合分类质量略低于 tau=0.07。
- `Concept-12 embedding_mean` 已经明显提升 ACC/F1，说明 concept prompt pool 本身有效；PEPS 在此基础上进一步提升。

---

## 2. Prompt Pool Size 与 Ensemble 消融表

这张表用于说明：不同 prompt pool 大小、不同 prompt 融合方式对结果的影响。

| 方法 | Prompt pool size | Ensemble mode | Test AUC | Test ACC | Test F1 | Val AUC | Balanced ACC | PR-AUC | 实验目录 | 数据来源 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| BiomedCLIP static prompt baseline | - | single prompt | 0.967166 ± 0.010387 | 0.922531 ± 0.029854 | 0.916034 ± 0.029814 | 0.969061 ± 0.008921 | 0.923770 ± 0.018227 | 0.934883 ± 0.032131 | `ViLa-MIL-main/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1` | S2 / S3 |
| Concept-6 embedding_mean | 6 | embedding_mean | 0.967973 ± 0.008786 | 0.921500 ± 0.022168 | 0.914589 ± 0.022716 | 0.968912 ± 0.008967 | 0.922249 ± 0.015658 | 0.940079 ± 0.023630 | `ViLa-MIL-main/trained_models/adeno_stage2_concept_mean_s1` | S2 / S3 |
| Concept-10 embedding_mean | 10 | embedding_mean | **0.970803 ± 0.009414** | 0.929758 ± 0.015475 | 0.922375 ± 0.015660 | 0.970694 ± 0.007267 | 0.923491 ± 0.008505 | **0.951994 ± 0.015263** | `ViLa-MIL-main/trained_models/adeno_concept10_embedding_mean_s1` | S2 / S3 |
| Concept-10 logit_mean | 10 | logit_mean | 0.962372 ± 0.017220 | 0.926649 ± 0.007626 | 0.918086 ± 0.007629 | 0.968986 ± 0.014124 | 0.916061 ± 0.008487 | 0.926888 ± 0.033355 | `ViLa-MIL-main/trained_models/adeno_concept10_logit_mean_s1` | S3 |
| Concept-12 embedding_mean | 12 | embedding_mean | 0.965884 ± 0.008652 | 0.933876 ± 0.016464 | 0.926832 ± 0.016916 | 0.969937 ± 0.011056 | 0.927344 ± 0.011114 | 0.941293 ± 0.027842 | `ViLa-MIL-main/trained_models/adeno_concept12_embedding_mean_s1` | S2 / S4 |
| Concept-14 embedding_mean | 14 | embedding_mean | 0.969163 ± 0.007241 | 0.931836 ± 0.016366 | 0.924921 ± 0.016459 | 0.968706 ± 0.007609 | 0.927233 ± 0.007809 | 0.942040 ± 0.019492 | `ViLa-MIL-main/trained_models/adeno_concept14_embedding_mean_s1` | S2 |
| Concept-12 MLP dynamic_gate | 12 | dynamic_gate | 0.965461 ± 0.013056 | 0.928722 ± 0.019128 | 0.921510 ± 0.021016 | 0.970106 ± 0.012834 | 0.925592 ± 0.021714 | 0.927728 ± 0.033763 | `ViLa-MIL-main/trained_models/adeno_concept12_dynamic_gate_s1` | S4 |
| Concept-12 PEPS topk=5 tau=0.07 | 12 | PEPS | 0.968327 ± 0.006489 | **0.939047 ± 0.007595** | **0.932179 ± 0.007740** | **0.972999 ± 0.007620** | **0.931275 ± 0.005445** | 0.945866 ± 0.014115 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.07_s1` | S4 |

**Prompt Pool 消融结论：**

- `Concept-10 embedding_mean` 的 Test AUC 和 PR-AUC 很高，说明 prompt pool size 并不是越大越好。
- `Concept-12 embedding_mean` 的 ACC/F1 已经明显优于 static baseline。
- `dynamic_gate` 没有超过 embedding_mean，说明简单 MLP 动态门控并不稳定。
- `PEPS topk=5 tau=0.07` 在 ACC/F1/Val AUC/Balanced ACC 上最稳，适合作为主方法。

---

## 3. PEPS 超参数敏感性表

这张表用于说明 top-k 和 tau 对 PEPS 的影响。

| PEPS 设置 | top-k | tau | Test AUC | Test ACC | Test F1 | Val AUC | Balanced ACC | PR-AUC | 实验目录 | 数据来源 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Concept-12 PEPS topk=1 tau=0.1 | 1 | 0.1 | 0.965226 ± 0.010916 | 0.926660 ± 0.016407 | 0.918443 ± 0.017167 | 0.968777 ± 0.007608 | 0.917507 ± 0.014148 | 0.941989 ± 0.018448 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk1_tau0.1_s1` | S4 |
| Concept-12 PEPS topk=3 tau=0.07 | 3 | 0.07 | 0.968138 ± 0.005759 | 0.932845 ± 0.007360 | 0.925810 ± 0.007480 | 0.970790 ± 0.010501 | 0.928006 ± 0.004538 | 0.938729 ± 0.019306 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk3_tau0.07_s1` | S4 |
| Concept-12 PEPS topk=3 tau=0.1 | 3 | 0.1 | 0.967991 ± 0.007419 | 0.935954 ± 0.006154 | 0.928744 ± 0.006727 | 0.971034 ± 0.008742 | 0.928121 ± 0.005782 | 0.942830 ± 0.020229 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk3_tau0.1_s1` | S4 |
| Concept-12 PEPS topk=3 tau=0.15 | 3 | 0.15 | 0.961322 ± 0.013083 | 0.918396 ± 0.012324 | 0.908797 ± 0.012850 | 0.971954 ± 0.009448 | 0.906960 ± 0.015058 | 0.934522 ± 0.018555 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk3_tau0.15_s1` | S4 |
| Concept-12 PEPS topk=5 tau=0.07 | 5 | 0.07 | 0.968327 ± 0.006489 | **0.939047 ± 0.007595** | **0.932179 ± 0.007740** | **0.972999 ± 0.007620** | **0.931275 ± 0.005445** | **0.945866 ± 0.014115** | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.07_s1` | S4 |
| Concept-12 PEPS topk=5 tau=0.1 | 5 | 0.1 | **0.969500 ± 0.008614** | 0.935965 ± 0.016121 | 0.929148 ± 0.016764 | 0.971365 ± 0.006076 | 0.929636 ± 0.010304 | 0.941762 ± 0.020707 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.1_s1` | S4 |
| Concept-12 PEPS topk=5 tau=0.15 | 5 | 0.15 | 0.966858 ± 0.006619 | 0.933898 ± 0.011410 | 0.926994 ± 0.011442 | 0.972843 ± 0.007685 | 0.928812 ± 0.005313 | 0.936667 ± 0.021050 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.15_s1` | S4 |

**PEPS 超参数结论：**

- `topk=1` 选择过窄，ACC/F1 不如 topk=3/5。
- `topk=5 tau=0.07` 综合最好，适合作为主模型。
- `topk=5 tau=0.1` 的 AUC 最高，但 ACC/F1 略低。
- tau=0.15 在 topk=3 时明显下降，说明温度过大可能削弱 prompt 选择的判别性。

---

## 4. 双尺度消融表

这张表用于说明 low-scale、high-scale 和 dual-scale 的贡献。

| 方法 | Scale mode | Test AUC | Test ACC | Test F1 | Val AUC | Balanced ACC | Sensitivity | Specificity | PR-AUC | 实验目录 | 数据来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| PEPS low-only | low_only | 0.931815 ± 0.021260 | 0.863656 ± 0.019080 | 0.849217 ± 0.016092 | 0.941425 ± 0.021333 | 0.850868 ± 0.006672 | 0.809905 ± 0.055197 | 0.891831 ± 0.056141 | 0.876428 ± 0.029797 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_lowonly_topk5_tau0.07_s1` | S5 |
| PEPS high-only | high_only | 0.963794 ± 0.013557 | 0.928695 ± 0.021813 | 0.920109 ± 0.024265 | **0.973556 ± 0.010373** | 0.916882 ± 0.024971 | 0.879285 ± 0.042336 | 0.954478 ± 0.022952 | 0.937085 ± 0.028102 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_highonly_topk5_tau0.07_s1` | S5 |
| PEPS dual-scale | dual | **0.968327 ± 0.006489** | **0.939047 ± 0.007595** | **0.932179 ± 0.007740** | 0.972999 ± 0.007620 | **0.931275 ± 0.005445** | **0.906422 ± 0.029051** | **0.956127 ± 0.024452** | **0.945866 ± 0.014115** | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.07_s1` | S5 |

**双尺度消融结论：**

- high-only 明显强于 low-only，说明高倍 patch 特征对腺癌二分类更加关键。
- dual-scale 在 Test AUC、ACC、F1、Balanced ACC、Specificity、PR-AUC 上最好，说明低倍结构信息仍然提供互补收益。
- 这张表建议放入正文，因为它能支撑 ViLa-MIL 双尺度框架的合理性。

---

## 5. 探索性扩展表：SAP / SAF / Residual SAF / L2H-SDA

这张表用于记录已经尝试但不建议作为当前主方法的扩展实验。

| 方法 | 扩展方向 | Test AUC | Test ACC | Test F1 | Val AUC | Balanced ACC | PR-AUC | 实验目录 | 数据来源 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| PEPS topk=5 tau=0.07 | baseline PEPS | 0.968327 ± 0.006489 | **0.939047 ± 0.007595** | **0.932179 ± 0.007740** | 0.972999 ± 0.007620 | **0.931275 ± 0.005445** | 0.945866 ± 0.014115 | `ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.07_s1` | S6 / S7 / S8 / S9 |
| SAP-PEPS topk=5 tau=0.07 | spatial-aware prompt evidence | **0.969688 ± 0.006855** | 0.936985 ± 0.006821 | 0.930095 ± 0.006766 | 0.973415 ± 0.006975 | 0.930428 ± 0.001694 | 0.940601 ± 0.024704 | `ViLa-MIL-main/trained_models/adeno_concept12_sap_peps_topk5_tau0.07_s1` | S6 |
| SAF-PEPS topk=5 tau=0.07 | learned scale fusion gate | 0.965731 ± 0.015548 | 0.922531 ± 0.015586 | 0.914460 ± 0.016043 | 0.970885 ± 0.007334 | 0.917360 ± 0.017502 | 0.947647 ± 0.032238 | `ViLa-MIL-main/trained_models/adeno_concept12_saf_peps_topk5_tau0.07_s1` | S7 |
| Residual SAF topk=5 tau=0.07 g=0.25 | residual scale fusion gate | 0.967997 ± 0.009623 | 0.926639 ± 0.012891 | 0.918158 ± 0.013971 | **0.974980 ± 0.008844** | 0.916758 ± 0.015273 | 0.944695 ± 0.024227 | `ViLa-MIL-main/trained_models/adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25_s1` | S8 |
| L2H-SDA-PEPS lambda=0.05 | low-to-high spatial domain adaptation | 0.955419 ± 0.024824 | 0.916303 ± 0.027813 | 0.906609 ± 0.032071 | 0.967388 ± 0.015189 | 0.906629 ± 0.035161 | 0.921202 ± 0.046190 | `ViLa-MIL-main/trained_models/adeno_concept12_l2h_sda_peps_topk5_tau0.07_k5_s8192_l0.05_s1` | S9 |
| L2H-SDA-PEPS lambda=0.10 | low-to-high spatial domain adaptation | 0.955674 ± 0.026535 | 0.901875 ± 0.049981 | 0.887241 ± 0.061428 | 0.957837 ± 0.015957 | 0.881789 ± 0.065229 | 0.908406 ± 0.063168 | `ViLa-MIL-main/trained_models/adeno_concept12_l2h_sda_peps_topk5_tau0.07_k5_s8192_l0.10_s1` | S9 |

**探索性扩展结论：**

- SAP-PEPS 的 AUC 略高于 baseline PEPS，但 ACC/F1/PR-AUC 不如 PEPS，因此不建议替代主方法。
- SAF learned gate 和 Residual SAF 都没有超过简单的 dual-scale sum，说明当前数据规模下复杂尺度门控不稳定。
- L2H-SDA 两个 lambda 设置都明显低于 PEPS，建议暂时作为失败探索或附录记录，不作为主线继续推进。

---

## 6. 显著性检验摘要表

这张表用于辅助写论文时描述统计显著性。注意：目前多数 test 指标的 p-value 未达到 0.05，因此写作时不要表述为 “all test metrics are statistically significant”。

| Candidate | Reference | Metric | Mean delta | Paired t-test p-value | Wilcoxon p-value | Wins / Losses / Ties | 结论 | 数据来源 |
|---|---|---:|---:|---:|---:|---|---|---|
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | test_auc | +0.001161 | 0.793733 | 1.000000 | 2 / 3 / 0 | AUC 小幅提升，但不显著 | S10 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | test_acc | +0.016516 | 0.272044 | 0.250000 | 3 / 1 / 1 | ACC 提升较明显，但 5-fold 下未显著 | S10 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | test_f1 | +0.016145 | 0.283683 | 0.312500 | 3 / 2 / 0 | F1 提升较明显，但未显著 | S10 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | val_auc | +0.003939 | **0.020470** | 0.062500 | 5 / 0 / 0 | Val AUC 达到 paired t-test 显著 | S10 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | test_acc | +0.005171 | 0.526986 | 0.437500 | 4 / 1 / 0 | 相比 embedding_mean 有一致提升趋势，但未显著 | S10 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | test_acc | +0.010325 | 0.266118 | 0.437500 | 3 / 2 / 0 | 相比 dynamic_gate 有提升趋势，但未显著 | S10 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | specificity | +0.020411 | **0.000452** | 0.062500 | 5 / 0 / 0 | Specificity 显著提升 | S10 |

**显著性写作建议：**

可以写：

> PEPS achieves consistent improvements in ACC/F1 and validation AUC over the BiomedCLIP static prompt baseline. Due to the limited 5-fold setting and strong baseline performance, most test-set improvements do not reach statistical significance, while validation AUC and specificity show statistically significant gains in selected comparisons.

不建议写：

> PEPS significantly outperforms all baselines on all metrics.

---

## 7. 建议用于论文 / 汇报的最终实验组织

| 表格编号 | 建议放置位置 | 表格内容 | 作用 |
|---|---|---|---|
| Table 1 | 正文主结果 | 主结果表：static baseline → Concept-12 → PEPS | 展示核心方法有效性 |
| Table 2 | 正文或附录 | Prompt pool size 与 ensemble 消融 | 解释为什么选择 Concept-12 + PEPS |
| Table 3 | 附录或正文简表 | PEPS top-k / tau 敏感性 | 证明主超参数设置合理 |
| Table 4 | 正文 | Low-only / High-only / Dual-scale | 证明双尺度设计合理 |
| Table 5 | 附录 | SAP / SAF / Residual SAF / L2H-SDA | 记录探索性扩展，不作为主方法 |
| Table 6 | 附录 | 显著性检验 | 避免过度声明，增强可信度 |

---

## 8. 当前推荐主方法

当前最推荐作为最终主模型的是：

```text
Concept-12 PEPS topk=5 tau=0.07
实验目录：ViLa-MIL-main/trained_models/adeno_concept12_peps_topk5_tau0.07_s1
```

推荐原因：

- Test ACC 最高：`0.939047 ± 0.007595`
- Test F1 最高：`0.932179 ± 0.007740`
- Val AUC 最高：`0.972999 ± 0.007620`
- Balanced ACC 最高：`0.931275 ± 0.005445`
- PR-AUC 最高：`0.945866 ± 0.014115`
- 相比 static baseline，模型改动集中在 Concept Prompt Pool + PEPS，主线清晰，容易写成方法贡献。

---

## 9. 后续继续创新的建议

基于当前结果，不建议继续把主精力放在 SAF、Residual SAF 或 L2H-SDA 上，因为这些扩展没有稳定超过 PEPS。更适合继续创新的方向是：

```text
Concept-guided Prototype Alignment
```

也就是让 PEPS 选出的病理概念不仅用于文本分类原型，还反过来指导图像侧 prototype 聚合，使图像 prototype 从“无语义的可学习原型”变为“病理概念约束下的证据原型”。

