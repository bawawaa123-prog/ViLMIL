# Step40 Experiment Tables Plan

## Table 1：主性能对比表
- 指标：`AUC`、`ACC`、`F1`、`Balanced ACC`、`PR-AUC`。
- 数据来源：`results_stage39/final_evidence_package/stage39_final_performance_summary.csv`，并可结合 Stage24/28/31/35/37 的主干行。
- 主要结论：`RCE-v4-CSG-a01-rq16 / DEG skeleton` 是当前最稳默认主模型。
- 摆放建议：正文主表。

## Table 2：RCE / CSG / rq 消融表
- 指标：`delta_auc`、`delta_acc`、`delta_f1`、`delta_balanced_acc`、`delta_pr_auc`。
- 数据来源：`results_stage39/final_evidence_package/stage39_ablation_summary.csv`。
- 主要结论：`CSG a01 > a005`，`rq16 > rq8/rq32`，最终主干收敛到 `RCE-v4-CSG-a01-rq16`。
- 摆放建议：正文。

## Table 3：negative ablation 表
- 指标：可不强调数值完整性，重点放 tested module、failure/trade-off reason、final decision。
- 数据来源：`results_stage39/final_evidence_package/stage39_negative_ablation_summary.csv`。
- 主要结论：region graph、concept graph、scalar gate 都有研究价值，但不进入最终默认主模型。
- 摆放建议：正文或正文+附录扩展。

## Table 4：skeleton vs lh_l001_m0 evidence calibration trade-off
- 指标：`fixed`、`regressed`、`persistent`、`low_high_conflict`、`both_support_wrong`、`visual_residual_override`、`AUC delta`、`PR-AUC delta`。
- 数据来源：`results_stage39/final_evidence_package/stage39_evidence_calibration_summary.csv` 与 `stage39_failure_comparison_summary.csv`。
- 主要结论：low-high consistency 可减少 conflict，但不是最终默认主模型。
- 摆放建议：正文。

## Table 5：fixed / regressed / persistent case examples
- 指标：案例组别、真实标签、两模型预测、primary failure type、margin、low-high joint state、selection note。
- 数据来源：`results_stage39/final_evidence_package/stage39_fixed_regressed_persistent_cases.csv`，必要时配合 top concepts / top CSG pairs。
- 主要结论：可以直观看到哪些错误被修复、哪些错误被引入、哪些错误持续存在。
- 摆放建议：正文展示少量代表案例，完整表放附录。
