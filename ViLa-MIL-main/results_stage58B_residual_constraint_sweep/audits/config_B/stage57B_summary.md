# Step57B Summary

## Answers

- 本 Step 是否修改了原始 RCE 文件：`否`。`models/model_RCE_MIL_BiomedCLIP.py` 未被修改。
- RCE-v2 是否仍保持原训练逻辑不变：`是`。仅在 `models/model_RCE_MIL_BiomedCLIP_v2.py` 中新增了默认关闭的 `last_logit_breakdown` 审计缓存，`forward` 仍返回原有的 `(Y_prob, Y_hat, loss)`。
- 是否成功导出 logit breakdown：`是`。审计同时缓存了 pre-calibration 与 post-calibration logits；branch-level metrics 与 sample-level margins 采用 post-calibration logits 统计，以对齐最终预测。
- full / concept_only / visual_only / full_without_visual 哪个分支表现最好：`visual_only`（按 `balanced_acc` 选取）。
- visual residual 是否明显主导 final logits：`no_clear_dominance`。 依据：mean visual ratio=0.4601，median=0.4076，ratio>0.5 占比=0.3093。
- CSG 是否有可观察贡献：`yes`。 依据：mean csg ratio=0.0005，csg_only acc=0.33505154639175255。
- 下一步建议：`dynamic CSG`。

## Run Status

- 审计状态：`completed`
- 已执行 folds：`[0]`
- 审计样本数：`194`
- flip cases 数量：`9`

## Notes

- `stage57B_branch_metrics.csv`、`stage57B_sample_contribution.csv`、`stage57B_margin_stats.csv`、`stage57B_flip_cases.csv` 已写入固定输出目录。
- 若未能完整推理，请检查 `stage57B_audit_status.json` 中的 warnings / errors，并使用下方命令复现。

## Repro Command

```bash
python scripts/analysis/build_stage57B_logit_contribution_audit.py \
  --output_dir /home/ljh/ViLMIL/ViLa-MIL-main/results_stage58B_residual_constraint_sweep/audits/config_B \
  --split test \
  --fold 0 \
  --run_dir /home/ljh/ViLMIL/ViLa-MIL-main/results_stage58B_residual_constraint_sweep/rce_v2_rc_config_B_fold0_s1
```
