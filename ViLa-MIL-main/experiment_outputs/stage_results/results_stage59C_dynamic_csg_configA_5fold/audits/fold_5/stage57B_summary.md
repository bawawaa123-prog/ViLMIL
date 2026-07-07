# Step57B Summary

## Answers

- 本 Step 是否修改了原始 RCE 文件：`否`。`models/model_RCE_MIL_BiomedCLIP.py` 未被修改。
- RCE-v2 是否仍保持原训练逻辑不变：`是`。仅在 `models/model_RCE_MIL_BiomedCLIP_v2.py` 中新增了默认关闭的 `last_logit_breakdown` 审计缓存，`forward` 仍返回原有的 `(Y_prob, Y_hat, loss)`。
- 是否成功导出 logit breakdown：`部分/否`。审计同时缓存了 pre-calibration 与 post-calibration logits；branch-level metrics 与 sample-level margins 采用 post-calibration logits 统计，以对齐最终预测。
- full / concept_only / visual_only / full_without_visual 哪个分支表现最好：`无法判断`（按 `balanced_acc` 选取）。
- visual residual 是否明显主导 final logits：`insufficient_audit`。 依据：mean visual ratio=nan，median=nan，ratio>0.5 占比=nan。
- CSG 是否有可观察贡献：`insufficient_audit`。 依据：mean csg ratio=nan，csg_only acc=nan。
- 下一步建议：`先补充审计`。

## Run Status

- 审计状态：`static_only`
- 已执行 folds：`[]`
- 审计样本数：`0`
- flip cases 数量：`0`

## Notes

- `stage57B_branch_metrics.csv`、`stage57B_sample_contribution.csv`、`stage57B_margin_stats.csv`、`stage57B_flip_cases.csv` 已写入固定输出目录。
- 若未能完整推理，请检查 `stage57B_audit_status.json` 中的 warnings / errors，并使用下方命令复现。

## Repro Command

```bash
python scripts/analysis/build_stage57B_logit_contribution_audit.py \
  --output_dir /home/ljh/ViLMIL/ViLa-MIL-main/results_stage59C_dynamic_csg_configA_5fold/audits/fold_5 \
  --split test \
  --fold 5 \
  --run_dir /home/ljh/ViLMIL/ViLa-MIL-main/results_stage59C_dynamic_csg_configA_5fold/rce_v2_rcD_dynCSG_A_5fold_e20_s1
```

## Warnings

- missing checkpoint for fold 5: /home/ljh/ViLMIL/ViLa-MIL-main/results_stage59C_dynamic_csg_configA_5fold/rce_v2_rcD_dynCSG_A_5fold_e20_s1/s_5_checkpoint.pt
- no fold plan available; only static audit outputs will be generated
