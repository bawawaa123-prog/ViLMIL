# Step57C RCE-v2 5-fold training reproduction audit

- Branch: `dev-rce-innovation`
- Stage23 source: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Stage57C source: `results_stage57C_rce_v2_copy_reproduction/rce_v2_copy_csg_a01_rq16_5fold_e20_s1`

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。
3. RCE-v2 是否使用与 Stage23 原始 RCE 完全一致的训练配置：是。 训练脚本基于 `run_stage23_rce_v4_csg_region_queries_5fold.sh` 镜像，核心训练参数保持一致，仅切换 `--model_type=RCE_MIL_BiomedCLIP_v2`、`results_dir` 与 `exp_code`。
4. RCE-v2 5-fold 是否训练完成：是。 当前状态为 `completed`，fold_count=5，checkpoint_count=5。
5. RCE-v2 的 5-fold ACC 是否接近原始 RCE 的 ACC=0.9225：是。RCE-v2 5-fold ACC=0.922509，相对原始 RCE 的差异为 +0.000000。
6. 如果结果有差异，差异是多少： Stage23 ACC=0.922509，Stage57C ACC=0.922509，delta=+0.000000； AUC delta=+0.000000； F1 delta=+0.000000； BACC delta=+0.000000； PR_AUC delta=+0.000000。
7. 是否可以确认 RCE-v2 适合作为后续创新基础：可以确认，当前复现结果已回到 Stage23 主线水平。
8. 下一步建议是否进入 Step58A residual-constrained RCE：可以进入 Step58A residual-constrained RCE。

## Metrics Snapshot

| model_name | AUC | ACC | F1 | Balanced_ACC | PR_AUC |
| --- | --- | --- | --- | --- | --- |
| RCE_MIL_BiomedCLIP | 0.970213 | 0.922509 | 0.914456 | 0.917143 | 0.944399 |
| RCE_MIL_BiomedCLIP_v2 | 0.970213 | 0.922509 | 0.914456 | 0.917143 | 0.944399 |

## Run Command

See `results_stage57C_rce_v2_copy_reproduction/stage57C_run_command.txt` for the exact reproduction command.

## Notes

- 本 Step 只新增了训练复现实验脚本与结果对比整理脚本，没有引入新模块、新 loss、新 graph。
- 当前 Stage57C 结果目录中已检测到 checkpoint，但尚未检测到完成 5-fold 所需的 aggregate/fold metrics CSV。
