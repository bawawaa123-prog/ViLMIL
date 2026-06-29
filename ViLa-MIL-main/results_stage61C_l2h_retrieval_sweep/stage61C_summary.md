# Step61C Low-to-High Retrieval sweep summary

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：是，做了最小窗口修复。
3. 本 Step 实际跑了哪些 L2H config：['A', 'B', 'C', 'D', 'E', 'F', 'G']。
4. 哪些 config completed / pending / skipped / failed：completed=['A', 'B', 'C', 'D', 'E', 'F', 'G'] pending=[] skipped=['H'] failed=[]。
5. 哪个 config 的 ACC/AUC/F1 最好：ACC=E AUC=E F1=F。
6. 哪个 config 的 L2H delta 或 learned alpha 最明显：alpha=C delta=D。
7. L2H 是否真的产生了非零 retrieval contribution：是。
8. retrieved_high_match_counts_mean 是否合理：是。
9. zero-match 比例是否低：是。
10. visual_ratio 是否仍保持在 Step58C config D 的低水平：是。
11. concept_ratio 是否仍保持较高：是。
12. 推荐进入 Step61D 的 selected config 是哪个：G。
13. 如果没有推荐配置，是否建议停止 L2H 并进入 final consolidation： 否。
14. 下一步建议是什么： 进入 Step61D 5-fold。

## Notes

- Step61C 开始前对 Step61B 做了最小修复：将 L2H patch extent 估计从坐标差分 median 调整为 p95，以更贴近 Step61A 的 patch footprint 审计；原因是原实现会把检索窗口压缩成 stride 级别，导致 retrieved match count 系统性偏低。
- selected_config decision: `selected_for_5fold`
- selected_config: `G`
- nonzero_l2h_signal: `True`
- retrieval_reasonable: `True`
- zero_match_low: `True`
- visual_low: `True`
- concept_high: `True`
