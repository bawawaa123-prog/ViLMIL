# Step52B RCE Ablation Statistics Plan

## Purpose

Step52B consolidates already-trained RCE core ablation results into paper-ready statistical tables, fold-level deltas, and interpretation notes.

## Inputs

- `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/`
- `results_stage52_rce_core_ablation/*/result.csv`
- `results_stage52_rce_core_ablation/*/fold_summary.csv` when available
- `results_stage52_rce_core_ablation/*/epoch_details.csv` only as a fallback note source

## Outputs

- `docs/stage52b_rce_ablation_table_paper.md`
- `docs/stage52b_rce_ablation_table_paper.csv`
- `docs/stage52b_rce_ablation_table_latex.tex`
- `docs/stage52b_rce_ablation_fold_delta.csv`
- `docs/stage52b_rce_ablation_interpretation.md`
- `docs/stage52b_rce_ablation_statistics_plan.md`

## Run Command

```bash
python scripts/analysis/build_stage52b_rce_ablation_paper_tables.py
```

## Using The LaTeX Table

The generated `docs/stage52b_rce_ablation_table_latex.tex` can be copied into the paper appendix or ablation section directly. It already uses `mean $\pm$ std`, includes the required rows and columns, and boldfaces the best value in each metric column.

## Statistical Limitations

- Paired t-tests are only valid when matched fold-level test metrics are available.
- `epoch_details.csv` stores train/val trajectories and is not treated as a source of final test metrics.
- If fold-level summaries are missing or incomplete, the script leaves p-values as `NA` instead of fabricating them.

