# Step55 Reproducibility Commands

These commands are recorded for reproducibility only. They were not executed in Step55.

## Main Model 5-Fold Reproduction
```bash
VARIANT=rq16 bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh
```

## Step52 Core Ablation
```bash
VARIANT=all bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh
```

## Step52B Paper Table Build
```bash
python scripts/analysis/build_stage52b_rce_ablation_paper_tables.py
```

## Step53 Paper-Ready Package Build
```bash
python scripts/analysis/build_stage53_rce_paper_ready_package.py
```

## Step54 Evidence Package Refresh
```bash
python scripts/analysis/build_stage54_rce_evidence_interpretability_package.py
python scripts/analysis/plot_stage54_rce_evidence_figures.py
```

## Step54B Evidence Audit / Metadata / Figures
```bash
python scripts/analysis/build_stage54b_evidence_source_audit.py
python scripts/analysis/build_stage54b_case_metadata.py
python scripts/analysis/plot_stage54b_rce_evidence_figures.py
```

## Step54 / Step54C Evidence Export Refresh
```bash
MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh
MODE=wo_csg bash scripts/experiments/run_stage54_export_rce_evidence.sh
MODE=both bash scripts/experiments/run_stage54_export_rce_evidence.sh
python scripts/analysis/build_stage54c_full_export_audit.py
```

## Step55 Frozen Release Build
```bash
python scripts/analysis/build_stage55_rce_frozen_release.py
python scripts/analysis/build_stage55_rce_frozen_release.py --check
```

## Suggested Manual Git Tag
```bash
git tag rce-paper-ready-v1
git push origin rce-paper-ready-v1
```
