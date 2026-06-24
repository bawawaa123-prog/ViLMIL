# Stage54C Next Steps

- Freeze the refreshed Step54B package with the direct full export as the preferred full-side source.
- Preserve `results_stage54_rce_evidence_interpretability/full/` and `results_stage54_rce_evidence_interpretability/wo_csg/` together as the paired Step54 evidence package.
- Keep `results_stage32/stage32_rce_v4_csg_evidence_export/` only as legacy reference, not as the active source in paper-facing summaries.
- If any future re-export is needed, rerun:
  - `MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh`
  - `python scripts/analysis/build_stage54c_full_export_audit.py`
  - `python scripts/analysis/build_stage54b_evidence_source_audit.py`
  - `python scripts/analysis/build_stage54b_case_metadata.py`
  - `python scripts/analysis/plot_stage54b_rce_evidence_figures.py`
- Step55 frozen release can proceed from the Step54C evidence/provenance standpoint.
