# Stage54B Next Steps

- If a direct full export is needed, run `MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh` manually.
- After a successful full direct export, rerun `python scripts/analysis/build_stage54b_evidence_source_audit.py`.
- Then rerun `python scripts/analysis/build_stage54b_case_metadata.py` and `python scripts/analysis/plot_stage54b_rce_evidence_figures.py` to replace fallback provenance where possible.
- Keep full vs w/o CSG comparison figures in supplementary material until provenance is fully same-source.
