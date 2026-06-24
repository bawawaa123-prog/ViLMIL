# Stage54C Summary

## Direct Answers

1. `MODE=full` export was run.
2. `MODE=full` did not trigger training. The shell wrapper only called `scripts/analysis/export_stage32_rce_v4_csg_evidence.py`, and the exporter restored a checkpoint, switched the model to `eval()`, and ran under `torch.no_grad()`.
3. Full direct export succeeded.
4. Full evidence no longer depends on the Stage32 fallback as the preferred source in the refreshed Step54B package.
5. `wo_csg` evidence remains the verified Step54 direct export.
6. Refreshed Step54B files include:
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_evidence_source_audit.md`
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_evidence_source_audit.csv`
   - `results_stage54_rce_evidence_interpretability/stage54b_evidence_source_audit.csv`
   - `results_stage54_rce_evidence_interpretability/stage54b_case_level_metadata.csv`
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.md`
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.csv`
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_paper_figure_caption_drafts.md`
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_summary.md`
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_claims_to_make_and_avoid.md`
   - `docs/stage54b_rce_evidence_figure_polish/stage54b_next_steps.md`
   - all 7 `docs/stage54b_rce_evidence_figure_polish/figures/stage54b_*.png`
7. All 7 refreshed Step54B figures switched from fallback-backed full provenance to Step54 full direct export on the full side.
8. Remaining limitations are interpretability-scope limitations, not source-mismatch limitations. The figures still cannot support localization-benchmark or training-significance claims.
9. Step55 frozen release can proceed from the Step54 evidence/provenance perspective.
10. No hard blocker remains for Step55 inside Step54C. Optional future work would only be paper-curation work, not evidence-source repair.

## Full Export Outcome

- Full export directory: `results_stage54_rce_evidence_interpretability/full/`
- Verified results directory: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Verified checkpoint: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt`
- Verified model type: `RCE_MIL_BiomedCLIP`
- Exported slide count: `194`

## Minimal Validation Commands

- `bash -n scripts/experiments/run_stage54_export_rce_evidence.sh`
- `MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh`
- `python scripts/analysis/build_stage54c_full_export_audit.py`
- `python scripts/analysis/build_stage54b_evidence_source_audit.py`
- `python scripts/analysis/build_stage54b_case_metadata.py`
- `python scripts/analysis/plot_stage54b_rce_evidence_figures.py`

## Outcome

- Step54C replaced the full-side preferred evidence source with a verified Step54 direct export.
- The refreshed Step54B package now supports matched direct-export-based interpretability comparisons between full and `wo_csg`, with claims still kept deliberately conservative.
