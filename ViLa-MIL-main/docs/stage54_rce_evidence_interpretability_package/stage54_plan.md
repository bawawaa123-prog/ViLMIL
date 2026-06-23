# Stage54 Plan

## Inputs

- `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- `results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1`
- `results_stage32/stage32_rce_v4_csg_evidence_export`
- `results_stage39/final_evidence_package`
- `docs/stage53_rce_paper_ready_package/stage53_rce_figure_plan.md`

## Outputs

- `results_stage54_rce_evidence_interpretability/stage54_case_level_summary.csv`
- `docs/stage54_rce_evidence_interpretability_package/stage54_case_selection.csv`
- `docs/stage54_rce_evidence_interpretability_package/stage54_case_selection.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_evidence_resource_audit.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_rce_evidence_report.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_interpretability_claims_to_make_and_avoid.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_paper_figure_caption_drafts.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_figure_index.md`

## Run Commands

```bash
python scripts/analysis/build_stage54_rce_evidence_interpretability_package.py
bash -n scripts/experiments/run_stage54_export_rce_evidence.sh
python scripts/analysis/plot_stage54_rce_evidence_figures.py
```

## Potentially Time-Consuming Steps

- Evidence export for Stage23 full or Step52 `wo_csg` can be inference-heavy, especially if run for multiple folds.
- Figure generation itself is light, but it depends on whether the export files already exist.

## Recommended Order

1. Build the Step54 package and audit available resources.
2. Run `MODE=dry_run` in the export script to verify commands.
3. Export `wo_csg` evidence first, because that is the main missing comparison resource.
4. Export direct Stage23 full evidence into the Step54 directory only if a direct-RCE copy is required for packaging symmetry.
5. Re-run the figure script after evidence export.

## Relation To Step53

- Step53 fixed the paper main model and the allowed claims.
- Step54 converts that stable result package into an evidence and interpretability package.

## Relation To Step55

- Step54 remains fully within the RCE main-model scope.
- Step55 should audit DEG any-on purity separately and should not be mixed into the Step54 main-model narrative.
