# Stage54 Evidence Resource Audit

> Historical note: this Step54 package records the pre-Step54C audit state. It is superseded for paper-facing full-model provenance by `results_stage54_rce_evidence_interpretability/full/`, whose manifest now points to `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1` and `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt` with `model_type=RCE_MIL_BiomedCLIP`.

## Audit Table

| resource | path | status | note |
| --- | --- | --- | --- |
| stage53_figure_plan | docs/stage53_rce_paper_ready_package/stage53_rce_figure_plan.md | ready | paper figure plan from Step53 |
| stage32_export_script | scripts/analysis/export_stage32_rce_v4_csg_evidence.py | ready | existing evidence export implementation |
| stage32_run_script | scripts/analysis/run_stage32_export_evidence.sh | ready | existing shell wrapper for evidence export |
| stage32_full_evidence_manifest | results_stage32/stage32_rce_v4_csg_evidence_export/stage32_manifest.json | ready | existing main-model-equivalent evidence manifest |
| stage32_full_slide_summary | results_stage32/stage32_rce_v4_csg_evidence_export/stage32_slide_evidence_summary.csv | ready | slide-level evidence summary for fold0/test |
| stage32_full_top_concepts | results_stage32/stage32_rce_v4_csg_evidence_export/stage32_top_concepts_long.csv | ready | concept-level evidence details |
| stage32_full_top_csg_pairs | results_stage32/stage32_rce_v4_csg_evidence_export/stage32_top_csg_pairs.csv | ready | cross-scale concept interaction details |
| stage39_final_evidence_package | results_stage39/final_evidence_package | ready | previous paper-ready evidence package |
| stage39_case_summary | results_stage39/final_evidence_package/stage39_fixed_regressed_persistent_cases.csv | ready | previous selected cases from another comparison branch |
| full_stage23_checkpoint_fold0 | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt | ready | direct full-RCE checkpoint for export |
| wo_csg_checkpoint_fold0 | results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1/s_0_checkpoint.pt | ready | matched w/o CSG checkpoint for export |
| expected_full_stage54_export | results_stage54_rce_evidence_interpretability/full/stage32_slide_evidence_summary.csv | missing | future Step54 full export location |
| expected_wo_csg_stage54_export | results_stage54_rce_evidence_interpretability/wo_csg/stage32_slide_evidence_summary.csv | ready | future Step54 w/o CSG export location |

## Reuse Decision

- Step54 reuses `scripts/analysis/export_stage32_rce_v4_csg_evidence.py` as the main evidence export backend.
- Existing `results_stage32/stage32_rce_v4_csg_evidence_export/` is documented here as the historical pre-Step54C full-model-equivalent evidence source for fold0/test.
- The current preferred full evidence source after Step54C is `results_stage54_rce_evidence_interpretability/full/`.
- Stage39 summaries are reused as reference assets, but they compare skeleton vs low-high consistency rather than full vs `wo_csg`.
- New Step54 export orchestration focuses on direct Stage23 full RCE and Step52 `wo_csg`.

## Current Gaps

- Historical Step54 note: before Step54C, the full model evidence export existed only as the earlier Stage32-equivalent package.
- Current provenance status: the preferred full direct export now exists under `results_stage54_rce_evidence_interpretability/full/` and points to the Stage23 main model `RCE-v4-CSG-a01-rq16`.
- `wo_csg` evidence export is now available under `results_stage54_rce_evidence_interpretability/wo_csg/`.
- Region thumbnails or image overlays are not present in the audited resources, so image-based qualitative figures remain pending.

## Warnings

- none
