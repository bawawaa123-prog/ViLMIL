# Stage54C Provenance Change Report

## Before Step54C

- Step54B preferred full evidence source was `results_stage32/stage32_rce_v4_csg_evidence_export/`.
- That source was a legacy fallback tied to `results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1`, not a direct export of `RCE-v4-CSG-a01-rq16`.
- `wo_csg` evidence already came from `results_stage54_rce_evidence_interpretability/wo_csg/` and was a verified Step54 direct export.
- Full vs `wo_csg` evidence figures therefore had to be described as provenance-mismatched comparisons.

## After Step54C

- Preferred full evidence source is now `results_stage54_rce_evidence_interpretability/full/`.
- The refreshed full manifest points to:
  - `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
  - `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt`
- `model_type` and `requested_model_type` are both `RCE_MIL_BiomedCLIP`.
- Full and `wo_csg` slide-level and concept-level export tables are field-aligned.

## Fallback Status

- `results_stage32/stage32_rce_v4_csg_evidence_export/` is still retained for reference.
- It is no longer the preferred full-side source in the refreshed Step54B package.
- Current refreshed Step54B metadata reports:
  - `evidence_source_full = stage54_full_direct_export`
  - `uses_stage32_fallback = False`

## Figures That Switched From Fallback To Direct Export

- `stage54b_region_concept_heatmap.png`
- `stage54b_low_high_concept_comparison.png`
- `stage54b_correct_case_evidence_components.png`
- `stage54b_failure_case_evidence_components.png`
- `stage54b_aggregate_top_concept_frequency.png`
- `stage54b_full_vs_wo_csg_concept_ranking_delta.png`
- `stage54b_csg_benefit_case_comparison.png`

## Figures Still Carrying Provenance Limitations

- No refreshed Step54B figure currently depends on the Stage32 legacy fallback as its preferred source.
- Provenance limitations that remain are interpretability-scope limitations rather than source-mismatch limitations:
  - figures remain selected-case or aggregate illustrations;
  - figures are not training-significance evidence;
  - figures are not localization benchmarks.

## Comparison Wording After Refresh

- It is now acceptable to describe full vs `wo_csg` evidence figures as direct-export-based matched interpretability comparisons.
- It is still not acceptable to describe them as statistical proof of training-time superiority or localization-quality improvement.
