# Stage54 RCE Evidence Report

## 1. Purpose

Step54 packages interpretability evidence for the fixed RCE main model so that the paper can support the claims of region-concept evidence reasoning and CSG-driven evidence ranking / cross-scale concept interaction.

## 2. How Interpretability Supports The Paper

- RCE should be explained as moving from slide-level vision-language alignment to region-concept evidence reasoning.
- Evidence exports should show which concepts and regions support the prediction.
- Matched full vs `wo_csg` comparisons should be used to explain CSG mainly through ranking, confidence, and cross-scale interaction rather than through ACC inflation.

## 3. Evidence Resources Found

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

## 4. What Can Be Generated Now

- Prediction-level matched case selection across all 5 folds is available now from Stage23 full and Step52 `wo_csg`.
- Fold0/test concept-level evidence plots can be generated now from the existing Stage32 export.
- Aggregate concept-frequency and concept-contribution plots can be generated now from the existing Stage32 long-form concept table.
- Full vs `wo_csg` evidence-ranking figures can now be generated from the matched export files.

## 5. Missing Evidence

- Matched `wo_csg` evidence is available for fold0/test, but not yet for folds 1-4.
- No Step54-format direct full export has been generated yet; Step54 currently reuses the earlier Stage32 full-equivalent evidence package.
- No region image crops or overlay assets have been found for direct pathology visual panels.
- No direct fold1-4 evidence exports have been found; the audited full evidence package is fold0/test only.

## 6. Can Full vs w/o CSG Be Compared As Matched Cases?

- Prediction-level matched comparison: yes.
- Evidence-level matched comparison: yes.

## 7. Correct / Failure Case Summary

- Full correct cases across all folds: `893`
- Full wrong cases across all folds: `75`
- Cases where full is correct but `wo_csg` is wrong: `14`
- Cases with the same prediction but a notable confidence shift: `62`

These counts support a targeted qualitative analysis plan instead of an exhaustive slide dump.

## 8. Writing Advice

- Use current evidence to support interpretability as an analysis view, not as a strict localization benchmark.
- Use CSG comparisons to discuss evidence ranking and concept interaction.
- Avoid claiming that the current heatmaps are equivalent to pathologist annotations or causal explanations.
