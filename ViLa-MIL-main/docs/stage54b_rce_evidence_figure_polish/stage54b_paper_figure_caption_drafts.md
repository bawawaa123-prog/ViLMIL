# Stage54B Paper Figure Caption Drafts

## stage54b_region_concept_heatmap.png
single-case region-concept heatmap. Source: stage32_legacy_deg_export_fallback. Context: case, slide=25034940B12, fold=0, true=NonAdenocarcinoma, pred=NonAdenocarcinoma, confidence=1.000. Example region-concept evidence can be inspected at low and high scales. This figure is not a localization benchmark or expert annotation surrogate. Provenance note: Single-case full-side evidence uses fallback provenance.

## stage54b_low_high_concept_comparison.png
single-case low-vs-high concept bar chart. Source: stage32_legacy_deg_export_fallback. Context: case, slide=25034940B12, fold=0, true=NonAdenocarcinoma, pred=NonAdenocarcinoma, confidence=1.000. Concept contributions from low and high scales can be compared within one evidence example. This figure cannot prove systematic localization quality or statistically significant benefit. Provenance note: Caption should state fallback provenance for the full-side example.

## stage54b_correct_case_evidence_components.png
single-case component decomposition. Source: stage32_legacy_deg_export_fallback. Context: case, slide=25034940B12, fold=0, true=NonAdenocarcinoma, pred=NonAdenocarcinoma, confidence=1.000. Evidence components can be decomposed into low-scale, high-scale, visual residual, and CSG terms. This figure cannot prove that the model localizes pathology at pathologist level. Provenance note: Full-side component plot uses fallback evidence.

## stage54b_full_vs_wo_csg_concept_ranking_delta.png
matched-case concept ranking delta. Source: full=stage32_legacy_deg_export_fallback; wo_csg=stage54_wo_csg_direct_export. Context: case, slide=25024855B4, fold=0, true=NonAdenocarcinoma, pred=full=NonAdenocarcinoma; wo_csg=NonAdenocarcinoma, confidence=full=0.996; wo_csg=0.777. In a selected matched case, CSG may alter concept ranking and confidence behavior. This cross-source example does not establish a fully same-source evidence comparison or statistical significance. Provenance note: Full side uses fallback while wo_csg side uses direct Step54 export.

## stage54b_failure_case_evidence_components.png
failure-case component decomposition. Source: stage32_legacy_deg_export_fallback. Context: case, slide=2485275-B2, fold=0, true=Adenocarcinoma, pred=NonAdenocarcinoma, confidence=1.000. Failure cases can be inspected for component imbalance or conflicting evidence. This figure cannot prove the exact clinical reason for the model error. Provenance note: Failure example uses fallback provenance on the full side.

## stage54b_csg_benefit_case_comparison.png
selected full-vs-wo_csg comparison. Source: full=stage32_legacy_deg_export_fallback; wo_csg=stage54_wo_csg_direct_export. Context: case, slide=25015422B, fold=0, true=Adenocarcinoma, pred=full=Adenocarcinoma; wo_csg=NonAdenocarcinoma, confidence=full=0.901; wo_csg=0.754. A selected case suggests CSG can improve margin or confidence behavior. This single case cannot support large ACC gains or a same-source causal proof. Provenance note: Comparison mixes full fallback evidence with direct wo_csg export.

## stage54b_aggregate_top_concept_frequency.png
aggregate concept frequency bar chart. Source: stage32_legacy_deg_export_fallback. Context: aggregate, slide=aggregate_figure, fold=aggregate_figure, true=aggregate_figure, pred=aggregate_predicted_class_frequency, confidence=aggregate_figure. Frequently recurring top predicted-class concepts can be summarized over the audited export. This aggregate count does not measure localization accuracy or cross-model statistical significance. Provenance note: Aggregate figure inherits full-side provenance.
