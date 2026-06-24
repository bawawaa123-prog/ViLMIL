# Step55 Quality Audit

## Status Summary
- Blocker count: `0`
- Warning count: `2`

## Warnings
- Legacy Step54 docs still describe Stage32 fallback as the current full-equivalent source.
- Some Step54B caption/index text still uses the pre-Step54C same-source mismatch wording.

## Audit Table

| check_id | status | severity | details | evidence | action |
| --- | --- | --- | --- | --- | --- |
| stage23_main_results_exist | pass | info | Stage23 main result.csv exists and was parsed. | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv | none |
| step52_full_matches_stage23 | pass | info | Step52 full metrics match Stage23 exactly across AUC/ACC/F1/BACC/PR-AUC. | results_stage52_rce_core_ablation/full_rce_v4_csg_rq16_5fold_e20_s1/result.csv vs results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv | none |
| step52b_ablation_table_exists | pass | info | Step52B ablation markdown and CSV exist. | docs/stage52b_rce_ablation_table_paper.md; docs/stage52b_rce_ablation_table_paper.csv | none |
| step53_package_exists | pass | info | Step53 package exists. | docs/stage53_rce_paper_ready_package | none |
| step54_package_exists | pass | info | Step54 package exists. | docs/stage54_rce_evidence_interpretability_package | none |
| step54b_package_exists | pass | info | Step54B package exists. | docs/stage54b_rce_evidence_figure_polish | none |
| step54c_package_exists | pass | info | Step54C package exists. | docs/stage54c_rce_full_direct_export_refresh | none |
| full_evidence_dir_exists | pass | info | Full evidence directory exists. | results_stage54_rce_evidence_interpretability/full | none |
| full_evidence_points_to_stage23_main | pass | info | Full evidence manifest points to the Stage23 main run and s_0 checkpoint. | results_stage54_rce_evidence_interpretability/full/stage32_manifest.json | none |
| full_evidence_model_type_rce | pass | info | Full evidence manifest model_type/requested_model_type are both RCE_MIL_BiomedCLIP. | results_stage54_rce_evidence_interpretability/full/stage32_manifest.json | none |
| wo_csg_evidence_exists | pass | info | wo_csg evidence directory exists. | results_stage54_rce_evidence_interpretability/wo_csg | none |
| stage54b_case_level_metadata_exists | pass | info | stage54b_case_level_metadata.csv exists. | results_stage54_rce_evidence_interpretability/stage54b_case_level_metadata.csv | none |
| stage54b_figure_index_exists | pass | info | stage54b figure index CSV and markdown exist. | docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.csv; docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.md | none |
| stage54b_seven_figures_exist | pass | info | All 7 Step54B figures exist. | docs/stage54b_rce_evidence_figure_polish/figures | none |
| uses_stage32_fallback_false | pass | info | Step54B figure index and case metadata both record uses_stage32_fallback=False. | docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.csv; results_stage54_rce_evidence_interpretability/stage54b_case_level_metadata.csv | none |
| historical_docs_still_reference_stage32_as_current_source | warning | warning | Legacy Step54 docs still describe Stage32 fallback as the current full-equivalent source. | docs/stage54_rce_evidence_interpretability_package/stage54_evidence_resource_audit.md:24 | Do not use the old Step54 package as the paper-facing preferred source; prefer Step54B/Step54C/Step55 docs. |
| positive_statistically_significant_claims_present | pass | info | No positive statistically significant claim was found in the paper-facing docs that were audited. | docs/stage53_rce_paper_ready_package/stage53_rce_method_claim_draft.md; docs/stage53_rce_paper_ready_package/stage53_rce_main_results.md; docs/stage53_rce_paper_ready_package/stage53_rce_paper_ready_summary.md; docs/stage54b_rce_evidence_figure_polish/stage54b_summary.md; docs/stage54b_rce_evidence_figure_polish/stage54b_paper_figure_caption_drafts.md; docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.md; docs/stage54c_rce_full_direct_export_refresh/stage54c_summary.md; docs/stage54c_rce_full_direct_export_refresh/stage54c_provenance_change_report.md; docs/stage54c_rce_full_direct_export_refresh/stage54c_claims_update.md | none |
| positive_localization_claims_present | pass | info | No positive localization benchmark or pathologist-level claim was found in the paper-facing docs that were audited. | docs/stage53_rce_paper_ready_package/stage53_rce_method_claim_draft.md; docs/stage53_rce_paper_ready_package/stage53_rce_main_results.md; docs/stage53_rce_paper_ready_package/stage53_rce_paper_ready_summary.md; docs/stage54b_rce_evidence_figure_polish/stage54b_summary.md; docs/stage54b_rce_evidence_figure_polish/stage54b_paper_figure_caption_drafts.md; docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.md; docs/stage54c_rce_full_direct_export_refresh/stage54c_summary.md; docs/stage54c_rce_full_direct_export_refresh/stage54c_provenance_change_report.md; docs/stage54c_rce_full_direct_export_refresh/stage54c_claims_update.md | none |
| deg_hcrc_prarc_main_model_claim_present | pass | info | No audited paper-facing doc places DEG/HCRC/PRARC into the frozen main-model definition. | docs/stage53_rce_paper_ready_package/stage53_rce_method_claim_draft.md; docs/stage53_rce_paper_ready_package/stage53_rce_main_results.md; docs/stage53_rce_paper_ready_package/stage53_rce_paper_ready_summary.md | none |
| outdated_same_source_caption_wording_present | warning | warning | Some Step54B caption/index text still uses the pre-Step54C same-source mismatch wording. | docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.md:8; docs/stage54b_rce_evidence_figure_polish/stage54b_paper_figure_caption_drafts.md:13; docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.md:8; docs/stage54b_rce_evidence_figure_polish/stage54b_paper_figure_caption_drafts.md:13 | If these old captions are reused, revise them to direct-export matched-comparison wording. |
