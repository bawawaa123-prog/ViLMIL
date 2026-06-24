# Stage54C Full Export Audit

## Key Findings
- Full direct export is present and verified against the full RCE main-model lineage.
- Full export results_dir: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Full export checkpoint: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt`
- Full export model_type/requested_model_type: `RCE_MIL_BiomedCLIP` / `RCE_MIL_BiomedCLIP`
- Full and wo_csg slide/concept tables are field-aligned for matched interpretability comparisons.
- wo_csg top CSG pair table is empty/unreadable, which is expected because cross-scale graph is disabled.

## Audit Table
| check_name | status | details |
| --- | --- | --- |
| full_export_directory_exists | True | results_stage54_rce_evidence_interpretability/full |
| full_manifest_exists | True | results_stage54_rce_evidence_interpretability/full/stage32_manifest.json |
| full_slide_summary_exists | True | results_stage54_rce_evidence_interpretability/full/stage32_slide_evidence_summary.csv |
| full_top_concepts_exists | True | results_stage54_rce_evidence_interpretability/full/stage32_top_concepts_long.csv |
| full_top_csg_pairs_exists | True | results_stage54_rce_evidence_interpretability/full/stage32_top_csg_pairs.csv |
| full_num_slides_reasonable | True | 194 |
| full_model_type_rce | True | requested=RCE_MIL_BiomedCLIP; model=RCE_MIL_BiomedCLIP |
| full_points_to_stage23_or_stage52_full | True | results_dir=/home/ljh/ViLMIL/ViLa-MIL-main/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1; ckpt_path=/home/ljh/ViLMIL/ViLa-MIL-main/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt |
| full_still_relies_on_stage32_fallback | False | false means Step54 full direct export is usable as preferred source |
| slide_fields_align_with_wo_csg | True | full and wo_csg slide summaries have identical columns |
| concept_fields_align_with_wo_csg | True | full and wo_csg concept tables have identical columns |
| csg_table_context | True | full has non-empty csg pair table; wo_csg csg table may be empty because the module is disabled |
| full_direct_export_verified | True | verified if manifest/model/checkpoint/results_dir all match an accepted full-RCE source |

## Interpretation
- Stage32 legacy fallback is no longer required as the preferred full-side source.
- This audit checks export provenance and field readiness only; it does not claim any new training-time statistical result.
- Legacy fallback directory retained for reference: `results_stage32/stage32_rce_v4_csg_evidence_export`
