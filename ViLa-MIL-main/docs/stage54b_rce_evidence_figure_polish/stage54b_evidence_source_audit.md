# Stage54B Evidence Source Audit

## Key Findings
- Full direct export under `results_stage54_rce_evidence_interpretability/full/` is present and verified against `RCE-v4-CSG-a01-rq16`.
- `wo_csg` direct export is present and points to `results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1`.
- The Stage32 legacy fallback remains available only as backup and should no longer be the preferred full-side source.
- Full vs `wo_csg` evidence-level figures can now be refreshed from Step54 direct exports on both sides.

## Manual Export Commands
- `MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh`
- `MODE=wo_csg bash scripts/experiments/run_stage54_export_rce_evidence.sh`
- `MODE=both bash scripts/experiments/run_stage54_export_rce_evidence.sh`
- These commands were audited only; Step54B did not execute them.

## Audit Table
| source_id | variant | source_role | evidence_dir | provenance_status | matches_expected_results_dir | matches_expected_checkpoint | matches_expected_model_type | num_slides | uses_stage32_fallback | fallback_consumers | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_direct_expected | full | expected_step54_direct_export | results_stage54_rce_evidence_interpretability/full | direct_export_verified | True | True | True | 194 | False | none | none |
| full_stage32_fallback | full | legacy_fallback_candidate | results_stage32/stage32_rce_v4_csg_evidence_export | fallback_reuse | False | False | False | 194 | True | stage54b_case_level_metadata.csv; stage54b_region_concept_heatmap.png; stage54b_low_high_concept_comparison.png; stage54b_full_vs_wo_csg_concept_ranking_delta.png(full side only); stage54b_correct_case_evidence_components.png; stage54b_failure_case_evidence_components.png; stage54b_csg_benefit_case_comparison.png(full side only); stage54b_aggregate_top_concept_frequency.png | fallback source only; not a direct Step54 full export; manifest results_dir does not match expected variant directory; manifest ckpt_path does not match expected checkpoint; manifest model_type does not match expected model type |
| wo_csg_direct_export | wo_csg | step54_direct_export | results_stage54_rce_evidence_interpretability/wo_csg | direct_export_verified | True | True | True | 194 | False | none | top csg pair table missing or unreadable; CSG-related columns exist but contain no non-null values |

## Interpretation Guardrails
- Preferred full-side provenance is now the Step54 direct export.
- Full vs `wo_csg` evidence comparisons can be described as matched Step54 direct-export illustrations, but not as training-significance evidence.
- Single-case evidence illustrations from the fallback may still be used for interpretability examples if provenance is stated explicitly.
- CSG-related fields in the `wo_csg` export are structurally present but expected to be empty because the module is disabled.
