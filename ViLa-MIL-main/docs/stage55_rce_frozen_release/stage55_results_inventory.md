# Step55 Results Inventory

This inventory records the source artifacts used to define the Step55 frozen release.

| artifact_id | stage | artifact_type | path | exists | lock_recommended | role |
| --- | --- | --- | --- | --- | --- | --- |
| stage23_main_results_dir | Step23 | directory | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1 | True | True | Main frozen result directory. |
| stage23_main_result_csv | Step23 | csv | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv | True | True | Main metrics source. |
| stage23_main_fold_summary | Step23 | csv | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/fold_summary.csv | True | True | Fold-level test metrics. |
| stage23_main_checkpoint_s0 | Step23 | checkpoint | results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt | True | True | Full evidence manifest target. |
| stage52_ablation_dir | Step52 | directory | results_stage52_rce_core_ablation | True | True | Core ablation result directory. |
| stage52b_table_md | Step52B | markdown | docs/stage52b_rce_ablation_table_paper.md | True | True | Human-readable ablation table. |
| stage52b_table_csv | Step52B | csv | docs/stage52b_rce_ablation_table_paper.csv | True | True | Machine-readable ablation table. |
| stage52b_interpretation_md | Step52B | markdown | docs/stage52b_rce_ablation_interpretation.md | True | True | Interpretation guardrails. |
| stage53_package_dir | Step53 | directory | docs/stage53_rce_paper_ready_package | True | True | Paper-ready package. |
| stage54_package_dir | Step54 | directory | docs/stage54_rce_evidence_interpretability_package | True | True | Initial evidence package. |
| stage54b_package_dir | Step54B | directory | docs/stage54b_rce_evidence_figure_polish | True | True | Figure polish package. |
| stage54c_package_dir | Step54C | directory | docs/stage54c_rce_full_direct_export_refresh | True | True | Full direct export refresh package. |
| stage54_full_evidence_dir | Step54C | directory | results_stage54_rce_evidence_interpretability/full | True | True | Preferred full evidence source. |
| stage54_wo_csg_evidence_dir | Step54 | directory | results_stage54_rce_evidence_interpretability/wo_csg | True | True | Preferred wo_csg evidence source. |
| stage54b_case_metadata | Step54B/54C | csv | results_stage54_rce_evidence_interpretability/stage54b_case_level_metadata.csv | True | True | Matched-case metadata. |
| stage54b_figure_index | Step54B | csv | docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.csv | True | True | Figure provenance index. |
| main_model_file | Code | python | models/model_RCE_MIL_BiomedCLIP.py | True | True | Frozen main model definition. |
| main_training_script | Code | shell | scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh | True | True | Main 5-fold training command. |
| ablation_training_script | Code | shell | scripts/experiments/run_stage52_rce_core_ablation_5fold.sh | True | True | Core ablation training command. |
| evidence_export_script | Code | shell | scripts/experiments/run_stage54_export_rce_evidence.sh | True | True | Evidence export command. |
| main_py | Code | python | main.py | True | True | Training entrypoint. |
| core_utils_py | Code | python | utils/core_utils.py | True | True | Training orchestration. |
| stage55_build_script | Step55 | python | scripts/analysis/build_stage55_rce_frozen_release.py | True | True | Frozen release builder. |
