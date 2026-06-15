# CODEX Handoff

## Step1: RCE-MIL Base Implementation

- Modified files:
  - `main.py`
  - `utils/core_utils.py`
  - `models/model_RCE_MIL_BiomedCLIP.py`
- Added model:
  - `RCE_MIL_BiomedCLIP`
- Key behavior:
  - Aggregates patch features into region features with learnable region queries.
  - Computes region-concept evidence with BiomedCLIP concept prompt features.
  - Supports `scale_mode` in `dual`, `low_only`, and `high_only`.
- Compatibility:
  - Existing `ViLa_MIL_BiomedCLIP` and PEPS logic were kept unchanged.
- Validation:
  - `py_compile` passed for `main.py`, `utils/core_utils.py`, and `models/model_RCE_MIL_BiomedCLIP.py`.

## Step2: RCE-MIL Smoke Script

- Modified files:
  - `scripts/experiments/run_stage9_rce_smoke.sh`
- Added script behavior:
  - Runs `RCE_MIL_BiomedCLIP`
  - Uses `fold0`, `1 epoch`, and `--testing`
  - Keeps environment-variable overrides for paths and Python binary
- Validation:
  - `bash -n` passed
  - No backslash line continuations were used
- Next suggested step: run the smoke script and confirm the full train/val/test loop executes.

## Step3: RCE-MIL Smoke Run

- Modified files:
  - No code changes; executed the Step2 script only
- Run artifacts:
  - Log: `results_smoke/logs/rce_mil_smoke_step3.log`
  - Results: `results_smoke/rce_mil_smoke_s1`
- Outcome:
  - Smoke run completed successfully through train / val / test
  - Final test metrics: `AUC=0.6000`, `ACC=0.7895`, `F1=0.4412`, `Balanced ACC=0.5000`
  - `Sensitivity=0.0000`, `Specificity=1.0000`
- Interpretation:
  - The training pipeline was functional, but the small `--testing` setup was not suitable for judging learning quality.
- Next suggested step: run a non-testing fold0 pilot.

## Step4: RCE-MIL Fold0 Pilot

- Modified files:
  - `scripts/experiments/run_stage9_rce_pilot_fold0.sh`
- Added script behavior:
  - Runs `RCE_MIL_BiomedCLIP`
  - Uses `fold0`, `20 epochs`, and no `--testing`
  - Keeps environment-variable overrides for paths and Python binary
- Validation:
  - `bash -n` passed
  - No backslash line continuations were used
- Run artifacts:
  - Log: `results_stage9/logs/rce_mil_pilot_fold0_step4.log`
  - Results: `results_stage9/rce_mil_pilot_fold0_s1`
- Outcome:
  - Pilot completed successfully
  - Best validation AUC reached `0.9573` at epoch 12
  - Final test metrics: `AUC=0.9743`, `ACC=0.9021`, `F1=0.8897`, `Balanced ACC=0.8854`
  - `Sensitivity=0.8333`, `Specificity=0.9375`
- Interpretation:
  - Base RCE-MIL learned successfully on fold0 without class collapse.

## Step5: RCE-MIL 5-Fold Formal Evaluation

- Modified files:
  - `scripts/experiments/run_stage9_rce_5fold.sh`
- Added script behavior:
  - Runs `RCE_MIL_BiomedCLIP`
  - Uses `5 folds`, `20 epochs`, and no `--testing`
  - Keeps environment-variable overrides for paths and Python binary
- Validation:
  - `bash -n` passed
  - No backslash line continuations were used
- Run artifacts:
  - Log: `results_stage9/logs/rce_mil_5fold_e20_step5.log`
  - Results: `results_stage9/rce_mil_5fold_e20_s1`
- Outcome:
  - All 5 folds completed successfully
  - Mean test metrics: `AUC=0.9633 +/- 0.0182`, `ACC=0.8822 +/- 0.0264`, `F1=0.8667 +/- 0.0298`
  - `Balanced ACC=0.8613 +/- 0.0316`, `Sensitivity=0.7948 +/- 0.0646`, `Specificity=0.9278 +/- 0.0355`
  - `PR-AUC=0.9275 +/- 0.0388`
- Interpretation:
  - No fold showed `sensitivity=0`, and no clear class-collapse behavior was observed.
- Next suggested step: improve logit calibration and class decision boundaries while keeping base RCE-MIL behavior available.

## Step6: RCE-MIL v2 Concept Prior + Logit Calibration

- Modified files:
  - `main.py`
  - `utils/core_utils.py`
  - `models/model_RCE_MIL_BiomedCLIP.py`
  - `docs/CODEX_HANDOFF.md`
- Added args:
  - `--rce_use_logit_calibration`
  - `--rce_use_concept_prior`
  - `--rce_logit_scale_init`
  - `--rce_concept_prior_strength`
- Default is fully backward compatible: all new options are off by default, so base `RCE_MIL_BiomedCLIP` behavior is unchanged.
- Next suggested step: run a fold0 pilot comparison against base RCE-MIL.

## Step7: RCE-MIL v2 Fold0 Pilot Ablation Script

- Modified files:
  - `scripts/experiments/run_stage9_rce_v2_pilot_ablation.sh`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/experiments/run_stage9_rce_v2_pilot_ablation.sh`
- Supported `VARIANT` values:
  - `calib_only`
  - `prior_only`
  - `prior_calib`
  - `all`
- Behavior:
  - Runs fold0 pilot ablations for the Step6 optional enhancements only.
  - Uses bash arrays and keeps environment-variable overrides for paths and Python binary.
- This step did not run any long training.
- Next suggested step: the user manually runs the script and shares the results for comparison against base RCE-MIL.

## Step8: RCE-MIL v2 5-Fold Evaluation Script

- Modified files:
  - `scripts/experiments/run_stage9_rce_v2_5fold.sh`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/experiments/run_stage9_rce_v2_5fold.sh`
- Supported `VARIANT` values:
  - `prior_calib`
  - `prior_only`
  - `all`
- Default `VARIANT`:
  - `prior_calib`
- Behavior:
  - Runs 5-fold evaluation for the two strongest Step7 candidates only.
  - Uses bash arrays and keeps environment-variable overrides for paths and Python binary.
- This step did not run any long training.
- Next suggested step: the user manually runs the script and shares the 5-fold results.

## Step9: RCE-MIL v3 Residual Visual Evidence Branch

- Modified files:
  - `main.py`
  - `utils/core_utils.py`
  - `models/model_RCE_MIL_BiomedCLIP.py`
  - `docs/CODEX_HANDOFF.md`
- Added args:
  - `--rce_use_visual_residual`
  - `--rce_visual_residual_init`
- Default is fully backward compatible: the visual residual branch is off by default, so base and v2 `RCE_MIL_BiomedCLIP` behavior is unchanged.
- Residual visual evidence is added after concept logits are fused and before optional logit calibration.
- Next suggested step: run a fold0 pilot comparison between `prior_calib` and `prior_calib + visual residual`.

## Step10: RCE-MIL v3 Visual Residual Fold0 Pilot Script

- Modified files:
  - `scripts/experiments/run_stage9_rce_v3_pilot_visual_residual.sh`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/experiments/run_stage9_rce_v3_pilot_visual_residual.sh`
- Supported `VARIANT` values:
  - `prior_calib_vr_a01`
  - `prior_calib_vr_a005`
  - `prior_calib_vr_a02`
  - `all`
- Default `VARIANT`:
  - `prior_calib_vr_a01`
- Behavior:
  - Runs fold0 pilot experiments for `prior_calib + visual_residual` with three alpha initializations.
  - Uses bash arrays and keeps environment-variable overrides for paths and Python binary.
- This step did not run any long training.
- Next suggested step: the user manually runs the script and shares the fold0 results.

## Step11: RCE-MIL v3 Visual Residual 5-Fold Script

- Modified files:
  - `scripts/experiments/run_stage9_rce_v3_5fold_visual_residual.sh`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/experiments/run_stage9_rce_v3_5fold_visual_residual.sh`
- Supported `VARIANT` values:
  - `vr_a01`
  - `vr_a005`
  - `all`
- Default `VARIANT`:
  - `vr_a01`
- Behavior:
  - Runs 5-fold evaluation for the two strongest Step10 visual residual candidates.
  - Uses bash arrays and keeps environment-variable overrides for paths and Python binary.
- This step did not run any long training.
- Next suggested step: the user manually runs the script and shares the 5-fold results.

## Step12: Stage9 RCE Final Analysis

- Modified files:
  - `scripts/analysis/build_stage9_rce_final_analysis.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/build_stage9_rce_final_analysis.py`
- Default compatibility:
  - Yes. This step only reads existing result CSV files and writes a new analysis directory.
- Training / 5-fold / feature extraction:
  - No training run
  - No 5-fold run
  - No feature extraction
- Generated outputs:
  - `results_stage9/stage9_rce_final_analysis/rce_stage9_main_comparison.csv`
  - `results_stage9/stage9_rce_final_analysis/rce_stage9_main_comparison.md`
  - `results_stage9/stage9_rce_final_analysis/rce_stage9_metric_deltas.csv`
  - `results_stage9/stage9_rce_final_analysis/rce_stage9_final_report.md`
- Recommendation captured in report:
  - Current recommended RCE variant is `RCE-MIL v3 prior_calib + visual_residual_init=0.05` (`RCE-v3-VR-a005`).
- Next suggested step:
  - Step13: RCE region-concept evidence export

## Step13: RCE Region-Concept Evidence Export

- Modified files:
  - `models/model_RCE_MIL_BiomedCLIP.py`
  - `scripts/analysis/export_stage13_rce_evidence.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/export_stage13_rce_evidence.py`
- Added export-only model attributes:
  - `last_low_region_concept_sim`
  - `last_high_region_concept_sim`
  - `last_low_region_features`
  - `last_high_region_features`
- Default compatibility:
  - Yes. Forward return values, logits, loss, `Y_prob`, and `Y_hat` are unchanged; new attributes are detach/cpu export buffers only.
- Training / 5-fold / feature extraction:
  - No training run
  - No 5-fold run
  - No feature extraction
- Small-sample export run:
  - Attempted `--fold 0 --split test --max_slides 10`
  - Script exited safely with warnings because local BiomedCLIP cache / HF access was unavailable
- Generated outputs:
  - `results_stage9/stage13_rce_evidence_export/stage13_rce_evidence_export_report.md`
  - When checkpoint + data + BiomedCLIP cache are available, the script also writes:
    - `results_stage9/stage13_rce_evidence_export/slide_prediction_evidence.csv`
    - `results_stage9/stage13_rce_evidence_export/slide_top_concepts.csv`
    - `results_stage9/stage13_rce_evidence_export/region_concept_evidence.pkl`
- Next suggested step:
  - Step14: concept-class graph or evidence visualization

## Step13c: Stage13 Evidence Quality Check

- Modified files:
  - `scripts/analysis/check_stage13_rce_evidence_quality.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/check_stage13_rce_evidence_quality.py`
- Checked evidence files:
  - `results_stage9/stage13_rce_evidence_export/slide_prediction_evidence.csv`
  - `results_stage9/stage13_rce_evidence_export/slide_top_concepts.csv`
  - `results_stage9/stage13_rce_evidence_export/region_concept_evidence.pkl`
  - `results_stage9/stage13_rce_evidence_export/stage13_rce_evidence_export_report.md`
- Generated quality-check outputs:
  - `results_stage9/stage13_rce_evidence_export/stage13_rce_evidence_quality_report.md`
  - `results_stage9/stage13_rce_evidence_export/stage13_rce_evidence_quality_summary.csv`
  - `results_stage9/stage13_rce_evidence_export/stage13_top_concept_frequency.csv`
- Training / 5-fold / feature extraction:
  - No training run
  - No 5-fold run
  - No feature extraction
- Outcome:
  - Small-sample evidence files were aligned by `slide_id`, tensor shapes were consistent, and no NaN/Inf issues were detected.
- Next suggested step:
  - Step14: concept-class graph or evidence visualization, or a larger fold0 export if broader evidence coverage is needed first

## Step14: Post-hoc Concept-Class Evidence Graph

- Modified files:
  - `scripts/analysis/build_stage14_concept_class_graph.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/build_stage14_concept_class_graph.py`
- Read Step13d evidence files:
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_prediction_evidence.csv`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_top_concepts.csv`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/region_concept_evidence.pkl`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/stage13_rce_evidence_quality_report.md`
- Generated graph outputs:
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_edges.csv`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_nodes.csv`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_graph.json`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_report.md`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_summary.csv`
- Training / 5-fold / feature extraction:
  - No training run
  - No 5-fold run
  - No feature extraction
- Outcome:
  - Built a post-hoc concept-to-class evidence graph using evidence score, prompt weight, rank, label, and prediction correctness rather than frequency alone.
- Next suggested step:
  - Step15: evidence visualization, or a learnable concept-class graph prototype if you want to move beyond post-hoc analysis

## Step15: RCE Evidence Visualization

- Modified files:
  - `scripts/analysis/visualize_stage15_rce_evidence.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/visualize_stage15_rce_evidence.py`
- Read upstream inputs:
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_prediction_evidence.csv`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_top_concepts.csv`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/region_concept_evidence.pkl`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_edges.csv`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_graph.json`
- Generated visualization outputs:
  - `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_selected_slides.csv`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_slide_evidence_paths.csv`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_visualization_summary.csv`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_rce_evidence_visualization_report.md`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/figures/top_concept_class_edges_by_class.png`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/figures/top_concept_class_edges_by_scale.png`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/figures/selected_slide_probability_overview.png`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/figures/slide_*_predclass_region_concept_heatmap.png`
- Visualization behavior:
  - Selects representative correct high-confidence slides for each true label and includes a small number of incorrect high-confidence cases when available.
  - Builds `slide -> region -> concept -> class` evidence-path rows by combining Step13 concept evidence with Step14 edge strengths and predicted-class region peaks from the pickle tensors.
  - Produces post-hoc plots only; no model change, no training, and no feature extraction.
- Checks run:
  - `python -m py_compile scripts/analysis/visualize_stage15_rce_evidence.py`
  - `python scripts/analysis/visualize_stage15_rce_evidence.py`
- Outcome:
  - Step15 ran successfully on fold0 full-export inputs and produced `8` selected slides, `54` evidence-path rows, `3` overview PNG figures, and `8` slide-level heatmaps.
- Notes:
  - Runtime showed local matplotlib/font cache warnings in the sandbox environment, but the visualization outputs were still generated successfully.
- Next suggested step:
  - Review the selected failure-case slides and decide whether the next step should emphasize richer per-slide narrative summaries or more structured region-level ranking diagnostics.

## Step16: Failure-Case Evidence Narrative Summary

- Modified files:
  - `scripts/analysis/build_stage16_failure_case_narratives.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/build_stage16_failure_case_narratives.py`
- Read upstream inputs:
  - `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_selected_slides.csv`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_slide_evidence_paths.csv`
  - `results_stage9/stage15_rce_evidence_visualization_fold0/stage15_visualization_summary.csv`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_prediction_evidence.csv`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_top_concepts.csv`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_edges.csv`
- Generated narrative outputs:
  - `results_stage9/stage16_failure_case_narratives_fold0/stage16_selected_slide_narratives.csv`
  - `results_stage9/stage16_failure_case_narratives_fold0/stage16_failure_cases.csv`
  - `results_stage9/stage16_failure_case_narratives_fold0/stage16_success_cases.csv`
  - `results_stage9/stage16_failure_case_narratives_fold0/stage16_case_narrative_report.md`
  - `results_stage9/stage16_failure_case_narratives_fold0/stage16_case_narrative_summary.csv`
- Narrative behavior:
  - Generates a per-slide narrative summary for all Step15 selected slides with probabilities, selection reason, low/high-scale concept evidence, and strongest evidence paths.
  - Separates incorrect slides into a failure-case table with error direction, predicted-vs-true concept gaps, low/high-scale relation, and a machine-generated failure hypothesis.
  - Separates correct slides into a success-case table showing whether both scales support the predicted class and whether the strongest support aligns with Step14 top edges.
- Checks run:
  - `python -m py_compile scripts/analysis/build_stage16_failure_case_narratives.py`
  - `python scripts/analysis/build_stage16_failure_case_narratives.py`
- Outcome:
  - Step16 ran successfully on the existing fold0 Step15/13/14 artifacts and produced `8` slide narratives, `2` failure cases, and `6` success cases with no warnings.
  - Failure narratives highlighted one cross-scale wrong-class drift case (`Adenocarcinoma -> NonAdenocarcinoma`) and one high-scale-dominant conflict case (`NonAdenocarcinoma -> Adenocarcinoma`).
- Training / 5-fold / feature extraction:
  - No training run
  - No 5-fold run
  - No feature extraction
- Next suggested step:
  - Step17 learnable concept-class graph prototype, or Step17 cross-scale evidence conflict analysis if you want to focus on why one scale overrides the other in failure cases.

## Step17: Cross-Scale Evidence Conflict Analysis

- Modified files:
  - `scripts/analysis/analyze_stage17_cross_scale_conflicts.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/analyze_stage17_cross_scale_conflicts.py`
- Read upstream inputs:
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_prediction_evidence.csv`
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_top_concepts.csv`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_edges.csv`
  - `results_stage9/stage16_failure_case_narratives_fold0/stage16_failure_cases.csv`
  - `results_stage9/stage16_failure_case_narratives_fold0/stage16_selected_slide_narratives.csv`
- Generated outputs:
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_cross_scale_slide_scores.csv`
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_conflict_type_summary.csv`
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_failure_pattern_cases.csv`
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_scale_conflict_report.md`
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_cross_scale_summary.csv`
- Analysis behavior:
  - Aggregates slide-level scale/class support from top concept paths using `evidence_score * prompt_weight * edge_strength * (1 / rank)`.
  - Produces per-slide low/high support classes, signed margins, agreement flags, and conflict-type labels over the full fold0 test export.
  - Summarizes conflict-type distributions by correctness, true label, and predicted label, and maps Step16 failure cases back into the full-slide conflict taxonomy.
- Checks run:
  - `python -m py_compile scripts/analysis/analyze_stage17_cross_scale_conflicts.py`
  - `python scripts/analysis/analyze_stage17_cross_scale_conflicts.py`
- Outcome:
  - Step17 analyzed `194` fold0 test slides with `0` warnings.
  - Low/high support agreement rate was `0.4381`.
  - Incorrect slides were dominated by `consistent_wrong_class_drift` (`8/14`) and `high_scale_dominant_wrong` (`5/14`).
  - Step16 failure cases mapped cleanly:
    - `25001929B3` -> `consistent_wrong_class_drift`
    - `25023972B3` -> `high_scale_dominant_wrong`
- Training / 5-fold / feature extraction:
  - No training run
  - No 5-fold run
  - No feature extraction
- Next suggested step:
  - Step18 cross-scale evidence graph prototype, or Step18 learnable concept-class graph prototype if you want to move from post-hoc analysis toward an explicit graph-based mechanism.

## Step18: Post-hoc Cross-Scale Evidence Graph Prototype

- Modified files:
  - `scripts/analysis/build_stage18_cross_scale_evidence_graph.py`
  - `docs/CODEX_HANDOFF.md`
- Added script:
  - `scripts/analysis/build_stage18_cross_scale_evidence_graph.py`
- Read upstream inputs:
  - `results_stage9/stage13_rce_evidence_export_fold0_test_full/slide_top_concepts.csv`
  - `results_stage9/stage14_concept_class_graph_fold0/stage14_concept_class_edges.csv`
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_cross_scale_slide_scores.csv`
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_conflict_type_summary.csv`
  - `results_stage9/stage17_cross_scale_conflict_analysis_fold0/stage17_failure_pattern_cases.csv`
- Generated outputs:
  - `results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_cross_scale_nodes.csv`
  - `results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_cross_scale_edges.csv`
  - `results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_low_high_concept_pairs.csv`
  - `results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_conflict_pattern_edges.csv`
  - `results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_cross_scale_graph.json`
  - `results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_cross_scale_graph_report.md`
  - `results_stage9/stage18_cross_scale_evidence_graph_fold0/stage18_cross_scale_graph_summary.csv`
- Graph behavior:
  - Builds `low_concept`, `high_concept`, `class`, and `conflict_type` nodes.
  - Builds three edge families:
    - `concept_to_class` edges from Step14 class-support statistics
    - `low_high_pair` edges from Step17 slide-level support concepts and conflict types
    - `conflict_to_class` edges summarizing which classes each conflict type predicts toward
  - Aggregates pair statistics including co-occurrence count/rate, mean joint evidence, correct/incorrect counts, dominant conflict type, high-scale override rate, and wrong-class drift rate.
- Checks run:
  - `python -m py_compile scripts/analysis/build_stage18_cross_scale_evidence_graph.py`
  - `python scripts/analysis/build_stage18_cross_scale_evidence_graph.py`
- Outcome:
  - Step18 ran successfully on fold0 full-export artifacts with `194` slides analyzed and `0` warnings.
  - Produced `50` nodes, `246` combined edges, and `195` retained low/high concept-pair edges.
  - High-scale override graph patterns concentrated around low nonadeno concepts such as `nonadeno_hamartoma_low` / `nonadeno_fibroinflammatory_low` paired with high adeno concepts such as `adeno_papillary_high` / `adeno_atypical_glands_high`.
  - Wrong-class drift graph patterns showed recurring low/high nonadeno concept-pair associations, and Step17 failure cases remained interpretable in graph form:
    - `25001929B3` -> low `nonadeno_hamartoma_low` paired with high `nonadeno_intercellular_bridges_high`
    - `25023972B3` -> low `nonadeno_hamartoma_low` paired with high `adeno_papillary_high`
- Training / 5-fold / feature extraction:
  - No training run
  - No 5-fold run
  - No feature extraction
- Next suggested step:
  - Step19 learnable cross-scale graph module prototype, or Step19 learnable concept-class graph module prototype if you want to convert the post-hoc graph patterns into trainable structure.

## 2026-06-05 - Step19: Learnable Cross-Scale Graph Module Prototype

### Goal
- Add an optional learnable cross-scale graph residual to `RCE_MIL_BiomedCLIP` for low/high concept evidence interaction.
- Keep old RCE base / v2 / v3 / PEPS behavior unchanged by default.
- Add a fold0 testing smoke script for the new RCE-v4 / CSG prototype.

### Files changed
- `main.py`: added CLI flags and experiment settings export for the cross-scale graph prototype.
- `utils/core_utils.py`: passed new cross-scale graph options into the RCE config.
- `models/model_RCE_MIL_BiomedCLIP.py`: added the optional learnable adjacency residual and debug exports.
- `scripts/experiments/run_stage19_rce_v4_csg_smoke.sh`: added a Stage19 fold0 testing smoke script based on the RCE-v3-VR-a005 setup.
- `docs/CODEX_HANDOFF.md`: appended this Step19 record.

### Behavior / tensor flow
- New optional args:
  - `--rce_use_cross_scale_graph`
  - `--rce_cross_scale_graph_init`
  - `--rce_cross_scale_graph_norm`
- When enabled in `scale_mode=dual`, the model builds a learnable adjacency with shape `num_classes x num_low_concepts x num_high_concepts`.
- The residual path computes:
  - `effective_adj = tanh(adjacency)`
  - `cross_scale_logits = einsum(low_prompt_evidence, effective_adj, high_prompt_evidence)`
  - optional normalization by `sqrt(num_low_concepts * num_high_concepts)`
  - `final_logits = final_logits + alpha * cross_scale_logits`
- `alpha` is a learnable scalar initialized from `rce_cross_scale_graph_init`.
- Debug exports were added:
  - `last_cross_scale_logits`
  - `last_cross_scale_alpha`
  - `last_cross_scale_adj`
- Safe behavior:
  - default is off
  - if `scale_mode` is `low_only` or `high_only`, the cross-scale graph is skipped even if the flag is on
  - `forward` return values and loss/probability formats are unchanged

### Checks run
- `python -m py_compile ViLa-MIL-main/main.py`: passed
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py`: passed
- `python -m py_compile ViLa-MIL-main/models/model_RCE_MIL_BiomedCLIP.py`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage19_rce_v4_csg_smoke.sh`: passed
- `bash scripts/experiments/run_stage19_rce_v4_csg_smoke.sh`: passed

### Commands not run
- No long training beyond the requested smoke run
- No 5-fold run
- No feature extraction

### Results / observations
- The Stage19 smoke run completed successfully through fold0 train / val / test with:
  - task `task_adenocarcinoma`
  - `--testing`
  - `--max_epochs 1`
  - output dir `results_stage19/rce_v4_csg_smoke_s1`
- The enabled configuration used the RCE-v3-VR-a005 base plus:
  - `--rce_use_cross_scale_graph`
  - `--rce_cross_scale_graph_init 0.05`
- No shape mismatch was observed with the new cross-scale residual path.
- Smoke metrics:
  - test `AUC=0.5833`
  - test `ACC=0.2105`
  - test `F1=0.1739`
  - balanced `ACC=0.5000`

### Next suggested step
- Inspect `last_cross_scale_logits` / `last_cross_scale_adj` on a few slides to verify whether learned cross-scale interactions resemble the Step18 post-hoc graph patterns before scheduling a non-testing fold0 pilot.

## 2026-06-05 - Step20: CSG Smoke Artifact Inspection

### Goal
- Inspect the Step19 smoke checkpoint for learnable cross-scale graph artifacts only.
- Verify `rce_cross_scale_graph_adj` / `rce_cross_scale_graph_alpha` presence, shape, and value range.
- Compare top learned low/high concept pairs against Step18 post-hoc cross-scale graph patterns.

### Files changed
- `scripts/analysis/inspect_stage20_csg_smoke_artifacts.py`: added a lightweight checkpoint inspection script with graceful handling for missing inputs.
- `docs/CODEX_HANDOFF.md`: appended this Step20 record.

### Behavior / tensor flow
- Default checkpoint search order:
  - `results_stage19/rce_v4_csg_smoke_s1/s_0_checkpoint.pt`
  - `results_stage19/rce_v4_csg_smoke_s1/s_1_checkpoint.pt`
  - `results_stage19/rce_v4_csg_smoke_s1/checkpoint.pt`
- Reads from `state_dict` with or without `module.` prefix:
  - `rce_cross_scale_graph_adj`
  - `rce_cross_scale_graph_alpha`
- Computes:
  - adjacency shape check
  - `NaN` / `Inf` check
  - raw `adj` stats
  - `tanh(adj)` stats
  - alpha value check
- Exports per-class top absolute learned low/high concept pairs and lightweight exact-text overlap against Step18 pair patterns.
- Missing inputs emit warnings and still produce a report instead of crashing.

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/inspect_stage20_csg_smoke_artifacts.py`: passed
- `/home/ljh/anaconda3/envs/vila_mil/bin/python ViLa-MIL-main/scripts/analysis/inspect_stage20_csg_smoke_artifacts.py`: passed

### Commands not run
- No training run
- No 5-fold run
- No feature extraction
- No Step19 smoke re-run

### Results / observations
- Generated outputs under:
  - `results_stage20/stage20_csg_smoke_inspection/`
- Output files:
  - `stage20_csg_adj_stats.csv`
  - `stage20_csg_top_learned_pairs.csv`
  - `stage20_csg_step18_overlap.csv`
  - `stage20_csg_smoke_inspection_report.md`
  - `stage20_csg_smoke_inspection_summary.csv`
- Checkpoint selected:
  - `results_stage19/rce_v4_csg_smoke_s1/s_0_checkpoint.pt`
- Parameter presence:
  - `rce_cross_scale_graph_adj`: found
  - `rce_cross_scale_graph_alpha`: found
- Shape check:
  - adjacency shape = `(2, 12, 12)`
  - expected shape = `(2, 12, 12)`
  - shape matched expected `num_classes x num_low_concepts x num_high_concepts`
- Global value summary:
  - raw `adj`: `min=-0.002617`, `max=0.002605`, `mean=-0.001261`, `std=0.001254`, `abs_max=0.002617`
  - `tanh(adj)`: effectively identical at smoke scale
  - alpha: `0.050335`
- Pattern inspection:
  - Learned top pairs were dominated by class-consistent adeno/adeno and nonadeno/nonadeno pairs after the short smoke run.
  - Exact-text overlap with Step18 post-hoc pairs was limited but non-zero:
    - `step18_top_mean_joint_evidence`: overlap `3`
    - `step18_high_scale_override`: overlap `0`
    - `step18_wrong_class_drift`: overlap `2`
  - The overlaps came from class `1` nonadeno learned pairs; no class `0` overlap appeared in the top learned-pair slice.
- Interpretation:
  - The adjacency is present, finite, shape-correct, and non-zero, but still very small as expected for `--testing` + `1 epoch`.
  - Step20 does not judge model quality or final graph usefulness.

### Next suggested step
- Step21 non-testing fold0 pilot, because the cross-scale graph parameters exist and basic smoke artifact stats look normal; the smoke run is too short to say much more about learned structure.

## 2026-06-05 - Step21: RCE-v4-CSG Fold0 Non-Testing Pilot Script

### Goal
- Add a fold0 non-testing pilot script for the RCE-v4 cross-scale graph variant.
- Keep the Stage21 script aligned with the current best RCE-v3-VR-a005 baseline and only add the learnable CSG switches.

### Files changed
- `scripts/experiments/run_stage21_rce_v4_csg_pilot_fold0.sh`: added a configurable fold0 pilot launcher for RCE-v4-CSG.
- `docs/CODEX_HANDOFF.md`: appended this Step21 record.

### Behavior / script config
- Default environment-variable overrides:
  - `PYTHON_BIN`
  - `DATA_ROOT_DIR`
  - `RESULTS_DIR`
  - `SEED`
  - `MAX_EPOCHS`
  - `FOLD`
- Additional path/runtime overrides retained:
  - `VARIANT`
  - `SPLIT_DIR`
  - `TEXT_PROMPT_PATH`
  - `CONCEPT12_PATH`
  - `HF_HUB_OFFLINE_FLAG`
  - `TRANSFORMERS_OFFLINE_FLAG`
- Default run shape:
  - `FOLD=0`
  - `MAX_EPOCHS=20`
  - `SEED=1`
  - `RESULTS_DIR=results_stage21`
  - no `--testing`
- Supported `VARIANT` values:
  - `csg_a005`: `--rce_cross_scale_graph_init 0.05`
  - `csg_a01`: `--rce_cross_scale_graph_init 0.1`
  - `all`: runs `csg_a005` then `csg_a01`
- Fixed exp codes:
  - `rce_v4_csg_a005_pilot_fold0_e20`
  - `rce_v4_csg_a01_pilot_fold0_e20`
- The script prints:
  - current `VARIANT`
  - `RESULTS_DIR`
  - `EXP_CODE`
  - `FOLD`
  - `MAX_EPOCHS`
  - `SEED`
  - full command line before execution

### Checks run
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage21_rce_v4_csg_pilot_fold0.sh`: passed

### Commands not run
- No pilot training run
- No 5-fold run
- No feature extraction

### Results / observations
- The script is ready for a non-testing fold0 pilot using the RCE-v3-VR-a005 base config plus:
  - `--rce_use_cross_scale_graph`
  - `--rce_cross_scale_graph_init <variant>`
  - `--rce_cross_scale_graph_norm sqrt`
- No model, dataset, analysis script, or historical result directory was modified in this step.

### Next suggested step
- Run the Stage21 pilot script with `VARIANT=csg_a005` first, then compare against `csg_a01` if needed.

## 2026-06-09 - Step22: RCE-v4-CSG 5-Fold Formal Script

### Goal
- Add the formal 5-fold launcher for the RCE-v4 cross-scale graph variant without changing any model or training code.

### Files changed
- `scripts/experiments/run_stage22_rce_v4_csg_5fold.sh`: added a configurable 5-fold launcher for the current RCE-v4-CSG setup.
- `docs/CODEX_HANDOFF.md`: appended this Step22 record.

### Behavior / script config
- Default environment-variable overrides:
  - `PYTHON_BIN`
  - `DATA_ROOT_DIR`
  - `RESULTS_DIR`
  - `SEED`
  - `MAX_EPOCHS`
- Additional path/runtime overrides retained:
  - `VARIANT`
  - `SPLIT_DIR`
  - `TEXT_PROMPT_PATH`
  - `CONCEPT12_PATH`
  - `HF_HUB_OFFLINE_FLAG`
  - `TRANSFORMERS_OFFLINE_FLAG`
- Fixed model/config switches:
  - `--model_type RCE_MIL_BiomedCLIP`
  - `--scale_mode dual`
  - `--data_folder_s features_biomedclip_5x`
  - `--data_folder_l features_biomedclip_20x`
  - `--use_concept_prompt_pool`
  - `--concept_prompt_path dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json` via `CONCEPT12_PATH`
  - `--prototype_number 16`
  - `--rce_use_logit_calibration`
  - `--rce_use_concept_prior`
  - `--rce_use_visual_residual`
  - `--rce_use_cross_scale_graph`
  - `--rce_cross_scale_graph_norm sqrt`
- Fixed 5-fold run shape:
  - `--k 5`
  - `--k_start 0`
  - `--k_end 4`
  - `RESULTS_DIR=results_stage22`
  - `MAX_EPOCHS=20`
  - `SEED=1`
- Supported `VARIANT` values:
  - `csg_a005`: `--rce_cross_scale_graph_init 0.05`
  - `csg_a01`: `--rce_cross_scale_graph_init 0.1`
  - `all`: runs both variants in sequence
- Fixed exp codes:
  - `rce_v4_csg_a005_5fold_e20`
  - `rce_v4_csg_a01_5fold_e20`

### Checks run
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage22_rce_v4_csg_5fold.sh`: passed

### Commands not run
- No 5-fold training run
- No pilot run
- No feature extraction

### Results / observations
- The new script follows the Stage21 pilot structure but switches to formal 5-fold execution with the same RCE-v4-CSG core settings.
- During follow-up debugging, confirmed that `main.py` treats `--k_end` as inclusive, so a true 5-fold run must use `--k_start 0 --k_end 4`; using `--k_end 5` incorrectly attempts a sixth fold and looks for `splits_5.csv`.
- This step only added the launcher and documentation; it did not modify `main.py`, `utils/core_utils.py`, or any model file.

### Next suggested step
- User runs one of:
  - `VARIANT=csg_a005 bash scripts/experiments/run_stage22_rce_v4_csg_5fold.sh`
  - `VARIANT=csg_a01 bash scripts/experiments/run_stage22_rce_v4_csg_5fold.sh`
  - `VARIANT=all bash scripts/experiments/run_stage22_rce_v4_csg_5fold.sh`

## 2026-06-09 - Step23: RCE-v4-CSG Region Query Number Sensitivity 5-Fold Script

### Goal
- Add a formal 5-fold experiment launcher to compare region query number sensitivity for the current RCE-v4-CSG main candidate.

### Files changed
- `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`: added a configurable 5-fold launcher for region query number sensitivity.
- `docs/CODEX_HANDOFF.md`: appended this Step23 record.

### Behavior / script config
- Experiment purpose:
  - region query number sensitivity under the current RCE-v4-CSG setup
- Default environment-variable overrides:
  - `PYTHON_BIN`
  - `DATA_ROOT_DIR`
  - `RESULTS_DIR`
  - `SEED`
  - `MAX_EPOCHS`
- Additional path/runtime overrides retained:
  - `VARIANT`
  - `SPLIT_DIR`
  - `TEXT_PROMPT_PATH`
  - `CONCEPT12_PATH`
  - `HF_HUB_OFFLINE_FLAG`
  - `TRANSFORMERS_OFFLINE_FLAG`
- Fixed model/config switches:
  - `--model_type RCE_MIL_BiomedCLIP`
  - `--scale_mode dual`
  - `--data_folder_s features_biomedclip_5x`
  - `--data_folder_l features_biomedclip_20x`
  - `--use_concept_prompt_pool`
  - `--concept_prompt_path dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json` via `CONCEPT12_PATH`
  - `--rce_use_logit_calibration`
  - `--rce_use_concept_prior`
  - `--rce_use_visual_residual`
  - `--rce_visual_residual_init 0.05`
  - `--rce_use_cross_scale_graph`
  - `--rce_cross_scale_graph_init 0.1`
  - `--rce_cross_scale_graph_norm sqrt`
- Why fix `CSG init=0.1`:
  - Step22 established `csg_a01` as the current RCE-v4-CSG main candidate, so Step23 only sweeps region query count on that branch.
- Fixed 5-fold run shape:
  - `--k 5`
  - `--k_start 0`
  - `--k_end 4`
  - `RESULTS_DIR=results_stage23`
  - `MAX_EPOCHS=20`
  - `SEED=1`
- Supported `VARIANT` values:
  - `rq8`
  - `rq16`
  - `rq32`
  - `all`
- Prototype settings:
  - `rq8`: `--prototype_number 8`
  - `rq16`: `--prototype_number 16`
  - `rq32`: `--prototype_number 32`
- Exp code bases and resulting default output directories:
  - `rq8`: exp code `rce_v4_csg_a01_rq8_5fold_e20`, output directory `rce_v4_csg_a01_rq8_5fold_e20_s1`
  - `rq16`: exp code `rce_v4_csg_a01_rq16_5fold_e20`, output directory `rce_v4_csg_a01_rq16_5fold_e20_s1`
  - `rq32`: exp code `rce_v4_csg_a01_rq32_5fold_e20`, output directory `rce_v4_csg_a01_rq32_5fold_e20_s1`

### Checks run
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`: passed

### Commands not run
- No 5-fold training run
- No pilot run
- No feature extraction

### Results / observations
- The new script keeps the full Stage22 RCE-v4-CSG config fixed and only changes `--prototype_number` across three variants.
- The script preserves the inclusive `k_end` correction from Step22 and therefore uses folds `0..4` instead of attempting a nonexistent sixth split.
- This step did not modify `main.py`, `utils/core_utils.py`, or any model file.

### Next suggested step
- User runs one of:
  - `VARIANT=rq8 bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
  - `VARIANT=rq16 bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
  - `VARIANT=rq32 bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
  - `VARIANT=all bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`

## 2026-06-09 - Step24: Stage22/Stage23 RCE-v4-CSG Summary Analysis and Launcher Cleanup

### Goal
- Clean up the Stage22 and Stage23 launcher formatting without changing experiment semantics.
- Add a read-only analysis script to summarize Stage22 CSG init results and Stage23 region query number sensitivity.

### Files changed
- `scripts/experiments/run_stage22_rce_v4_csg_5fold.sh`: moved the inclusive `k_end` note outside the bash array and kept the launcher as a readable multi-line script.
- `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`: moved the inclusive `k_end` note outside the bash array and kept the launcher as a readable multi-line script.
- `scripts/analysis/build_stage24_rce_v4_csg_summary.py`: added a read-only Stage24 summary builder for Stage22, Stage23, and optional Stage9 reference comparisons.
- `docs/CODEX_HANDOFF.md`: appended this Step24 record.

### Behavior / script config
- The Stage24 analysis script reads:
  - `results_stage22/rce_v4_csg_a005_5fold_e20_s1/fold_summary.csv`
  - `results_stage22/rce_v4_csg_a01_5fold_e20_s1/fold_summary.csv`
  - `results_stage23/rce_v4_csg_a01_rq8_5fold_e20_s1/fold_summary.csv`
  - `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/fold_summary.csv`
  - `results_stage23/rce_v4_csg_a01_rq32_5fold_e20_s1/fold_summary.csv`
  - optional `results_stage9/stage9_rce_final_analysis/rce_stage9_main_comparison.csv`
- Supported environment-variable overrides:
  - `RESULTS_STAGE22_DIR`
  - `RESULTS_STAGE23_DIR`
  - `STAGE9_ANALYSIS_CSV`
  - `OUTPUT_DIR`
- Generated outputs under:
  - `results_stage24/stage24_rce_v4_csg_summary/`

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage24_rce_v4_csg_summary.py`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage22_rce_v4_csg_5fold.sh`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`: passed
- `python ViLa-MIL-main/scripts/analysis/build_stage24_rce_v4_csg_summary.py`: passed

### Commands not run
- No training run
- No 5-fold evaluation run
- No feature extraction

### Results / observations
- Generated output files:
  - `results_stage24/stage24_rce_v4_csg_summary/stage24_stage22_csg_init_summary.csv`
  - `results_stage24/stage24_rce_v4_csg_summary/stage24_stage23_region_query_summary.csv`
  - `results_stage24/stage24_rce_v4_csg_summary/stage24_metric_deltas.csv`
  - `results_stage24/stage24_rce_v4_csg_summary/stage24_rce_v4_csg_summary_report.md`
- Stage22 core conclusion:
  - `csg_a01` outperformed `csg_a005` on the main 5-fold summary metrics and is the preferred CSG init branch.
- Stage23 core conclusion:
  - `rq16` outperformed both `rq8` and `rq32` on the main 5-fold summary metrics and remains the preferred region query count.
- Current recommended main configuration:
  - `RCE-v4-CSG-a01-rq16`
- Follow-on default:
  - continue to use `prototype_number=16` for DEG-MIL by default.
- This step did not modify any model file, `main.py`, or `utils/core_utils.py`.

### Next suggested step
- Use `RCE-v4-CSG-a01-rq16` as the Stage24 main configuration for the next DEG-MIL step and keep `prototype_number=16` as the default.

## 2026-06-09 - Step25: DEG-MIL Skeleton With Region Coordinate Export

### Goal
- Add a new `DEG_MIL_BiomedCLIP` skeleton model that stays aligned with the current `RCE-v4-CSG-a01-rq16` logits path.
- Export region attention and attention-weighted region coordinates without changing the original RCE model behavior.

### Files changed
- `models/model_DEG_MIL_BiomedCLIP.py`: added the new DEG skeleton model.
- `main.py`: added `DEG_MIL_BiomedCLIP` to `--model_type` choices.
- `utils/core_utils.py`: added model initialization support for `DEG_MIL_BiomedCLIP`.
- `scripts/experiments/run_stage25_deg_skeleton_smoke.sh`: added a DEG skeleton smoke launcher.
- `docs/CODEX_HANDOFF.md`: appended this Step25 record.

### Behavior / tensor flow
- `DEG_MIL_BiomedCLIP` inherits from `RCE_MIL_BiomedCLIP` and keeps the same main logits path, concept prior path, visual residual path, logit calibration path, and cross-scale graph path.
- Step25 does not add graph reasoning, region graph message passing, or concept graph message passing.
- The only new Step25 behavior is:
  - keep low/high normalized region attention weights
  - compute low/high attention-weighted region coordinates from input patch coords
  - export these debug attributes:
    - `last_low_region_attn`
    - `last_high_region_attn`
    - `last_low_region_coords`
    - `last_high_region_coords`
    - `last_slide_id`
- Existing RCE-style debug exports remain available in the DEG skeleton.

### Checks run
- `python -m py_compile ViLa-MIL-main/models/model_DEG_MIL_BiomedCLIP.py`: passed
- `python -m py_compile ViLa-MIL-main/main.py`: passed
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage25_deg_skeleton_smoke.sh`: passed
- `bash ViLa-MIL-main/scripts/experiments/run_stage25_deg_skeleton_smoke.sh`: passed

### Commands not run
- No formal 5-fold training run
- No feature extraction

### Results / observations
- `main.py` now accepts `--model_type DEG_MIL_BiomedCLIP`.
- `utils/core_utils.py` now routes `DEG_MIL_BiomedCLIP` through the same config fields currently used by `RCE_MIL_BiomedCLIP`.
- The original `models/model_RCE_MIL_BiomedCLIP.py` file was not modified.
- Smoke run status:
  - executed successfully
  - results directory: `results_stage25/deg_skeleton_smoke_s1`
  - smoke metrics are only a short-path sanity check and should not be treated as model-quality evidence

### Next suggested step
- Step26: add Spatial Region Graph on top of the DEG skeleton.

## 2026-06-09 - Step26: Spatial Region Graph For DEG-MIL

### Goal
- Add an optional Spatial Region Graph to `DEG_MIL_BiomedCLIP` while keeping the Step25 skeleton behavior unchanged when the graph flag is off.

### Files changed
- `models/model_DEG_MIL_BiomedCLIP.py`: added optional same-scale Spatial Region Graph logic and new debug exports.
- `main.py`: added DEG region graph argparse parameters and settings logging.
- `utils/core_utils.py`: passed DEG region graph config fields into `DEG_MIL_BiomedCLIP`.
- `scripts/experiments/run_stage26_deg_region_graph_smoke.sh`: added a smoke launcher for the DEG Spatial Region Graph path.
- `docs/CODEX_HANDOFF.md`: appended this Step26 record.

### Behavior / tensor flow
- Current graph scope:
  - low-scale region graph
  - high-scale region graph
- Current graph exclusions:
  - no cross-scale region graph
  - no concept graph
  - no region-concept bipartite graph
- When `deg_use_region_graph=False`, `DEG_MIL_BiomedCLIP` stays aligned with the Step25 skeleton path.
- When `deg_use_region_graph=True`:
  - the model first computes Step25 low/high region features and region coordinates
  - saves graph-pre-update features to:
    - `last_low_region_features_before_graph`
    - `last_high_region_features_before_graph`
  - builds per-scale kNN adjacency from region coordinates
  - applies residual message passing with row-normalized adjacency and per-scale projection/norm
  - uses graph-updated region features for downstream region-concept evidence, visual residual, and cross-scale graph logits
- New argparse/config parameters:
  - `--deg_use_region_graph`
  - `--deg_region_graph_k`
  - `--deg_region_graph_alpha`
- New debug attributes:
  - `last_low_region_adj`
  - `last_high_region_adj`
  - `last_low_region_graph_alpha`
  - `last_high_region_graph_alpha`
  - `last_low_region_features_before_graph`
  - `last_high_region_features_before_graph`

### Checks run
- `python -m py_compile ViLa-MIL-main/models/model_DEG_MIL_BiomedCLIP.py`: passed
- `python -m py_compile ViLa-MIL-main/main.py`: passed
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage26_deg_region_graph_smoke.sh`: passed
- `bash ViLa-MIL-main/scripts/experiments/run_stage26_deg_region_graph_smoke.sh`: passed

### Commands not run
- No formal 5-fold training run
- No feature extraction

### Results / observations
- `main.py` now accepts the DEG region graph flags and records them in experiment settings.
- `utils/core_utils.py` now forwards the DEG region graph settings through the existing DEG config path.
- The original `models/model_RCE_MIL_BiomedCLIP.py` file was not modified.
- Smoke run status:
  - executed successfully
  - results directory: `results_stage26/deg_region_graph_smoke_s1`
  - smoke metrics are only a short-path sanity check and should not be treated as model-quality evidence

### Next suggested step
- If the Step26 smoke passes, either run a Step27 5-fold pilot for DEG region graph or extend the model with cross-scale region graph reasoning.

## 2026-06-09 - Step27: DEG Spatial Region Graph 5-Fold Ablation Script

### Goal
- Add a formal 5-fold launcher to evaluate whether the DEG Spatial Region Graph is useful and how sensitive it is to the kNN neighbor count.

### Files changed
- `scripts/experiments/run_stage27_deg_region_graph_5fold.sh`: added a configurable 5-fold DEG region-graph ablation launcher.
- `docs/CODEX_HANDOFF.md`: appended this Step27 record.

### Behavior / script config
- Experiment purpose:
  - formally evaluate whether Spatial Region Graph helps over the DEG skeleton path
  - compare different kNN neighborhood sizes under the current DEG main configuration
- Fixed base configuration:
  - `--model_type DEG_MIL_BiomedCLIP`
  - `--scale_mode dual`
  - `--data_folder_s features_biomedclip_5x`
  - `--data_folder_l features_biomedclip_20x`
  - `--use_concept_prompt_pool`
  - `--concept_prompt_path dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json` via `CONCEPT12_PATH`
  - `--prototype_number 16`
  - `--rce_use_logit_calibration`
  - `--rce_use_concept_prior`
  - `--rce_use_visual_residual`
  - `--rce_visual_residual_init 0.05`
  - `--rce_use_cross_scale_graph`
  - `--rce_cross_scale_graph_init 0.1`
  - `--rce_cross_scale_graph_norm sqrt`
  - `--deg_region_graph_alpha 0.1` for graph-enabled variants
- Why keep `prototype_number=16` and `rce_cross_scale_graph_init=0.1` fixed:
  - Step24 recommended `prototype_number=16` as the DEG default.
  - Step22 recommended `csg_a01` (`rce_cross_scale_graph_init=0.1`) as the preferred CSG branch.
- Fixed 5-fold run shape:
  - `--k 5`
  - `--k_start 0`
  - `--k_end 4`
  - `RESULTS_DIR=results_stage27`
  - `MAX_EPOCHS=20`
  - `SEED=1`
- Supported `VARIANT` values:
  - `skeleton`: no `--deg_use_region_graph`
  - `rg_k2`: `--deg_use_region_graph --deg_region_graph_k 2 --deg_region_graph_alpha 0.1`
  - `rg_k4`: `--deg_use_region_graph --deg_region_graph_k 4 --deg_region_graph_alpha 0.1`
  - `rg_k8`: `--deg_use_region_graph --deg_region_graph_k 8 --deg_region_graph_alpha 0.1`
  - `all`: runs all four variants in sequence
- Exp code bases and resulting default output directories:
  - `skeleton`: exp code `deg_skeleton_5fold_e20`, output directory `deg_skeleton_5fold_e20_s1`
  - `rg_k2`: exp code `deg_region_graph_k2_a01_5fold_e20`, output directory `deg_region_graph_k2_a01_5fold_e20_s1`
  - `rg_k4`: exp code `deg_region_graph_k4_a01_5fold_e20`, output directory `deg_region_graph_k4_a01_5fold_e20_s1`
  - `rg_k8`: exp code `deg_region_graph_k8_a01_5fold_e20`, output directory `deg_region_graph_k8_a01_5fold_e20_s1`

### Checks run
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage27_deg_region_graph_5fold.sh`: passed

### Commands not run
- No formal 5-fold training run
- No feature extraction

### Results / observations
- The script prints:
  - current `VARIANT`
  - `PROTOTYPE_NUMBER`
  - `CSG_INIT`
  - `REGION_GRAPH_K`
  - `REGION_GRAPH_ALPHA`
  - `RESULTS_DIR`
  - `EXP_CODE`
  - full command line before execution
- This step only added the Stage27 launcher and documentation; it did not modify any model file, dataset file, `main.py`, or `utils/core_utils.py`.

### Next suggested step
- User runs one of:
  - `VARIANT=skeleton bash scripts/experiments/run_stage27_deg_region_graph_5fold.sh`
  - `VARIANT=rg_k2 bash scripts/experiments/run_stage27_deg_region_graph_5fold.sh`
  - `VARIANT=rg_k4 bash scripts/experiments/run_stage27_deg_region_graph_5fold.sh`
  - `VARIANT=rg_k8 bash scripts/experiments/run_stage27_deg_region_graph_5fold.sh`
  - `VARIANT=all bash scripts/experiments/run_stage27_deg_region_graph_5fold.sh`

## 2026-06-10 - Step28: Stage27 DEG Spatial Region Graph Summary And Cleanup

### Goal
- Add a read-only Stage28 analysis script for the Stage27 DEG Spatial Region Graph ablations.
- Clean up the Stage27 launcher formatting without changing experiment semantics.
- Update the DEG model docstring to document the Step25 and Step26 scope.

### Files changed
- `scripts/analysis/build_stage28_deg_region_graph_summary.py`: added a read-only Stage28 summary builder for Stage27 DEG region-graph results.
- `scripts/experiments/run_stage27_deg_region_graph_5fold.sh`: reformatted the launcher into clearer common/variant argument blocks without changing parameters or behavior.
- `models/model_DEG_MIL_BiomedCLIP.py`: updated the top-level docstring only.
- `docs/CODEX_HANDOFF.md`: appended this Step28 record.

### Behavior / script config
- The Stage28 analysis script reads:
  - `results_stage27/deg_skeleton_5fold_e20_s1/fold_summary.csv`
  - `results_stage27/deg_region_graph_k2_a01_5fold_e20_s1/fold_summary.csv`
  - `results_stage27/deg_region_graph_k4_a01_5fold_e20_s1/fold_summary.csv`
  - `results_stage27/deg_region_graph_k8_a01_5fold_e20_s1/fold_summary.csv`
- Supported environment-variable overrides:
  - `RESULTS_STAGE27_DIR`
  - `OUTPUT_DIR`
- Generated outputs under:
  - `results_stage28/stage28_deg_region_graph_summary/`

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage28_deg_region_graph_summary.py`: passed
- `python -m py_compile ViLa-MIL-main/models/model_DEG_MIL_BiomedCLIP.py`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage27_deg_region_graph_5fold.sh`: passed
- `python ViLa-MIL-main/scripts/analysis/build_stage28_deg_region_graph_summary.py`: passed

### Commands not run
- No training run
- No 5-fold evaluation run
- No feature extraction

### Results / observations
- Generated output files:
  - `results_stage28/stage28_deg_region_graph_summary/stage28_deg_region_graph_summary.csv`
  - `results_stage28/stage28_deg_region_graph_summary/stage28_deg_region_graph_metric_deltas.csv`
  - `results_stage28/stage28_deg_region_graph_summary/stage28_deg_region_graph_report.md`
- Stage27 core conclusion:
  - `skeleton` remains the strongest main configuration.
  - `rg_k2`, `rg_k4`, and `rg_k8` all remain below `skeleton`.
  - `rg_k8` is the closest graph variant on mean `test_auc`, but still trails `skeleton` and is not consistently best on the other main metrics.
  - the current Spatial Region Graph should not be treated as the main performance module.
- Follow-up guidance:
  - not recommended to directly build current cross-scale region graph on top of this Region Graph version.
  - more promising follow-ups are a gated/zero-init Region Graph or a higher-priority Concept Prompt Graph line.
- This step did not run training and did not modify model computation logic.

### Next suggested step
- Keep `RCE-v4-CSG-a01-rq16 / DEG skeleton` as the main line, and only revisit Region Graph after adding a more stable gated or zero-init design.

## 2026-06-10 - Step29: DEG Concept Prompt Graph Smoke

### Goal
- Add an optional Concept Prompt Graph to `DEG_MIL_BiomedCLIP` for intra-class / intra-scale prompt-pool message passing only.
- Expose Step29 Concept Graph args in `main.py` and pass them through `utils/core_utils.py`.
- Add a smoke launcher that enables Concept Prompt Graph without enabling Spatial Region Graph.

### Files changed
- `models/model_DEG_MIL_BiomedCLIP.py`: added optional low/high Concept Prompt Graph modules, safe class-wise top-k adjacency construction, prompt-feature debug exports, and prompt-side graph application before region-concept evidence and existing cross-scale graph logic.
- `main.py`: added `--deg_use_concept_graph`, `--deg_concept_graph_topk`, and `--deg_concept_graph_alpha`, and recorded them in the printed run settings.
- `utils/core_utils.py`: passed the new DEG Concept Graph args into the `DEG_MIL_BiomedCLIP` config.
- `scripts/experiments/run_stage29_deg_concept_graph_smoke.sh`: added the Step29 smoke launcher using the current DEG/RCE main config, `--testing`, fold0-only, and Concept Prompt Graph enabled with Spatial Region Graph left off.
- `docs/CODEX_HANDOFF.md`: appended this Step29 record.

### Behavior / tensor flow
- The new prompt graph only operates on `low_prompt_features` and `high_prompt_features` with shape `[C, P, D]`.
- Graph construction is intra-class and intra-scale only; it does not mix classes, does not replace the existing RCE cross-scale graph, does not add cross-scale concept graph replacement, and does not add a region-concept bipartite graph.
- For each class and scale, prompt features are normalized, cosine similarity is used to choose top-k neighbors, self-loops are masked, and the row-normalized adjacency is used for a residual prompt update before `_compute_scale_logits()`.
- When `deg_use_concept_graph=False`, the DEG path falls back to the Step28 / DEG skeleton prompt flow.
- Added debug attributes:
  - `last_low_concept_adj`
  - `last_high_concept_adj`
  - `last_low_prompt_features_before_graph`
  - `last_high_prompt_features_before_graph`
  - `last_low_prompt_features_after_graph`
  - `last_high_prompt_features_after_graph`
  - `last_low_concept_graph_alpha`
  - `last_high_concept_graph_alpha`

### Checks run
- `python -m py_compile ViLa-MIL-main/models/model_DEG_MIL_BiomedCLIP.py`: passed
- `python -m py_compile ViLa-MIL-main/main.py`: passed
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage29_deg_concept_graph_smoke.sh`: passed

### Commands not run
- No formal 5-fold training was run.
- No feature extraction was run.
- No dataset files were modified.

### Results / observations
- Smoke command run:
  - `bash ViLa-MIL-main/scripts/experiments/run_stage29_deg_concept_graph_smoke.sh`
- Smoke outcome:
  - Completed successfully through train / val / test on fold0-only testing mode.
- Smoke results directory:
  - `results_stage29/deg_concept_graph_smoke_s1`
- Final smoke test metrics:
  - `AUC=0.5833`, `ACC=0.2105`, `F1=0.1739`, `Balanced ACC=0.5000`, `Sensitivity=1.0000`, `Specificity=0.0000`, `PR-AUC=0.3375`
- The smoke run confirms that the new Concept Prompt Graph args, prompt update path, existing RCE cross-scale graph path, and DEG training/eval loop are all compatible in the Step29 configuration.

### Next suggested step
- If the Step29 smoke path is accepted, proceed to Step30: Concept Prompt Graph 5-fold ablation / sensitivity on top of the current DEG skeleton-style configuration.

## 2026-06-10 - Step30: DEG Concept Prompt Graph 5-Fold Ablation Script

### Goal
- Add a Stage30 5-fold launcher to formally evaluate whether the Step29 Concept Prompt Graph improves over the current DEG skeleton / `RCE-v4-CSG-a01-rq16` style main line.
- Keep the sweep focused on Concept Prompt Graph top-k sensitivity only, with `alpha=0.05` fixed.

### Files changed
- `scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`: added the Stage30 5-fold launcher with `skeleton`, `cg_k2_a005`, `cg_k4_a005`, `cg_k8_a005`, and `all` variants.
- `docs/CODEX_HANDOFF.md`: appended this Step30 record.

### Behavior / script config
- Fixed main configuration across all variants:
  - `model_type=DEG_MIL_BiomedCLIP`
  - `scale_mode=dual`
  - `data_folder_s=features_biomedclip_5x`
  - `data_folder_l=features_biomedclip_20x`
  - `use_concept_prompt_pool=True`
  - `concept_prompt_path=dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
  - `prototype_number=16`
  - `rce_use_logit_calibration=True`
  - `rce_use_concept_prior=True`
  - `rce_use_visual_residual=True`
  - `rce_visual_residual_init=0.05`
  - `rce_use_cross_scale_graph=True`
  - `rce_cross_scale_graph_init=0.1`
  - `rce_cross_scale_graph_norm=sqrt`
- `prototype_number=16` and `rce_cross_scale_graph_init=0.1` stay fixed because Step30 is intended to isolate the Concept Prompt Graph effect on top of the current main DEG/RCE setting rather than reopen unrelated hyperparameters.
- `deg_use_region_graph` stays off for all Step30 variants because Stage28 already showed the current Spatial Region Graph is not the preferred main branch, and Step30 is meant to evaluate prompt-side graph reasoning only.
- Concept Prompt Graph remains intra-class / intra-scale only. This script does not add cross-scale region graph, cross-scale concept-graph replacement, or region-concept bipartite graph settings.
- Supported variants:
  - `skeleton`: no Concept Prompt Graph flag; control run against the current DEG skeleton line.
  - `cg_k2_a005`: `--deg_use_concept_graph --deg_concept_graph_topk 2 --deg_concept_graph_alpha 0.05`
  - `cg_k4_a005`: `--deg_use_concept_graph --deg_concept_graph_topk 4 --deg_concept_graph_alpha 0.05`
  - `cg_k8_a005`: `--deg_use_concept_graph --deg_concept_graph_topk 8 --deg_concept_graph_alpha 0.05`
  - `all`: runs the four variants above sequentially.
- Fold and epoch setup:
  - `k=5`
  - `k_start=0`
  - `k_end=4`
  - `max_epochs=20`
  - `seed=1`
- The script prints:
  - `VARIANT`
  - whether Concept Prompt Graph is enabled
  - Concept Prompt Graph `topk`
  - Concept Prompt Graph `alpha`
  - `PROTOTYPE_NUMBER`
  - `CSG_INIT`
  - `RESULTS_DIR`
  - `EXP_CODE`
  - full command line before execution
- To avoid duplicated seed suffixes in output paths, the script keeps `exp_code` as the base name and relies on `main.py` to append `_s${SEED}`.

### Checks run
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage29_deg_concept_graph_smoke.sh`: passed

### Commands not run
- No formal 5-fold training run was executed in this step.
- No smoke run was executed in this step.
- No model computation logic was modified in this step.

### Results / observations
- Stage30 only adds the formal 5-fold launcher and documentation; it does not modify `models/model_DEG_MIL_BiomedCLIP.py`, `models/model_RCE_MIL_BiomedCLIP.py`, dataset files, or existing results.
- Recommended user commands:
  - `cd /xiangmu/ViLMIL/ViLa-MIL-main && VARIANT=skeleton bash scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`
  - `cd /xiangmu/ViLMIL/ViLa-MIL-main && VARIANT=cg_k2_a005 bash scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`
  - `cd /xiangmu/ViLMIL/ViLa-MIL-main && VARIANT=cg_k4_a005 bash scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`
  - `cd /xiangmu/ViLMIL/ViLa-MIL-main && VARIANT=cg_k8_a005 bash scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`
  - `cd /xiangmu/ViLMIL/ViLa-MIL-main && VARIANT=all bash scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`

### Next suggested step
- After the user runs the Stage30 launcher, compare the 5-fold summaries against the Step28 / DEG skeleton baseline to decide whether Concept Prompt Graph should remain on the main branch or only as an ablation line.

## 2026-06-10 - Step31: Stage30 Concept Prompt Graph Summary And Cleanup

### Goal
- Add a read-only Stage31 analysis script for the Stage30 Concept Prompt Graph 5-fold ablations.
- Clean up the Stage30 launcher formatting without changing experiment semantics.

### Files changed
- `scripts/analysis/build_stage31_deg_concept_graph_summary.py`: added a read-only Stage31 summary builder for Stage30 Concept Prompt Graph results.
- `scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`: cleaned up variant-list handling and launcher formatting without changing parameters, variant names, output directories, or experiment semantics.
- `docs/CODEX_HANDOFF.md`: appended this Step31 record.

### Behavior / script config
- The Stage31 analysis script reads:
  - `results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1/fold_summary.csv`
  - `results_stage30/deg_concept_graph_k2_a005_5fold_e20_s1/fold_summary.csv`
  - `results_stage30/deg_concept_graph_k4_a005_5fold_e20_s1/fold_summary.csv`
  - `results_stage30/deg_concept_graph_k8_a005_5fold_e20_s1/fold_summary.csv`
- Supported environment-variable overrides:
  - `RESULTS_STAGE30_DIR`
  - `OUTPUT_DIR`
- Generated outputs under:
  - `results_stage31/stage31_deg_concept_graph_summary/`
- Generated files:
  - `stage31_deg_concept_graph_summary.csv`
  - `stage31_deg_concept_graph_metric_deltas.csv`
  - `stage31_deg_concept_graph_report.md`
- The analysis script is warning-tolerant: missing or malformed result CSVs produce graceful warnings and still allow partial report generation.

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage31_deg_concept_graph_summary.py`: passed
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage30_deg_concept_graph_5fold.sh`: passed
- `python ViLa-MIL-main/scripts/analysis/build_stage31_deg_concept_graph_summary.py`: passed

### Commands not run
- No training run
- No 5-fold evaluation run
- No feature extraction
- No model computation logic change

### Results / observations
- Stage30 core conclusion:
  - `skeleton` remains the best main configuration.
  - `cg_k2_a005`, `cg_k4_a005`, and `cg_k8_a005` all remain below `skeleton`.
  - `cg_k8_a005` is the closest Concept Prompt Graph variant on `test_auc` / `pr_auc`, but it still trails `skeleton` on `sensitivity`, `test_f1`, and `balanced_acc`.
  - the current Concept Prompt Graph should not be treated as a mainline performance module.
- Current main line remains:
  - `RCE-v4-CSG-a01-rq16 / DEG skeleton`
- Combined interpretation across Stage27/28 and Stage30:
  - both Spatial Region Graph and Concept Prompt Graph suggest that directly stacking ordinary feature-level message passing can weaken already-learned evidence discrimination.
- Recommended follow-up direction:
  - do not prioritize more ordinary region/concept feature graphs right now.
  - prioritize evidence export / interpretability first.
  - if adding a new module, prefer evidence-level gated residual or evidence consistency loss over plain graph message passing.
- This step did not run training and did not modify any model computation file.

### Next suggested step
- Keep `RCE-v4-CSG-a01-rq16 / DEG skeleton` as the main line and shift the next iteration toward evidence-level analysis or lighter evidence-side control modules instead of more feature-graph stacking.

## 2026-06-13 - Step32: Evidence Export For Current Main Model

### Goal
- Add a read-only Step32 evidence export path for the current main configuration `RCE-v4-CSG-a01-rq16 / DEG skeleton`.
- Export slide-level prediction summaries, prompt evidence, CSG pair evidence, and error-case subsets without changing any training or model computation logic.

### Files changed
- `scripts/analysis/export_stage32_rce_v4_csg_evidence.py`: added the Step32 evidence exporter for the current main model.
- `scripts/analysis/run_stage32_export_evidence.sh`: added an optional launcher with environment-variable overrides for fold/split/results/checkpoint paths.
- `docs/CODEX_HANDOFF.md`: appended this Step32 record.

### Behavior / script config
- Default input priority:
  - `results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1/`
  - fallback: `results_stage22/rce_v4_csg_a01_5fold_e20_s1/`
- Default model path:
  - `DEG_MIL_BiomedCLIP`
  - with `deg_use_region_graph=False`
  - with `deg_use_concept_graph=False`
- Forced mainline config equivalence:
  - `scale_mode=dual`
  - `prototype_number=16`
  - `use_concept_prompt_pool=True`
  - `rce_use_concept_prior=True`
  - `rce_use_visual_residual=True`
  - `rce_visual_residual_init=0.05`
  - `rce_use_logit_calibration=True`
  - `rce_use_cross_scale_graph=True`
  - `rce_cross_scale_graph_init=0.1`
  - `rce_cross_scale_graph_norm=sqrt`
- Exported outputs under:
  - `results_stage32/stage32_rce_v4_csg_evidence_export/`
- Generated files:
  - `stage32_slide_evidence_summary.csv`
  - `stage32_top_concepts_long.csv`
  - `stage32_top_csg_pairs.csv`
  - `stage32_error_cases.csv`
  - `stage32_manifest.json`
  - `stage32_evidence_export_report.md`
- The exporter is warning-tolerant:
  - missing checkpoint / split / optional debug attributes produce graceful warnings and a report instead of hard crashing.

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/export_stage32_rce_v4_csg_evidence.py`: passed
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage32_export_evidence.sh`: passed
- `/home/ljh/anaconda3/envs/vila_mil/bin/python ViLa-MIL-main/scripts/analysis/export_stage32_rce_v4_csg_evidence.py --fold 0 --split test`: passed

### Commands not run
- No training run
- No 5-fold evaluation run
- No feature extraction
- No model computation file change

### Results / observations
- Step32 only adds read-only evidence export tooling and does not modify:
  - `models/model_RCE_MIL_BiomedCLIP.py`
  - `models/model_DEG_MIL_BiomedCLIP.py`
  - `main.py`
  - `utils/core_utils.py`
  - dataset files
  - Stage22/23/24/25/26/27/28/29/30/31 result directories
- Verified fold0 test export outputs:
  - `results_stage32/stage32_rce_v4_csg_evidence_export/stage32_slide_evidence_summary.csv`
  - `results_stage32/stage32_rce_v4_csg_evidence_export/stage32_top_concepts_long.csv`
  - `results_stage32/stage32_rce_v4_csg_evidence_export/stage32_top_csg_pairs.csv`
  - `results_stage32/stage32_rce_v4_csg_evidence_export/stage32_error_cases.csv`
  - `results_stage32/stage32_rce_v4_csg_evidence_export/stage32_manifest.json`
  - `results_stage32/stage32_rce_v4_csg_evidence_export/stage32_evidence_export_report.md`
- Verified fold0 test export metrics:
  - `AUC=0.9757`
  - `ACC=0.9072`
  - `F1=0.8988`
  - `Balanced ACC=0.9077`
  - `PR-AUC=0.9548`
- Verified fold0 test export counts:
  - `194` slides exported
  - `176` correct
  - `18` error cases
- The exporter is designed to support later evidence-side work:
  - evidence failure / conflict analysis
  - evidence-level gated residual
  - evidence consistency / margin objectives

### Next suggested step
- Run the Step32 exporter on fold0 test first, inspect the top concepts / top CSG pairs / error cases, then move to Step33 evidence failure and conflict analysis.

## 2026-06-13 - Step33: Evidence Failure / Conflict Analysis

### Goal
- Add a read-only Step33 analysis path for diagnosing error cases, evidence-source conflicts, visual residual dominance, and CSG weakness from Step32 evidence exports.

### Files changed
- `scripts/analysis/build_stage33_evidence_failure_analysis.py`: added the Step33 evidence failure/conflict analyzer.
- `scripts/analysis/run_stage33_evidence_failure_analysis.sh`: added an optional launcher with environment-variable overrides for input/output paths and thresholds.
- `docs/CODEX_HANDOFF.md`: appended this Step33 record.

### Behavior / script config
- Default input directory:
  - `results_stage32/stage32_rce_v4_csg_evidence_export/`
- Default output directory:
  - `results_stage33/stage33_evidence_failure_analysis/`
- Read Step32 files:
  - `stage32_slide_evidence_summary.csv`
  - `stage32_top_concepts_long.csv`
  - `stage32_top_csg_pairs.csv`
  - `stage32_error_cases.csv`
  - `stage32_manifest.json`
  - optional `stage32_evidence_export_report.md`
- Generated files:
  - `stage33_slide_failure_labels.csv`
  - `stage33_error_failure_cases.csv`
  - `stage33_evidence_source_stats.csv`
  - `stage33_low_high_conflict_summary.csv`
  - `stage33_visual_residual_diagnostics.csv`
  - `stage33_csg_diagnostics.csv`
  - `stage33_prompt_confusion_summary.csv`
  - `stage33_prompt_reliability_preview.csv`
  - `stage33_failure_type_counts.csv`
  - `stage33_recommendations.json`
  - `stage33_evidence_failure_report.md`
- Important diagnosis convention:
  - visual and CSG dominance are computed from effective residual contribution:
    - `visual_alpha * visual_margin`
    - `csg_alpha * csg_margin`
  - this avoids overstating raw residual logits when comparing them against low/high concept margins.

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage33_evidence_failure_analysis.py`: passed
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage33_evidence_failure_analysis.sh`: passed
- `/home/ljh/anaconda3/envs/vila_mil/bin/python ViLa-MIL-main/scripts/analysis/build_stage33_evidence_failure_analysis.py`: passed

### Commands not run
- No training run
- No 5-fold evaluation run
- No feature extraction
- No model computation file change

### Results / observations
- Step33 only adds read-only analysis tooling and does not modify:
  - `models/model_RCE_MIL_BiomedCLIP.py`
  - `models/model_DEG_MIL_BiomedCLIP.py`
  - `main.py`
  - `utils/core_utils.py`
  - dataset files
  - Stage22/23/24/25/26/27/28/29/30/31/32 result files
- Verified Step33 input set:
  - reads the exported Step32 fold0/test evidence from `results_stage32/stage32_rce_v4_csg_evidence_export/`
- Verified Step33 output set:
  - all requested Stage33 CSV / JSON / Markdown files were generated under `results_stage33/stage33_evidence_failure_analysis/`
- Core fold0/test counts:
  - `194` slides
  - `176` correct
  - `18` error
- Primary failure counts:
  - `visual_residual_override`: `13 / 18`
  - `low_high_conflict`: `4 / 18`
  - `concept_wrong_class_drift`: `1 / 18`
- Visual residual diagnosis:
  - after alpha-scaling, mean visual source ratio is still high at about `0.8286`
  - visual supports the wrong class on `100%` of error slides
  - Step34 should compare a scalar visual gate first, with init around `0.00` or `0.01`
- CSG diagnosis:
  - effective mean abs CSG margin is about `0.000037`
  - raw mean abs CSG margin is about `0.000377`
  - mean top pair score is about `0.000009`
  - current conclusion is that CSG is present but far weaker than low/high concept evidence and the visual residual on exported fold0/test evidence
  - pair-export mismatch does not appear to be the main issue on this fold
- Low/high conflict diagnosis:
  - conflict on `10 / 18` error slides
  - both low and high support the wrong class on `7 / 18` error slides
  - high-scale wrong-class dominance appears often enough to justify later high-branch margin control
- Prompt confusion preview:
  - `salt-and-pepper chromatin` is the clearest high-risk error concept on the exported fold0/test evidence
- Step34 recommendation:
  - start with evidence-level scalar gating for the visual residual
  - compare `gate_init=0.00` vs `0.01` vs current-equivalent `0.05`
  - keep CSG gate only as a secondary ablation while visual gating is tested

### Next suggested step
- Step34 should implement evidence-level gated residual fusion, starting with a scalar visual gate before trying slide-adaptive gating, and Step35 can follow with high-scale margin control or low-high consistency loss if the Step34 gate does not remove the dominant error patterns.

## 2026-06-14 - Step34: Visual Evidence-Level Gate smoke

### Files changed
- `main.py`: added CLI args for the optional visual evidence gate.
- `utils/core_utils.py`: passed the new gate args into `DEG_MIL_BiomedCLIP` config.
- `models/model_DEG_MIL_BiomedCLIP.py`: added the scalar visual evidence gate and export buffers for gated visual residual diagnostics.
- `scripts/experiments/run_stage34_visual_gate_smoke.sh`: added the Stage34 smoke launcher.
- `docs/CODEX_HANDOFF.md`: appended this Step34 record.

### Added args
- `--rce_use_visual_evidence_gate`
- `--rce_visual_gate_init`

### Behavior
- Step34 only changes `DEG_MIL_BiomedCLIP` visual residual fusion.
- Legacy behavior is preserved when `--rce_use_visual_evidence_gate` is not set:
  - `final_logits = final_logits + alpha * visual_logits`
- Gated behavior when enabled:
  - `visual_residual_contribution = alpha * visual_logits`
  - `gate = sigmoid(rce_visual_evidence_gate)`
  - `final_logits = final_logits + gate * visual_residual_contribution`
- `rce_visual_gate_init` is clamped to `[1e-6, 1 - 1e-6]` before `torch.logit(...)`, so `0.00 / 0.01 / 0.05 / 1.0` stay numerically safe.
- If `rce_use_visual_evidence_gate=True` but `rce_use_visual_residual=False`, the model logs a warning and the gate does not affect forward.
- Added forward export buffers:
  - `last_visual_evidence_gate`
  - `last_visual_residual_contribution`
  - `last_visual_gated_contribution`
- Existing Step32/Step33 evidence fields remain intact.

### Smoke variants
- `gate0`: `--rce_use_visual_evidence_gate --rce_visual_gate_init 0.00`
- `gate001`: `--rce_use_visual_evidence_gate --rce_visual_gate_init 0.01`
- `gate005`: `--rce_use_visual_evidence_gate --rce_visual_gate_init 0.05`
- `all`: runs `gate0`, `gate001`, and `gate005` in sequence

### Checks run
- `python -m py_compile ViLa-MIL-main/main.py`
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py`
- `python -m py_compile ViLa-MIL-main/models/model_DEG_MIL_BiomedCLIP.py`
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage34_visual_gate_smoke.sh`
- `VARIANT=gate001 bash ViLa-MIL-main/scripts/experiments/run_stage34_visual_gate_smoke.sh`

### Commands not run
- No 5-fold evaluation run
- No dataset file change

### Smoke run result
- Output directory:
  - `results_stage34/stage34_visual_gate_smoke_gate001_s1`
- Fold0 / 1 epoch / testing metrics:
  - `AUC=0.5000`
  - `ACC=0.2105`
  - `F1=0.1739`
  - `Balanced ACC=0.5000`
  - `Sensitivity=1.0000`
  - `Specificity=0.0000`
- This smoke run confirms the Stage34 gate path executes end-to-end with the requested DEG skeleton configuration and does not replace formal 5-fold comparison.

### Next suggested step
- Step35: Visual Gate 5-fold，比较 skeleton、gate0、gate001、gate005、no_visual_residual，并观察 AUC/ACC/F1/Balanced ACC/Sensitivity 以及 visual_residual_override 是否减少。

## 2026-06-14 - Step35: Visual Gate 5-fold + Integrity Audit + Summary

### Files changed
- `scripts/analysis/check_stage35_visual_gate_integrity.py`: added the Step35 visual-gate integrity audit.
- `scripts/experiments/run_stage35_visual_gate_5fold.sh`: added the Stage35 5-fold launcher.
- `scripts/analysis/build_stage35_visual_gate_summary.py`: added the Stage35 result summarizer and recommender.
- `docs/CODEX_HANDOFF.md`: appended this Step35 record.

### Integrity checks
- Confirms `main.py` contains:
  - `--rce_use_visual_evidence_gate`
  - `--rce_visual_gate_init`
- Confirms `utils/core_utils.py` passes both args into `DEG_MIL_BiomedCLIP`.
- Confirms `models/model_DEG_MIL_BiomedCLIP.py` contains:
  - `rce_use_visual_evidence_gate`
  - `rce_visual_evidence_gate`
  - `last_visual_evidence_gate`
  - `last_visual_residual_contribution`
  - `last_visual_gated_contribution`
- Confirms the forward path keeps:
  - `visual_residual_contribution = alpha * visual_logits`
  - `gate = sigmoid(rce_visual_evidence_gate)`
  - `visual_gated_contribution = gate * visual_residual_contribution`
  - `final_logits += visual_gated_contribution`
- Confirms the gate-off path stays equivalent to the original skeleton by falling back to ungated visual residual contribution.

### Variants
- `skeleton`
- `no_visual_residual`
- `gate0`
- `gate001`
- `gate005`
- `gate05`
- `gate1`
- `all`

### Expected result directories
- `results_stage35/visual_gate_skeleton_5fold_e20_s1`
- `results_stage35/visual_gate_no_visual_residual_5fold_e20_s1`
- `results_stage35/visual_gate_gate0_5fold_e20_s1`
- `results_stage35/visual_gate_gate001_5fold_e20_s1`
- `results_stage35/visual_gate_gate005_5fold_e20_s1`
- `results_stage35/visual_gate_gate05_5fold_e20_s1`
- `results_stage35/visual_gate_gate1_5fold_e20_s1`

### How to run
- Integrity audit:
  - `python ViLa-MIL-main/scripts/analysis/check_stage35_visual_gate_integrity.py`
- Single variant 5-fold:
  - `cd ViLa-MIL-main`
  - `VARIANT=skeleton bash scripts/experiments/run_stage35_visual_gate_5fold.sh`
- Short Step35 smoke:
  - `cd ViLa-MIL-main`
  - `MAX_EPOCHS=1 K_START=0 K_END=0 VARIANT=gate1 bash scripts/experiments/run_stage35_visual_gate_5fold.sh`
- Summary after 5-fold results exist:
  - `python ViLa-MIL-main/scripts/analysis/build_stage35_visual_gate_summary.py`

### Checks run
- `python -m py_compile ViLa-MIL-main/main.py`
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py`
- `python -m py_compile ViLa-MIL-main/models/model_DEG_MIL_BiomedCLIP.py`
- `python -m py_compile ViLa-MIL-main/scripts/analysis/check_stage35_visual_gate_integrity.py`
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage35_visual_gate_summary.py`
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage35_visual_gate_5fold.sh`
- `python ViLa-MIL-main/scripts/analysis/check_stage35_visual_gate_integrity.py`
- `cd ViLa-MIL-main && MAX_EPOCHS=1 K_START=0 K_END=0 VARIANT=gate1 bash scripts/experiments/run_stage35_visual_gate_5fold.sh`
- `cd ViLa-MIL-main && MAX_EPOCHS_FILTER=1 SEED_FILTER=1 python scripts/analysis/build_stage35_visual_gate_summary.py`

### Commands not run
- Formal `VARIANT=all` 5-fold
- Any full 5-fold gate sweep

### Smoke run result
- Output directory:
  - `results_stage35/visual_gate_gate1_5fold_e1_s1`
- Fold0 / 1 epoch metrics:
  - `AUC=0.9704`
  - `ACC=0.8505`
  - `F1=0.8129`
  - `Balanced ACC=0.7876`
  - `Sensitivity=0.5909`
  - `Specificity=0.9844`
  - `PR-AUC=0.9427`
- Partial summary check:
  - `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_summary.csv`
  - `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_metric_deltas.csv`
  - `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_rankings.csv`
  - `results_stage35/stage35_visual_gate_summary/stage35_visual_gate_report.md`
  - `results_stage35/stage35_visual_gate_summary/stage35_recommendations.json`

### Next suggested step
- 如果 Step35 找到优于 skeleton 或 sensitivity/specificity 更均衡的 gate variant，则 Step36 对最佳 gate variant 做 Step32/33 风格的 evidence re-export 和 failure analysis，检查 visual_residual_override 是否下降。如果所有 gate variant 都不如 skeleton，则保留 visual gate 为 negative/diagnostic ablation，下一步转向 Low-High Evidence Consistency Loss。

## 2026-06-14 - Step36: Low-High Evidence Consistency Loss smoke

### Files changed
- `main.py`: added Step36 CLI switches.
- `utils/core_utils.py`: passed the Step36 switches into `DEG_MIL_BiomedCLIP` config.
- `models/model_DEG_MIL_BiomedCLIP.py`: added the optional low-high consistency auxiliary loss and diagnostics export.
- `scripts/experiments/run_stage36_lh_consistency_smoke.sh`: added the Stage36 smoke launcher.
- `scripts/analysis/check_stage36_lh_consistency_integrity.py`: added the Stage36 integrity audit.
- `docs/CODEX_HANDOFF.md`: appended this Step36 record.

### Added args
- `--rce_use_low_high_consistency_loss`
- `--rce_lh_consistency_lambda`
- `--rce_lh_consistency_margin`
- Defaults:
  - `rce_use_low_high_consistency_loss = False`
  - `rce_lh_consistency_lambda = 0.0`
  - `rce_lh_consistency_margin = 0.0`

### Loss definition
- Scope: `DEG_MIL_BiomedCLIP` only.
- Active only when:
  - `scale_mode == dual`
  - `rce_use_low_high_consistency_loss == True`
- Margin helper:
  - true-class logit minus the max wrong-class logit, computed independently for `logits_low` and `logits_high`
- Auxiliary loss:
  - `low_loss = relu(margin - low_margin)`
  - `high_loss = relu(margin - high_margin)`
  - `lh_consistency_loss = mean(low_loss + high_loss)`
- Final training loss:
  - `loss = ce_loss + lambda * lh_consistency_loss`
- Default-off behavior:
  - `loss` remains the original CE loss path
  - `final_logits` computation is unchanged
  - no region graph / concept graph / visual gate behavior is modified

### Exported diagnostics
- `last_low_scale_logits`
- `last_high_scale_logits`
- `last_low_true_wrong_margin`
- `last_high_true_wrong_margin`
- `last_lh_margin_gap`
- `last_lh_consistency_loss`
- `last_total_loss`

### Smoke variants
- `lh_l001_m0`
- `lh_l005_m0`
- `lh_l001_m005`
- `lh_l005_m005`
- `all`

### How to run
- Integrity audit:
  - `python ViLa-MIL-main/scripts/analysis/check_stage36_lh_consistency_integrity.py`
- Single smoke variant:
  - `cd ViLa-MIL-main`
  - `VARIANT=lh_l001_m005 bash scripts/experiments/run_stage36_lh_consistency_smoke.sh`
- Run all smoke variants:
  - `cd ViLa-MIL-main`
  - `VARIANT=all bash scripts/experiments/run_stage36_lh_consistency_smoke.sh`

### Expected result directories
- `results_stage36/stage36_lh_consistency_smoke_lh_l001_m0_s1`
- `results_stage36/stage36_lh_consistency_smoke_lh_l005_m0_s1`
- `results_stage36/stage36_lh_consistency_smoke_lh_l001_m005_s1`
- `results_stage36/stage36_lh_consistency_smoke_lh_l005_m005_s1`

### Checks run
- `python -m py_compile ViLa-MIL-main/main.py`
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py`
- `python -m py_compile ViLa-MIL-main/models/model_DEG_MIL_BiomedCLIP.py`
- `python -m py_compile ViLa-MIL-main/scripts/analysis/check_stage36_lh_consistency_integrity.py`
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage36_lh_consistency_smoke.sh`
- `python ViLa-MIL-main/scripts/analysis/check_stage36_lh_consistency_integrity.py`
- `cd ViLa-MIL-main && bash scripts/experiments/run_stage36_lh_consistency_smoke.sh`

### Commands not run
- No `VARIANT=all` multi-variant smoke sweep
- No 5-fold Step36 evaluation
- No evidence re-export / failure analysis

### Smoke run result
- Output directory:
  - `results_stage36/stage36_lh_consistency_smoke_lh_l001_m0_s1`
- Fold0 / 1 epoch / testing metrics:
  - `AUC=0.5833`
  - `ACC=0.2105`
  - `F1=0.1739`
  - `Balanced ACC=0.5000`
  - `Sensitivity=1.0000`
  - `Specificity=0.0000`
  - `PR-AUC=0.3375`

### Next suggested step
- 跑 `lh_l005_m0`、`lh_l001_m005`、`lh_l005_m005` 三个 smoke 变体，先看是否有比 `lh_l001_m0` 更稳定的 validation/test 方向；若 smoke 没有明显异常，再进入 5-fold 比较。

## 2026-06-14 - Step37: Low-High Consistency 5-fold + Summary

### Files changed
- `scripts/experiments/run_stage37_lh_consistency_5fold.sh`: added the formal Stage37 5-fold launcher.
- `scripts/analysis/build_stage37_lh_consistency_summary.py`: added the Stage37 summary/ranking/recommendation builder.
- `scripts/analysis/check_stage37_lh_consistency_5fold_integrity.py`: added the Stage37 run-script and summary-script integrity audit.
- `docs/CODEX_HANDOFF.md`: appended this Step37 record.

### Variants
- `skeleton`: DEG skeleton control; does not pass `--rce_use_low_high_consistency_loss`.
- `lh_l0001_m0`: lambda `0.001`, margin `0.00`.
- `lh_l0005_m0`: lambda `0.005`, margin `0.00`.
- `lh_l001_m0`: lambda `0.01`, margin `0.00`.
- `lh_l0005_m002`: lambda `0.005`, margin `0.02`.
- `lh_l001_m002`: lambda `0.01`, margin `0.02`.
- `lh_l001_m005`: lambda `0.01`, margin `0.05`.
- `lh_l005_m0`: lambda `0.05`, margin `0.00`.
- `lh_l005_m005`: lambda `0.05`, margin `0.05`.
- `all`: runs all variants in sequence; not run automatically.

### Expected result directories
- `results_stage37/lh_consistency_skeleton_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l0001_m0_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l0005_m0_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l0005_m002_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l001_m002_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l001_m005_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l005_m0_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l005_m005_5fold_e20_s1`

### Summary outputs
- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_summary.csv`
- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_metric_deltas.csv`
- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_rankings.csv`
- `results_stage37/stage37_lh_consistency_summary/stage37_lh_consistency_report.md`
- `results_stage37/stage37_lh_consistency_summary/stage37_recommendations.json`

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage37_lh_consistency_summary.py`
- `python -m py_compile ViLa-MIL-main/scripts/analysis/check_stage37_lh_consistency_5fold_integrity.py`
- `bash -n ViLa-MIL-main/scripts/experiments/run_stage37_lh_consistency_5fold.sh`
- `python ViLa-MIL-main/scripts/analysis/check_stage37_lh_consistency_5fold_integrity.py`
- `cd ViLa-MIL-main && MAX_EPOCHS=1 K_START=0 K_END=0 VARIANT=lh_l0005_m0 bash scripts/experiments/run_stage37_lh_consistency_5fold.sh`
- `cd ViLa-MIL-main && MAX_EPOCHS_FILTER=1 SEED_FILTER=1 python scripts/analysis/build_stage37_lh_consistency_summary.py`

### Commands not run
- Formal `VARIANT=all` 5-fold.
- Any full e20 5-fold Stage37 sweep.
- Step38 evidence re-export / failure analysis.
- Any model-body logic change.

### Short smoke result
- Output directory:
  - `results_stage37/lh_consistency_lh_l0005_m0_5fold_e1_s1`
- Fold0 / 1 epoch metrics:
  - `AUC=0.9705`
  - `ACC=0.8505`
  - `F1=0.8129`
  - `Balanced ACC=0.7876`
  - `Sensitivity=0.5909`
  - `Specificity=0.9844`
  - `PR-AUC=0.9429`
- Partial summary check:
  - Only `lh_l0005_m0` e1 was loaded; `skeleton` and other variants were expectedly marked `missing`.
  - This validates summary generation but is not a formal Step37 conclusion.

### Next suggested step
- 如果 Step37 找到优于 skeleton 或 sensitivity/specificity 更均衡的 consistency variant，则 Step38 对最佳 variant 做 Step32/33 风格的 evidence re-export + failure analysis，重点检查 low_high_conflict 是否下降。如果所有 consistency variants 都不如 skeleton，则保留 Step36/37 为 negative diagnostic ablation，下一步转向 Prompt Reliability / Refined Prompt Pool。

## 2026-06-14 - Step38: Skeleton vs LH-Consistency Evidence Re-export + Failure Comparison

### Files changed
- `scripts/analysis/build_stage33_evidence_failure_analysis.py`: added optional `--variant_name` for report/recommendation labeling while keeping default behavior compatible.
- `scripts/analysis/run_stage38_lh_consistency_evidence_compare.sh`: added the Step38 end-to-end runner for evidence export, failure analysis, and comparison.
- `scripts/analysis/build_stage38_lh_consistency_failure_comparison.py`: added the Step38 comparison builder.
- `docs/CODEX_HANDOFF.md`: appended this Step38 record.

### Compared variants
- `skeleton`
- `lh_l001_m0`

### Input result directories
- `results_stage37/lh_consistency_skeleton_5fold_e20_s1`
- `results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1`

### Generated outputs
- `results_stage38/evidence_export_skeleton_fold0_test/`
- `results_stage38/evidence_export_lh_l001_m0_fold0_test/`
- `results_stage38/failure_analysis_skeleton_fold0_test/`
- `results_stage38/failure_analysis_lh_l001_m0_fold0_test/`
- `results_stage38/stage38_lh_consistency_failure_comparison/`

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage38_lh_consistency_failure_comparison.py`
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage38_lh_consistency_evidence_compare.sh`
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage33_evidence_failure_analysis.py`
- `cd ViLa-MIL-main && bash scripts/analysis/run_stage38_lh_consistency_evidence_compare.sh`

### Commands not run
- No training command.
- No model-body logic change.
- No region graph / concept graph / visual evidence gate ablation.

### Key comparison results
- Metric deltas (`lh_l001_m0 - skeleton`):
  - `AUC=-0.0039`
  - `ACC=+0.0206`
  - `F1=+0.0208`
  - `Balanced ACC=+0.0120`
  - `Sensitivity=-0.0152`
  - `Specificity=+0.0391`
  - `PR-AUC=-0.0034`
- Error-set comparison:
  - `fixed cases = 6`
  - `regressed cases = 2`
  - `persistent errors = 12`
- Failure-type comparison:
  - `low_high_conflict`: `10 -> 8`
  - `visual_residual_override`: `13 -> 14`
- Low/high conflict diagnostics:
  - error `low_high_conflict` count delta: `-2`
  - error `both_support_wrong` count delta: `-2`
  - fixed cases with repaired `low_high_conflict`: `4`
- Visual residual diagnostics:
  - `visual_residual_override` delta: `+1`
  - error-time `visual_source_ratio` mean increased from `0.6286` to `0.7949`

### Recommendation
- Current recommendation remains: keep `skeleton` as the final main model.
- Reason:
  - `lh_l001_m0` fixes more cases than it regresses and does reduce some `low_high_conflict`,
  - but it also increases `visual_residual_override` and slightly lowers `AUC/PR-AUC`.
- Interpretation:
  - `lh_l001_m0` is a meaningful diagnostic / secondary trade-off variant,
  - but the evidence is not clean enough to replace `skeleton` as the final primary model.

### Next suggested step
- 如果 Step38 证明 lh_l001_m0 明确减少 low_high_conflict 且 fixed cases 多于 regressed cases，则 Step39 做 final model evidence visualization / paper-ready figures。否则 Step39 转向 Prompt Reliability / Refined Prompt Pool，或直接固定 skeleton 为最终主模型并整理论文主图。

## 2026-06-15 - Step39: Final Evidence Package and Paper-Ready Summary

### Files changed
- `scripts/analysis/build_stage39_final_evidence_package.py`: added the Step39 final aggregation/report builder.
- `scripts/analysis/run_stage39_final_evidence_package.sh`: added the Step39 runner with configurable input/output paths.
- `docs/CODEX_HANDOFF.md`: appended this Step39 record.

### Input result dirs
- `results_stage24/stage24_rce_v4_csg_summary/`
- `results_stage28/stage28_deg_region_graph_summary/`
- `results_stage31/stage31_deg_concept_graph_summary/`
- `results_stage35/`
- `results_stage37/stage37_lh_consistency_summary/`
- `results_stage38/stage38_lh_consistency_failure_comparison/`
- `results_stage38/evidence_export_skeleton_fold0_test/`
- `results_stage38/evidence_export_lh_l001_m0_fold0_test/`

### Generated outputs
- `results_stage39/final_evidence_package/stage39_final_model_recommendation.json`
- `results_stage39/final_evidence_package/stage39_final_performance_summary.csv`
- `results_stage39/final_evidence_package/stage39_ablation_summary.csv`
- `results_stage39/final_evidence_package/stage39_negative_ablation_summary.csv`
- `results_stage39/final_evidence_package/stage39_evidence_calibration_summary.csv`
- `results_stage39/final_evidence_package/stage39_failure_comparison_summary.csv`
- `results_stage39/final_evidence_package/stage39_fixed_regressed_persistent_cases.csv`
- `results_stage39/final_evidence_package/stage39_top_concepts_for_examples.csv`
- `results_stage39/final_evidence_package/stage39_top_csg_pairs_for_examples.csv`
- `results_stage39/final_evidence_package/stage39_final_innovation_points.md`
- `results_stage39/final_evidence_package/stage39_paper_ready_summary.md`
- `results_stage39/final_evidence_package/stage39_final_next_steps.md`

### Final recommendation
- Recommended default model:
  - `RCE-v4-CSG-a01-rq16 / DEG skeleton`
- Secondary trade-off variant:
  - `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0`
- Reason:
  - `skeleton` keeps the stronger AUC / PR-AUC and remains the most robust default.
  - `lh_l001_m0` reduces fold0/test errors and low-high conflict, but increases `visual_residual_override` and slightly reduces AUC / PR-AUC.

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage39_final_evidence_package.py`
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage39_final_evidence_package.sh`
- `cd ViLa-MIL-main && bash scripts/analysis/run_stage39_final_evidence_package.sh`

### Commands not run
- No training command.
- No model-body logic modification.
- No region graph / concept graph / visual evidence gate experiment rerun.
- No new evidence export beyond existing Step38 inputs.

### Next suggested step
- Step40：根据 stage39_paper_ready_summary.md 生成最终论文主图/方法图说明、实验表格说明和答辩汇报材料。如果还要继续模型创新，可以转向 Prompt Reliability / Refined Prompt Pool，但不建议再盲目堆 graph 或 gate。

## 2026-06-15 - Step40: Paper-Ready Figures / Tables / Defense Package

### Files changed
- `scripts/analysis/build_stage40_paper_ready_assets.py`: added the Step40 paper-ready asset builder.
- `scripts/analysis/run_stage40_paper_ready_assets.sh`: added the Step40 runner with `PYTHON_BIN` / `STAGE39_DIR` / `OUTPUT_DIR` overrides.
- `docs/CODEX_HANDOFF.md`: appended this Step40 record.

### Input dirs
- `results_stage39/final_evidence_package/`

### Generated outputs
- `results_stage40/paper_ready_assets/stage40_method_overview.md`
- `results_stage40/paper_ready_assets/stage40_main_figure_mermaid.md`
- `results_stage40/paper_ready_assets/stage40_evidence_pipeline_mermaid.md`
- `results_stage40/paper_ready_assets/stage40_experiment_tables_plan.md`
- `results_stage40/paper_ready_assets/stage40_ablation_table_paper_ready.csv`
- `results_stage40/paper_ready_assets/stage40_negative_ablation_table_paper_ready.csv`
- `results_stage40/paper_ready_assets/stage40_failure_case_table_paper_ready.csv`
- `results_stage40/paper_ready_assets/stage40_defense_slide_outline.md`
- `results_stage40/paper_ready_assets/stage40_paper_section_draft.md`
- `results_stage40/paper_ready_assets/stage40_final_claims_and_limitations.md`
- `results_stage40/paper_ready_assets/stage40_next_research_branch.md`

### Final default model
- `RCE-v4-CSG-a01-rq16 / DEG skeleton`

### Secondary trade-off variant
- `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0`

### Final paper claims
- `RCE-v4-CSG-a01-rq16` is the most stable final default model.
- `CSG a01` and `rq16` are experimentally supported.
- `region graph / concept graph / scalar gate` should be retained as negative ablation evidence rather than promoted to the final model.
- `low-high consistency` reduces `low-high conflict` but remains a secondary trade-off variant.
- The final pipeline supports evidence source decomposition and failure diagnosis.

### Checks run
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage40_paper_ready_assets.py`
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage40_paper_ready_assets.sh`
- `cd ViLa-MIL-main && bash scripts/analysis/run_stage40_paper_ready_assets.sh`

### Commands not run
- No training command.
- No model-body logic modification.
- No new graph / gate / consistency experiment.
- No overwrite of Step39 original outputs.

### Next suggested step
- 如果当前目标是论文/答辩，下一步根据 Step40 输出生成最终 Word/PPT。如果还要继续模型创新，Step41 可启动 Prompt Reliability / Refined Prompt Pool 分支，但不要再盲目堆 graph 或 gate。

## 2026-06-15 - Step41: Low-High Coordinate Correspondence Audit

### Step 名称
- `Low-High Coordinate Correspondence Audit`

### 目标
- 不训练模型、不修改模型前向逻辑，只审计 `features_biomedclip_5x` 和 `features_biomedclip_20x` 的原始 patch coords 是否支持真实 low-high 空间对应。
- 重点结论是：后续 HCRC / Step42 是否可以基于真实 coords 继续做 `CGSP Anchor Selection Audit`，而不是继续依赖 `DEG` 中 attention-centroid 风格的 region coords。

### 新增文件
- `scripts/analysis/build_stage41_low_high_coordinate_audit.py`
- `scripts/analysis/run_stage41_low_high_coordinate_audit.sh`
- `docs/CODEX_HANDOFF.md`

### 默认输入路径
- `DATA_ROOT_DIR=/xiangmu/data/VILMIL`
- `DATA_FOLDER_S=features_biomedclip_5x`
- `DATA_FOLDER_L=features_biomedclip_20x`
- `CSV_PATH=dataset_csv/all_data.csv`
- `SPLIT_DIR=splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- `FOLD=0`
- `SPLIT=test`

### 输出文件
- `results_stage41/low_high_coordinate_audit/low_high_coordinate_match_stats.csv`
- `results_stage41/low_high_coordinate_audit/child_count_distribution.csv`
- `results_stage41/low_high_coordinate_audit/empty_match_cases.csv`
- `results_stage41/low_high_coordinate_audit/example_low_high_pairs.csv`
- `results_stage41/low_high_coordinate_audit/coordinate_match_report.md`
- `results_stage41/low_high_coordinate_audit/stage41_manifest.json`

### 如何运行
- `cd ViLa-MIL-main && bash scripts/analysis/run_stage41_low_high_coordinate_audit.sh`

### 验证命令
- `python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage41_low_high_coordinate_audit.py`
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage41_low_high_coordinate_audit.sh`
- `cd ViLa-MIL-main && MAX_SLIDES=3 OUTPUT_DIR=results_stage41/low_high_coordinate_audit_smoke bash scripts/analysis/run_stage41_low_high_coordinate_audit.sh`
- `cd ViLa-MIL-main && bash scripts/analysis/run_stage41_low_high_coordinate_audit.sh`

### 是否实际运行
- 是。
- smoke run:
  - `results_stage41/low_high_coordinate_audit_smoke/`
  - `MAX_SLIDES=3`
- formal run:
  - `results_stage41/low_high_coordinate_audit/`
  - `fold=0`
  - `split=test`
  - `processed slides = 194`
  - `missing slides = 0`
  - `failed slides = 0`

### 初步结论
- 当前 low/high 原始 coords 在坐标范围上高度一致，`same_coord_system_heuristic rate = 1.0000`。
- low/high h5 结构稳定：
  - 两侧都使用 `coords` key。
  - 本次审计覆盖的 194 张 slide 中，`patch_level` / `patch_size` attrs 均缺失，因此不能依赖 attrs 做尺度转换，只能靠统计匹配判断。
- 推荐组合：
  - `coord_mode = top_left`
  - `scale_ratio = 1.0`
- 推荐组合统计：
  - `total low patches = 41068`
  - `total high patches = 583366`
  - `weighted empty match ratio = 0.1888`
  - `median child count = 1.0`
  - `mean child count = 0.7922`
- 其他 ratio (`2 / 4 / 0.5 / 0.25`) 的 empty match ratio 几乎全部接近 `1.0`，不支持显式尺度放缩假设。
- 这说明当前 `features_biomedclip_5x` 与 `features_biomedclip_20x` 更像是已经处于同一 WSI 坐标系，至少在原始 patch coords 层面不需要再乘 `2x` 或 `4x` 才能建立 low-high correspondence。

### 下一步建议
- 可以进入 Step42，做 `CGSP Anchor Selection Audit`，并且必须基于 Step41 审计通过的真实 patch coords，而不是 `DEG` 里的 attention-weighted centroid。
- 但要保留一个明确风险：
  - 即使推荐组合通过，当前推荐组合仍有 `weighted empty match ratio = 0.1888`，说明不是所有 low patches 都有高倍 child patches。
  - Step42 应继续检查 anchor 选择覆盖率、empty-anchor 分布，以及是否需要对无 child 的 low patch 做跳过或单独标记。

## 2026-06-15 - Step42: CGSP Anchor Selection Audit

### Step 名称
- `CGSP Anchor Selection Audit`

### 目标
- 不训练模型、不修改模型前向逻辑，只验证 Concept-Guided Spatial Proposal 是否能从 low patch 层面选出合理 spatial anchors。
- 审计 selected anchors 是否能稳定匹配 high child patches，为 Step43 `HCRC-Light` 做准备。
- 本步骤严格基于 Step41 推荐的真实原始 patch coords：
  - `coord_mode=top_left`
  - `scale_ratio=1.0`

### 新增文件
- `scripts/analysis/build_stage42_cgsp_anchor_selection_audit.py`
- `scripts/analysis/run_stage42_cgsp_anchor_selection_audit.sh`
- `docs/CODEX_HANDOFF.md`

### 输入
- low features:
  - `/xiangmu/data/VILMIL/features_biomedclip_5x`
- high features:
  - `/xiangmu/data/VILMIL/features_biomedclip_20x`
- slide CSV:
  - `dataset_csv/all_data.csv`
- split:
  - `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100/splits_0.csv`
  - `split=test`
- Step41 manifest:
  - `results_stage41/low_high_coordinate_audit/stage41_manifest.json`
- concept prompt pool:
  - `dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
- prompt feature source used for actual runs:
  - `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt`
  - source type: checkpoint buffers `low_prompt_features` / `high_prompt_features`

### 输出
- `results_stage42/cgsp_anchor_selection_audit/cgsp_patch_scores.csv`
- `results_stage42/cgsp_anchor_selection_audit/cgsp_candidate_proposals.csv`
- `results_stage42/cgsp_anchor_selection_audit/cgsp_selected_anchors.csv`
- `results_stage42/cgsp_anchor_selection_audit/cgsp_anchor_child_match_stats.csv`
- `results_stage42/cgsp_anchor_selection_audit/cgsp_anchor_distribution_summary.csv`
- `results_stage42/cgsp_anchor_selection_audit/cgsp_example_anchor_child_pairs.csv`
- `results_stage42/cgsp_anchor_selection_audit/cgsp_anchor_selection_report.md`
- `results_stage42/cgsp_anchor_selection_audit/stage42_manifest.json`

### 如何运行
- 默认脚本保留 `CHECKPOINT_PATH=` 为空；若本地 BiomedCLIP cache 不稳定，建议显式传入已有 checkpoint：
  - `cd ViLa-MIL-main && CHECKPOINT_PATH=results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt bash scripts/analysis/run_stage42_cgsp_anchor_selection_audit.sh`

### 验证命令
- `/home/ljh/anaconda3/envs/vila_mil/bin/python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage42_cgsp_anchor_selection_audit.py`
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage42_cgsp_anchor_selection_audit.sh`
- `cd ViLa-MIL-main && CHECKPOINT_PATH=results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt MAX_SLIDES=3 OUTPUT_DIR=results_stage42/cgsp_anchor_selection_audit_smoke bash scripts/analysis/run_stage42_cgsp_anchor_selection_audit.sh`
- `cd ViLa-MIL-main && CHECKPOINT_PATH=results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt bash scripts/analysis/run_stage42_cgsp_anchor_selection_audit.sh`

### 是否实际运行
- 是。
- smoke run:
  - `results_stage42/cgsp_anchor_selection_audit_smoke/`
  - `MAX_SLIDES=3`
  - `processed slides = 3`
  - `failed slides = 0`
- formal run:
  - `results_stage42/cgsp_anchor_selection_audit/`
  - `fold=0`
  - `split=test`
  - `processed slides = 194`
  - `failed slides = 0`
  - `warning count = 0`

### 核心统计结论
- Prompt feature 来源：
  - checkpoint buffer，未触发 BiomedCLIP 在线构建。
- patch score 分布：
  - `patch_score mean = 0.4128`
  - `patch_score median = 0.4158`
  - `class_margin median = 0.0260`
- proposal 数量：
  - 每张 slide candidate proposals median = `64`
- selected anchors：
  - 每张 slide 都选满 `16 / 16`
  - `selected_anchor_count mean = 16.0`
  - `selected_anchor_count median = 16.0`
- anchor 空间分散性：
  - `anchor_pair_distance_mean median = 45716.9484`
  - `coverage_ratio_x median = 0.8670`
  - `coverage_ratio_y median = 0.7862`
  - 说明 anchors 并未高度集中在单个小区域。
- anchor class / concept 多样性：
  - `NonAdenocarcinoma anchors = 1592`
  - `Adenocarcinoma anchors = 1512`
  - `unique_anchor_concepts median = 6`
- high child 覆盖：
  - `bbox_expand=1.0`: `empty_anchor_ratio=0.2242`, `median_used_child_count=1.0`
  - `bbox_expand=1.5`: `empty_anchor_ratio=0.2236`, `median_used_child_count=1.0`
  - `bbox_expand=2.0`: `empty_anchor_ratio=0.2233`, `median_used_child_count=1.0`
  - `bbox_expand=3.0`: `empty_anchor_ratio=0.2226`, `median_used_child_count=1.0`
- 推荐 bbox_expand:
  - `2.0`
  - 注意：`3.0` 的 empty ratio 略低，但 child count 仍然没有实质改善；脚本选择 `2.0` 作为更保守的 HCRC-Light 候选。

### 是否建议进入 Step43
- 当前不建议直接进入 Step43 `HCRC-Light Smoke`。
- 原因：
  - anchor 选择稳定、分散、多样性够；
  - 但 high child 覆盖不足，推荐 `bbox_expand=2.0` 时 `median_used_child_count=1.0`，远低于期望的 `>=4`。

### 如果不建议进入 Step43，先调哪些参数
- 优先增大或重定义 low anchor bbox：
  - 当前 proposal bbox 基于低倍 patch 邻域，但 high child 仍然偏少，说明 patch 坐标网格/patch size 语义可能需要重新校准。
- 调整 `proposal_radius`：
  - 尝试 `1024 / 2048`，让 proposal bbox 覆盖更多 low-neighbor 组织区域。
- 调整 `nms_radius`：
  - 当前 anchors 已经分散，`nms_radius=512` 不构成主要瓶颈；可保持或略增。
- 调整 `bbox_expand_values`：
  - 增加 `4.0 / 6.0 / 8.0` 做专门 child 覆盖 sweep。
- 调整 `candidate_top_l` 与 concept coverage：
  - anchors 已能选满，`candidate_top_l=64` 当前够用；若后续扩大 bbox 后仍偏少，再增加到 `128`。
- Step43 前建议先做 Step42b：
  - `bbox_expand / proposal_radius` child coverage sweep，目标是把 `median_used_child_count` 提升到 `>=4`，再进入 HCRC-Light。

## 2026-06-15 - Step42b: CGSP Child Coverage Sweep

### Step 名称
- `CGSP Child Coverage Sweep`

### 目标
- 不训练模型、不修改模型前向逻辑，基于 Step42 的 CGSP anchor selection 做系统 sweep。
- 重点比较：
  - `bbox_containment`
  - `center_radius`
  - `bbox_then_nearest`
- 判断 high child 覆盖不足是否能通过 `proposal_radius / bbox_expand / high_radius` 解决；如果不能，再考虑 nearest fallback 或 weak correspondence。

### 新增文件
- `scripts/analysis/build_stage42b_cgsp_child_coverage_sweep.py`
- `scripts/analysis/run_stage42b_cgsp_child_coverage_sweep.sh`
- `docs/CODEX_HANDOFF.md`

### 输入
- Step41 manifest:
  - `results_stage41/low_high_coordinate_audit/stage41_manifest.json`
- Step42 manifest:
  - `results_stage42/cgsp_anchor_selection_audit/stage42_manifest.json`
- Step42 anchors:
  - `results_stage42/cgsp_anchor_selection_audit/cgsp_selected_anchors.csv`
- low/high feature dirs:
  - `/xiangmu/data/VILMIL/features_biomedclip_5x`
  - `/xiangmu/data/VILMIL/features_biomedclip_20x`
- prompt pool:
  - `dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
- checkpoint prompt buffers:
  - `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt`

### 输出
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_child_coverage_sweep_summary.csv`
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_anchor_level_child_stats_part_001.csv`
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_anchor_level_child_stats_part_002.csv`
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_anchor_stability_summary.csv`
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_strategy_comparison.csv`
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_example_anchor_child_pairs.csv`
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_child_coverage_report.md`
- `results_stage42b/cgsp_child_coverage_sweep/stage42b_manifest.json`

### 如何运行
- `cd ViLa-MIL-main && bash scripts/analysis/run_stage42b_cgsp_child_coverage_sweep.sh`
- smoke:
  - `cd ViLa-MIL-main && MAX_SLIDES=3 OUTPUT_DIR=results_stage42b/cgsp_child_coverage_sweep_smoke bash scripts/analysis/run_stage42b_cgsp_child_coverage_sweep.sh`

### 验证命令
- `/home/ljh/anaconda3/envs/vila_mil/bin/python -m py_compile ViLa-MIL-main/scripts/analysis/build_stage42b_cgsp_child_coverage_sweep.py`
- `bash -n ViLa-MIL-main/scripts/analysis/run_stage42b_cgsp_child_coverage_sweep.sh`
- `cd ViLa-MIL-main && MAX_SLIDES=3 OUTPUT_DIR=results_stage42b/cgsp_child_coverage_sweep_smoke bash scripts/analysis/run_stage42b_cgsp_child_coverage_sweep.sh`
- `cd ViLa-MIL-main && bash scripts/analysis/run_stage42b_cgsp_child_coverage_sweep.sh`

### 是否实际运行
- 是。
- smoke run:
  - `results_stage42b/cgsp_child_coverage_sweep_smoke/`
  - `MAX_SLIDES=3`
  - `processed slides = 3`
  - `failed slides = 0`
- formal run:
  - `results_stage42b/cgsp_child_coverage_sweep/`
  - 默认完整 sweep：
    - `proposal_radius_values=512,1024,2048,4096`
    - `nms_radius_values=512,1024`
    - `bbox_expand_values=2,3,4,6,8,10`
    - `high_radius_values=512,1024,2048,4096,8192`
  - `processed slides = 194`
  - `failed slides = 0`
  - `warning count = 0`

### 核心统计结论
- Step42 的失败原因不是 anchors 不稳定：
  - 所有 proposal/nms 组合都能稳定选满 `16` anchors。
  - `full_anchor_slide_ratio = 1.0`。
  - class 多样性保留：`unique_classes = 2`。
  - concept 多样性保留：`unique_concepts = 23`。
- Step42 的主要问题是 `bbox_expand<=3` 太小：
  - `proposal_radius=512, bbox_expand=2/3/4/6` 时，`median_used_child_count` 仍为 `1.0`。
  - 当 `bbox_expand=8/10` 时，child 覆盖明显改善。
- 最佳纯 bbox containment：
  - `proposal_radius=4096`
  - `nms_radius=512`
  - `bbox_expand=8`
  - `empty_anchor_ratio=0.0023`
  - `median_used_child_count=16.0`
  - `mean_used_child_count=15.7784`
  - `mean_fallback_ratio=0.0`
- bbox_then_nearest 也可行，但不是首选：
  - `proposal_radius=4096`
  - `nms_radius=512`
  - `bbox_expand=10`
  - `empty_anchor_ratio=0.0`
  - `median_used_child_count=16.0`
  - `mean_fallback_ratio=0.0127`
- center_radius 也可行，但空间约束更弱：
  - `proposal_radius=4096`
  - `nms_radius=512`
  - `high_radius=4096`
  - `empty_anchor_ratio=0.0`
  - `median_used_child_count=16.0`

### 推荐给 Step43 的参数
- `child_selection_strategy=bbox_containment`
- `proposal_radius=4096`
- `nms_radius=512`
- `bbox_expand=8`
- `num_anchors=16`
- `num_high_children=16`
- `use_bbox_then_nearest_fallback=false`

### 是否建议进入 Step43
- 建议进入 Step43 `HCRC-Light Smoke`。
- 理由：
  - 纯 bbox containment 已满足阈值：
    - `median_used_child_count >= 4`
    - `empty_anchor_ratio <= 0.15`
  - selected anchors 仍能稳定选满 16。
  - anchor class/concept 多样性没有退化。
  - 不需要 nearest fallback，因此 Step43 仍可以保持真实空间 containment 解释。

### 风险与后续建议
- `proposal_radius=4096` 相比 Step42 默认 `512` 会明显改变 proposal bbox 和部分 anchor 坐标：
  - `rankwise_anchor_drift_median_vs_min_radius ≈ 35224.86`
  - Step43 smoke 必须确认更大 spatial proposal 不会引入过宽的 weak evidence。
- `bbox_expand=8` 是较大的 bbox 扩张：
  - 虽然 child 覆盖达标，但可能混入更宽区域的 high patches。
  - Step43 应保留 ablation：`bbox_expand=6/8/10` 或至少记录实际 child distance 分布。
- 如果 Step43 smoke 不稳定：
  - 优先退到 `center_radius=4096` 或 `bbox_then_nearest` 作为 HCRC-KNN / weak correspondence 版本。
  - 若仍无收益，应暂停 HCRC 分支，转向 PRARC / Prompt Reliability 分支。

### GitHub 文件大小处理
- 原 `results_stage42b/cgsp_child_coverage_sweep/stage42b_anchor_level_child_stats.csv` 大小约 `132.20 MB`，超过 GitHub 单文件 `100 MB` 限制。
- 已删除原单体 CSV，并按行顺序拆成两个 GitHub-safe part 文件：
  - `stage42b_anchor_level_child_stats_part_001.csv`
    - rows `0` 到 `273025`
    - size ≈ `85.39 MB`
  - `stage42b_anchor_level_child_stats_part_002.csv`
    - rows `273026` 到 `422143`
    - size ≈ `46.46 MB`
- `stage42b_manifest.json` 已记录 `anchor_level_write_info`：
  - `split=true`
  - `row_count=422144`
  - `parts=[part_001, part_002]`
- 后续步骤如果需要 anchor-level child stats，应按 part 编号顺序 concat：
  - `stage42b_anchor_level_child_stats_part_001.csv`
  - `stage42b_anchor_level_child_stats_part_002.csv`
- `scripts/analysis/build_stage42b_cgsp_child_coverage_sweep.py` 已更新为自动分片：以后若 `stage42b_anchor_level_child_stats.csv` 超过约 `95 MB`，会自动删除单体 CSV 并输出 part 文件，同时写入 manifest/report。
