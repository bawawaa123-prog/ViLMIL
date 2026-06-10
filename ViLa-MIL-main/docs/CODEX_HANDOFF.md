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
