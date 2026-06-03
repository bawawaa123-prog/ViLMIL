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
