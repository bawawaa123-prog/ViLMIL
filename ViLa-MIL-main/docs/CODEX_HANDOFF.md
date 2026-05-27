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
