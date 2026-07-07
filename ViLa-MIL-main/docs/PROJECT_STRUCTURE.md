# Project Structure

This document summarizes the intended high-level structure of `ViLa-MIL-main` after the Step63B cleanup.

## Core Code Directories

- `models/`: model definitions and architecture modules. Do not modify archived Step63 freeze behavior unless a new step explicitly requires it.
- `utils/`: training and utility helpers, including existing core pipeline utilities.
- `main.py`: training entrypoint. Step63B did not modify it.
- `wsi_core/`: WSI processing components.
- `feature_extraction/`: feature preparation and extraction helpers.
- `text_prompt/`: prompt resources used by the concept-based pipeline.

## Data And Split Directories

- `datasets/`: dataset wrappers and loading support code.
- `dataset_csv/`: dataset metadata tables and CSV assets.
- `splits/`: split definitions used for training and evaluation.
- `data/`: local runtime data area, typically ignored by Git.

## Script Directories

- `scripts/experiments/`: experiment launch scripts.
- `scripts/analysis/`: analysis and report-generation helpers.
- `scripts/maintenance/`: repository maintenance and structure validation utilities.

## Archived Experiment Outputs

- `results/`, `results_smoke/`, `eval_results/`, and `trained_models/` remain in their original roles.
- Historical `results_stage*` directories are archived under `experiment_outputs/stage_results/`.
- `experiment_outputs/stage_results_index.md` provides a scanned index of archived stage directories.

## Future Step64A Placement

- New Step64A audit outputs should be written to `experiment_outputs/stage_results/results_stage64A_performance_gap_audit/`.
- Future stage runs should keep using unique stage names and should prefer the `experiment_outputs/stage_results/` archive layout instead of creating new root-level `results_stage*` directories.

## Freeze Preservation

- The historical state represented by tag `rce-step63-freeze-v1` must remain reviewable.
- Do not modify or delete archived artifacts associated with the Step63 freeze baseline during later cleanup steps.
