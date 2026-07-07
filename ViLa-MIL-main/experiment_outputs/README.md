# Experiment Outputs

`experiment_outputs/` stores experiment artifacts and stage-level result archives for this repository.

## Layout

- `stage_results/` keeps historical `results_stage*` directories that were previously placed at the repository root.
- The repository root should no longer accumulate direct `results_stage*` directories.

## Guidance

- Use unique stage names for future experiments.
- Prefer saving new stage outputs under `experiment_outputs/stage_results/`.
- Keep archived outputs intact so historical stage evidence remains reproducible and reviewable.
