# Step64K Final Report

## Scope

Step64K targeted initialization-equivalence and controlled single-fold diagnosis only. No core model files were modified, no Python packages were changed, and no existing Step58C, Step64I, or Step64J results were overwritten.

Current workspace:

- branch: `dev-rce-aofei-split-repro-206200`
- commit: `13f7982f990966b1ebd4d408b662487522dc60b2`

Historical worktree:

- path: `/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main`
- commit: `c1253e8b122f6c3e97188d2b482e89e4d29e54ed`

Precheck artifacts were written under `precheck/`, including workspace state, Python version, `nvidia-smi`, `pip freeze`, and sorted environment variables.

## Initialization Equivalence Probe

Artifacts:

- `equivalence_probe/old_probe.json`
- `equivalence_probe/current_probe.json`
- `equivalence_probe/probe_compare.json`
- `equivalence_probe/probe_compare.md`

Result:

- `biomedclip_weights_consistent = True`
- `trainable_parameters_consistent = True`
- `token_ids_consistent = True`
- `text_embeddings_consistent = True`
- `concept_embeddings_consistent = True`
- `initial_logits_consistent = True`
- `probabilities_consistent = True`
- `training_pre_behavior_equivalent = True`

All shared numeric comparisons reported `max_abs_diff = 0.0`, `mean_abs_diff = 0.0`, and `allclose = True`. This rules out a training-pre initialization or forward-path difference between the old commit and current commit under the Step64K controlled setup.

## Controlled Fold Diagnosis

Three-way comparison artifacts were written under `run_compare/`.

### Fold 2

Historical Step58C:

- test_auc `0.945904356060606`
- test_acc `0.9072164948453608`
- test_f1 `0.8966619318181819`
- val_auc `0.9936537783523328`

New server old code:

- test_auc `0.927911931818182`
- test_acc `0.8865979381443299`
- test_f1 `0.877073732718894`
- val_auc `0.9938888235985428`

New server current code:

- test_auc `0.927911931818182`
- test_acc `0.8865979381443299`
- test_f1 `0.877073732718894`
- val_auc `0.9938888235985428`

Key deltas:

- historical vs old/current: test_auc `0.017992424242423977`, test_acc `0.020618556701030966`, test_f1 `0.019588199099287884`
- old vs current: all monitored metrics exactly `0.0`

Epoch summary:

- historical best_val_auc_epoch `9`, best_vs_last_val_auc_delta `0.0015277941003644102`
- old/current best_val_auc_epoch `8`, best_vs_last_val_auc_delta `0.0009401809848396114`

### Fold 5

Historical Step58C:

- test_auc `0.9621808637556668`
- test_acc `0.927461139896373`
- test_f1 `0.9220157007619488`
- val_auc `0.9644886363636364`

New server old code:

- test_auc `0.9677881173944166`
- test_acc `0.8963730569948186`
- test_f1 `0.8898401826484018`
- val_auc `0.9745501893939394`

New server current code:

- test_auc `0.9677881173944166`
- test_acc `0.8963730569948186`
- test_f1 `0.8898401826484018`
- val_auc `0.9745501893939394`

Key deltas:

- historical vs old/current: test_auc `0.0056072536387498095`, test_acc `0.031088082901554404`, test_f1 `0.03217551811354702`, val_auc `0.010061553030303094`
- old vs current: all monitored metrics exactly `0.0`

Epoch summary:

- historical best_val_auc_epoch `2`, best_vs_last_val_auc_delta `0.020123106060606077`
- old/current best_val_auc_epoch `3`, best_vs_last_val_auc_delta `0.010416666666666519`

## Integrity Notes

- Old-code Fold 2 and Fold 5 integrity checks passed for result-file presence, target-fold matching, finite epoch metrics, and absence of `NaN`/`Inf` tokens in the saved logs.
- Captured logs contain the known SciPy/NumPy warning about the installed NumPy version. This warning was observed already and was not changed in Step64K.
- `prompt_path_matches = false` in the check JSONs because the experiment settings file does not persist `text_prompt_path`; this is a settings-capture limitation, not a training failure.
- Current-code Fold 5 completed successfully, but `training_logs/04_currentcode_fold5.log` does not exist because the wrapper did not tee stdout/stderr to disk. This was not backfilled with synthetic content. The status is recorded in `training_logs/04_currentcode_fold5_log_status.md` and `training_logs/04_currentcode_fold5_checks.json`.

## Manifest Compare Status

No local old-server manifest bundle was found under the Step64J audit directory, so `compare_stage64J_manifests.py` was not executed in Step64K.

Artifacts:

- `manifest_compare/old_new_manifest_compare.json`
- `manifest_compare/old_new_manifest_compare.md`

## Conclusion

Step64K rules out a current-code mathematical regression as the cause of the Step58C versus new-server discrepancy:

1. The old commit and current commit are initialization-equivalent before training.
2. Under the same new-server environment and controlled folds, the old commit and current commit produce identical Fold 2 and Fold 5 final metrics.
3. The discrepancy remains between historical Step58C and both new-server runs.

The strongest supported diagnosis is therefore:

- the root cause is more likely cross-server environment differences, asset/cache provenance differences not yet captured by a missing old-server manifest bundle, or server-level training nondeterminism,
- not a change in current code initialization or forward/training behavior.

This matches `run_compare/decision.md`, which classifies the outcome as `情况2：服务器环境、H5、BiomedCLIP cache provenance 或非确定性更可疑。`
