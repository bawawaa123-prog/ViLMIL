# Stage64J Code Behavior Audit

## Scope

Compared:

- `main.py`
- `utils/core_utils.py`
- `utils/utils.py`
- `utils/metric_utils.py`
- `datasets/dataset_generic.py`
- `datasets/dataset_h5.py`
- `models/model_RCE_MIL_BiomedCLIP_v2.py`
- `models/model_ViLa_MIL_BiomedCLIP.py`
- `scripts/experiments/run_stage58C_residual_constrained_configD_5fold.sh`

## Confirmed Differences

1. `main.py`
   - Common Step58C arguments keep the same defaults and the same `k_start` / `k_end` semantics.
   - New code adds extra CLI surface:
     - `ViLa_MIL_BiomedCLIP_AofeiClean`
     - `rce_use_dynamic_csg` and related `rce_dynamic_csg_*`
     - `rce_use_ccra` and related `rce_ccra_*`
     - `rce_use_l2h_retrieval` and related `rce_l2h_*`
   - `settings` persistence now records the `rce_l2h_*` values.

2. `utils/core_utils.py`
   - Added an `AofeiClean` model branch.
   - For `RCE_MIL_BiomedCLIP_v2`, the config object now carries `rce_dynamic_csg_*`, `rce_ccra_*`, and `rce_l2h_*`.
   - Offline lock logic now includes `ViLa_MIL_BiomedCLIP_AofeiClean`.
   - `np.Inf` changed to `np.inf`. This is a NumPy 2 compatibility fix, not a training-logic change.

3. `models/model_RCE_MIL_BiomedCLIP_v2.py`
   - BiomedCLIP loading logic changed materially:
     - old code: direct `hf-hub:` load through ambient Hugging Face cache / env
     - current code: explicit cache discovery, snapshot resolution, `local-dir:` fallback, and offline env bootstrapping
   - Added dynamic CSG / CCRA / L2H code paths and config fields.
   - Added many diagnostic state fields for breakdown export.

4. `models/model_ViLa_MIL_BiomedCLIP.py`
   - Same BiomedCLIP cache/snapshot resolution logic was added here as well.
   - This matters because `RCE_MIL_BiomedCLIP_v2` imports `BiomedCLIPTextEncoder` from this file.

5. `utils/utils.py`
   - Only relevant diff is widening BiomedCLIP parameter-group logic to include `ViLa_MIL_BiomedCLIP_AofeiClean`.
   - RCE optimizer path is unchanged.

6. No content change detected in:
   - `utils/metric_utils.py`
   - `datasets/dataset_generic.py`
   - `datasets/dataset_h5.py`
   - `scripts/experiments/run_stage58C_residual_constrained_configD_5fold.sh`

## Behavior Audit

1. `argparse` defaults
   - Confirmed unchanged for the common Step58C arguments that were actually used in Step58C / Step64I:
     - `lr`, `max_epochs`, `seed`, `opt`, `bag_loss`, `label_frac`, `drop_out`, `weighted_sample`, `prototype_number`, `scale_mode`, `prompt_ensemble_mode`
   - New code only adds extra defaulted arguments for features not present in old Step58C code.

2. `k_start` / `k_end`
   - Confirmed inclusive in both old and current code:
     - `end = args.k if args.k_end == -1 else args.k_end + 1`
   - Diagnostic single-fold commands must therefore use `--k_start N --k_end N`.

3. Random seed reset
   - Confirmed both old and current code call `seed_torch(args.seed)`:
     - once before training starts
     - once again at the start of each fold
   - This is unchanged.

4. Model initialization order
   - RCE core model math path changed only by adding new optional branches and cache/bootstrap logic.
   - The active Step58C Config-D path still initializes:
     - region queries
     - attention modules
     - logit calibration
     - concept prior
     - visual residual
     - residual constraint
     - cross-scale graph
   - The newly added optional branches are gated.

5. Newly added modules when flags are `False`
   - `rce_use_dynamic_csg=False`: no new parameter is created.
   - `rce_use_ccra=False`: no CCRA parameter/module is created.
   - `rce_use_l2h_retrieval=False`: no L2H parameter/module is created.
   - Additional `None` attributes and debug holders are created, but these do not consume random initialization.

6. Optimizer parameter collection
   - For RCE, unchanged.
   - Only the non-RCE BiomedCLIP ViLa path changed its parameter-group condition.
   - This should not affect Step58C / Step64I RCE runs.

7. DataLoader and sampler behavior
   - Unchanged for Linux CUDA runs:
     - train: `RandomSampler` unless `weighted_sample=True`
     - val/test: `SequentialSampler`
     - `num_workers=4` on CUDA
   - No generator is passed to DataLoader.
   - No `worker_init_fn` is defined.

8. Checkpoint strategy
   - Unchanged.
   - If `--early_stopping` is enabled:
     - checkpoint is updated when validation error improves
     - after training, checkpoint is reloaded
   - If `--early_stopping` is disabled:
     - checkpoint is saved once after the loop
     - it is the last-epoch model

9. Early stopping monitor
   - Despite the `EarlyStopping` naming and `val_loss_min` variable name, the actual monitored score is validation error:
     - `early_stopping(epoch, val_error, model, ...)`
   - This is unchanged between old and current code.

10. Validation / test metrics
   - Unchanged.
   - `utils/metric_utils.py` is byte-identical.
   - Validation AUC / F1 logic and test summary logic are unchanged.

11. Hugging Face / BiomedCLIP cache behavior
   - This is the strongest confirmed runtime behavior change.
   - Old code depends on ambient Hugging Face cache resolution and whatever snapshot is already available to `open_clip` / `transformers`.
   - Current code actively:
     - searches multiple cache directories
     - resolves a specific local snapshot
     - converts `hf-hub:` to `local-dir:...`
     - sets offline env vars from discovered cache
   - This can change reproducibility if the old server and the new server resolved different snapshots or different cache roots.

12. Random-number consumption order
   - For the active Step58C Config-D path, no confirmed RNG-consuming parameter initialization was added when the new optional branches remain disabled.
   - Therefore:
     - added CLI fields by themselves should not change RNG consumption
     - added debug placeholders should not change RNG consumption
   - No confirmed evidence yet that current code changes the active Config-D math path enough to explain the observed AUC/ACC gap by itself.

## Assessment

### Confirmed differences likely to matter

- BiomedCLIP / Hugging Face cache resolution logic changed.
- Current runtime config includes extra `rce_l2h_*` fields by default, but these remain disabled and should not affect active math.

### Confirmed differences unlikely to matter for Step58C Config-D

- `np.Inf` -> `np.inf`
- `AofeiClean` registration
- `utils/utils.py` BiomedCLIP optimizer branch expansion
- New debug state fields in `RCE_MIL_BiomedCLIP_v2`

### Confirmed no-difference areas

- Fold indexing semantics
- Fold seed reset behavior
- DataLoader sampler behavior
- `worker_init_fn` / DataLoader generator usage
- Early stopping policy
- Last-epoch checkpoint behavior when early stopping is off
- Validation / test metric implementation
- Stage58C launcher script content

### Still unconfirmed

- Whether the old Step58C server used the same BiomedCLIP snapshot bytes as the current server
- Whether old-server feature H5 files were byte-identical to the current server copies
- Whether old-server Python/CUDA package versions matched the current manifest
