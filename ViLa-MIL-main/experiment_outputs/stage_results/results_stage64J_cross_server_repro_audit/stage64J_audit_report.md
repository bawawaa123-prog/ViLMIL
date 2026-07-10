# Stage64J Cross-Server Repro Audit

## 1. Repository State

- Current repo: `/private/ljh-data/shared/Linux_school/ViLMIL`
- Current project: `/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main`
- Historical repo worktree: `/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8`
- Historical project: `/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main`
- Current branch at audit start: `dev-rce-aofei-split-repro-206200`
- Current commit at audit start: `05403ec6246d93b81105568e7931d4ebb1a24b42`
- Historical worktree commit: `c1253e8b122f6c3e97188d2b482e89e4d29e54ed`
- Initial `git status --short`: clean
- Current manifest `git_dirty=true` because this audit created new scripts and reports under the Step64J output directory. No existing tracked files were overwritten.

## 2. Worktree Creation

- Historical detached worktree creation: success
- Command used:
  - `git worktree add --detach /private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8 c1253e8`

## 3. Config Comparison: Step58C vs Step64I

Source files:

- old:
  - `/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main/results_stage58C_residual_constrained_configD_5fold/rce_v2_rcD_l003_t050_aux020_5fold_e20_s1/experiment_rce_v2_rcD_l003_t050_aux020_5fold_e20.txt`
- new:
  - `/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/experiment_outputs/stage_results/results_stage64I_rce_step58C_server_only_206200/adenocarcinoma_rce_step58C_server_only_206200_s1/experiment_adenocarcinoma_rce_step58C_server_only_206200.txt`

Artifacts:

- `config_compare/old_config.json`
- `config_compare/new_config.json`
- `config_compare/config_diff.csv`
- `config_compare/config_diff.md`

Key results:

- Core training hyperparameters are identical:
  - `lr=1e-4`
  - `max_epochs=20`
  - `seed=1`
  - `opt=adam`
  - `label_frac=1.0`
  - `mode=transformer`
  - `model_type=RCE_MIL_BiomedCLIP_v2`
  - `split_dir=splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
  - all active Config-D `rce_*` parameters used in Step58C
- Path-like differences:
  - `data_root_dir`: old `/xiangmu/data/VILMIL` vs new `data/yiyuan`
  - `concept_prompt_path`: old absolute path vs new relative path
  - `results_dir` / `experiment`
- New-only config keys in Step64I:
  - `rce_use_l2h_retrieval` and all `rce_l2h_*` defaults
- Interpretation:
  - Step64I did not change the active Config-D hyperparameters.
  - Step64I did run under a codebase that exposes more defaulted RCE options than old Step58C.
  - Those new fields are disabled in the saved config.

## 4. Code Comparison

Artifacts:

- `code_compare/file_hashes.csv`
- `code_compare/git_diff_stat.txt`
- `code_compare/relevant_code_diff.patch`
- `code_compare/code_behavior_audit.md`

Byte-identical files:

- `utils/metric_utils.py`
- `datasets/dataset_generic.py`
- `datasets/dataset_h5.py`
- `scripts/experiments/run_stage58C_residual_constrained_configD_5fold.sh`

Changed files:

- `main.py`
- `utils/core_utils.py`
- `utils/utils.py`
- `models/model_RCE_MIL_BiomedCLIP_v2.py`
- `models/model_ViLa_MIL_BiomedCLIP.py`

Behavior summary:

- `k_start` / `k_end` semantics are unchanged and inclusive.
- Fold-level seed reset is unchanged.
- DataLoader sampler / worker behavior is unchanged on Linux CUDA.
- Validation / test metrics are unchanged.
- Early stopping and final checkpoint behavior are unchanged:
  - with `--early_stopping`: reload best validation-error checkpoint
  - without `--early_stopping`: save last-epoch checkpoint and evaluate that model
- The major confirmed runtime change is BiomedCLIP cache/snapshot resolution:
  - old code relied on ambient Hugging Face resolution
  - current code searches explicit cache roots and resolves a local snapshot
- New RCE branches were added:
  - dynamic CSG
  - CCRA
  - L2H retrieval
- Under Step64I saved config these branches are all disabled, and no confirmed additional trainable parameters are initialized for them.

## 5. Checkpoint Strategy Analysis

- The observed Step64I vs Step58C difference is not explained by a checkpoint-policy drift in code.
- Both old and current code save the last-epoch model when `--early_stopping` is not enabled.
- The saved Step58C and Step64I configs both show no early-stopping flag in the runtime config.
- Therefore both runs should be interpreted as last-epoch evaluations unless external launch conditions differed.

## 6. DataLoader and Determinism Analysis

- `seed_torch(args.seed)` is called before training and again before each fold in both old and current code.
- `torch.backends.cudnn.benchmark=False` and `torch.backends.cudnn.deterministic=True` are set in both old and current `main.py`.
- `torch.are_deterministic_algorithms_enabled()` is currently `False` in the runtime manifest.
- No DataLoader generator is provided.
- No `worker_init_fn` is provided.
- Training uses `RandomSampler`; validation/test use `SequentialSampler`.
- Current runtime environment does not set `PYTHONHASHSEED`, `CUDA_VISIBLE_DEVICES`, or `CUBLAS_WORKSPACE_CONFIG` globally in the manifest process.

Assessment:

- Some deterministic controls exist.
- Full deterministic replay is not guaranteed.
- If old-code diagnostic runs on the new server still drift materially from historical Step58C, deterministic audit is justified.

## 7. Data and Research Asset Hash Results

Current real data path:

- `readlink -f data/yiyuan` -> `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/yiyuan`

Current ordinary assets:

- `dataset_csv/all_data.csv`
  - old SHA256 = new SHA256 = `f432b81135a8b2bdce9932dbd6ebdc7b8b5049e4c6d21a9468059fc5bd6e12d0`
- `dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
  - old SHA256 = new SHA256 = `d8f4c0907eb847db402b662260a96b34bcabff90ec055d2e395623131b7bd856`
- `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
  - old SHA256 = `a54e2d9221f56f1e19ed130cbd589404c882a6b70c832f0024ed550ffad9d8aa`
  - new SHA256 = `fb96a2fbc0b6444eec09c09df9742163296d7a207f489958ee9202309ab948d4`
  - `diff -u` shows only a trailing blank line in the current file
  - `pandas.read_csv(old).equals(pandas.read_csv(new)) == True`
  - Conclusion: raw file hash differs, parsed prompt table is semantically identical

Current split assets:

- old and current `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100` tree aggregate SHA256:
  - `7a84349fc8d2459017475a8e92b368687854893fc60d0d3db06fdf6af6d3ade3`
- Split files are byte-identical between old worktree and current project.

Current H5 assets:

- `features_biomedclip_5x`: 968 files
- `features_biomedclip_20x`: 968 files
- No NaN detected in feature tensors
- No Inf detected in feature tensors
- Current asset aggregate SHA256:
  - `0dd52ca021ae52ee96cc745ba79b3d7d6e60232c0e900259cd9f0860fa255060`

Limitation:

- Old-server H5 manifests were not available in this environment.
- Cross-server content equality for features remains unconfirmed until the old-server manifest is collected and compared.

## 8. BiomedCLIP Snapshot Audit

Artifacts:

- `manifests/biomedclip_cache_manifest.json`
- `manifests/biomedclip_cache_files.csv`

Findings:

- Model name used by code:
  - `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Selected cache dir:
  - `/private/ljh-data/shared/ViLMIL/hf_cache`
- Selected snapshot:
  - `/private/ljh-data/shared/ViLMIL/hf_cache/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/9f341de24bfb00180f1b847274256e9b65a3a32e`
- Only one model snapshot was found in the current cache manifest.
- Key files present with SHA256 recorded:
  - `open_clip_config.json`
  - `open_clip_pytorch_model.bin`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `vocab.txt`
  - BiomedBERT `config.json`
  - BiomedBERT `pytorch_model.bin`

Assessment:

- Current server cache looks complete for offline loading.
- Old Step58C code did not have the same explicit snapshot-selection logic.
- Whether old Step58C resolved the same exact snapshot bytes on the old server remains unconfirmed.

## 9. Environment and Compatibility

Artifacts:

- `manifests/current_environment_manifest.json`
- `manifests/current_pip_freeze.txt`
- `manifests/current_nvidia_smi.txt`

Current environment:

- Python: `3.12.6`
- Conda env: `vila_mil_overlay_rt`
- Torch: `2.4.1+cu124`
- CUDA runtime reported by torch: `12.4`
- cuDNN: `90100`
- GPUs: `2 x NVIDIA A30`
- NumPy: `2.5.0`
- SciPy: `1.14.1`
- scikit-learn: `1.5.2`
- h5py: `3.12.1`
- pandas: `2.2.3`
- open_clip: `3.3.0`
- transformers: `5.12.0`
- huggingface_hub: `1.19.0`
- tokenizers: `0.22.2`
- ml_collections: `1.1.0`

NumPy / SciPy conclusion:

- Current manifest confirms:
  - SciPy requirement: `numpy>=1.23.5,<2.3`
  - installed NumPy: `2.5.0`
  - `numpy_satisfies_all_scipy_requirements = false`
- This is a confirmed environment compatibility conflict.
- It does not prove this conflict caused the Step58C vs Step64I metric gap, but it is a real environment drift risk and should be treated as high-priority audit input.

## 10. Confirmed vs Unconfirmed Differences

### Confirmed differences

- Current code has extra disabled RCE argument surface and extra disabled branches.
- Current code has materially different BiomedCLIP cache/snapshot resolution logic.
- Current prompt CSV raw hash differs from historical prompt CSV by a trailing blank line.
- Current ordinary assets `all_data.csv`, concept JSON, and strict split tree match historical worktree copies.
- Current H5 inventory is fully hashed and internally clean for NaN/Inf.
- Current environment has a confirmed SciPy / NumPy version-spec mismatch.

### Not confirmed yet

- Old-server H5 feature bytes vs current-server H5 feature bytes
- Old-server prompt/concept/split manifests vs current manifests
- Old-server BiomedCLIP cache snapshot bytes vs current snapshot bytes
- Old-server Python / CUDA / package versions vs current manifest
- Whether current-code behavior differences alone are sufficient to explain the observed Step64I metric drift

## 11. Root-Cause Priority Ranking

1. BiomedCLIP cache / snapshot provenance mismatch across servers
   - highest leverage confirmed code-path difference
   - old and new servers may have resolved different snapshot content or cache roots
2. Data / feature asset mismatch across servers
   - current server assets are hashed
   - old server assets still need manifest capture for proof
3. Python / NumPy / SciPy / CUDA environment drift
   - confirmed compatibility mismatch exists in current env
4. Training randomness / determinism gap
   - some controls exist, but deterministic algorithms are not fully enforced
5. Prompt file raw hash drift
   - low priority because parsed CSV content is identical
6. Evaluation / checkpoint policy drift
   - low priority because code path is unchanged

## 12. Diagnostic Training Scripts

Generated, not executed:

- `commands/01_run_old_commit_fold2.sh`
- `commands/02_run_current_commit_fold2.sh`
- `commands/03_run_old_commit_fold5.sh`
- `commands/04_run_current_commit_fold5.sh`

Design choices:

- `k_start` / `k_end` use the confirmed inclusive semantics.
- Fold 2 uses `--k_start 1 --k_end 1`.
- Fold 5 uses `--k_start 4 --k_end 4`.
- All four scripts use:
  - the same absolute data root
  - the same absolute split directory
  - the same absolute prompt file
  - the same absolute concept JSON
- Prompt path chosen for the four scripts is the historical worktree prompt path because:
  - the historical file is the stricter reproduction target
  - the current prompt differs only by a trailing blank line

## 13. Recommended Manual Execution Order

1. Run `commands/01_run_old_commit_fold2.sh`
2. Run `commands/02_run_current_commit_fold2.sh`
3. Run `commands/03_run_old_commit_fold5.sh`
4. Run `commands/04_run_current_commit_fold5.sh`
5. After all four runs finish, run:
   - `tools/compare_stage64J_diagnostic_runs.py`
   - or a follow-up script built around the produced result directories

Reasoning:

- Fold 2 was identified as the strongest ranking-metric drift candidate.
- Fold 5 was identified as a threshold-heavy mismatch candidate where AUC stayed closer but ACC dropped more.

## 14. Interpretation Logic After Manual Runs

- If old code on the new server is already close to historical Step58C:
  - server environment is less likely to be the main cause
  - current-code drift becomes more suspicious
- If old code on the new server is already far from historical Step58C:
  - environment, cache, or data drift remains the leading explanation
- If old and current code on the new server are close to each other:
  - committed code differences are less likely to explain the original gap
- If AUC remains close but ACC / sensitivity / specificity move more:
  - threshold / calibration / logit-scale drift is more plausible than ranking drift
- If last-epoch performance is clearly below best-validation epoch in both old and current diagnostic runs:
  - a best-checkpoint audit is justified

## 15. Next-Step Recommendation

1. Collect old-server manifests with `commands/run_manifest_on_old_server.sh`
2. Compare old/new manifests with `tools/compare_stage64J_manifests.py`
3. Run the four diagnostic fold scripts in the order listed above
4. Compare old/new fold outputs
5. Only if old/new fold diagnostics remain ambiguous, start a deterministic audit or best-checkpoint audit

## 16. Training Status

- No training was started in Step64J.
- No diagnostic script was executed.
- This audit only performed:
  - worktree creation
  - path checks
  - file hashing
  - static code comparison
  - manifest generation
  - shell/Python script generation
