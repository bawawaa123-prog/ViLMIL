# Step51B DEG Skeleton Equivalence Repair

## Background

Step51 compared the Stage23 source-of-truth RCE path (`RCE-v4-CSG-a01-rq16`) against the current `DEG_MIL_BiomedCLIP` skeleton path. The RCE reproduction path matched Stage23, but the DEG skeleton path did not.

The key requirement for Step51B is narrower than a new ablation: when all DEG-only modules are disabled, `DEG_MIL_BiomedCLIP` must behave as a strict shell over the current `RCE_MIL_BiomedCLIP` main path.

## Why Step51 Found a Mismatch

Static audit showed the Step51 reproduction shell script already matched the Stage23 RCE script on the main flags that define the source-of-truth path:

- `prototype_number=16`
- `rce_use_logit_calibration=True`
- `rce_use_concept_prior=True`
- `rce_use_visual_residual=True`
- `rce_use_cross_scale_graph=True`
- `rce_cross_scale_graph_init=0.1`
- `rce_cross_scale_graph_norm=sqrt`
- `concept_prompt_path=dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
- `split_dir=splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- `text_prompt_path=text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- `data_folder_s=features_biomedclip_5x`
- `data_folder_l=features_biomedclip_20x`
- `scale_mode=dual`
- `prompt_ensemble_mode=embedding_mean`
- `k=5`, `k_start=0`, `k_end=4`
- `max_epochs=20`
- `seed=1`

The mismatch therefore came from the model implementation, not from Step51 script drift.

## Audit Findings

Before repair, `models/model_DEG_MIL_BiomedCLIP.py` still changed skeleton behavior even when all DEG-only modules were disabled:

- It unconditionally registered extra trainable parameters for visual evidence gate, PRARC MLP, and HCRC.
- Those extra parameters changed the optimizer parameter set relative to RCE.
- Those extra parameters also changed random initialization order, so native-init comparisons could diverge even before any DEG-only branch was used.
- DEG used its own forward path instead of directly reusing the RCE source-of-truth forward path.

That means the previous skeleton mode was not a strict no-op shell around the RCE main model.

## Code Change

`models/model_DEG_MIL_BiomedCLIP.py` was modified.

The repair is intentionally minimal:

- A `deg_skeleton_passthrough` condition is computed from the DEG-only switches.
- When all of the following are disabled:
  - `deg_use_region_graph`
  - `deg_use_concept_graph`
  - `rce_use_visual_evidence_gate`
  - `rce_use_prarc_gate`
  - `rce_use_hcrc`
  - `rce_use_low_high_consistency_loss`
- DEG no longer registers DEG-only trainable parameters.
- In the same mode, `DEG_MIL_BiomedCLIP.forward(...)` immediately delegates to `super().forward(...)`.
- A small internal guard forces region aggregation to use the exact RCE implementation during passthrough mode.

## Before vs After

Before:

- skeleton mode still had DEG-only trainable parameters
- skeleton mode still used DEG's custom forward body
- native initialization and optimizer contents could differ from RCE

After:

- skeleton mode does not register DEG-only trainable parameters
- skeleton mode directly uses the RCE forward/logits/loss path
- optimizer-visible trainable parameters match RCE on the shared path
- native-init and shared-weight equivalence can now be audited directly

## Forward-Level Equivalence

The formal audit is implemented in:

- `scripts/analysis/audit_stage51b_deg_skeleton_equivalence.py`

It checks:

- Stage51 vs Stage23 flag consistency
- common parameter names, shapes, and initial values
- whether DEG exposes extra trainable parameters in skeleton mode
- native-init output equivalence
- shared-weight output equivalence
- whether any DEG-only extra parameters receive gradients

The generated audit outputs are:

- `docs/stage51b_deg_skeleton_equivalence_audit.md`
- `docs/stage51b_deg_skeleton_equivalence_audit.csv`

## Whether Full 5-Fold Needs Rerun

Yes, if you want to re-establish the Step51 DEG skeleton reproduction claim at the experiment-result level, the full DEG skeleton 5-fold should be rerun after this repair.

That rerun is not part of Step51B, because Step51B only requires audit, minimal repair, and lightweight verification.

## Recommendation for Future Ablations

Future DEG-family ablations should continue to use `models/model_DEG_MIL_BiomedCLIP.py` as the experimental shell, but only if:

- the skeleton passthrough path remains a strict no-op wrapper over RCE when DEG-only modules are disabled
- every new DEG module is opt-in and isolated from the default RCE-equivalent path

The source-of-truth main model path remains:

- `models/model_RCE_MIL_BiomedCLIP.py`
