# Step51 Reproduce RCE And DEG Skeleton Plan

## Purpose

Step51 provides two stable reproduction entrances for the current main model without changing training logic:

- Path A: direct RCE main-model reproduction
- Path B: DEG skeleton replay of the same main-model path

This step does not introduce a new ablation. It only standardizes the reproduction entrypoints for later verification work.

## Why Run Both Paths

We need both paths because they answer different questions:

- `RCE_MIL_BiomedCLIP` confirms the original main-model source path from Step23.
- `DEG_MIL_BiomedCLIP` confirms that the later DEG shell can replay the same logits path when all extra modules stay off.

That gives a clean separation between:

- the historical source-of-truth main model
- the later unified ablation shell

## Code Files

- RCE path code file: `models/model_RCE_MIL_BiomedCLIP.py`
- DEG skeleton path code file: `models/model_DEG_MIL_BiomedCLIP.py`

## Why The Results Should Match

Both Step51 paths keep the Step23 main-model configuration:

- `prototype_number=16`
- logit calibration on
- concept prior on
- visual residual on
- cross-scale graph on with `init=0.1` and `norm=sqrt`

The DEG path should match because the skeleton run does not enable any later add-on modules:

- `deg_use_region_graph`
- `deg_use_concept_graph`
- `rce_use_visual_evidence_gate`
- `rce_use_prarc_gate`
- `rce_use_hcrc`
- `rce_use_low_high_consistency_loss`

Under that setting, DEG should reduce to the same effective Stage23 logits path.

## Step51 Entry Script

Script:

`scripts/experiments/run_stage51_reproduce_rce_and_deg_skeleton.sh`

Default mode:

`MODE=both`

Outputs:

- `results_stage51_repro/rce_step23_rq16_5fold_e20_s1/`
- `results_stage51_repro/deg_skeleton_rq16_5fold_e20_s1/`

## Run Commands

Run only the direct RCE reproduction:

```bash
MODE=rce bash scripts/experiments/run_stage51_reproduce_rce_and_deg_skeleton.sh
```

Run only the DEG skeleton reproduction:

```bash
MODE=deg bash scripts/experiments/run_stage51_reproduce_rce_and_deg_skeleton.sh
```

Run both:

```bash
MODE=both bash scripts/experiments/run_stage51_reproduce_rce_and_deg_skeleton.sh
```

## Compare After Training

Comparison script:

`scripts/analysis/compare_stage51_rce_deg_repro.py`

Run it after Step51 training finishes:

```bash
python scripts/analysis/compare_stage51_rce_deg_repro.py
```

Outputs:

- `docs/stage51_rce_deg_reproduction_comparison.md`
- `docs/stage51_rce_deg_reproduction_comparison.csv`

The comparison checks:

- historical Stage23 main result
- Step51 RCE reproduction result
- Step51 DEG skeleton reproduction result

Metrics:

- `test_auc`
- `test_acc`
- `test_f1`
- `balanced_acc`
- `pr_auc`

## Scope Boundary

Step51 does not do any of the following:

- no new module ablation
- no model-logic rewrite
- no replacement of the main-model identity
- no training launch in this implementation task

It only prepares the scripts and documents needed to verify:

`Step23 main-model path == DEG skeleton replay path`
