# Step55 Main Model Definition

## Frozen Main Model
- Main model name: `RCE-v4-CSG-a01-rq16`
- model_type: `RCE_MIL_BiomedCLIP`
- Main model file: `models/model_RCE_MIL_BiomedCLIP.py`
- Main training script: `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
- Main result directory: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Runtime call chain: `main.py -> utils/core_utils.py -> models/model_RCE_MIL_BiomedCLIP.py`

## Frozen Configuration Summary
- Feature backbone: BiomedCLIP features (`features_biomedclip_5x` + `features_biomedclip_20x`).
- Scale mode: dual scale.
- Region queries: `rq16`.
- Active RCE components: concept prior, visual residual, cross-scale graph (CSG), logit calibration.
- CSG normalization: `sqrt`.
- Text encoder mode: BiomedCLIP text tower kept frozen in the main release run.

## Explicit Scope Boundary
- `DEG`, `HCRC`, and `PRARC` are not components of the current Step55 frozen main model.
- The frozen release is anchored on the direct RCE main line, not a DEG wrapper line.

## Git Snapshot
- Commit hash: `d6fdd9d4306fa984da5a5641c7953a4aa31de310`
- Branch: `main`
- Captured at: `2026-06-24T13:21:50+08:00`
- Uncommitted changes present: `True`
