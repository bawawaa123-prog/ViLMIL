# Stage53 RCE Main Results

## Final Main Model

- Main model: `RCE-v4-CSG-a01-rq16`
- Model file: `models/model_RCE_MIL_BiomedCLIP.py`
- Training script: `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
- Result directory: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`

## Main Metrics

| Metric | Stage23 Main Result | Step52 Full Check | Delta |
| --- | --- | --- | --- |
| AUC | 0.9702 ± 0.0130 | 0.9702 | +0.0000 |
| ACC | 0.9225 ± 0.0188 | 0.9225 | +0.0000 |
| F1 | 0.9145 ± 0.0207 | 0.9145 | +0.0000 |
| BACC | 0.9171 ± 0.0196 | 0.9171 | +0.0000 |
| PR-AUC | 0.9444 ± 0.0252 | 0.9444 | +0.0000 |

## Scope Clarification

- The preferred paper result is the Stage23 direct-RCE run, not a DEG wrapper run.
- Stage23 and Step52 full are used here as consistency-checked equivalents for the final RCE package.
- The final main model is not DEG.
- Step51B only supports the narrower statement that the DEG all-off skeleton can reproduce the RCE path.
- Step53 does not rely on, or claim, any DEG any-on conclusion.
