# Step55 Release Manifest

## Release Identity
- Release step: `Step55`
- Generated at: `2026-06-24T13:21:50+08:00`
- Main model: `RCE-v4-CSG-a01-rq16`
- model_type: `RCE_MIL_BiomedCLIP`

## Git State
- Commit hash: `d6fdd9d4306fa984da5a5641c7953a4aa31de310`
- Branch: `main`
- Uncommitted changes present: `True`

```text
?? docs/stage55_rce_frozen_release/
?? results_stage55_rce_frozen_release/
?? scripts/analysis/build_stage55_rce_frozen_release.py
```

## Main Results

| metric | mean | std |
| --- | --- | --- |
| AUC | 0.9702 | 0.0130 |
| ACC | 0.9225 | 0.0188 |
| F1 | 0.9145 | 0.0207 |
| BACC | 0.9171 | 0.0196 |
| PR-AUC | 0.9444 | 0.0252 |

## Audit Outcome
- Blockers: `0`
- Warnings: `2`
- Suggested tag readiness: `yes_with_warnings`
