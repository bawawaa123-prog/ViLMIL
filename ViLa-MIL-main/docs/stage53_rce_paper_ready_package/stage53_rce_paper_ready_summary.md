# Stage53 RCE Paper-Ready Summary

## Final Main Model Definition

- Main model: `RCE-v4-CSG-a01-rq16`
- Main model file: `models/model_RCE_MIL_BiomedCLIP.py`
- Main training script: `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
- Main result directory: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Runtime call chain: `main.py -> utils/core_utils.py -> models/model_RCE_MIL_BiomedCLIP.py`
- Scope note: Step53 is fixed on RCE. DEG is retained only as a later extension shell and is not the final paper model.

## Stage23 Main Result

| Metric | Stage23 Main Result | Step52 Full Check | Delta |
| --- | --- | --- | --- |
| AUC | 0.9702 ± 0.0130 | 0.9702 | +0.0000 |
| ACC | 0.9225 ± 0.0188 | 0.9225 | +0.0000 |
| F1 | 0.9145 ± 0.0207 | 0.9145 | +0.0000 |
| BACC | 0.9171 ± 0.0196 | 0.9171 | +0.0000 |
| PR-AUC | 0.9444 ± 0.0252 | 0.9444 | +0.0000 |

Consistency note: Stage23 and Step52 full are numerically identical across the tracked metrics.

## Step52 and Step52B Ablation Summary

| Variant | AUC | ACC | F1 | BACC | PR-AUC | ΔAUC vs full | ΔACC vs full | ΔF1 vs full | ΔBACC vs full | ΔPR-AUC vs full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full RCE-v4-CSG-rq16 | 0.9702 ± 0.0130 | 0.9225 ± 0.0188 | 0.9145 ± 0.0207 | 0.9171 ± 0.0196 | 0.9444 ± 0.0252 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| w/o CSG | 0.9655 ± 0.0154 | 0.9215 ± 0.0165 | 0.9127 ± 0.0191 | 0.9134 ± 0.0217 | 0.9278 ± 0.0361 | -0.0047 | -0.0010 | -0.0018 | -0.0037 | -0.0166 |
| w/o concept prior | 0.9592 ± 0.0221 | 0.9019 ± 0.0390 | 0.8903 ± 0.0445 | 0.8891 ± 0.0447 | 0.9078 ± 0.0495 | -0.0110 | -0.0206 | -0.0241 | -0.0280 | -0.0366 |
| w/o visual residual | 0.9645 ± 0.0192 | 0.9049 ± 0.0167 | 0.8946 ± 0.0160 | 0.8960 ± 0.0154 | 0.9239 ± 0.0467 | -0.0057 | -0.0176 | -0.0198 | -0.0211 | -0.0205 |
| w/o logit calibration | 0.9686 ± 0.0169 | 0.9215 ± 0.0277 | 0.9111 ± 0.0338 | 0.9076 ± 0.0394 | 0.9486 ± 0.0226 | -0.0017 | -0.0010 | -0.0034 | -0.0095 | +0.0042 |

## Current Innovation Line

- ViLa-MIL provides dual-scale vision-language alignment at the slide level.
- RCE extends that backbone into region-concept evidence reasoning rather than only global alignment.
- Concept prior is the dominant class-aware evidence constraint.
- Visual residual supplies complementary visual evidence that concept-only scoring does not fully capture.
- CSG mainly improves evidence ranking quality, reflected more clearly in AUC and PR-AUC than in raw ACC.
- Logit calibration improves balanced decision behavior, especially BACC and F1, with a small PR-AUC trade-off.

## Evidence Strength Assessment

- Strongest evidence: exact reproduction of the final main model in Stage23 and Step52 full, plus consistent descriptive ablation trends.
- Moderate evidence: concept prior and visual residual both show clear average drops across multiple metrics when removed.
- Cautious evidence: CSG and logit calibration effects are more metric-specific and should not be oversold as uniform gains.
- Statistical stance: paired t-tests are available as descriptive references, but the current 5-fold setup is better presented as trend evidence rather than strict statistical significance.

## Recommended Paper Narrative

- Position the final method as an RCE extension of ViLa-MIL from slide-level alignment to region-concept evidence reasoning.
- Emphasize that concept prior and visual residual are the two most visible contributors to the final operating point.
- Describe CSG as improving cross-scale concept interaction and evidence ranking quality, especially AUC and PR-AUC.
- Describe logit calibration as improving balanced decision behavior, while acknowledging that PR-AUC can be slightly higher without it.
- Keep DEG, HCRC, and PRARC out of the final main-method claim in Step53.

## Input Status

Missing inputs:
- none

Warnings:
- none
