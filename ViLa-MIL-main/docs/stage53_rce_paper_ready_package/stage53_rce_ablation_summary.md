# Stage53 RCE Ablation Summary

## Core Ablation Table

| Variant | AUC | ACC | F1 | BACC | PR-AUC | ΔAUC vs full | ΔACC vs full | ΔF1 vs full | ΔBACC vs full | ΔPR-AUC vs full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full RCE-v4-CSG-rq16 | 0.9702 ± 0.0130 | 0.9225 ± 0.0188 | 0.9145 ± 0.0207 | 0.9171 ± 0.0196 | 0.9444 ± 0.0252 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| w/o CSG | 0.9655 ± 0.0154 | 0.9215 ± 0.0165 | 0.9127 ± 0.0191 | 0.9134 ± 0.0217 | 0.9278 ± 0.0361 | -0.0047 | -0.0010 | -0.0018 | -0.0037 | -0.0166 |
| w/o concept prior | 0.9592 ± 0.0221 | 0.9019 ± 0.0390 | 0.8903 ± 0.0445 | 0.8891 ± 0.0447 | 0.9078 ± 0.0495 | -0.0110 | -0.0206 | -0.0241 | -0.0280 | -0.0366 |
| w/o visual residual | 0.9645 ± 0.0192 | 0.9049 ± 0.0167 | 0.8946 ± 0.0160 | 0.8960 ± 0.0154 | 0.9239 ± 0.0467 | -0.0057 | -0.0176 | -0.0198 | -0.0211 | -0.0205 |
| w/o logit calibration | 0.9686 ± 0.0169 | 0.9215 ± 0.0277 | 0.9111 ± 0.0338 | 0.9076 ± 0.0394 | 0.9486 ± 0.0226 | -0.0017 | -0.0010 | -0.0034 | -0.0095 | +0.0042 |

## Module-Wise Interpretation

- `concept prior`: contribution is the largest. Removing it causes the biggest mean drop across AUC, ACC, F1, BACC, and PR-AUC.
- `visual residual`: provides important complementary visual evidence. Removing it produces clear degradation in ACC, F1, BACC, and PR-AUC.
- `CSG`: mainly improves AUC and PR-AUC, which supports an evidence-ranking interpretation. The ACC change is small and should not be exaggerated.
- `logit calibration`: mainly improves balanced decision behavior, especially BACC and F1. PR-AUC is slightly higher without calibration, so this module should be described as a trade-off rather than a universal gain.

## Paired T-Test Reference

| Variant | p(AUC) | p(ACC) | p(F1) | p(BACC) | p(PR-AUC) | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| w/o CSG | 0.2282 | 0.8655 | 0.7938 | 0.5728 | 0.0878 | descriptive only; no robust significance claim |
| w/o concept prior | 0.1383 | 0.2344 | 0.2254 | 0.1742 | 0.0606 | descriptive only; no robust significance claim |
| w/o visual residual | 0.1743 | 0.2320 | 0.2271 | 0.2523 | 0.2008 | descriptive only; no robust significance claim |
| w/o logit calibration | 0.7506 | 0.9353 | 0.8246 | 0.5864 | 0.6665 | descriptive only; no robust significance claim |

Interpretation: these p-values are reported only as matched-fold descriptive references. Under the current 5-fold setting, the ablation evidence is better framed as stable directional trends than as strict statistical significance.

## Fold-Level Delta Preview

| Variant | Fold | AUC Δ | ACC Δ | F1 Δ | BACC Δ | PR-AUC Δ |
| --- | --- | --- | --- | --- | --- | --- |
| w/o CSG | 1 | -0.0039 | +0.0206 | +0.0214 | +0.0156 | -0.0044 |
| w/o CSG | 2 | -0.0045 | -0.0103 | -0.0132 | -0.0188 | -0.0122 |
| w/o CSG | 3 | -0.0002 | -0.0103 | -0.0114 | -0.0114 | +0.0008 |
| w/o CSG | 4 | -0.0169 | -0.0052 | -0.0061 | -0.0076 | -0.0382 |
| w/o CSG | 5 | +0.0021 | +0.0000 | +0.0006 | +0.0036 | -0.0292 |
| w/o concept prior | 1 | -0.0114 | +0.0052 | +0.0017 | -0.0144 | -0.0157 |
| w/o concept prior | 2 | -0.0264 | -0.0773 | -0.0901 | -0.0953 | -0.0443 |
| w/o concept prior | 3 | +0.0027 | -0.0052 | -0.0071 | -0.0180 | +0.0044 |
| w/o concept prior | 4 | -0.0220 | -0.0207 | -0.0198 | -0.0085 | -0.0769 |
| w/o concept prior | 5 | +0.0019 | -0.0052 | -0.0054 | -0.0039 | -0.0506 |
| w/o visual residual | 1 | +0.0027 | +0.0258 | +0.0257 | +0.0122 | +0.0097 |
| w/o visual residual | 2 | -0.0167 | -0.0103 | -0.0115 | -0.0115 | -0.0634 |

## Writing Constraint

- The current ablation package supports descriptive trend statements.
- It should not be written as proof that all modules are statistically significant.
