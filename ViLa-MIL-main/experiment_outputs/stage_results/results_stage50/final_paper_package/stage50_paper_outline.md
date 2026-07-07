# Stage50 Paper Outline

## Title Suggestion
- `Evidence-Guided Cross-Scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification`
- Optional shorter title: `Evidence-Guided Cross-Scale Vision-Language MIL for Whole Slide Classification`

## Abstract Structure
- Problem: WSI classification needs not only strong slide-level accuracy but also interpretable evidence paths.
- Method: summarize region-concept evidence, low/high concept evidence, CSG reasoning, visual residual, and calibrated logits.
- Main result: retain `RCE-v4-CSG-a01-rq16 / DEG skeleton` as the final default model.
- Boundary: report `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0` as a secondary calibration trade-off variant only.
- Failure analysis: highlight visual residual override as the main unresolved error type.

## Introduction Structure
- Clinical and technical motivation for evidence-aware WSI classification.
- Gap: stronger graphs or gates do not automatically yield more reliable evidence.
- Core idea: make region-concept evidence and cross-scale concept reasoning the main modeling axis.
- Contributions: final evidence-driven model selection, negative ablation package, and failure-analysis pipeline.

## Related Work Structure
- Vision-language MIL for pathology.
- Cross-scale reasoning for WSI classification.
- Evidence decomposition / interpretability in medical AI.
- Reliability calibration and residual correction.

## Method Structure
- BiomedCLIP low/high patch feature extraction.
- Learnable region queries and region-concept evidence.
- Concept prompt pool and low/high concept logits.
- CSG cross-scale concept reasoning.
- Visual residual branch, calibrated logits, and evidence decomposition.

## Experiments Structure
- Dataset, strict CV protocol, and metrics.
- Main comparison table centered on the final primary model.
- Main ablation on CSG strength and region query count.
- Calibration trade-off comparison with low-high consistency.

## Ablation Study Structure
- Positive ablation: CSG a01 and rq16.
- Final-method component table: concept prompt pool, logit calibration, visual residual.
- Secondary trade-off variant: low-high consistency.

## Failure Analysis Structure
- Failure type counts from Stage33.
- Skeleton vs consistency comparison from Stage39.
- Stage45 override-profile interpretation.

## Limitations Structure
- HCRC relies on loose proposal/bbox settings and can pull weak high-scale evidence.
- PRARC gates still collapse toward near-scalar behavior.
- Visual residual override remains unsolved.

## Conclusion Structure
- Re-state the final primary model and what is actually validated.
- Emphasize evidence-driven model choice rather than module stacking.
- Close with conservative future-work directions.
