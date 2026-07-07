# Stage50 Experiment Tables Plan

## Main Comparison Table
- Include the final primary model, the low-high consistency trade-off variant, and representative baseline/ablation references from Stage39.
- Core metrics: `AUC`, `ACC`, `F1`, `Balanced ACC`, `PR-AUC`.
- Final primary metrics snapshot: `AUC=0.9702`, `ACC=0.9225`, `F1=0.9145`, `Balanced ACC=0.9171`, `PR-AUC=0.9444`.

## Main Ablation Table
- Focus on positive and retained components: `CSG a01`, `rq16`, concept prompt pool, logit calibration, visual residual, and the low-high consistency trade-off variant.
- Do not invent unavailable standalone deltas; leave them blank and explain that the component is retained as part of the final pipeline.

## Negative Ablation Table
- Cover ordinary region graph, ordinary concept graph, scalar visual evidence gate, HCRC-Light, PRARC-v1, and PRARC-v2.
- Emphasize why they were not promoted rather than treating them as incomplete engineering attempts.

## HCRC/PRARC Exploratory Ablation Table
- Optional appendix table summarizing HCRC 5-fold metrics, PRARC-v1 5-fold metrics, and PRARC-v2 gate-diagnostic smoke metrics.
- Keep wording conservative: exploratory branches were completed and rejected by evidence.

## Failure Analysis Table
- Report failure-type counts from Stage33 plus the Stage39 consistency trade-off comparison and Stage45 override-profile interpretation.
- Make visual residual override the central unresolved failure axis.

## Optional Sensitivity Table
- If page budget allows, add sensitivity/specificity trade-offs for HCRC-Light and low-high consistency to show why they were not promoted despite selective gains.
