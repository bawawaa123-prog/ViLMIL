# Stage49 Final Consolidation Report

## Project Overview
- Up to Step48b, the project completed a full search over cross-scale evidence modeling, low-high consistency, HCRC, PRARC-v1, and PRARC-v2 diagnostics.
- The final objective of Step49 is consolidation rather than further training or architecture changes.

## Final Primary Model
- final_primary_model: `RCE-v4-CSG-a01-rq16 / DEG skeleton`
- final_primary_model_auc: `0.9702`
- final_primary_model_acc: `0.9225`
- final_primary_model_f1: `0.9145`
- final_primary_model_balanced_acc: `0.9171`
- final_primary_model_pr_auc: `0.9444`
- final_secondary_variant: `RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0`

## HCRC Summary
- enter_step45_from_stage44: `False`
- HCRC-Light completed clean 5-fold evaluation but did not exceed baseline on the main ranking metrics.
- The HCRC branch should be retained as a negative ablation / future-work branch rather than promoted to the main model.

## PRARC-v1 Summary
- recommend_enter_step48_from_stage47: `False`
- PRARC-v1 completed 5-fold evaluation but remained below baseline, and Step47 diagnostics indicated weak or near-scalar gate behavior.

## PRARC-v2 Summary
- recommend_enter_step49_from_stage48b: `False`
- PRARC-v2 smoke variants were engineering-stable, but Step48b still found insufficient gate dynamics for promotion.

## Negative Ablation Summary
- Ordinary region graph, ordinary concept graph, scalar visual evidence gate, HCRC-Light, PRARC-v1, and PRARC-v2 should all remain outside the final primary model.

## Paper Narrative
- The strongest paper narrative is that region-concept evidence modeling plus cross-scale concept reasoning is the current winning path.
- The story should emphasize systematic exploration and rejection of alternative graph/gate modules rather than over-claiming new modules as successful.
- Visual residual override should be presented as the main unresolved bottleneck.

## Next Step Recommendation
- Do not continue HCRC or PRARC training in the current branch.
- If continuing research, prioritize loss-level or uncertainty-aware residual calibration instead of more direct residual gating.
- Start organizing paper materials now because the primary model and negative-ablation package are already well supported.

## Execution Decision
- recommend_more_training_now: `False`
- recommend_start_writing_paper_materials: `True`
