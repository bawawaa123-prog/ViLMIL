# Stage49 Paper Claims And Evidence

## Paper-Ready Claims
- Final primary model should remain `RCE-v4-CSG-a01-rq16 / DEG skeleton` because the strongest evidence package still points to it as the most robust default model.
- Region-concept evidence modeling is effective because the Stage39 final package retained the cross-scale evidence model as the strongest overall design rather than reverting to plain graph or scalar-gate add-ons.
- Cross-scale concept relation modeling is a key contributor because Stage39 concluded `CSG a01 > CSG a005` and retained the CSG-equipped skeleton as the primary model.
- The current main failure source is visual residual override rather than missing graph complexity, because Stage33/39/45 repeatedly showed that wrong visual residual support explains a large share of residual errors.
- HCRC and PRARC should be framed as systematic exploratory branches rather than failed implementation attempts, because they were validated through dedicated smoke, 5-fold, and diagnostic stages before being rejected.
- The final model choice is evidence-driven: Stage39 selected the default model, Stage44 showed HCRC remained below baseline, Stage47 showed PRARC-v1 remained below baseline, and Stage48b showed PRARC-v2 gate dynamics stayed insufficient.

## Claims To Avoid
- Do not claim that HCRC improved the final model.
- Do not claim that PRARC already solved visual residual override.
- Do not claim that the PRARC gate became strongly sample-adaptive.
- Do not claim that low-high spatial correspondence is inherently superior to concept-level cross-scale evidence reasoning.

## Evidence Anchors
- Stage39 final recommendation and ablation summaries provide the main primary-model evidence.
- Stage44 HCRC 5-fold summary provides the formal negative result for HCRC-Light.
- Stage45 reliability audit supports the statement that visual residual override is still the main bottleneck.
- Stage47 PRARC 5-fold and gate diagnostics provide the formal PRARC-v1 negative result.
- Stage48b PRARC-v2 variant sweep provides the final PRARC-v2 negative-ablation evidence.

