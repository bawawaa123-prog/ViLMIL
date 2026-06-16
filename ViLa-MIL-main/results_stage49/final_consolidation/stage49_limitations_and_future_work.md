# Stage49 Limitations And Future Work

## Current Limitations
- Current HCRC relies on relatively large proposal radius and bbox expansion, which can pull in weak or spatially loose high-scale evidence.
- Current PRARC gates can still collapse toward scalar-like behavior instead of maintaining meaningful sample-adaptive spread.
- Visual residual override remains a major bottleneck and is not solved by simply stacking more ordinary graph or gating modules.

## Better Next Directions
- Loss-level visual residual calibration instead of another direct residual gate.
- Uncertainty-aware residual suppression to penalize visually confident but concept-inconsistent residual corrections.
- Train-split-only reliability learning so prompt or residual reliability signals are learned without test-derived leakage.
- Evidence-margin auxiliary loss to reward cleaner separation between concept-supported correct slides and residual-overridden failures.
- Stronger concept evidence construction rather than continued stacking of ordinary graph smoothing or scalar-style gates.

