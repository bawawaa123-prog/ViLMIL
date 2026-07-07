# Stage50 Limitations And Future Work

## Current Limitations
- HCRC currently needs large proposal radius and bbox expansion settings, which can pull in weak or spatially loose high-scale evidence.
- PRARC gates still tend to collapse toward scalar-like behavior instead of maintaining clear sample-adaptive spread.
- Visual residual override remains a major bottleneck and is not solved by the current HCRC/PRARC branches.

## Recommended Future Work
- Shift toward loss-level or uncertainty-aware residual calibration rather than adding another direct residual gate.
- Use train-split-only reliability learning if prompt or residual reliability signals are introduced into learning.
- Add an evidence-margin auxiliary loss to better separate concept-supported correct slides from residual-overridden failures.
- Strengthen concept evidence construction before adding new graph/gate complexity.
