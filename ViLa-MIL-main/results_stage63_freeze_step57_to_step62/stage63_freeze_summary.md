# Step63 Freeze Summary

## Frozen conclusion

- final primary model = Step58C Residual-Constrained RCE config D
- secondary variants = Step59C Dynamic CSG config A, Step60D CCRA config C
- rejected variant = Step61D L2H Retrieval config G
- next step should start from a new innovation branch, not overwrite the current frozen version

## Final reading of Step57-Step62

- Stage57C remains the reproduced RCE-v2 baseline reference with AUC=0.970213, ACC=0.922509, F1=0.914456, BACC=0.917143, PR-AUC=0.944399.
- Step58C is frozen as the selected main model because it improves hard classification metrics over Stage57C: ACC 0.922509 -> 0.927691, F1 0.914456 -> 0.919625, BACC 0.917143 -> 0.919700, while accepting a trade-off in AUC 0.970213 -> 0.964842 and PR-AUC 0.944399 -> 0.931551.
- Step58C also gives the key evidence shift: visual_ratio_mean 0.719580 -> 0.320575, concept_ratio_mean 0.280420 -> 0.679425.
- Step59C stays as a secondary variant: it preserves strong AUC=0.969323 and PR-AUC=0.941877, but ACC/F1/BACC are weaker than Step58C.
- Step60D stays as a secondary variant: it is performance-competitive with AUC=0.968601, ACC=0.922493, F1=0.915115, BACC=0.920087, but it is not the chosen primary checkpoint.
- Step61D is rejected as a final main model: AUC=0.967921, ACC=0.921479, F1=0.911256, BACC=0.904859, PR-AUC=0.938282.

## Freeze policy

- Step63 performs no training.
- Step63 does not modify any model forward logic.
- Step63 is a checkpoint-preparation step for frozen results, paper assets, and GitHub bookkeeping only.
