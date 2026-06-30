# Step63 Model Decision

## Selected primary model

- Step58C Residual-Constrained RCE config D is the final primary model.

## Why Step58C is selected

- Compared with Stage57C, Step58C improves ACC (0.922509 -> 0.927691), F1 (0.914456 -> 0.919625), and BACC (0.917143 -> 0.919700).
- visual_ratio_mean drops from about 0.719580 to 0.320575.
- concept_ratio_mean rises from about 0.280420 to 0.679425.
- The full branch remains stronger than concept_only: full_minus_concept_acc=+0.010325, full_minus_concept_auc=+0.003213.
- The AUC / PR-AUC trade-off is explicit rather than hidden: AUC 0.970213 -> 0.964842, PR-AUC 0.944399 -> 0.931551.

## Why the others are not the primary checkpoint

- Step59C Dynamic CSG stays as a secondary variant because its AUC=0.969323 and PR-AUC=0.941877 are good, but ACC/F1/BACC are weaker than Step58C.
- Step60D CCRA config C stays as a secondary variant because it is balanced and interpretable, but it is not the final selected primary checkpoint.
- Step61D L2H Retrieval does not enter the final primary model because ACC/F1/BACC remain weaker than Step58C, especially BACC=0.904859.
