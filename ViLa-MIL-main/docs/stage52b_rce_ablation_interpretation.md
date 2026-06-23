# Step52B RCE Ablation Interpretation

Step52 confirms that the final RCE main model is not driven by a single isolated trick. Instead, concept prior, visual residual, cross-scale graph, and logit calibration contribute to an evidence chain with different roles.

## Main Findings

1. Concept prior contributes the most. Removing it causes the largest degradation across the core classification metrics, including AUC (-0.0110), ACC (-0.0206), F1 (-0.0241), BACC (-0.0280), and PR-AUC (-0.0366).
2. Visual residual provides important complementary visual evidence. Removing it leads to notable drops in ACC (-0.0176), F1 (-0.0198), BACC (-0.0211), and PR-AUC (-0.0205), indicating that concept evidence alone is not sufficient.
3. CSG should be interpreted cautiously. The ACC change after removing CSG is small (-0.0010), but the AUC (-0.0047) and PR-AUC (-0.0166) drops are more visible. This suggests that CSG is more helpful for ranking quality and confidence refinement than for raw accuracy alone.
4. Logit calibration helps balanced accuracy and F1, but PR-AUC shows a trade-off. Removing calibration reduces F1 (-0.0034) and BACC (-0.0095), while PR-AUC changes in the opposite direction (+0.0042). The calibration module should therefore be described as improving decision balance rather than universally improving every metric.

## Paper-Oriented Conclusion

The ablation results support the paper's central claim that the RCE main model works by combining class-aware concept weighting, complementary visual residual evidence, cross-scale interaction, and calibrated final decision scores. The evidence chain is strongest when all four parts are present together.

This interpretation intentionally avoids claims about DEG modules and does not overstate the contribution of CSG to ACC.

