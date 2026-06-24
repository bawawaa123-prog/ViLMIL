# Step55 Claims To Make And Avoid

| Can Make | Avoid |
| --- | --- |
| The frozen main model is `RCE-v4-CSG-a01-rq16` with `model_type=RCE_MIL_BiomedCLIP`. | DEG/HCRC/PRARC is the current frozen main model. |
| RCE extends ViLa-MIL from slide-level vision-language alignment toward region-concept evidence reasoning. | The current release already proves broad external validation or multi-cancer generalization. |
| Concept prior is the strongest single contributor among the audited core ablations. | All modules are statistically significant. |
| Visual residual provides important complementary visual evidence. | Heatmaps or evidence figures provide pathologist-level localization. |
| CSG mainly improves AUC / PR-AUC / evidence ranking or confidence behavior and should not be oversold on ACC. | CSG significantly and substantially improves ACC. |
| Step54C makes the full and wo_csg evidence sources matched direct-export interpretability illustrations. | Evidence figures are training-significance proof or a strict localization benchmark. |
| Evidence visualization can support interpretability discussion of region-concept reasoning. | Evidence visualization is equivalent to expert annotation or formal localization evaluation. |

Paired t-test values from Step52B should be cited only as descriptive references in this release. No conventional `p < 0.05` conclusion should be claimed for the audited core ablations.
