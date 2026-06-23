# Stage53 RCE Figure Plan

## Step54 Figure Candidates

1. `RCE overall pipeline`
Purpose: summarize the final method from dual-scale inputs to region-concept evidence aggregation and slide prediction.
Input needs: final model diagram source, concept prior path, CSG logic sketch, Stage23 main-model metadata.
Placement: main paper.

2. `region-concept evidence heatmap`
Purpose: show which regions support which concepts in a correct case.
Input needs: exported region-level evidence scores and concept labels from the RCE evidence package.
Placement: main paper.

3. `low-high CSG concept interaction visualization`
Purpose: illustrate concept-level cross-scale interaction between low- and high-magnification evidence.
Input needs: cross-scale concept linkage or attention-style export from the evidence package.
Placement: main paper or supplementary depending on clarity.

4. `correct case evidence visualization`
Purpose: demonstrate that the final prediction is grounded in interpretable evidence regions and concepts.
Input needs: region thumbnails, concept labels, slide prediction, and evidence scores.
Placement: main paper.

5. `failure case analysis`
Purpose: show where region-concept evidence is incomplete, ambiguous, or misleading.
Input needs: failed cases with exported evidence, prediction, label, and selected region overlays.
Placement: supplementary material.

6. `w/o CSG vs full evidence ranking comparison`
Purpose: show how CSG changes evidence ordering or concept consistency even when ACC change is small.
Input needs: matched-case evidence exports from full and w/o CSG runs.
Placement: supplementary material, with one compact teaser panel potentially in the main paper.
