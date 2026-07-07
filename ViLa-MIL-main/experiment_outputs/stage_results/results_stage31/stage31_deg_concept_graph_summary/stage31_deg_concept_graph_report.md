# Stage31 DEG Concept Prompt Graph Summary

Step31 reads existing Stage30 result CSV files only. It does not run training, modify model computation, or extract features.

## Inputs

- `results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1/fold_summary.csv`
- `results_stage30/deg_concept_graph_k2_a005_5fold_e20_s1/fold_summary.csv`
- `results_stage30/deg_concept_graph_k4_a005_5fold_e20_s1/fold_summary.csv`
- `results_stage30/deg_concept_graph_k8_a005_5fold_e20_s1/fold_summary.csv`

## Mean ± Std Summary

- `skeleton`: `test_auc`=0.970213 ± 0.013016, `test_acc`=0.922509 ± 0.018779, `test_f1`=0.914456 ± 0.020681, `balanced_acc`=0.917143 ± 0.019648, `sensitivity`=0.900181 ± 0.026688, `pr_auc`=0.944399 ± 0.025168
- `cg_k2_a005`: `test_auc`=0.961367 ± 0.014348, `test_acc`=0.900828 ± 0.032986, `test_f1`=0.890196 ± 0.033231, `balanced_acc`=0.889839 ± 0.024851, `sensitivity`=0.854862 ± 0.052378, `pr_auc`=0.931659 ± 0.027872
- `cg_k4_a005`: `test_auc`=0.961241 ± 0.019549, `test_acc`=0.913210 ± 0.018338, `test_f1`=0.902744 ± 0.020501, `balanced_acc`=0.899983 ± 0.020845, `sensitivity`=0.857938 ± 0.051402, `pr_auc`=0.937577 ± 0.026995
- `cg_k8_a005`: `test_auc`=0.964563 ± 0.017400, `test_acc`=0.910096 ± 0.033296, `test_f1`=0.898092 ± 0.038442, `balanced_acc`=0.893081 ± 0.042660, `sensitivity`=0.839484 ± 0.074139, `pr_auc`=0.940142 ± 0.029548

## Deltas Vs Skeleton

- `cg_k2_a005 - skeleton`: `test_auc`=-0.008845, `test_acc`=-0.021682, `test_f1`=-0.024260, `balanced_acc`=-0.027304, `sensitivity`=-0.045319, `pr_auc`=-0.012740
- `cg_k4_a005 - skeleton`: `test_auc`=-0.008971, `test_acc`=-0.009300, `test_f1`=-0.011712, `balanced_acc`=-0.017160, `sensitivity`=-0.042243, `pr_auc`=-0.006822
- `cg_k8_a005 - skeleton`: `test_auc`=-0.005650, `test_acc`=-0.012414, `test_f1`=-0.016364, `balanced_acc`=-0.024061, `sensitivity`=-0.060697, `pr_auc`=-0.004257

## Recommendation

- `DEG skeleton` remains the recommended main configuration in Stage30.
- `cg_k2_a005`, `cg_k4_a005`, and `cg_k8_a005` all remain below `skeleton` on the main evidence-quality metrics.
- `cg_k8_a005` is the closest Concept Prompt Graph variant on `test_auc` / `pr_auc`, but still trails `skeleton` on `test_f1` (-0.016364), `balanced_acc` (-0.024061), and `sensitivity` (-0.060697).
- The current Concept Prompt Graph should not be treated as a mainline performance module.
- The current main line remains `RCE-v4-CSG-a01-rq16 / DEG skeleton`.
- Combined with Stage27/28 Spatial Region Graph results, Stage30 suggests that direct feature-level message passing can weaken already-learned evidence discrimination rather than improve it.
- Recommended follow-up direction: prioritize evidence-level analysis / evidence export / interpretability, or try evidence-level gated residual / evidence consistency loss instead of stacking ordinary graph message passing.

## Output Files

- `results_stage31/stage31_deg_concept_graph_summary/stage31_deg_concept_graph_summary.csv`
- `results_stage31/stage31_deg_concept_graph_summary/stage31_deg_concept_graph_metric_deltas.csv`
- `results_stage31/stage31_deg_concept_graph_summary/stage31_deg_concept_graph_report.md`

## Compact Tables

### Summary

| variant | method | status | test_auc_formatted | test_acc_formatted | test_f1_formatted | balanced_acc_formatted | sensitivity_formatted | pr_auc_formatted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton | DEG skeleton | ok | 0.970213 ± 0.013016 | 0.922509 ± 0.018779 | 0.914456 ± 0.020681 | 0.917143 ± 0.019648 | 0.900181 ± 0.026688 | 0.944399 ± 0.025168 |
| cg_k2_a005 | DEG Concept Prompt Graph k=2 alpha=0.05 | ok | 0.961367 ± 0.014348 | 0.900828 ± 0.032986 | 0.890196 ± 0.033231 | 0.889839 ± 0.024851 | 0.854862 ± 0.052378 | 0.931659 ± 0.027872 |
| cg_k4_a005 | DEG Concept Prompt Graph k=4 alpha=0.05 | ok | 0.961241 ± 0.019549 | 0.913210 ± 0.018338 | 0.902744 ± 0.020501 | 0.899983 ± 0.020845 | 0.857938 ± 0.051402 | 0.937577 ± 0.026995 |
| cg_k8_a005 | DEG Concept Prompt Graph k=8 alpha=0.05 | ok | 0.964563 ± 0.017400 | 0.910096 ± 0.033296 | 0.898092 ± 0.038442 | 0.893081 ± 0.042660 | 0.839484 ± 0.074139 | 0.940142 ± 0.029548 |

### Deltas

| comparison | test_auc_delta | test_acc_delta | test_f1_delta | balanced_acc_delta | sensitivity_delta | pr_auc_delta |
| --- | --- | --- | --- | --- | --- | --- |
| cg_k2_a005 - skeleton | -0.008845363598970035 | -0.021681534106084088 | -0.024260265659182934 | -0.027303869098494582 | -0.04531886024423348 | -0.012740200692518022 |
| cg_k4_a005 - skeleton | -0.00897138324127389 | -0.009299716895465049 | -0.011712167809225571 | -0.017160050232017943 | -0.042243328810493086 | -0.006821720970137668 |
| cg_k8_a005 - skeleton | -0.005650067041314433 | -0.01241386678062073 | -0.016364419321252877 | -0.02406134925764858 | -0.06069651741293547 | -0.004257088109662299 |
