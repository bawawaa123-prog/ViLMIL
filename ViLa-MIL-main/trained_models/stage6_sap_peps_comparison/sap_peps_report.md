# Stage6 SAP-PEPS Comparison

## Metric Summary

| experiment | status | test_auc_mean | test_acc_mean | test_f1_mean | val_auc_mean | balanced_acc_mean | sensitivity_mean | specificity_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=5 tau=0.07 | ok | 0.968327 | 0.939047 | 0.932179 | 0.972999 | 0.931275 | 0.906422 | 0.956127 | 0.945866 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | ok | 0.969688 | 0.936985 | 0.930095 | 0.973415 | 0.930428 | 0.909453 | 0.951403 | 0.940601 |
| Concept-12 PEPS topk=5 tau=0.1 | ok | 0.969500 | 0.935965 | 0.929148 | 0.971365 | 0.929636 | 0.909408 | 0.949865 | 0.941762 |
| Concept-12 SAP-PEPS topk=5 tau=0.1 | missing | NA | NA | NA | NA | NA | NA | NA | NA |

## Spatial Diagnostics

| experiment | fold | scale | semantic_evidence_mean | semantic_evidence_std | spatial_score_mean | spatial_score_std | final_evidence_mean | final_evidence_std | topk_proto_mean_dist_mean | topk_proto_mean_dist_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=5 tau=0.07 | 1 | high | 0.049891 | 0.029709 | 0.000000 | 0.000000 | 0.049891 | 0.029709 | 0.008840 | 0.003544 |
| Concept-12 PEPS topk=5 tau=0.07 | 1 | low | 0.092497 | 0.027217 | 0.000000 | 0.000000 | 0.092497 | 0.027217 | 0.007449 | 0.002152 |
| Concept-12 PEPS topk=5 tau=0.07 | 2 | high | 0.057990 | 0.038962 | 0.000000 | 0.000000 | 0.057990 | 0.038962 | 0.014301 | 0.008313 |
| Concept-12 PEPS topk=5 tau=0.07 | 2 | low | 0.103803 | 0.036739 | 0.000000 | 0.000000 | 0.103803 | 0.036739 | 0.024020 | 0.009447 |
| Concept-12 PEPS topk=5 tau=0.07 | 3 | high | 0.068746 | 0.033277 | 0.000000 | 0.000000 | 0.068746 | 0.033277 | 0.003997 | 0.001967 |
| Concept-12 PEPS topk=5 tau=0.07 | 3 | low | 0.107976 | 0.033311 | 0.000000 | 0.000000 | 0.107976 | 0.033311 | 0.005901 | 0.002412 |
| Concept-12 PEPS topk=5 tau=0.07 | 4 | high | 0.085577 | 0.038917 | 0.000000 | 0.000000 | 0.085577 | 0.038917 | 0.002917 | 0.003234 |
| Concept-12 PEPS topk=5 tau=0.07 | 4 | low | 0.126981 | 0.037672 | 0.000000 | 0.000000 | 0.126981 | 0.037672 | 0.004362 | 0.005878 |
| Concept-12 PEPS topk=5 tau=0.07 | 5 | high | 0.034181 | 0.032679 | 0.000000 | 0.000000 | 0.034181 | 0.032679 | 0.001889 | 0.000554 |
| Concept-12 PEPS topk=5 tau=0.07 | 5 | low | 0.074447 | 0.031938 | 0.000000 | 0.000000 | 0.074447 | 0.031938 | 0.002817 | 0.000618 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 1 | high | 0.058881 | 0.029409 | -0.012398 | 0.001922 | 0.046483 | 0.029286 | 0.011754 | 0.001865 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 1 | low | 0.095539 | 0.026627 | -0.011060 | 0.001471 | 0.084479 | 0.026459 | 0.010260 | 0.001621 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 2 | high | 0.062340 | 0.040780 | -0.018742 | 0.004024 | 0.043598 | 0.042252 | 0.017526 | 0.003193 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 2 | low | 0.114293 | 0.035711 | -0.022197 | 0.002921 | 0.092096 | 0.036057 | 0.020888 | 0.002702 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 3 | high | 0.058772 | 0.025966 | -0.011003 | 0.001070 | 0.047769 | 0.025983 | 0.010172 | 0.001323 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 3 | low | 0.099509 | 0.027777 | -0.008799 | 0.001208 | 0.090711 | 0.027795 | 0.007447 | 0.001222 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 4 | high | 0.077631 | 0.034806 | -0.014456 | 0.003368 | 0.063175 | 0.035758 | 0.013762 | 0.003440 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 4 | low | 0.121973 | 0.037256 | -0.017170 | 0.002473 | 0.104803 | 0.037920 | 0.015961 | 0.002635 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 5 | high | 0.047603 | 0.026855 | -0.008480 | 0.001586 | 0.039123 | 0.026912 | 0.007413 | 0.001514 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | 5 | low | 0.088875 | 0.031505 | -0.006338 | 0.000965 | 0.082536 | 0.031445 | 0.005706 | 0.001008 |

## Aggregated Spatial Effect

| experiment | scale | spatial_score_mean | final_evidence_mean | topk_proto_mean_dist_mean |
| --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=5 tau=0.07 | high | 0.000000 | 0.059277 | 0.006389 |
| Concept-12 PEPS topk=5 tau=0.07 | low | 0.000000 | 0.101141 | 0.008910 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | high | -0.013016 | 0.048030 | 0.012125 |
| Concept-12 SAP-PEPS topk=5 tau=0.07 | low | -0.013113 | 0.090925 | 0.012053 |