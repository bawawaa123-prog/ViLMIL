# Stage3 PEPS Extended Report

## Current Recommendation

Concept-12 PEPS topk=5 tau=0.1 is the current primary recommendation (test_auc=0.969500, test_acc=0.935965, test_f1=0.929148, balanced_acc=0.929636).

## All Current PEPS Results

| experiment | status | test_auc_mean | test_acc_mean | test_f1_mean | val_auc_mean | balanced_acc_mean | sensitivity_mean | specificity_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 embedding_mean | ok | 0.965884 | 0.933876 | 0.926832 | 0.969937 | 0.927344 | 0.906422 | 0.948265 | 0.941293 |
| Concept-12 MLP dynamic_gate | ok | 0.965461 | 0.928722 | 0.921510 | 0.970106 | 0.925592 | 0.915468 | 0.935716 | 0.927728 |
| Concept-12 PEPS topk=1 tau=0.1 | ok | 0.965226 | 0.926660 | 0.918443 | 0.968777 | 0.917507 | 0.888286 | 0.946727 | 0.941989 |
| Concept-12 PEPS topk=3 tau=0.1 | ok | 0.967991 | 0.935954 | 0.928744 | 0.971034 | 0.928121 | 0.903302 | 0.952940 | 0.942830 |
| Concept-12 PEPS topk=5 tau=0.1 | ok | 0.969500 | 0.935965 | 0.929148 | 0.971365 | 0.929636 | 0.909408 | 0.949865 | 0.941762 |
| Concept-12 PEPS topk=3 tau=0.07 | ok | 0.968138 | 0.932845 | 0.925810 | 0.970790 | 0.928006 | 0.912483 | 0.943529 | 0.938729 |
| Concept-12 PEPS topk=3 tau=0.15 | ok | 0.961322 | 0.918396 | 0.908797 | 0.971954 | 0.906960 | 0.870330 | 0.943590 | 0.934522 |
| Concept-12 PEPS topk=5 tau=0.07 | ok | 0.968327 | 0.939047 | 0.932179 | 0.972999 | 0.931275 | 0.906422 | 0.956127 | 0.945866 |
| Concept-12 PEPS topk=5 tau=0.15 | ok | 0.966858 | 0.933898 | 0.926994 | 0.972843 | 0.928812 | 0.912483 | 0.945140 | 0.936667 |

## Tau Sweep Results

| experiment | status | test_auc_mean | test_acc_mean | test_f1_mean | balanced_acc_mean | sensitivity_mean | specificity_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=3 tau=0.1 | ok | 0.967991 | 0.935954 | 0.928744 | 0.928121 | 0.903302 | 0.952940 | 0.942830 |
| Concept-12 PEPS topk=5 tau=0.1 | ok | 0.969500 | 0.935965 | 0.929148 | 0.929636 | 0.909408 | 0.949865 | 0.941762 |
| Concept-12 PEPS topk=3 tau=0.07 | ok | 0.968138 | 0.932845 | 0.925810 | 0.928006 | 0.912483 | 0.943529 | 0.938729 |
| Concept-12 PEPS topk=3 tau=0.15 | ok | 0.961322 | 0.918396 | 0.908797 | 0.906960 | 0.870330 | 0.943590 | 0.934522 |
| Concept-12 PEPS topk=5 tau=0.07 | ok | 0.968327 | 0.939047 | 0.932179 | 0.931275 | 0.906422 | 0.956127 | 0.945866 |
| Concept-12 PEPS topk=5 tau=0.15 | ok | 0.966858 | 0.933898 | 0.926994 | 0.928812 | 0.912483 | 0.945140 | 0.936667 |

## Topk=3 vs Topk=5

| experiment | test_auc_mean | test_acc_mean | test_f1_mean | balanced_acc_mean | specificity_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=3 tau=0.1 | 0.967991 | 0.935954 | 0.928744 | 0.928121 | 0.952940 | 0.942830 |
| Concept-12 PEPS topk=5 tau=0.1 | 0.969500 | 0.935965 | 0.929148 | 0.929636 | 0.949865 | 0.941762 |
| Concept-12 PEPS topk=3 tau=0.07 | 0.968138 | 0.932845 | 0.925810 | 0.928006 | 0.943529 | 0.938729 |
| Concept-12 PEPS topk=3 tau=0.15 | 0.961322 | 0.918396 | 0.908797 | 0.906960 | 0.943590 | 0.934522 |
| Concept-12 PEPS topk=5 tau=0.07 | 0.968327 | 0.939047 | 0.932179 | 0.931275 | 0.956127 | 0.945866 |
| Concept-12 PEPS topk=5 tau=0.15 | 0.966858 | 0.933898 | 0.926994 | 0.928812 | 0.945140 | 0.936667 |

## Fold-Level Delta Analysis

| candidate_experiment | reference_experiment | fold | delta_test_auc | delta_test_acc | delta_test_f1 | delta_balanced_acc | delta_sensitivity | delta_specificity | delta_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | 1 | -0.001657 | -0.020619 | -0.021979 | -0.015625 | 0.000000 | -0.031250 | -0.006741 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | 2 | -0.004735 | 0.000000 | -0.001191 | -0.007339 | -0.030303 | 0.015625 | -0.010701 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | 3 | 0.010930 | 0.005155 | 0.006035 | 0.007463 | 0.014925 | 0.000000 | 0.004536 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | 4 | 0.004056 | 0.020725 | 0.023197 | 0.023026 | 0.030303 | 0.015748 | -0.020284 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | 5 | 0.009485 | 0.005181 | 0.005517 | 0.003937 | 0.000000 | 0.007874 | 0.035534 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | 1 | -0.002012 | -0.020619 | -0.021979 | -0.015625 | 0.000000 | -0.031250 | -0.007452 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | 2 | -0.004616 | 0.020619 | 0.019011 | 0.000947 | -0.060606 | 0.062500 | -0.013536 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | 3 | 0.004936 | -0.005155 | -0.004629 | 0.003114 | 0.029851 | -0.023622 | 0.003106 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | 4 | 0.004414 | 0.015544 | 0.017153 | 0.015450 | 0.015152 | 0.015748 | -0.024046 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | 5 | 0.007814 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.049610 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | 1 | -0.005327 | -0.005155 | -0.007196 | -0.018584 | -0.060606 | 0.023438 | -0.012399 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | 2 | 0.000829 | 0.010309 | 0.012636 | 0.018821 | 0.045455 | -0.007812 | 0.019678 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | 3 | 0.012222 | 0.005155 | 0.005583 | 0.003937 | 0.000000 | 0.007874 | 0.007987 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | 4 | 0.015450 | 0.031088 | 0.034109 | 0.030900 | 0.030303 | 0.031496 | 0.028648 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | 5 | -0.002983 | -0.005181 | -0.006938 | -0.014853 | -0.045455 | 0.015748 | 0.026256 |

### Win/Loss Counts

| candidate_experiment | reference_experiment | metric | wins | losses | ties | mean_delta |
| --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | test_auc | 3 | 2 | 0 | 0.003616 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | test_acc | 3 | 1 | 1 | 0.002089 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | test_f1 | 3 | 2 | 0 | 0.002316 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | balanced_acc | 3 | 2 | 0 | 0.002292 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | sensitivity | 2 | 1 | 2 | 0.002985 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | specificity | 3 | 1 | 1 | 0.001599 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 embedding_mean | pr_auc | 2 | 3 | 0 | 0.000469 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | test_auc | 3 | 2 | 0 | 0.002107 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | test_acc | 2 | 2 | 1 | 0.002078 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | test_f1 | 2 | 2 | 1 | 0.001911 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | balanced_acc | 3 | 1 | 1 | 0.000777 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | sensitivity | 2 | 1 | 2 | -0.003121 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | specificity | 2 | 2 | 1 | 0.004675 |
| Concept-12 PEPS topk=3 tau=0.1 | Concept-12 embedding_mean | pr_auc | 2 | 3 | 0 | 0.001536 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | test_auc | 3 | 2 | 0 | 0.004038 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | test_acc | 3 | 2 | 0 | 0.007243 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | test_f1 | 3 | 2 | 0 | 0.007639 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | balanced_acc | 3 | 2 | 0 | 0.004044 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | sensitivity | 2 | 2 | 1 | -0.006061 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | specificity | 4 | 1 | 0 | 0.014149 |
| Concept-12 PEPS topk=5 tau=0.1 | Concept-12 MLP dynamic_gate | pr_auc | 4 | 1 | 0 | 0.014034 |

## Prompt Usage Diversity

| experiment | num_folds | low_top1_unique_prompts_mean | low_top1_unique_prompts_std | high_top1_unique_prompts_mean | high_top1_unique_prompts_std | low_prompt_entropy_mean | low_prompt_entropy_std | high_prompt_entropy_mean | high_prompt_entropy_std | low_prompt_collapse_rate_mean | low_prompt_collapse_rate_std | high_prompt_collapse_rate_mean | high_prompt_collapse_rate_std | num_low_samples_mean | num_low_samples_std | num_high_samples_mean | num_high_samples_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 MLP dynamic_gate | 5 | 2.600000 | 0.800000 | 3.000000 | 1.095445 | 0.750049 | 0.117365 | 0.856722 | 0.229846 | 0.643609 | 0.020887 | 0.559073 | 0.110467 | 193.600000 | 0.489898 | 193.600000 | 0.489898 |
| Concept-12 PEPS topk=3 tau=0.1 | 5 | 10.000000 | 3.847077 | 8.200000 | 2.785678 | 1.580997 | 0.453328 | 1.495941 | 0.425784 | 0.428375 | 0.183509 | 0.439960 | 0.122368 | 193.600000 | 0.489898 | 193.600000 | 0.489898 |
| Concept-12 PEPS topk=5 tau=0.1 | 5 | 9.000000 | 1.264911 | 6.200000 | 1.166190 | 1.544982 | 0.238865 | 1.216619 | 0.241739 | 0.422253 | 0.146006 | 0.559869 | 0.090429 | 193.600000 | 0.489898 | 193.600000 | 0.489898 |

## Prompt Collapse Check

| experiment | fold | low_top1_unique_prompts | high_top1_unique_prompts | low_prompt_entropy | high_prompt_entropy | low_prompt_collapse_rate | high_prompt_collapse_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=3 tau=0.1 | 1 | 8 | 9 | 1.546631 | 1.550615 | 0.335052 | 0.469072 |
| Concept-12 PEPS topk=3 tau=0.1 | 2 | 11 | 11 | 1.360097 | 1.883982 | 0.634021 | 0.309278 |
| Concept-12 PEPS topk=3 tau=0.1 | 3 | 6 | 3 | 1.041912 | 0.687370 | 0.649485 | 0.659794 |
| Concept-12 PEPS topk=3 tau=0.1 | 4 | 8 | 8 | 1.547217 | 1.552011 | 0.341969 | 0.409326 |
| Concept-12 PEPS topk=3 tau=0.1 | 5 | 17 | 10 | 2.409127 | 1.805725 | 0.181347 | 0.352332 |
| Concept-12 PEPS topk=5 tau=0.1 | 1 | 8 | 8 | 1.532525 | 1.571113 | 0.355670 | 0.443299 |
| Concept-12 PEPS topk=5 tau=0.1 | 2 | 10 | 6 | 1.481693 | 1.052078 | 0.515464 | 0.603093 |
| Concept-12 PEPS topk=5 tau=0.1 | 3 | 8 | 5 | 1.195492 | 0.935708 | 0.654639 | 0.680412 |
| Concept-12 PEPS topk=5 tau=0.1 | 4 | 8 | 7 | 1.572440 | 1.430029 | 0.347150 | 0.466321 |
| Concept-12 PEPS topk=5 tau=0.1 | 5 | 11 | 5 | 1.942757 | 1.094165 | 0.238342 | 0.606218 |
| Concept-12 MLP dynamic_gate | 1 | 2 | 5 | 0.669526 | 1.210487 | 0.608247 | 0.386598 |
| Concept-12 MLP dynamic_gate | 2 | 3 | 3 | 0.902669 | 0.732924 | 0.639175 | 0.639175 |
| Concept-12 MLP dynamic_gate | 3 | 4 | 3 | 0.883743 | 1.045895 | 0.670103 | 0.469072 |
| Concept-12 MLP dynamic_gate | 4 | 2 | 2 | 0.642333 | 0.642333 | 0.658031 | 0.658031 |
| Concept-12 MLP dynamic_gate | 5 | 2 | 2 | 0.651974 | 0.651974 | 0.642487 | 0.642487 |

## Most Frequent Top-1 Prompts by True Class

| experiment | true_class_name | rank | prompt_text | prompt_concept | count | frequency |
| --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=3 tau=0.1 | Adenocarcinoma | 1 | A high magnification histopathology image showing gland-forming adenocarcinoma cells with vesicular chromatin and prominent nucleoli. | prominent nucleoli in gland-forming tumor cells | 388 | 0.329652 |
| Concept-12 PEPS topk=3 tau=0.1 | Adenocarcinoma | 2 | A low magnification histopathology image showing multifocal irregular glandular proliferation consistent with lung adenocarcinoma. | multifocal irregular glandular proliferation | 277 | 0.235344 |
| Concept-12 PEPS topk=3 tau=0.1 | Adenocarcinoma | 3 | A low magnification histopathology image showing infiltrative gland-forming lung adenocarcinoma replacing alveolated lung parenchyma. | infiltrative glandular adenocarcinoma | 220 | 0.186916 |
| Concept-12 PEPS topk=3 tau=0.1 | Adenocarcinoma | 4 | A high magnification histopathology image showing atypical gland-forming tumor cells with pseudostratified nuclei. | atypical gland-forming tumor cells | 110 | 0.093458 |
| Concept-12 PEPS topk=3 tau=0.1 | Adenocarcinoma | 5 | A low magnification histopathology image showing lung adenocarcinoma with cribriform glandular architecture. | cribriform glandular pattern | 96 | 0.081563 |
| Concept-12 PEPS topk=3 tau=0.1 | NonAdenocarcinoma | 1 | A high magnification histopathology image showing inflammatory cells, granulomas, or fibrosis without adenocarcinoma morphology. | inflammatory cells and granulomas | 194 | 0.336222 |
| Concept-12 PEPS topk=3 tau=0.1 | NonAdenocarcinoma | 2 | A low magnification histopathology image showing benign reactive lung tissue changes without adenocarcinoma morphology. | benign reactive lung tissue | 69 | 0.119584 |
| Concept-12 PEPS topk=3 tau=0.1 | NonAdenocarcinoma | 3 | A low magnification histopathology image showing a fibroinflammatory lung lesion with scarring and no malignant gland formation. | fibroinflammatory scar-like lesion | 63 | 0.109185 |
| Concept-12 PEPS topk=3 tau=0.1 | NonAdenocarcinoma | 4 | A low magnification histopathology image showing metastatic tumor in the lung without primary lung adenocarcinoma morphology. | metastatic tumor in lung | 51 | 0.088388 |
| Concept-12 PEPS topk=3 tau=0.1 | NonAdenocarcinoma | 5 | A low magnification histopathology image showing inflammatory or granulomatous lung lesion without malignant glandular structures. | inflammatory or granulomatous lesion | 45 | 0.077990 |
| Concept-12 PEPS topk=5 tau=0.1 | Adenocarcinoma | 1 | A high magnification histopathology image showing gland-forming adenocarcinoma cells with vesicular chromatin and prominent nucleoli. | prominent nucleoli in gland-forming tumor cells | 543 | 0.438611 |
| Concept-12 PEPS topk=5 tau=0.1 | Adenocarcinoma | 2 | A low magnification histopathology image showing multifocal irregular glandular proliferation consistent with lung adenocarcinoma. | multifocal irregular glandular proliferation | 281 | 0.226979 |
| Concept-12 PEPS topk=5 tau=0.1 | Adenocarcinoma | 3 | A low magnification histopathology image showing infiltrative gland-forming lung adenocarcinoma replacing alveolated lung parenchyma. | infiltrative glandular adenocarcinoma | 215 | 0.173667 |
| Concept-12 PEPS topk=5 tau=0.1 | Adenocarcinoma | 4 | A low magnification histopathology image showing lung adenocarcinoma with cribriform glandular architecture. | cribriform glandular pattern | 84 | 0.067851 |
| Concept-12 PEPS topk=5 tau=0.1 | Adenocarcinoma | 5 | A high magnification histopathology image showing atypical gland-forming tumor cells with pseudostratified nuclei. | atypical gland-forming tumor cells | 44 | 0.035541 |
| Concept-12 PEPS topk=5 tau=0.1 | NonAdenocarcinoma | 1 | A high magnification histopathology image showing inflammatory cells, granulomas, or fibrosis without adenocarcinoma morphology. | inflammatory cells and granulomas | 138 | 0.231156 |
| Concept-12 PEPS topk=5 tau=0.1 | NonAdenocarcinoma | 2 | A high magnification histopathology image showing squamous cell carcinoma with keratinization and keratin pearls. | keratinization | 98 | 0.164154 |
| Concept-12 PEPS topk=5 tau=0.1 | NonAdenocarcinoma | 3 | A low magnification histopathology image showing a fibroinflammatory lung lesion with scarring and no malignant gland formation. | fibroinflammatory scar-like lesion | 73 | 0.122278 |
| Concept-12 PEPS topk=5 tau=0.1 | NonAdenocarcinoma | 4 | A low magnification histopathology image showing solid non-gland-forming tumor nests without adenocarcinoma-type acinar structures. | solid non-gland-forming tumor nests | 60 | 0.100503 |
| Concept-12 PEPS topk=5 tau=0.1 | NonAdenocarcinoma | 5 | A low magnification histopathology image showing inflammatory or granulomatous lung lesion without malignant glandular structures. | inflammatory or granulomatous lesion | 39 | 0.065327 |