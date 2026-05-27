# Final DCP-ViLa-MIL Analysis

## Fixed Models

- Primary model: `Concept-12 PEPS topk=5 tau=0.07`
- AUC-first reference: `Concept-12 PEPS topk=5 tau=0.1`

## Final Main Table

| method | test_auc | test_acc | test_f1 | val_auc | balanced_acc | sensitivity | specificity | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BiomedCLIP static prompt baseline | 0.967166 ± 0.010387 | 0.922531 ± 0.029854 | 0.916034 ± 0.029814 | 0.969061 ± 0.008921 | 0.923770 ± 0.018227 | 0.927499 ± 0.026045 | 0.920042 ± 0.056348 | 0.934883 ± 0.032131 |
| Concept-12 embedding_mean | 0.965884 ± 0.008652 | 0.933876 ± 0.016464 | 0.926832 ± 0.016916 | 0.969937 ± 0.011056 | 0.927344 ± 0.011114 | 0.906422 ± 0.029051 | 0.948265 ± 0.034754 | 0.941293 ± 0.027842 |
| Concept-12 MLP dynamic_gate | 0.965461 ± 0.013056 | 0.928722 ± 0.019128 | 0.921510 ± 0.021016 | 0.970106 ± 0.012834 | 0.925592 ± 0.021714 | 0.915468 ± 0.041170 | 0.935716 ± 0.024804 | 0.927728 ± 0.033763 |
| Concept-12 PEPS topk=5 tau=0.07 | 0.968327 ± 0.006489 | 0.939047 ± 0.007595 | 0.932179 ± 0.007740 | 0.972999 ± 0.007620 | 0.931275 ± 0.005445 | 0.906422 ± 0.029051 | 0.956127 ± 0.024452 | 0.945866 ± 0.014115 |
| Concept-12 PEPS topk=5 tau=0.1 | 0.969500 ± 0.008614 | 0.935965 ± 0.016121 | 0.929148 ± 0.016764 | 0.971365 ± 0.006076 | 0.929636 ± 0.010304 | 0.909408 ± 0.013208 | 0.949865 ± 0.029811 | 0.941762 ± 0.020707 |

## Statistical Significance

| candidate_method | reference_method | metric | mean_delta | paired_ttest_pvalue | wilcoxon_pvalue | bootstrap_ci_low | bootstrap_ci_high | wins | losses | ties | fold_outcomes | fold_deltas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | test_auc | 0.001161 | 0.793733 | 1.000000 | -0.005821 | 0.008172 | 2 | 3 | 0 | L,L,L,W,W | -0.000888,-0.001302,-0.011752,0.007039,0.012706 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | test_acc | 0.016516 | 0.272044 | 0.250000 | -0.001020 | 0.042279 | 3 | 1 | 1 | L,W,T,W,W | -0.005155,0.067010,0.000000,0.010363,0.010363 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | test_f1 | 0.016145 | 0.283683 | 0.312500 | -0.000920 | 0.042001 | 3 | 2 | 0 | L,W,L,W,W | -0.004430,0.067104,-0.001193,0.009359,0.009886 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | val_auc | 0.003939 | 0.020470 | 0.062500 | 0.002131 | 0.005785 | 5 | 0 | 0 | W,W,W,W,W | 0.006096,0.002938,0.006800,0.001432,0.002427 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | balanced_acc | 0.007504 | 0.501784 | 0.812500 | -0.006108 | 0.028345 | 3 | 2 | 0 | W,W,L,L,W | 0.003433,0.047112,-0.010577,-0.003042,0.000597 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | sensitivity | -0.021076 | 0.206477 | 0.187500 | -0.042153 | 0.006196 | 1 | 4 | 0 | W,L,L,L,L | 0.030303,-0.015152,-0.044776,-0.045455,-0.030303 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | specificity | 0.036085 | 0.165990 | 0.125000 | -0.001464 | 0.078223 | 4 | 1 | 0 | L,W,W,W,W | -0.023438,0.109375,0.023622,0.039370,0.031496 |
| Concept-12 PEPS topk=5 tau=0.07 | BiomedCLIP static prompt baseline | pr_auc | 0.010983 | 0.365458 | 0.625000 | -0.005246 | 0.032286 | 3 | 2 | 0 | L,W,L,W,W | -0.006936,0.000126,-0.006241,0.050199,0.017770 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | test_auc | 0.002443 | 0.491845 | 0.625000 | -0.002890 | 0.008381 | 3 | 2 | 0 | L,L,W,W,W | -0.003551,-0.004143,0.000940,0.005727,0.013243 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | test_acc | 0.005171 | 0.526986 | 0.437500 | -0.009268 | 0.017531 | 4 | 1 | 0 | L,W,W,W,W | -0.020619,0.025773,0.005155,0.010363,0.005181 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | test_f1 | 0.005347 | 0.533459 | 0.437500 | -0.009872 | 0.018030 | 4 | 1 | 0 | L,W,W,W,W | -0.021979,0.026518,0.005619,0.011599,0.004978 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | val_auc | 0.003063 | 0.254244 | 0.437500 | -0.000959 | 0.007085 | 3 | 2 | 0 | W,L,W,W,L | 0.008168,-0.002938,0.001909,0.008590,-0.000414 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | balanced_acc | 0.003931 | 0.541790 | 0.437500 | -0.007013 | 0.014081 | 4 | 1 | 0 | L,W,W,W,W | -0.015625,0.019531,0.003937,0.011513,0.000298 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | sensitivity | 0.000000 | 1.000000 | 1.000000 | -0.009091 | 0.009091 | 1 | 1 | 3 | T,T,T,W,L | 0.000000,0.000000,0.000000,0.015152,-0.015152 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | specificity | 0.007862 | 0.525794 | 0.375000 | -0.014026 | 0.026587 | 4 | 1 | 0 | L,W,W,W,W | -0.031250,0.039062,0.007874,0.007874,0.015748 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 embedding_mean | pr_auc | 0.004573 | 0.692027 | 0.812500 | -0.008984 | 0.026216 | 2 | 3 | 0 | L,L,W,L,W | -0.011004,-0.009187,0.000875,-0.004534,0.046716 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | test_auc | 0.002866 | 0.508153 | 0.437500 | -0.003731 | 0.010840 | 4 | 1 | 0 | L,W,W,W,W | -0.007221,0.001420,0.002233,0.017120,0.000775 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | test_acc | 0.010325 | 0.266118 | 0.437500 | -0.003103 | 0.024764 | 3 | 2 | 0 | L,W,W,W,L | -0.005155,0.036082,0.005155,0.020725,-0.005181 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | test_f1 | 0.010670 | 0.311506 | 0.625000 | -0.004836 | 0.027270 | 3 | 2 | 0 | L,W,W,W,L | -0.007196,0.040345,0.005167,0.022511,-0.007478 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | val_auc | 0.002894 | 0.477710 | 0.437500 | -0.004408 | 0.009515 | 4 | 1 | 0 | W,L,W,W,W | 0.003314,-0.009754,0.013481,0.003818,0.003610 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | balanced_acc | 0.005683 | 0.666254 | 0.625000 | -0.014748 | 0.027594 | 3 | 2 | 0 | L,W,W,W,L | -0.018584,0.045691,0.000411,0.019387,-0.018492 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | sensitivity | -0.009046 | 0.741865 | 1.000000 | -0.051470 | 0.036364 | 2 | 3 | 0 | L,W,L,W,L | -0.060606,0.075758,-0.014925,0.015152,-0.060606 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | specificity | 0.020411 | 0.000452 | 0.062500 | 0.017237 | 0.023585 | 5 | 0 | 0 | W,W,W,W,W | 0.023438,0.015625,0.015748,0.023622,0.023622 |
| Concept-12 PEPS topk=5 tau=0.07 | Concept-12 MLP dynamic_gate | pr_auc | 0.018139 | 0.178200 | 0.187500 | -0.001644 | 0.036973 | 4 | 1 | 0 | L,W,W,W,W | -0.016662,0.021192,0.004326,0.044398,0.037438 |

## Scale Ablation

| method | test_auc | test_acc | test_f1 | val_auc | balanced_acc | sensitivity | specificity | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PEPS low-only | 0.931815 ± 0.021260 | 0.863656 ± 0.019080 | 0.849217 ± 0.016092 | 0.941425 ± 0.021333 | 0.850868 ± 0.006672 | 0.809905 ± 0.055197 | 0.891831 ± 0.056141 | 0.876428 ± 0.029797 |
| PEPS high-only | 0.963794 ± 0.013557 | 0.928695 ± 0.021813 | 0.920109 ± 0.024265 | 0.973556 ± 0.010373 | 0.916882 ± 0.024971 | 0.879285 ± 0.042336 | 0.954478 ± 0.022952 | 0.937085 ± 0.028102 |
| PEPS dual-scale | 0.968327 ± 0.006489 | 0.939047 ± 0.007595 | 0.932179 ± 0.007740 | 0.972999 ± 0.007620 | 0.931275 ± 0.005445 | 0.906422 ± 0.029051 | 0.956127 ± 0.024452 | 0.945866 ± 0.014115 |

## Interpretability Summary

### Diversity Overview

| section | experiment | low_top1_unique_prompts_mean | high_top1_unique_prompts_mean | low_prompt_entropy_mean | high_prompt_entropy_mean | low_prompt_collapse_rate_mean | high_prompt_collapse_rate_mean | fold | low_top1_unique_prompts | high_top1_unique_prompts | low_prompt_entropy | high_prompt_entropy | low_prompt_collapse_rate | high_prompt_collapse_rate | true_class_name | rank | prompt_text | prompt_concept | count | frequency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diversity_overall | Concept-12 PEPS topk=5 tau=0.07 | 9.800000 | 6.600000 | 1.455861 | 1.185941 | 0.521559 | 0.519732 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| diversity_overall | Concept-12 MLP dynamic_gate | 2.600000 | 3.000000 | 0.750049 | 0.856722 | 0.643609 | 0.559073 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

### Top-10 Prompts for the Main Model

| section | experiment | low_top1_unique_prompts_mean | high_top1_unique_prompts_mean | low_prompt_entropy_mean | high_prompt_entropy_mean | low_prompt_collapse_rate_mean | high_prompt_collapse_rate_mean | fold | low_top1_unique_prompts | high_top1_unique_prompts | low_prompt_entropy | high_prompt_entropy | low_prompt_collapse_rate | high_prompt_collapse_rate | true_class_name | rank | prompt_text | prompt_concept | count | frequency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 1.000000 | A high magnification histopathology image showing gland-forming adenocarcinoma cells with vesicular chromatin and prominent nucleoli. | prominent nucleoli in gland-forming tumor cells | 493.000000 | 0.386970 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 2.000000 | A low magnification histopathology image showing multifocal irregular glandular proliferation consistent with lung adenocarcinoma. | multifocal irregular glandular proliferation | 419.000000 | 0.328885 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 3.000000 | A low magnification histopathology image showing infiltrative gland-forming lung adenocarcinoma replacing alveolated lung parenchyma. | infiltrative glandular adenocarcinoma | 93.000000 | 0.072998 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 4.000000 | A low magnification histopathology image showing lung adenocarcinoma with lepidic growth pattern. | lepidic growth pattern | 81.000000 | 0.063579 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 5.000000 | A high magnification histopathology image showing atypical gland-forming tumor cells with pseudostratified nuclei. | atypical gland-forming tumor cells | 53.000000 | 0.041601 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 6.000000 | A high magnification histopathology image showing adenocarcinoma cells forming glandular lumina. | glandular lumen formation | 27.000000 | 0.021193 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 7.000000 | A low magnification histopathology image showing lung adenocarcinoma with cribriform glandular architecture. | cribriform glandular pattern | 25.000000 | 0.019623 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 8.000000 | A high magnification histopathology image showing solid sheets of adenocarcinoma cells with nuclear atypia. | solid sheets with nuclear atypia | 23.000000 | 0.018053 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 9.000000 | A high magnification histopathology image showing adenocarcinoma cells with intracytoplasmic mucin vacuoles. | intracytoplasmic mucin vacuoles | 21.000000 | 0.016484 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Adenocarcinoma | 10.000000 | A high magnification histopathology image showing lepidic tumor cells growing along alveolar septa. | lepidic tumor cells along alveolar septa | 20.000000 | 0.015699 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 1.000000 | A high magnification histopathology image showing inflammatory cells, granulomas, or fibrosis without adenocarcinoma morphology. | inflammatory cells and granulomas | 233.000000 | 0.351964 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 2.000000 | A low magnification histopathology image showing solid non-gland-forming tumor nests without adenocarcinoma-type acinar structures. | solid non-gland-forming tumor nests | 83.000000 | 0.125378 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 3.000000 | A low magnification histopathology image showing metastatic tumor in the lung without primary lung adenocarcinoma morphology. | metastatic tumor in lung | 76.000000 | 0.114804 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 4.000000 | A low magnification histopathology image showing inflammatory or granulomatous lung lesion without malignant glandular structures. | inflammatory or granulomatous lesion | 58.000000 | 0.087613 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 5.000000 | A high magnification histopathology image showing squamous cell carcinoma with keratinization and keratin pearls. | keratinization | 57.000000 | 0.086103 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 6.000000 | A low magnification histopathology image showing organoid nests and trabeculae consistent with neuroendocrine morphology. | organoid neuroendocrine nesting pattern | 40.000000 | 0.060423 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 7.000000 | A low magnification histopathology image showing a fibroinflammatory lung lesion with scarring and no malignant gland formation. | fibroinflammatory scar-like lesion | 35.000000 | 0.052870 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 8.000000 | A low magnification histopathology image showing benign reactive lung tissue changes without adenocarcinoma morphology. | benign reactive lung tissue | 22.000000 | 0.033233 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 9.000000 | A high magnification histopathology image showing fibrosis and chronic inflammatory infiltrates without adenocarcinoma cells. | fibrosis and chronic inflammation | 19.000000 | 0.028701 |
| top_prompts | Concept-12 PEPS topk=5 tau=0.07 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NonAdenocarcinoma | 10.000000 | A high magnification histopathology image showing atypical spindle tumor cells without gland formation or mucin production. | spindle cell atypia without gland formation | 12.000000 | 0.018127 |