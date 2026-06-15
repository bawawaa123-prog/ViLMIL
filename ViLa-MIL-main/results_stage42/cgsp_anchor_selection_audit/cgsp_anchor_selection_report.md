# Step42 CGSP Anchor Selection Audit

## Step42 目的
- 验证 Concept-Guided Spatial Proposal 是否能从 low patch 层面选出合理 spatial anchors。
- 检查 selected anchors 在不同 bbox_expand 下能否稳定匹配 high child patches，为 Step43 HCRC-Light 做准备。
- 本步骤不训练模型，不修改模型前向逻辑。

## Step41 推荐设置读取情况
- loaded_from_stage41: `True`
- coord_mode: `top_left`
- scale_ratio: `1.0`

## 输入路径
- low directory: `/xiangmu/data/VILMIL/features_biomedclip_5x`
- high directory: `/xiangmu/data/VILMIL/features_biomedclip_20x`
- slide source: `split` from `/home/ljh/ViLMIL/ViLa-MIL-main/splits/adenocarcinoma/task_adenocarcinoma_strictcv_100/splits_0.csv`
- concept/checkpoint source: `checkpoint`
- checkpoint_path: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt`
- concept_prompt_path: `/home/ljh/ViLMIL/ViLa-MIL-main/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`

## 处理概况
- requested slides: `194`
- processed slides: `194`
- failed slides: `0`
- warning count: `0`

## patch score 分布
| metric | mean | median | p90 | p95 |
| --- | --- | --- | --- | --- |
| patch_score | 0.4128 | 0.4158 | 0.4547 | 0.4666 |
| concept_relevance | 0.3980 | 0.4023 | 0.4382 | 0.4474 |
| class_margin | 0.0296 | 0.0260 | 0.0589 | 0.0686 |
| top_prompt_score | 0.4134 | 0.4177 | 0.4545 | 0.4644 |

## proposal 生成数量统计
| stat | candidate_proposals |
| --- | --- |
| count | 194.0 |
| mean | 64.47938144329896 |
| std | 3.4539288478298853 |
| min | 40.0 |
| 25% | 64.0 |
| 50% | 64.0 |
| 75% | 65.0 |
| max | 76.0 |

## selected anchors 数量统计
| stat | selected_anchors |
| --- | --- |
| count | 194.0 |
| mean | 16.0 |
| std | 0.0 |
| min | 16.0 |
| 25% | 16.0 |
| 50% | 16.0 |
| 75% | 16.0 |
| max | 16.0 |

## anchor 空间分散性统计
| metric | mean | median | min | max |
| --- | --- | --- | --- | --- |
| selected_anchor_count | 16.0000 | 16.0000 | 16.0000 | 16.0000 |
| anchor_pair_distance_mean | 45370.3035 | 45716.9484 | 997.0562 | 165219.6466 |
| coverage_ratio_x | 0.8482 | 0.8670 | 0.1928 | 1.0000 |
| coverage_ratio_y | 0.7540 | 0.7862 | 0.2635 | 1.0000 |

## anchor top_class 分布
| top_class | count |
| --- | --- |
| NonAdenocarcinoma | 1592 |
| Adenocarcinoma | 1512 |

## anchor top_concept 分布
| top_concept | count |
| --- | --- |
| benign reactive lung tissue | 992 |
| alveolar replacement by atypical glands | 386 |
| infiltrative glandular adenocarcinoma | 339 |
| papillary architecture | 262 |
| lepidic growth pattern | 255 |
| metastatic tumor in lung | 238 |
| inflammatory or granulomatous lesion | 143 |
| solid growth pattern | 132 |
| diffuse small blue cell growth pattern | 67 |
| fibroinflammatory scar-like lesion | 54 |
| acinar growth pattern | 48 |
| mixed acinar papillary architecture | 41 |

## 不同 bbox_expand 下 high child count 分布
| bbox_expand | empty_anchor_ratio | median_raw_child_count | median_used_child_count | mean_raw_child_count | mean_used_child_count |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 0.2242 | 1.0000 | 1.0000 | 0.7758 | 0.7758 |
| 1.5 | 0.2236 | 1.0000 | 1.0000 | 0.7764 | 0.7764 |
| 2.0 | 0.2233 | 1.0000 | 1.0000 | 0.7767 | 0.7767 |
| 3.0 | 0.2226 | 1.0000 | 1.0000 | 0.7774 | 0.7774 |

## 推荐给 HCRC-Light 的设置
- bbox_expand: `2.0`
- num_anchors: `16`
- num_high_children: `16`
- decision: 暂不建议直接进入 Step43，建议先调参。
- decision reason: median_anchor_count=16.00, best_empty_anchor_ratio=0.2233, best_median_used_child_count=1.00, diversity_ok=True, spread_ok=True.

## 风险说明
- median used_child_count 偏低：推荐 bbox_expand=2.0 时为 1.00。
