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
- requested slides: `3`
- processed slides: `3`
- failed slides: `0`
- warning count: `0`

## patch score 分布
| metric | mean | median | p90 | p95 |
| --- | --- | --- | --- | --- |
| patch_score | 0.4003 | 0.3993 | 0.4521 | 0.4622 |
| concept_relevance | 0.3846 | 0.3845 | 0.4302 | 0.4438 |
| class_margin | 0.0314 | 0.0280 | 0.0594 | 0.0712 |
| top_prompt_score | 0.3996 | 0.3995 | 0.4494 | 0.4604 |

## proposal 生成数量统计
| stat | candidate_proposals |
| --- | --- |
| count | 3.0 |
| mean | 65.66666666666667 |
| std | 1.5275252316519468 |
| min | 64.0 |
| 25% | 65.0 |
| 50% | 66.0 |
| 75% | 66.5 |
| max | 67.0 |

## selected anchors 数量统计
| stat | selected_anchors |
| --- | --- |
| count | 3.0 |
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
| anchor_pair_distance_mean | 47803.3782 | 47895.6657 | 2541.0677 | 111424.2178 |
| coverage_ratio_x | 0.8197 | 0.8267 | 0.7026 | 0.9299 |
| coverage_ratio_y | 0.8442 | 0.8671 | 0.7197 | 0.9458 |

## anchor top_class 分布
| top_class | count |
| --- | --- |
| NonAdenocarcinoma | 25 |
| Adenocarcinoma | 23 |

## anchor top_concept 分布
| top_concept | count |
| --- | --- |
| diffuse small blue cell growth pattern | 10 |
| benign reactive lung tissue | 9 |
| alveolar replacement by atypical glands | 7 |
| solid growth pattern | 4 |
| lepidic growth pattern | 4 |
| infiltrative glandular adenocarcinoma | 3 |
| sarcomatoid spindle cell morphology | 3 |
| metastatic tumor in lung | 2 |
| mixed acinar papillary architecture | 2 |
| papillary architecture | 2 |
| acinar growth pattern | 1 |
| inflammatory or granulomatous lesion | 1 |

## 不同 bbox_expand 下 high child count 分布
| bbox_expand | empty_anchor_ratio | median_raw_child_count | median_used_child_count | mean_raw_child_count | mean_used_child_count |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 0.2708 | 1.0000 | 1.0000 | 0.7292 | 0.7292 |
| 1.5 | 0.2708 | 1.0000 | 1.0000 | 0.7292 | 0.7292 |
| 2.0 | 0.2708 | 1.0000 | 1.0000 | 0.7292 | 0.7292 |
| 3.0 | 0.2708 | 1.0000 | 1.0000 | 0.7292 | 0.7292 |

## 推荐给 HCRC-Light 的设置
- bbox_expand: `2.0`
- num_anchors: `16`
- num_high_children: `16`
- decision: 暂不建议直接进入 Step43，建议先调参。
- decision reason: median_anchor_count=16.00, best_empty_anchor_ratio=0.2708, best_median_used_child_count=1.00, diversity_ok=True, spread_ok=True.

## 风险说明
- median used_child_count 偏低：推荐 bbox_expand=2.0 时为 1.00。
