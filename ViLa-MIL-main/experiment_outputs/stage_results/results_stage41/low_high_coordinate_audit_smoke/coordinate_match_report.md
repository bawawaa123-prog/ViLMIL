# Step41 Low-High Coordinate Correspondence Audit

## Step41 目的
- 审计 `features_biomedclip_5x` 与 `features_biomedclip_20x` 原始 patch coords 是否支持真实 low-high 空间对应。
- 不训练模型，不修改模型前向逻辑，只为后续 HCRC / Step42 提供坐标可靠性结论。

## 输入路径
- low directory: `/xiangmu/data/VILMIL/features_biomedclip_5x`
- high directory: `/xiangmu/data/VILMIL/features_biomedclip_20x`
- csv_path: `dataset_csv/all_data.csv`
- split_dir: `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- fold: `0`
- split: `test`
- slide source: `split` from `/home/ljh/ViLMIL/ViLa-MIL-main/splits/adenocarcinoma/task_adenocarcinoma_strictcv_100/splits_0.csv`

## 处理概况
- requested slides: `3`
- processed slides: `3`
- missing slides: `0`
- failed slides: `0`
- warning count: `0`

## low/high h5 key 与 attrs 总结
| scale | coord_keys | patch_level_values | patch_size_values |
| --- | --- | --- | --- |
| low | coords (3) | None (3) | None (3) |
| high | coords (3) | None (3) | None (3) |

## low/high 坐标范围统计
- mean low span: `x=122506.3333`, `y=65240.3333`
- mean high span: `x=124849.3333`, `y=68243.0000`
- same-coordinate-system heuristic rate: `1.0000`

## 不同 coord_mode 与 scale_ratio 的匹配结果
| coord_mode | scale_ratio | slides | total_low_patches | weighted_empty_ratio | mean_child_count | median_child_count | median_p95 | same_coord_system_rate | expected_child_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_left | 1.0 | 3 | 619 | 0.2294 | 0.7639 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| top_left | 2.0 | 3 | 619 | 0.9758 | 0.0267 | 0.0000 | 0.0000 | 1.0000 | 4.0000 |
| top_left | 4.0 | 3 | 619 | 0.9919 | 0.0085 | 0.0000 | 0.0000 | 1.0000 | 16.0000 |
| top_left | 0.5 | 3 | 619 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| top_left | 0.25 | 3 | 619 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| center | 1.0 | 3 | 619 | 0.2294 | 0.7639 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| center | 2.0 | 3 | 619 | 0.9774 | 0.0249 | 0.0000 | 0.0000 | 1.0000 | 4.0000 |
| center | 4.0 | 3 | 619 | 0.9903 | 0.0102 | 0.0000 | 0.0000 | 1.0000 | 16.0000 |
| center | 0.5 | 3 | 619 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| center | 0.25 | 3 | 619 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## empty match ratio 总结
| coord_mode | scale_ratio | high_empty_slides | mean_slide_empty_ratio | max_slide_empty_ratio |
| --- | --- | --- | --- | --- |
| top_left | 1.0 | 0 | 0.2361 | 0.3155 |
| top_left | 2.0 | 3 | 0.9733 | 1.0000 |
| top_left | 4.0 | 3 | 0.9915 | 0.9959 |
| top_left | 0.5 | 3 | 1.0000 | 1.0000 |
| top_left | 0.25 | 3 | 1.0000 | 1.0000 |
| center | 1.0 | 0 | 0.2361 | 0.3155 |
| center | 2.0 | 3 | 0.9751 | 1.0000 |
| center | 4.0 | 3 | 0.9898 | 0.9959 |
| center | 0.5 | 3 | 1.0000 | 1.0000 |
| center | 0.25 | 3 | 1.0000 | 1.0000 |

## child count 分布总结
| slide_id | coord_mode | scale_ratio | child_count_mean | child_count_median | child_count_p90 | child_count_p95 | empty_match_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2460239-B2 | center | 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2460628-B2 | center | 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2460641-B2 | center | 0.25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2460239-B2 | center | 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2460628-B2 | center | 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2460641-B2 | center | 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 2460239-B2 | center | 1.0 | 0.6845 | 1.0000 | 1.0000 | 1.0000 | 0.3155 |
| 2460628-B2 | center | 1.0 | 0.7684 | 1.0000 | 1.0000 | 1.0000 | 0.2316 |
| 2460641-B2 | center | 1.0 | 0.8388 | 1.0000 | 1.0000 | 1.0000 | 0.1612 |
| 2460239-B2 | center | 2.0 | 0.0695 | 0.0000 | 0.0000 | 1.0000 | 0.9305 |
| 2460628-B2 | center | 2.0 | 0.0053 | 0.0000 | 0.0000 | 0.0000 | 0.9947 |
| 2460641-B2 | center | 2.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## 推荐结论
- Recommended coord_mode: `top_left`; recommended scale_ratio: `1.0`.
- 可以进入 Step42（CGSP Anchor Selection Audit），因为至少存在一组 coord_mode + scale_ratio 在大多数 low patches 上给出非空匹配。

## 风险与建议
- 当前推荐组合的加权 empty match ratio 为 0.2294；仍需在 Step42 使用真实 coords 而不是 attention centroid，并继续检查 anchor 选择是否稳定。 推荐组合的 slide-level child count median 约为 1.0000。

## High Empty-Match Cases
| slide_id | coord_mode | scale_ratio | empty_match_ratio | child_count_mean | child_count_p95 |
| --- | --- | --- | --- | --- | --- |
| 2460239-B2 | top_left | 0.5 | 1.0000 | 0.0000 | 0.0000 |
| 2460239-B2 | top_left | 0.25 | 1.0000 | 0.0000 | 0.0000 |
| 2460239-B2 | center | 0.5 | 1.0000 | 0.0000 | 0.0000 |
| 2460239-B2 | center | 0.25 | 1.0000 | 0.0000 | 0.0000 |
| 2460628-B2 | top_left | 0.5 | 1.0000 | 0.0000 | 0.0000 |
| 2460628-B2 | top_left | 0.25 | 1.0000 | 0.0000 | 0.0000 |
| 2460628-B2 | center | 0.5 | 1.0000 | 0.0000 | 0.0000 |
| 2460628-B2 | center | 0.25 | 1.0000 | 0.0000 | 0.0000 |
| 2460641-B2 | top_left | 2.0 | 1.0000 | 0.0000 | 0.0000 |
| 2460641-B2 | top_left | 0.5 | 1.0000 | 0.0000 | 0.0000 |
| 2460641-B2 | top_left | 0.25 | 1.0000 | 0.0000 | 0.0000 |
| 2460641-B2 | center | 2.0 | 1.0000 | 0.0000 | 0.0000 |
| 2460641-B2 | center | 0.5 | 1.0000 | 0.0000 | 0.0000 |
| 2460641-B2 | center | 0.25 | 1.0000 | 0.0000 | 0.0000 |
| 2460641-B2 | top_left | 4.0 | 0.9959 | 0.0041 | 0.0000 |
| 2460641-B2 | center | 4.0 | 0.9959 | 0.0041 | 0.0000 |
| 2460628-B2 | top_left | 2.0 | 0.9947 | 0.0053 | 0.0000 |
| 2460628-B2 | center | 2.0 | 0.9947 | 0.0053 | 0.0000 |
| 2460628-B2 | top_left | 4.0 | 0.9895 | 0.0105 | 0.0000 |
| 2460239-B2 | top_left | 4.0 | 0.9893 | 0.0107 | 0.0000 |
