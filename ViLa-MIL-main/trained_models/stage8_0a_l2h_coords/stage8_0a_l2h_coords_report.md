# Stage 8.0-A L2H Coordinate Diagnosis

## Scope

- split_dir: `/xiangmu/ViLMIL/ViLa-MIL-main/splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- split_column: `test`
- total slides listed: `968`
- slides with valid low/high coords: `968`
- skipped or missing slides: `0`

## Coordinate Range Summary

| metric | mean | median | p10 | p90 |
| --- | --- | --- | --- | --- |
| range_ratio_x | 1.033846 | 1.024970 | 1.007931 | 1.057587 |
| range_ratio_y | 1.060459 | 1.043692 | 1.014064 | 1.117122 |

结论观察：
- high/low 的中位 range_ratio_x 约为 `1.0250`，最接近的候选比例是 `1.0`。
- high/low 的中位 range_ratio_y 约为 `1.0437`，最接近的候选比例是 `1.0`。

## Nearest Distance Summary By Scale

| scale_ratio | nearest_dist_mean | nearest_dist_median | nearest_dist_p10 | nearest_dist_p90 | coverage_256 | coverage_512 | coverage_1024 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 1807.446515 | 1646.760430 | 1024.000000 | 2843.629666 | 0.056835 | 0.057027 | 0.282411 |
| 2.000000 | 11926.256706 | 9545.867514 | 2903.297412 | 24744.389366 | 0.000994 | 0.004349 | 0.019394 |
| 4.000000 | 36072.842073 | 34454.103914 | 15054.277803 | 59938.231089 | 0.000157 | 0.000609 | 0.002438 |
| 8.000000 | 121760.180949 | 122081.293982 | 89453.486312 | 153766.062427 | 0.000001 | 0.000011 | 0.000045 |

结论观察：
- 最近邻中位距离最小的 scale_ratio 是 `1.0`。
- `coverage_512` 最高的 scale_ratio 是 `1.0`。
- `coverage_1024` 最高的 scale_ratio 是 `1.0`。

## Recommendation

- 初步建议将 `l2h_coord_scale` 优先设为 `1.0`。
- 在 `1.0` 与 `4.0` 的直接对比中，`1.0` 的最近邻/coverage 综合表现更优。
- 仍建议保留对 `4.0` 的人工复核，避免只依据单一统计量下结论。