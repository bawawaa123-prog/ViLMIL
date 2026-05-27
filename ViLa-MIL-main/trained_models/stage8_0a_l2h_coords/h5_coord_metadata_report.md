# H5 Coordinate Metadata Report

用于最小化确认 5x 与 20x h5 中的 `coords` 是否更像处于同一 WSI 原始坐标系。

# Slide 2460239-B2

## 5x

- path: `/xiangmu/data/VILMIL/features_biomedclip_5x/2460239-B2.h5`
- h5 keys: `['coords', 'features']`
- features.shape: `(187, 512)`
- features.dtype: `float32`
- coords.shape: `(187, 2)`
- coords.dtype: `int64`

### Datasets

| dataset | shape | dtype | attrs |
| --- | --- | --- | --- |
| coords | (187, 2) | int64 | {} |
| features | (187, 512) | float32 | {} |

### coords Head 10

```text
[101700.0, 43681.0]
[102196.0, 6944.0]
[105284.0, 56983.0]
[105284.0, 61079.0]
[105796.0, 31393.0]
[105796.0, 35489.0]
[105796.0, 39585.0]
[105796.0, 47777.0]
[106292.0, 6944.0]
[106340.0, 64168.0]
```

### coords Range

| key | value |
| --- | --- |
| x_min | 17328.000000 |
| x_max | 149110.000000 |
| y_min | 2848.000000 |
| y_max | 78472.000000 |
| range_x | 131782.000000 |
| range_y | 75624.000000 |

### coords Step Stats

| key | value |
| --- | --- |
| x_step_median | 1664.000000 |
| x_step_p10 | 272.000000 |
| x_step_p90 | 4096.000000 |
| y_step_median | 1216.000000 |
| y_step_p10 | 441.000000 |
| y_step_p90 | 2880.000000 |

### h5 File Attrs

_No attributes._

### coords Dataset Attrs

_No attributes._

### features Dataset Attrs

_No attributes._

## 20x

- path: `/xiangmu/data/VILMIL/features_biomedclip_20x/2460239-B2.h5`
- h5 keys: `['coords', 'features']`
- features.shape: `(2592, 512)`
- features.dtype: `float32`
- coords.shape: `(2592, 2)`
- coords.dtype: `int64`

### Datasets

| dataset | shape | dtype | attrs |
| --- | --- | --- | --- |
| coords | (2592, 2) | int64 | {} |
| features | (2592, 512) | float32 | {} |

### coords Head 10

```text
[35760.0, 47969.0]
[118084.0, 43681.0]
[130372.0, 37537.0]
[141636.0, 43681.0]
[41904.0, 42849.0]
[56240.0, 54113.0]
[39856.0, 50017.0]
[109892.0, 40609.0]
[20400.0, 19297.0]
[136516.0, 31393.0]
```

### coords Range

| key | value |
| --- | --- |
| x_min | 17328.000000 |
| x_max | 150134.000000 |
| y_min | 2848.000000 |
| y_max | 79496.000000 |
| range_x | 132806.000000 |
| range_y | 76648.000000 |

### coords Step Stats

| key | value |
| --- | --- |
| x_step_median | 384.000000 |
| x_step_p10 | 33.400000 |
| x_step_p90 | 1024.000000 |
| y_step_median | 368.000000 |
| y_step_p10 | 48.400000 |
| y_step_p90 | 832.000000 |

### h5 File Attrs

_No attributes._

### coords Dataset Attrs

_No attributes._

### features Dataset Attrs

_No attributes._

## 5x vs 20x Comparison

- range_ratio_x: `1.007770`
- range_ratio_y: `1.013541`
- 坐标范围是否接近: `接近，倾向同一原始坐标系`
- 是否建议 l2h_coord_scale=1.0: `建议优先尝试 l2h_coord_scale=1.0`

# Slide 2460242-B2

## 5x

- path: `/xiangmu/data/VILMIL/features_biomedclip_5x/2460242-B2.h5`
- h5 keys: `['coords', 'features']`
- features.shape: `(426, 512)`
- features.dtype: `float32`
- coords.shape: `(426, 2)`
- coords.dtype: `int64`

### Datasets

| dataset | shape | dtype | attrs |
| --- | --- | --- | --- |
| coords | (426, 2) | int64 | {} |
| features | (426, 512) | float32 | {} |

### coords Head 10

```text
[137719.0, 46992.0]
[55265.0, 56560.0]
[145911.0, 79760.0]
[145911.0, 26512.0]
[51169.0, 72944.0]
[26593.0, 31984.0]
[30689.0, 27888.0]
[150007.0, 63376.0]
[109047.0, 22416.0]
[67553.0, 72944.0]
```

### coords Range

| key | value |
| --- | --- |
| x_min | 18401.000000 |
| x_max | 173341.000000 |
| y_min | 4240.000000 |
| y_max | 83856.000000 |
| range_x | 154940.000000 |
| range_y | 79616.000000 |

### coords Step Stats

| key | value |
| --- | --- |
| x_step_median | 4096.000000 |
| x_step_p10 | 860.000000 |
| x_step_p90 | 4096.000000 |
| y_step_median | 1376.000000 |
| y_step_p10 | 288.000000 |
| y_step_p90 | 2720.000000 |

### h5 File Attrs

_No attributes._

### coords Dataset Attrs

_No attributes._

### features Dataset Attrs

_No attributes._

## 20x

- path: `/xiangmu/data/VILMIL/features_biomedclip_20x/2460242-B2.h5`
- h5 keys: `['coords', 'features']`
- features.shape: `(6128, 512)`
- features.dtype: `float32`
- coords.shape: `(6128, 2)`
- coords.dtype: `int64`

### Datasets

| dataset | shape | dtype | attrs |
| --- | --- | --- | --- |
| coords | (6128, 2) | int64 | {} |
| features | (6128, 512) | float32 | {} |

### coords Head 10

```text
[145911.0, 21392.0]
[120311.0, 48016.0]
[74721.0, 51440.0]
[67553.0, 40176.0]
[35809.0, 40176.0]
[144887.0, 30608.0]
[132599.0, 51088.0]
[118263.0, 38800.0]
[64481.0, 82160.0]
[114167.0, 46992.0]
```

### coords Range

| key | value |
| --- | --- |
| x_min | 18401.000000 |
| x_max | 176413.000000 |
| y_min | 1568.000000 |
| y_max | 84208.000000 |
| range_x | 158012.000000 |
| range_y | 82640.000000 |

### coords Step Stats

| key | value |
| --- | --- |
| x_step_median | 1024.000000 |
| x_step_p10 | 67.000000 |
| x_step_p90 | 1024.000000 |
| y_step_median | 352.000000 |
| y_step_p10 | 96.000000 |
| y_step_p90 | 672.000000 |

### h5 File Attrs

_No attributes._

### coords Dataset Attrs

_No attributes._

### features Dataset Attrs

_No attributes._

## 5x vs 20x Comparison

- range_ratio_x: `1.019827`
- range_ratio_y: `1.037982`
- 坐标范围是否接近: `接近，倾向同一原始坐标系`
- 是否建议 l2h_coord_scale=1.0: `建议优先尝试 l2h_coord_scale=1.0`

# Slide 25034929B2

## 5x

- path: `/xiangmu/data/VILMIL/features_biomedclip_5x/25034929B2.h5`
- h5 keys: `['coords', 'features']`
- features.shape: `(274, 512)`
- features.dtype: `float32`
- coords.shape: `(274, 2)`
- coords.dtype: `int64`

### Datasets

| dataset | shape | dtype | attrs |
| --- | --- | --- | --- |
| coords | (274, 2) | int64 | {} |
| features | (274, 512) | float32 | {} |

### coords Head 10

```text
[101574.0, 30817.0]
[101574.0, 34913.0]
[101574.0, 39009.0]
[101574.0, 43105.0]
[101574.0, 47201.0]
[101574.0, 55393.0]
[101574.0, 59489.0]
[101574.0, 67681.0]
[105670.0, 34913.0]
[105670.0, 39009.0]
```

### coords Range

| key | value |
| --- | --- |
| x_min | 8816.000000 |
| x_max | 126150.000000 |
| y_min | 5440.000000 |
| y_max | 76401.000000 |
| range_x | 117334.000000 |
| range_y | 70961.000000 |

### coords Step Stats

| key | value |
| --- | --- |
| x_step_median | 4096.000000 |
| x_step_p10 | 1185.200000 |
| x_step_p90 | 4096.000000 |
| y_step_median | 1281.000000 |
| y_step_p10 | 273.000000 |
| y_step_p90 | 3568.000000 |

### h5 File Attrs

_No attributes._

### coords Dataset Attrs

_No attributes._

### features Dataset Attrs

_No attributes._

## 20x

- path: `/xiangmu/data/VILMIL/features_biomedclip_20x/25034929B2.h5`
- h5 keys: `['coords', 'features']`
- features.shape: `(3840, 512)`
- features.dtype: `float32`
- coords.shape: `(3840, 2)`
- coords.dtype: `int64`

### Datasets

| dataset | shape | dtype | attrs |
| --- | --- | --- | --- |
| coords | (3840, 2) | int64 | {} |
| features | (3840, 512) | float32 | {} |

### coords Head 10

```text
[29296.0, 36465.0]
[110790.0, 57441.0]
[100550.0, 39009.0]
[22128.0, 67185.0]
[47728.0, 68209.0]
[84166.0, 45153.0]
[49776.0, 61041.0]
[114886.0, 46177.0]
[12912.0, 76401.0]
[40560.0, 55921.0]
```

### coords Range

| key | value |
| --- | --- |
| x_min | 8816.000000 |
| x_max | 126202.000000 |
| y_min | 6464.000000 |
| y_max | 79473.000000 |
| range_x | 117386.000000 |
| range_y | 73009.000000 |

### coords Step Stats

| key | value |
| --- | --- |
| x_step_median | 972.000000 |
| x_step_p10 | 52.000000 |
| x_step_p90 | 1024.000000 |
| y_step_median | 494.000000 |
| y_step_p10 | 46.500000 |
| y_step_p90 | 528.000000 |

### h5 File Attrs

_No attributes._

### coords Dataset Attrs

_No attributes._

### features Dataset Attrs

_No attributes._

## 5x vs 20x Comparison

- range_ratio_x: `1.000443`
- range_ratio_y: `1.028861`
- 坐标范围是否接近: `接近，倾向同一原始坐标系`
- 是否建议 l2h_coord_scale=1.0: `建议优先尝试 l2h_coord_scale=1.0`
