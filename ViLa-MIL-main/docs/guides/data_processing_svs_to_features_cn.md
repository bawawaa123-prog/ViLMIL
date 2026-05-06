# 数据处理手册：从 SVS 到双分支特征向量（CLIP + BiomedCLIP）

本文档只讲数据处理链路：从原始 SVS 到最终 h5 特征文件。
覆盖两条分支（低倍率/高倍率）与两种编码器（CLIP / BiomedCLIP）。

---

## 1. 目标与输出

输入：
- 原始病理切片 `*.svs`
- 样本索引 CSV（至少含 `case_id, slide_id, label`）

输出：
- 低倍率特征目录（每个 slide 一个 `.h5`，含 `features` 和 `coords`）
- 高倍率特征目录（每个 slide 一个 `.h5`，含 `features` 和 `coords`）

建议输出目录示例：
- CLIP 低倍率：`features_clip_5x`
- CLIP 高倍率：`features_clip_20x`
- BiomedCLIP 低倍率：`features_biomedclip_5x`
- BiomedCLIP 高倍率：`features_biomedclip_20x`

---

## 2. 流程总览

1. 生成 patch 坐标（WSI 级 h5 坐标文件）
2. 按坐标裁剪 patch 图像（低倍率分支 + 高倍率分支）
3. 用编码器提取 patch 特征（CLIP 或 BiomedCLIP）
4. 得到训练可直接读取的双分支 h5 特征

---

## 3. 前置准备

### 3.1 数据 CSV

建议使用：
- `dataset_csv/all_data.csv`

格式（示例）：
```csv
case_id,slide_id,label
2460239-B2,2460239-B2,Adenocarcinoma
...
```

### 3.2 目录约定（示例）

- 项目根目录：`/xiangmu/ViLMIL/ViLa-MIL-main`
- SVS 根目录：`/path/to/svs_root`
- 坐标输出根目录（低倍率）：`/xiangmu/ViLMIL/ViLa-MIL-main/patches_coords_5x`
- 坐标输出根目录（高倍率）：`/xiangmu/ViLMIL/ViLa-MIL-main/patches_coords_20x`
- patch 图像输出（低倍率）：`/xiangmu/ViLMIL/ViLa-MIL-main/patches_5x`
- patch 图像输出（高倍率）：`/xiangmu/ViLMIL/ViLa-MIL-main/patches_20x`

---

## 4. 第一步：生成坐标（SVS -> coords h5）

使用脚本：`create_patches_fp.py`

> 该步骤可分别跑两次，得到低倍率分支与高倍率分支坐标。

### 4.1 低倍率分支坐标（示例）

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python create_patches_fp.py \
  --source /path/to/svs_root \
  --slide_name_file /xiangmu/ViLMIL/ViLa-MIL-main/dataset_csv/all_data.csv \
  --preset tcga.csv \
  --save_dir /xiangmu/ViLMIL/ViLa-MIL-main/patches_coords_5x \
  --patch_size 256 \
  --step_size 256 \
  --patch_level 2 \
  --seg --patch
```

### 4.2 高倍率分支坐标（示例）

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python create_patches_fp.py \
  --source /path/to/svs_root \
  --slide_name_file /xiangmu/ViLMIL/ViLa-MIL-main/dataset_csv/all_data.csv \
  --preset tcga.csv \
  --save_dir /xiangmu/ViLMIL/ViLa-MIL-main/patches_coords_20x \
  --patch_size 256 \
  --step_size 256 \
  --patch_level 1 \
  --seg --patch
```

输出（示例）：
- `patches_coords_5x/patches_1024/<slide_id>.h5`
- `patches_coords_20x/patches_256/<slide_id>.h5`

---

## 5. 第二步：裁剪 patch 图像（coords -> png）

推荐使用可参数化脚本：`patch_generation_5x.py`（可用于低/高分支）

### 5.1 裁剪低倍率 patch

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python patch_generation_5x.py \
  --source /path/to/svs_root \
  --csv /xiangmu/ViLMIL/ViLa-MIL-main/dataset_csv/all_data.csv \
  --coords-root /xiangmu/ViLMIL/ViLa-MIL-main/patches_coords_5x \
  --patch-size 256 \
  --patch-level 2 \
  --output-root /xiangmu/ViLMIL/ViLa-MIL-main/patches_5x \
  --workers 8 \
  --skip-existing
```

### 5.2 裁剪高倍率 patch

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python patch_generation_5x.py \
  --source /path/to/svs_root \
  --csv /xiangmu/ViLMIL/ViLa-MIL-main/dataset_csv/all_data.csv \
  --coords-root /xiangmu/ViLMIL/ViLa-MIL-main/patches_coords_20x \
  --patch-size 256 \
  --patch-level 1 \
  --output-root /xiangmu/ViLMIL/ViLa-MIL-main/patches_20x \
  --workers 8 \
  --skip-existing
```

输出（示例）：
- `patches_5x/<slide_id>/*.png`
- `patches_20x/<slide_id>/*.png`

---

## 6. 第三步：提取特征（patch -> h5）

下面分别给出 CLIP 和 BiomedCLIP 两套命令。

## 6A. CLIP 分支

使用脚本：`feature_extraction/patch_extraction.py`

### 6A.1 低倍率（CLIP）

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python feature_extraction/patch_extraction.py \
  --patches_path /xiangmu/ViLMIL/ViLa-MIL-main/patches_5x \
  --library_path /xiangmu/ViLMIL/ViLa-MIL-main/features_clip_5x \
  --model_name clip_RN50 \
  --batch_size 64 \
  --dataset adenocarcinoma
```

### 6A.2 高倍率（CLIP）

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python feature_extraction/patch_extraction.py \
  --patches_path /xiangmu/ViLMIL/ViLa-MIL-main/patches_20x \
  --library_path /xiangmu/ViLMIL/ViLa-MIL-main/features_clip_20x \
  --model_name clip_RN50 \
  --batch_size 64 \
  --dataset adenocarcinoma
```

输出：
- `features_clip_5x/<slide_id>.h5`
- `features_clip_20x/<slide_id>.h5`

> 若切换到 CLIP ViT，可把 `--model_name` 改为 `clip_ViTB32`。

## 6B. BiomedCLIP 分支

使用脚本：`feature_extraction/patch_extraction_biomedclip.py`

### 6B.1 低倍率（BiomedCLIP）

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path /xiangmu/ViLMIL/ViLa-MIL-main/patches_5x \
  --library_path /xiangmu/ViLMIL/ViLa-MIL-main/features_biomedclip_5x \
  --batch_size 32 \
  --dataset adenocarcinoma \
  --model_path hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

### 6B.2 高倍率（BiomedCLIP）

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path /xiangmu/ViLMIL/ViLa-MIL-main/patches_20x \
  --library_path /xiangmu/ViLMIL/ViLa-MIL-main/features_biomedclip_20x \
  --batch_size 32 \
  --dataset adenocarcinoma \
  --model_path hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

输出：
- `features_biomedclip_5x/<slide_id>.h5`
- `features_biomedclip_20x/<slide_id>.h5`

---

## 7. 输入输出与参数解释（数据处理相关）

### 7.1 `create_patches_fp.py`
- `--source`: 原始SVS根目录
- `--slide_name_file`: 样本CSV路径
- `--save_dir`: 坐标输出根目录
- `--patch_size`: patch边长
- `--step_size`: 滑窗步长
- `--patch_level`: openslide层级
- `--seg`: 是否做组织分割
- `--patch`: 是否导出patch坐标

输入：SVS + CSV
输出：每个slide一个坐标h5

### 7.2 `patch_generation_5x.py`
- `--source`: SVS根目录
- `--csv`: 样本CSV
- `--coords-root`: 坐标根目录（含 `patches_<size>`）
- `--patch-size`: 与坐标匹配的patch_size
- `--patch-level`: 读取层级
- `--output-root`: png输出目录
- `--workers`: 并行进程数
- `--skip-existing`: 跳过已完成slide

输入：coords h5 + SVS
输出：patch png

### 7.3 `patch_extraction.py`（CLIP）
- `--patches_path`: patch目录
- `--library_path`: 特征输出目录
- `--model_name`: `clip_RN50` / `clip_ViTB32` / 其他
- `--batch_size`: 批大小

输入：patch png
输出：h5特征（features+coords）

### 7.4 `patch_extraction_biomedclip.py`（BiomedCLIP）
- `--patches_path`: patch目录
- `--library_path`: 特征输出目录
- `--batch_size`: 批大小
- `--model_path`: BiomedCLIP模型路径

输入：patch png
输出：h5特征（features+coords）

---

## 8. 结果自检（建议）

可随机检查一个 h5 文件：

```bash
python - << 'PY'
import h5py
p = '/xiangmu/ViLMIL/ViLa-MIL-main/features_biomedclip_5x/2460239-B2.h5'
with h5py.File(p, 'r') as f:
    print('features:', f['features'].shape)
    print('coords:', f['coords'].shape)
PY
```

期望：
- `features` 第二维 = 512（BiomedCLIP）
- `coords` 第二维 = 2

---

## 9. 推荐实践

1. 先固定一套 patch 参数跑小样本，确认全链路无错再批量。
2. 低/高分支目录命名清晰，避免训练时混用。
3. CLIP 与 BiomedCLIP 特征不要混目录。
4. 大规模提特征建议先 `--skip-existing` 断点续跑。

---

## 10. 与训练衔接

训练时只需要把 `data_root_dir + data_folder_s + data_folder_l` 指向上述特征目录即可。

例如：
- `--data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main`
- `--data_folder_s features_biomedclip_5x`
- `--data_folder_l features_biomedclip_20x`

这样就完成了“SVS -> 双分支特征向量 -> 训练”的完整数据处理闭环。
