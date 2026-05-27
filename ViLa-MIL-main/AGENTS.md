# AGENTS.md - ViLMIL 项目 Codex 协作指南（中文版）

本文件用于指导 Codex 在 `ViLMIL` 仓库中进行代码修改、实验脚本编写、结果分析和阶段交接。每次开始新会话或执行新 Step 前，都应先阅读本文件。

当前主要项目根目录通常是：

```text
ViLa-MIL-main/
```

当前研究方向正在从原始 ViLa-MIL / BiomedCLIP / PEPS baseline，逐步转向 **RCE-MIL / DEG-MIL**。新的目标是把 WSI 分类从单纯的 slide-level 图文匹配，扩展为 **region-concept evidence aggregation（区域-概念证据聚合）**。

---

## 1. Codex 核心协作规则

1. **小步迭代**。每次只完成用户明确要求的当前 Step，不要提前实现后续 Step。
2. **不要默认查看所有文件**。先根据本文件的目录地图判断任务可能涉及哪些文件，再按需打开相关文件。
3. **保护已有 baseline**。不要破坏以下已有路径：
   - `ViLa_MIL`
   - `ViLa_MIL_BiomedCLIP`
   - PEPS / SAP-PEPS / SAF 实验路径
   - 已有 `trained_models/`、`eval_results/`、历史结果目录
4. 优先**新增模型文件/脚本文件**，不要直接重写已经稳定的模型文件。
5. 保持公共接口兼容，尤其是：
   - dataset 返回格式：`features_s, coords_s, features_l, coords_l, label, slide_id`
   - 模型 forward 签名：`forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None)`
   - 训练返回：`Y_prob, Y_hat, loss`
6. 代码应结构清晰、命名明确、可读性好；涉及 tensor 操作时应写必要的 shape 注释。
7. 新脚本避免硬编码绝对路径。应使用环境变量默认值和基于脚本位置推导的 `ROOT_DIR`。
8. 新 shell 脚本优先使用 **bash 数组**，避免长反斜杠 `\` 续行。
9. 除非用户明确要求，否则只运行快速检查，不直接运行耗时任务。
10. 每次完成大的代码改动 Step 后，都要按照第 3 节要求，向 `docs/CODEX_HANDOFF.md` 追加简洁交接记录。

---

## 2. 长时间任务执行规则

除非用户明确要求 Codex 直接执行，否则不要直接运行耗时任务。

耗时任务包括但不限于：

- 非 smoke 的完整单折训练
- 5-fold 完整训练/评估
- 特征提取
- patch 生成
- WSI 分割 / patch 坐标生成
- 多张 slide 的 heatmap 重新生成
- 遍历所有 folds/results 的大型分析脚本

遇到耗时任务时，Codex 应该：

1. 检查脚本或命令是否写得正确。
2. 只运行快速静态检查，例如：
   - `python -m py_compile <file.py>`
   - `bash -n <script.sh>`
   - `grep -n '\\$' <script.sh> || true`，检查 shell 续行问题
3. 给用户明确的执行命令。
4. 让用户执行后把日志/结果贴回来。
5. 根据用户返回的日志继续分析，不要一直等待长任务完成。

允许在合适情况下运行的短任务：

- `py_compile`
- `bash -n`
- 精确的 `grep` / `find` 检查
- 只有用户明确要求时，才运行极小的 smoke test

---

## 3. 交接文档要求

维护一个简洁的阶段交接文档：

```text
docs/CODEX_HANDOFF.md
```

如果该文件不存在，需要创建。每次完成一个较大的代码改动 Step 后，追加一条新记录。记录要短，但足够让新的 Codex 会话快速接上上下文。

推荐格式：

```markdown
## YYYY-MM-DD - Step <N>: <简短标题>

### Goal
- <本 Step 想完成什么>

### Files changed
- `<path>`: <一句话说明>

### Behavior / tensor flow
- <仅当模型或数据流程变化时填写>

### Checks run
- `<command>`: passed/failed

### Commands not run
- <未运行的长任务及原因，如果有>

### Results / observations
- <指标、日志路径或关键观察>

### Next suggested step
- <一个简洁的下一步建议>
```

不要在该文件中粘贴长日志。长日志应放在 `results_*/*/logs/` 下，并在交接文档中引用路径。

---

## 4. 当前项目状态

### 4.1 稳定 baseline 路线

仓库中保留了原始 ViLa-MIL 实现，以及更强的 BiomedCLIP 适配版本。BiomedCLIP 版本已经支持 concept prompt pool 和 PEPS 风格的 prompt 选择。

此前的主要 baseline 是：

```text
Concept-12 PEPS topk=5 tau=0.07
```

该 baseline 应保持可复现，不要随意改动相关路径。

### 4.2 新创新路线

当前新增了一个模型类型：

```text
RCE_MIL_BiomedCLIP
```

RCE-MIL 表示 **Region-Concept Evidence MIL（区域-概念证据 MIL）**。它是更大 DEG-MIL 设想的最小可行版本。

当前 RCE-MIL 的核心流程是：

```text
patch features
  -> learnable region queries
  -> region features
  -> region-concept cosine evidence
  -> prompt evidence / concept evidence
  -> class logits
```

除非任务明确要求整合，否则 RCE-MIL 应保持和已有 PEPS 代码路径分离。

---

## 5. 项目目录与文件地图

使用本节判断应该查看哪些文件。不要为了一个小任务打开全部文件。

### `ViLa-MIL-main/`

项目主体目录。

#### 根目录文件

- `README.md`  
  原始 ViLa-MIL 使用说明和项目介绍。相对于当前 BiomedCLIP / RCE 新增内容，可能已经部分过时。

- `main.py`  
  主训练入口。定义 CLI 参数、任务配置、split 处理，并调用训练函数。新增 `model_type`、任务、参数或实验模式时查看。

- `eval.py`  
  训练后模型评估入口。新增模型需要支持独立评估时查看。

- `create_patches_fp.py`  
  从 WSI 生成 patch 坐标 / patch 元数据。通常耗时，不要未经用户确认直接执行。

- `patch_generation_5x.py`  
  5x / 低倍 patch 生成脚本。通常耗时。

- `patch_generation_20x.py`  
  20x / 高倍 patch 生成脚本。通常耗时。

- `convert_dataset_format.py`  
  数据集 CSV 格式转换工具。只有数据格式问题时查看。

- `create_splits_seq.py`  
  原始 split 生成工具。

- `create_splits_strict_cv.py`  
  strict cross-validation split 生成工具，腺癌实验常用。

- `regenerate_heatmaps.py`  
  重新生成解释性 heatmap。通常耗时，不要直接执行。

- `transfer_svs_files.py`  
  SVS 文件转移或辅助脚本。除非任务涉及数据迁移，否则不要修改。

---

### `dataset_csv/`

数据集元数据与 concept prompt 文件。

常见文件包括：

- `all_data.csv`  
  当前腺癌任务主数据 CSV。

- `private_lung_concept_prompt_pool_stage2_core12.json`  
  当前 PEPS 和 RCE 实验常用的 Concept-12 prompt pool。

当任务涉及标签、slide ID、类别名、prompt pool 内容时查看该目录。

---

### `datasets/`

数据集类和数据读取逻辑。

- `dataset_generic.py`  
  主 MIL dataset / split loader。返回低尺度/高尺度 features、coords、label、slide_id。模型输入 shape 或 split 行为不清楚时查看。

- `dataset_h5.py`  
  HDF5 feature / patch dataset 辅助工具。

- `wsi_dataset.py`  
  WSI 级数据集工具。

- `BatchWSI.py`  
  批量 WSI 辅助类。

- `__init__.py`  
  包标记文件。

除非任务明确要求修改数据格式，否则避免改动 dataset 代码。

---

### `models/`

模型定义目录。

- `model_ViLa_MIL.py`  
  原始 ViLa-MIL 模型。

- `model_ViLa_MIL_BiomedCLIP.py`  
  BiomedCLIP 适配模型，包含 concept prompt pool、PEPS/SAP-PEPS/scale-fusion 等路径。视为稳定 baseline，除非任务明确涉及该模型，否则不要修改。

- `model_RCE_MIL_BiomedCLIP.py`  
  当前新增 RCE-MIL 模型。使用 region queries 聚合 patch features 为 region features，再与 concept prompt features 匹配生成 class logits。RCE/DEG 相关任务优先查看。

- `model_mil.py`  
  MIL baseline 模型或 pooling baseline。

- `model_utils.py`  
  共享模型工具和辅助层。

- `__init__.py`  
  包标记文件。

新增模型时，优先新建文件并新增 `model_type` 分支，不要直接修改稳定模型。

---

### `utils/`

训练、评估、loss、指标、prompt 辅助工具。

- `core_utils.py`  
  主训练循环、模型初始化、验证、测试、summary 和保存逻辑。新增模型类型或改变训练行为时查看。

- `eval_utils.py`  
  `eval.py` 使用的评估辅助函数。

- `loss_utils.py`  
  CE、focal 或其他 bag-level loss。

- `metric_utils.py`  
  AUC、F1、balanced accuracy、sensitivity、specificity、PR-AUC 等指标。

- `prompt_utils.py`  
  Concept prompt pool 的读取、分组和校验工具。修改 prompt 文件格式或 prompt grouping 时查看。

- `file_utils.py`  
  文件保存/读取、pickle 工具、文件系统辅助。

- `heatmap_utils.py`  
  heatmap / 可解释性辅助函数。

- `utils.py`  
  通用工具函数。

- `__init__.py`  
  包标记文件。

---

### `feature_extraction/`

patch 特征提取脚本和辅助函数。

仅当任务涉及从 patch 图像提取 CLIP / BiomedCLIP 特征时查看。特征提取通常耗时，通常只给用户执行命令，不要直接运行。

---

### `wsi_core/`

WSI 处理核心工具，包括 slide 读取、分割、可视化、patching 等，继承自 CLAM / ViLa-MIL 风格流程。

除非任务明确涉及 WSI 预处理或 patch 生成，否则不要改动。

---

### `splits/`

训练/验证/测试 split CSV 文件。

当前腺癌实验常用：

```text
splits/adenocarcinoma/task_adenocarcinoma_strictcv_100
```

除非任务明确要求重新生成 split，否则不要覆盖 split 文件。

---

### `text_prompt/`

静态双尺度类别 prompt，通常为 CSV 文件。原始 ViLa-MIL / BiomedCLIP prompt 路径会使用。

---

### `scripts/`

实验、分析、数据工具脚本目录。

- `scripts/README_CN.md`  
  中文脚本目录说明。

- `scripts/experiments/`  
  主实验 shell 脚本目录。Stage 脚本放在这里，包括 PEPS / SAP / SAF / RCE 实验。

- `scripts/analysis/`  
  结果聚合和分析脚本。

- `scripts/data_tools/`  
  数据集和文件管理辅助脚本。

- `scripts/run_stage7_1_residual_saf_peps.sh`  
  较早的 residual SAF-PEPS 脚本，除非任务明确要求，否则视为历史脚本。

- `scripts/build_stage7_1_residual_saf_analysis.py`  
  residual SAF 实验分析脚本，通常作为历史参考。

新增实验脚本时，应满足：

- 使用 `#!/usr/bin/env bash`
- 使用 `set -euo pipefail`
- 从脚本位置推导 `ROOT_DIR`
- 路径变量允许环境变量覆盖
- 使用 bash 数组构造命令
- 避免 `\` 续行
- 除非用户明确要求，否则不直接运行耗时任务

---

### `docs/`

项目笔记、分析文档和生成文档。交接记录放在这里。

必须维护的重要文件：

```text
docs/CODEX_HANDOFF.md
```

如果不存在，应创建。

---

### `knowledge_base/`

项目知识、笔记、参考资料或 prompt/实验材料。仅当任务涉及项目知识库或已有研究笔记时查看。

---

### `trained_models/`

历史训练模型 checkpoint、summary 和最终分析输出。

不要修改或删除。该目录是实验历史证据。可用于对比 baseline 和先前实验。

重要目录：

```text
trained_models/final_dcp_vila_analysis/
```

包含最终 PEPS / DCP-ViLa summary tables 和显著性检验结果。

---

### `eval_results/`

训练模型的评估输出。除非任务明确要求生成新的评估结果，否则不要修改。

---

### `results/`、`results_smoke/`、`results_stage9/`

训练/评估输出目录。

- `results/`：通用训练输出。
- `results_smoke/`：smoke test 输出。
- `results_stage9/`：当前 RCE-MIL Stage9 输出。

不要删除已有结果。新实验应使用唯一的 `exp_code`。

---

### `history_code/`

历史/归档代码。仅在明确需要参考旧实现时查看。

---

### `presets/`

patch / segmentation preset 文件。只有 patch 生成或 WSI 预处理任务时查看。

---

### `shared/`

共享项目资源。仅当任务明确引用时查看。

---

### `tools/`

杂项工具。仅当任务涉及工具维护或工具脚本时查看。

---

### `image/ readme/`

原始仓库的图片或 README 资源目录。通常与模型修改无关。

---

## 6. 常见任务路由

用本节判断该改哪些文件。

### 新增模型

可能涉及文件：

- `models/model_<NewModel>.py`
- `main.py`
- `utils/core_utils.py`
- 如果评估也需要显式支持，可能涉及 `eval.py` / `utils/eval_utils.py`

不要改稳定模型文件，除非确实必要。

### 新增实验脚本

通常放在：

- `scripts/experiments/`

同时更新：

- `docs/CODEX_HANDOFF.md`

检查命令：

```bash
bash -n scripts/experiments/<script>.sh
grep -n '\\$' scripts/experiments/<script>.sh || true
```

### 新增分析脚本

通常放在：

- `scripts/analysis/`

优先读取 `trained_models/`、`results_stage9/` 或 `eval_results/` 中已有 CSV summary。不要为了分析重新跑昂贵实验。

### 修改 concept prompt 处理

可能涉及文件：

- `utils/prompt_utils.py`
- `models/model_ViLa_MIL_BiomedCLIP.py`
- `models/model_RCE_MIL_BiomedCLIP.py`
- `dataset_csv/*.json`

注意保持已有 prompt JSON 的兼容性。

### 修改训练逻辑

主要文件：

- `utils/core_utils.py`

务必谨慎。训练逻辑影响所有模型。优先做模型局部修改。

### 修改指标或 summary 输出

可能涉及文件：

- `utils/metric_utils.py`
- `utils/core_utils.py`
- `utils/eval_utils.py`

除非任务要求，否则不要重命名已有 CSV 列。

---

## 7. 当前 RCE-MIL / Stage9 工作流

已知完成步骤：

1. 已新增 `RCE_MIL_BiomedCLIP` 模型。
2. 已新增 `scripts/experiments/run_stage9_rce_smoke.sh`。
3. Smoke test 已成功运行：`--testing`、fold0、1 epoch。
4. 已新增 `scripts/experiments/run_stage9_rce_pilot_fold0.sh`。
5. Pilot fold0 已成功运行：20 epochs、非 testing，并得到较强 fold0 结果。
6. 下一步常见任务：新增/运行 `scripts/experiments/run_stage9_rce_5fold.sh`，或在已运行后分析其输出。

继续 Stage9 时，应保持改动和 PEPS baseline 代码隔离。

---

## 8. 代码修改后的验证清单

只运行与本次修改相关的检查。

### Python 文件

```bash
python -m py_compile <changed_file.py>
```

如果修改了模型或训练逻辑，也编译可能的 import 链：

```bash
python -m py_compile main.py
python -m py_compile utils/core_utils.py
python -m py_compile models/model_RCE_MIL_BiomedCLIP.py
```

### Shell 脚本

```bash
bash -n <script.sh>
grep -n '\\$' <script.sh> || true
```

新脚本不应包含 `\` 续行，优先使用 bash 数组。

### 不要在未检查时声称成功

如果某个命令没有运行，应明确说明没有运行以及原因。

---

## 9. 代码风格与兼容性要求

- 复杂 tensor 操作附近保留 shape 注释。
- 图像/文本 embedding 做 cosine similarity 前应先 normalize。
- 避免静默 shape 假设。必要时兼容 `[N, 512]`、`[1, N, 512]`、`[N, 1, 512]` 等常见输入。
- 保持 CPU/GPU device 处理与已有 tensor 一致。
- 非训练 prompt features 如果需要随模型移动设备，应注册为 buffer。
- 避免不必要新增依赖。
- 脚本路径应通过环境变量保持可移植性。
- 不要删除或覆盖历史结果。
- 新实验使用唯一 exp code，例如 `rce_mil_5fold_e20_s1`。

---

## 10. 向用户汇报的格式

每次任务完成后，总结以下内容：

1. 修改了哪些文件。
2. 做了什么修改，用 3-6 条说明。
3. 运行了哪些检查，是否通过。
4. 哪些长任务没有运行，尤其要说明原因。
5. 如果有输出文件，给出输出路径。
6. 下一步建议。

如果出现错误，应包含失败命令和关键 traceback / 日志片段。
