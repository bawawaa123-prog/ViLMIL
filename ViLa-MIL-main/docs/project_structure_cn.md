# 项目结构说明（中文版）

本文档用于说明 `/xiangmu/ViLMIL/ViLa-MIL-main` 当前整理后的目录结构、各目录用途，以及关键文件的作用。

本次整理遵循两个原则：

1. **不破坏核心训练/评估入口**
   - `main.py`
   - `eval.py`
   - `models/`
   - `utils/`
   - `datasets/`

2. **优先整理最杂乱的说明文档、实验脚本和历史调试脚本**
   - `docs/`
   - `scripts/`
   - root 下的历史/调试型脚本

---

## 一、顶层目录总览

### 1. 核心训练与评估入口

- `main.py`
  - 项目主训练入口
  - 负责参数解析、数据集构建、交叉验证训练、结果保存

- `eval.py`
  - 项目主评估入口
  - 负责加载训练好的 checkpoint，进行 test/val/train 集评估与导出

- `README.md`
  - 项目总说明入口

### 2. 核心代码目录

- `models/`
  - 模型定义目录
  - 包含 `ViLa_MIL` 与 `ViLa_MIL_BiomedCLIP` 的主要实现

- `utils/`
  - 训练、评估、指标、文件读写、prompt 相关的通用工具

- `datasets/`
  - 数据集定义与 h5 / csv 读取逻辑

- `wsi_core/`
  - Whole Slide Image 相关底层工具

- `feature_extraction/`
  - patch / feature 提取流程相关脚本

### 3. 数据与配置目录

- `dataset_csv/`
  - 任务 csv、concept pool、数据表、阶段性计划文件

- `text_prompt/`
  - 静态 prompt / 双尺度 prompt / 动态 prompt 模板

- `splits/`
  - 交叉验证 split 文件

- `presets/`
  - 预设配置

### 4. 实验与结果目录

- `trained_models/`
  - 各阶段训练得到的模型目录与最终对比结果

- `eval_results/`
  - 各阶段评估导出结果

- `results/`
  - 较早期的一些结果缓存与历史训练结果

- `results_smoke/`
  - 冒烟测试相关结果

### 5. 辅助与历史目录

- `docs/`
  - 项目说明文档（本次已整理分类）

- `scripts/`
  - 实验运行、分析汇总、数据辅助脚本（本次已整理分类）

- `tools/`
  - 杂项工具脚本，例如裁切、比对、特征检查、SVS 处理等

- `history_code/`
  - 历史代码、旧版本实现、旧调试入口

- `shared/`
  - 共用的小工具

- `knowledge_base/`
  - 知识库与结构化文本材料

- `image/`
  - 图像类附加材料

- `data/`
  - 一些附加数据与统计文件

---

## 二、核心代码目录说明

## `models/`

用于存放模型主体。

关键文件：

- `models/model_ViLa_MIL_BiomedCLIP.py`
  - 当前主模型核心实现
  - 包含：
    - BiomedCLIP 文本编码
    - 双尺度 low/high 图像分支
    - prompt 融合逻辑
    - PEPS / SAP-PEPS / SAF-PEPS 等阶段性扩展

- `models/model_ViLa_MIL.py`
  - 原始 `ViLa_MIL` 版本

- `models/model_mil.py`
  - 基础 MIL 模型

- `models/model_utils.py`
  - 模型辅助组件

## `utils/`

用于存放训练与评估过程中会复用的工具。

关键文件：

- `utils/core_utils.py`
  - 主训练流程相关函数
  - 包括：
    - train
    - train_loop
    - validate
    - checkpoint 保存/加载

- `utils/eval_utils.py`
  - 主评估流程相关函数
  - 包括：
    - 模型初始化
    - checkpoint 加载
    - summary
    - prompt diagnostics 导出

- `utils/metric_utils.py`
  - 分类指标计算
  - 包括：
    - AUC
    - ACC
    - F1
    - balanced_acc
    - sensitivity
    - specificity
    - pr_auc

- `utils/prompt_utils.py`
  - concept prompt pool 构建与编码逻辑

- `utils/file_utils.py`
  - pkl / csv 等结果文件保存

- `utils/heatmap_utils.py`
  - 热力图相关辅助函数

- `utils/loss_utils.py`
  - 损失函数工具

- `utils/utils.py`
  - 通用工具函数

## `datasets/`

负责把 csv、h5 特征和 split 组织成训练/验证/测试集。

关键文件：

- `datasets/dataset_generic.py`
  - 当前最重要的数据读取逻辑
  - 负责：
    - slide 列表读取
    - h5 特征读取
    - coords 读取
    - train/val/test split 对接

- `datasets/dataset_h5.py`
  - h5 数据读取相关辅助

- `datasets/wsi_dataset.py`
  - WSI 数据集接口

- `datasets/BatchWSI.py`
  - batch 形式的数据组织辅助

## `feature_extraction/`

负责从 patch / 图像到特征的前处理脚本。

关键文件：

- `feature_extraction/patch_extraction.py`
- `feature_extraction/patch_extraction_biomedclip.py`
- `feature_extraction/patch_extraction_utils.py`
- `feature_extraction/patch_extraction_utils_biomedclip.py`
- `feature_extraction/patch_generation.py`

用途：

- patch 提取
- BiomedCLIP feature 提取相关流程

## `wsi_core/`

底层 WSI 处理类和工具。

关键文件：

- `wsi_core/WholeSlideImage.py`
  - Whole Slide Image 封装类

- `wsi_core/wsi_utils.py`
  - WSI 常用工具函数

- `wsi_core/batch_process_utils.py`
  - 批量处理工具

---

## 三、文档目录说明（docs/）

本次已整理为 3 类：

### `docs/guides/`

用途：

- 面向流程、使用说明、数据处理的“说明型文档”

当前文件：

- `docs/guides/biomedclip_vs_clip.md`
  - BiomedCLIP 与原 CLIP 的差异说明

- `docs/guides/data_processing_svs_to_features_cn.md`
  - 从 SVS 到特征的流程说明

- `docs/guides/dual_scale_prompt_usage.md`
  - 双尺度 prompt 的使用说明

- `docs/guides/end_to_end_pipeline_and_switches_cn.md`
  - 整个 pipeline 与开关说明

- `docs/guides/feature_inspection.md`
  - 特征/coords 检查相关记录

### `docs/experiment_notes/`

用途：

- 面向各阶段实验与改动说明的“实验记录型文档”

当前文件：

- `docs/experiment_notes/ppt_model_improvement_summary_cn.md`
  - 模型迭代阶段总结

- `docs/experiment_notes/stage7_saf_peps_modifications.md`
  - Stage 7 SAF-PEPS 修改说明

- `docs/experiment_notes/vilamil_biomedclip_timeline_cn.md`
  - 项目阶段演进时间线

### `docs/root_notes/`

用途：

- 收纳原先散落在项目根目录的说明/笔记文档

当前文件：

- `docs/root_notes/BIOMEDCLIP_INTEGRATION_GUIDE.md`
- `docs/root_notes/BIOMEDCLIP_TODO.md`
- `docs/root_notes/CODE_MODIFICATION_GUIDE.md`
- `docs/root_notes/SUMMARY_BIOMEDCLIP.md`
- `docs/root_notes/Xiugai.md`

这些文档主要是历史说明、阶段记录和临时笔记，保留用于追溯，但不建议作为当前主入口。

---

## 四、脚本目录说明（scripts/）

本次已整理为 3 类：

### `scripts/experiments/`

用途：

- 直接运行某一阶段实验的脚本

当前文件：

- `scripts/experiments/run_stage2_dynamic_gate_ablation.sh`
  - Stage 2 dynamic gate 消融实验

- `scripts/experiments/run_stage2_prompt_pool_4way.sh`
  - Stage 2 四组 prompt pool 对比

- `scripts/experiments/run_stage2_prompt_pool_size_sweep.sh`
  - Stage 2 prompt 数量 sweep

- `scripts/experiments/run_stage3_peps_5way.sh`
  - Stage 3 PEPS 五组实验

- `scripts/experiments/run_stage3_peps_tau_sweep.sh`
  - Stage 3 tau sweep

- `scripts/experiments/run_stage6_sap_peps.sh`
  - Stage 6 SAP-PEPS 实验

- `scripts/experiments/run_stage7_saf_peps.sh`
  - Stage 7 SAF-PEPS 实验

- `scripts/experiments/run_final_dcp_vila_analysis.sh`
  - 最终汇总分析运行脚本

### `scripts/analysis/`

用途：

- 读取已有训练/评估结果并生成对比表、差值分析、报告

当前文件：

- `scripts/analysis/aggregate_stage2_ablation_results.py`
- `scripts/analysis/aggregate_stage2_size_sweep_results.py`
- `scripts/analysis/aggregate_stage3_peps_results.py`
- `scripts/analysis/aggregate_stage3_peps_extended_results.py`
- `scripts/analysis/analyze_stage3_peps_fold_deltas.py`
- `scripts/analysis/analyze_stage3_peps_prompt_usage.py`
- `scripts/analysis/build_stage3_peps_extended_report.py`
- `scripts/analysis/build_stage6_sap_peps_analysis.py`
- `scripts/analysis/build_stage7_saf_peps_analysis.py`
- `scripts/analysis/build_final_dcp_vila_analysis.py`

### `scripts/data_tools/`

用途：

- 和 patch 裁切、文本特征抽取相关的辅助脚本

当前文件：

- `scripts/data_tools/crop_patches_from_coords.py`
- `scripts/data_tools/crop_patches_grid.py`
- `scripts/data_tools/extract_modified_text_features.py`

---

## 五、数据与配置目录说明

## `dataset_csv/`

用途：

- 存放训练数据 csv、concept pool、病理/影像概念表等

关键文件：

- `dataset_csv/all_data.csv`
  - 当前 `task_adenocarcinoma` 的主数据表

- `dataset_csv/private_lung_concept_prompt_pool_stage2.json`
  - Stage 2 初版 concept pool

- `dataset_csv/private_lung_concept_prompt_pool_stage2_core10.json`
- `dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
- `dataset_csv/private_lung_concept_prompt_pool_stage2_core14.json`
  - 不同数量版本的 concept pool

- `dataset_csv/private_lung_concept_dictionary_with_regex.json`
  - concept 字典

- `dataset_csv/private_lung_case_concepts_extracted.csv`
  - case-level 概念提取结果

- `dataset_csv/private_lung_pathology_concept_summary.csv`
  - 病理概念汇总

- `dataset_csv/private_lung_radiology_concept_summary.csv`
  - 影像概念汇总

- `dataset_csv/codex_stage2_implementation_plan.md`
  - 某阶段实现规划记录

说明：

- 这个目录既有“正式输入数据”，也有“实验中间材料”
- 后续如果想进一步精细化，还可以再拆成：
  - `dataset_csv/core/`
  - `dataset_csv/concept_pool/`
  - `dataset_csv/history/`

## `text_prompt/`

用途：

- 存放静态 prompt 和 prompt 模板

关键文件：

- `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
  - 当前 adenocarcinoma 任务的双尺度静态 prompt

- `text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv`
  - 动态 prompt 池模板

## `splits/`

用途：

- 存放各任务 split

关键目录：

- `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
  - 当前主任务的 strict 5-fold split

---

## 六、实验结果目录说明

## `trained_models/`

用途：

- 存放正式训练出的模型目录
- 也存放各阶段自动生成的比较表与分析结果

当前包含：

- 主模型训练目录，例如：
  - `adeno_concept12_peps_topk5_tau0.07_s1`
  - `adeno_concept12_saf_peps_topk5_tau0.07_s1`

- 阶段性对比目录，例如：
  - `stage2_ablation_comparison`
  - `stage2_size_sweep_comparison`
  - `stage3_peps_comparison`
  - `stage6_sap_peps_comparison`
  - `stage7_saf_peps_comparison`
  - `final_dcp_vila_analysis`

## `eval_results/`

用途：

- 存放模型评估结果
- 包括：
  - `summary.csv`
  - `fold_metrics.csv`
  - `result.csv`
  - `prompt diagnostics csv`

命名习惯：

- `EVAL_实验名`

## `results/`

用途：

- 更早期或过渡阶段产生的缓存、历史结果、训练中间材料

## `results_smoke/`

用途：

- 冒烟测试结果

---

## 七、历史与辅助目录说明

## `history_code/`

用途：

- 存放历史代码与旧入口

当前新增了：

- `history_code/legacy_entrypoints/`

这里收纳了之前散落在 root 的调试型/历史型脚本：

- `BiomedCLIP_01.py`
- `BiomedCLIP_02.py`
- `eval_enhanced_with_overfitting.py`
- `eval_full_dataset_with_heatmap.py`
- `test_biomedclip_integration.py`

这样做的目的是：

- 减少根目录噪音
- 把“历史调试脚本”和“当前正式入口”明确分开

## `tools/`

用途：

- 存放各种辅助脚本
- 偏向：
  - SVS 处理
  - 特征检查
  - 裁切
  - 数据对比

关键文件示例：

- `tools/build_stage2_core_prompt_pools.py`
  - 构建 Stage 2 不同大小的 core prompt pool

- `tools/inspect_biomedclip_feature.py`
  - 检查 BiomedCLIP 特征

- `tools/compare_slides_with_h5.py`
  - 对比 slide 与 h5 覆盖情况

- `tools/export_svs_preview_images.py`
  - 导出 SVS 预览图

## `shared/`

用途：

- 少量共享辅助脚本

## `knowledge_base/`

用途：

- 存放知识库文本、结构化概念材料

---

## 八、当前最重要的“入口文件”

如果你现在继续做实验，最常用的是这几类：

### 训练/评估主入口

- `main.py`
- `eval.py`

### 当前核心模型

- `models/model_ViLa_MIL_BiomedCLIP.py`

### 当前主任务数据

- `dataset_csv/all_data.csv`
- `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
- `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- `dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`

### 当前阶段实验脚本

- `scripts/experiments/run_stage7_saf_peps.sh`

### 当前阶段分析脚本

- `scripts/analysis/build_stage7_saf_peps_analysis.py`

---

## 九、整理后的使用建议

后续建议按下面的方式找文件：

1. **找训练/评估入口**
   - 看 root：
     - `main.py`
     - `eval.py`

2. **找某一阶段实验脚本**
   - 看：
     - `scripts/experiments/`

3. **找结果汇总与报告脚本**
   - 看：
     - `scripts/analysis/`

4. **找数据辅助脚本**
   - 看：
     - `tools/`
     - `scripts/data_tools/`

5. **找正式说明文档**
   - 看：
     - `docs/guides/`
     - `docs/experiment_notes/`

6. **找历史实现或旧调试脚本**
   - 看：
     - `history_code/`
     - `docs/root_notes/`

---

## 十、一句话总结

整理后的结构可以这样理解：

- **root**：只放核心入口和核心目录
- **models / utils / datasets**：放真正的训练与评估核心代码
- **scripts/experiments**：放“跑实验”的脚本
- **scripts/analysis**：放“做对比分析”的脚本
- **docs/**：放说明文档，并按用途分组
- **history_code/**：放历史和调试遗留
- **trained_models / eval_results**：放跑出来的结果

这样后续继续做新阶段实验时，会更容易知道：

- 去哪里改模型
- 去哪里跑实验
- 去哪里看分析脚本
- 去哪里找历史记录
