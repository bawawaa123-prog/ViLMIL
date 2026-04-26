# BiomedCLIP 基线训练与评估说明

本文档只描述当前仓库保留的干净基线流程：

- `ViLa_MIL_BiomedCLIP`
- 双尺度 prompt
- 双尺度 patch 特征输入
- 标准训练 / 验证 / 测试

当前仓库已移除以下旧增强功能，不再支持对应命令行参数或脚本：

- dynamic prompt retrieval
- VCP / HyperPrompt
- RAG / LLM rewrite

## 1. 需要准备的内容

### 1.1 数据与特征

- 样本 CSV：`dataset_csv/all_data.csv`
- 5x 特征目录：`/xiangmu/data/VILMIL/features_biomedclip_5x`
- 20x 特征目录：`/xiangmu/data/VILMIL/features_biomedclip_20x`
- split 目录示例：`splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`

### 1.2 文本提示

当前基线仍使用双尺度 prompt CSV。

推荐文件：

- `text_prompt/adenocarcinoma_dual_scale_prompt.csv`

期望列：

- `class_name`
- `low_resolution_description`
- `high_resolution_description`

`main.py` 与 `eval.py` 会把它解析成：

```text
[low_class0, low_class1, ..., high_class0, high_class1, ...]
```

对于二分类任务，最终应得到 4 条文本提示。

## 2. 训练命令

下面是一条可直接用于腺癌 strict 5-fold 的 BiomedCLIP 基线训练命令。

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main

PYTHONPATH=/xiangmu/ViLMIL/ViLa-MIL-main \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/ljh/anaconda3/envs/vila_mil/bin/python main.py \
  --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --data_root_dir /xiangmu/data/VILMIL \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --split_dir splits/adenocarcinoma/task_adenocarcinoma_strictcv_100 \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --exp_code adeno_biomedclip_clean \
  --k 5 \
  --k_start 0 \
  --k_end 4 \
  --max_epochs 40 \
  --lr 1e-4 \
  --seed 1 \
  --prototype_number 16 \
  --early_stopping
```

### 2.1 单折快速验证命令

如果你只想先验证环境和链路是否正常：

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main

PYTHONPATH=/xiangmu/ViLMIL/ViLa-MIL-main \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/ljh/anaconda3/envs/vila_mil/bin/python main.py \
  --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --data_root_dir /xiangmu/data/VILMIL \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --split_dir splits/adenocarcinoma/task_adenocarcinoma_strictcv_100 \
  --results_dir /tmp/vilmil_smoke \
  --exp_code smoke \
  --k 5 \
  --k_start 0 \
  --k_end 0 \
  --max_epochs 1 \
  --seed 1 \
  --prototype_number 16 \
  --testing
```

## 3. 评估命令

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main

PYTHONPATH=/xiangmu/ViLMIL/ViLa-MIL-main \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/ljh/anaconda3/envs/vila_mil/bin/python eval.py \
  --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --data_root_dir /xiangmu/data/VILMIL \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --models_exp_code adeno_biomedclip_clean_s1 \
  --save_exp_code adeno_biomedclip_clean_eval \
  --k 5 \
  --k_start 0 \
  --k_end 4 \
  --split test
```

如果只评估某一折，可加：

```bash
--fold 0
```

## 4. 输出文件

训练目录示例：

- `trained_models/<exp_code>_s<seed>/s_0_checkpoint.pt`
- `trained_models/<exp_code>_s<seed>/epoch_details.csv`
- `trained_models/<exp_code>_s<seed>/fold_summary.csv`
- `trained_models/<exp_code>_s<seed>/result.csv`

评估目录示例：

- `eval_results/EVAL_<save_exp_code>/summary.csv`
- `eval_results/EVAL_<save_exp_code>/result.csv`
- `eval_results/EVAL_<save_exp_code>/timing.csv`

## 5. 当前可用的核心参数

### 5.1 训练主参数

- `--task`
- `--model_type`
- `--data_root_dir`
- `--data_folder_s`
- `--data_folder_l`
- `--text_prompt_path`
- `--split_dir`
- `--results_dir`
- `--exp_code`
- `--k`
- `--k_start`
- `--k_end`
- `--max_epochs`
- `--lr`
- `--seed`
- `--prototype_number`
- `--early_stopping`

### 5.2 BiomedCLIP 微调相关参数

- `--finetune_text_encoder`
- `--prompt_lr`
- `--text_lr`
- `--text_finetune_mode`
- `--text_unfreeze_last_n`

如果你只是跑干净基线，默认不需要打开文本编码器微调。

## 6. 代码入口

建议按这个顺序看代码：

1. `main.py`
2. `utils/core_utils.py`
3. `datasets/dataset_generic.py`
4. `models/model_ViLa_MIL_BiomedCLIP.py`
5. `eval.py`
6. `utils/eval_utils.py`

## 7. 故障排查

### 7.1 提示词数量不对

如果看到：

```text
Text prompts loaded: X items (expected 4 = 2 x n_classes)
```

说明你的 prompt CSV 不符合当前二分类双尺度格式。

### 7.2 找不到 split

检查：

- `--split_dir`
- `splits_0.csv` 到 `splits_4.csv` 是否存在

### 7.3 找不到 checkpoint

检查：

- `--models_exp_code`
- 训练目录下是否有 `s_<fold>_checkpoint.pt`

### 7.4 无 GPU 也能跑吗

可以。当前主链路已经兼容 CPU，但会明显更慢。

## 8. 结论

当前仓库的推荐使用方式就是：

- 预提取 BiomedCLIP 5x/20x 特征
- 使用双尺度 prompt
- 跑 `ViLa_MIL_BiomedCLIP` 基线训练与评估

如果后续再做方法增强，建议新开分支，不要再直接污染这条基线链路。
