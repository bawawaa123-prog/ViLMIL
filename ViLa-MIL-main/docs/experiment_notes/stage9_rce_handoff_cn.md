# Stage9 RCE-MIL Handoff

本文用于交接 Step1-4 的需求、修改结果与运行结论，方便在新会话中继续推进。

## 总览

- 项目主体目录：`/xiangmu/ViLMIL/ViLa-MIL-main`
- 本阶段主题：新增并验证 `RCE_MIL_BiomedCLIP`
- 当前结论：
  - `RCE_MIL_BiomedCLIP` 已完成最小可行实现
  - smoke test 已成功
  - fold0 pilot（20 epochs, 非 testing）已成功
  - pilot 结果显示模型确实能学习，未出现类别完全塌缩

## Step1

### 用户需求摘要

- 新增模型类型 `RCE_MIL_BiomedCLIP`
- 不沿用 slide-level image-text similarity
- 新模型流程为：
  - patch features
  - region queries 聚合为 region features
  - region 与 concept prompt features 计算 evidence
  - 输出 class logits
- 要求：
  - 新增 `models/model_RCE_MIL_BiomedCLIP.py`
  - `forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None)`
  - 返回 `Y_prob, Y_hat, loss`
  - 接入 `main.py` 和 `utils/core_utils.py`
  - 不修改现有 PEPS / `ViLa_MIL_BiomedCLIP` 逻辑

### 实际修改

- 新增文件：
  - `models/model_RCE_MIL_BiomedCLIP.py`
- 修改文件：
  - `main.py`
  - `utils/core_utils.py`

### 核心实现

- 固定输入特征维度 `512`
- 使用 `config.prototype_number` 作为 `region_number`
- 强制要求：
  - `config.use_concept_prompt_pool=True`
  - `config.concept_prompt_path` 非空
- 复用 BiomedCLIP 文本编码器与 concept prompt pool 编码逻辑
- 构造：
  - `low_prompt_features: [num_classes, num_prompts, 512]`
  - `high_prompt_features: [num_classes, num_prompts, 512]`
- 新增可学习参数：
  - `region_queries_low: [region_number, 1, 512]`
  - `region_queries_high: [region_number, 1, 512]`
- 使用 `MultiheadAttention` 聚合 low/high patch 为 region features
- evidence 逻辑：
  - `sim: [batch, num_classes, region_number, num_prompts]`
  - 对 region 取 `max`
  - 对 prompt 做 `softmax(prompt_evidence / peps_tau)`
  - 加权求和得到每个尺度 logits
- `scale_mode` 支持：
  - `low_only`
  - `high_only`
  - `dual`
- loss 为 `CrossEntropyLoss`

### forward 张量流

- `x_s/x_l`
  -> region attention
  -> `[batch, region_number, 512]`
  -> 与 prompt features 做 cosine similarity
  -> prompt evidence / prompt weights
  -> `logits_low`, `logits_high`
  -> 按 `scale_mode` 融合
  -> `Y_prob, Y_hat, loss`

### 检查结果

- `python -m py_compile ViLa-MIL-main/main.py` 通过
- `python -m py_compile ViLa-MIL-main/utils/core_utils.py` 通过
- `python -m py_compile ViLa-MIL-main/models/model_RCE_MIL_BiomedCLIP.py` 通过

## Step2

### 用户需求摘要

- 新增一个最小 smoke test 脚本
- 只跑：
  - `RCE_MIL_BiomedCLIP`
  - `fold0`
  - `1 epoch`
  - `testing` 模式
- 不改模型和训练逻辑

### 实际修改

- 新增文件：
  - `scripts/experiments/run_stage9_rce_smoke.sh`

### 脚本要点

- 自动推导 `ROOT_DIR`
- 环境变量可覆盖：
  - `PYTHON_BIN`
  - `DATA_ROOT_DIR`
  - `RESULTS_DIR`
  - `SPLIT_DIR`
  - `TEXT_PROMPT_PATH`
  - `CONCEPT12_PATH`
- 使用 bash 数组 `CMD=(...)`
- 实际命令参数：
  - `--task task_adenocarcinoma`
  - `--model_type RCE_MIL_BiomedCLIP`
  - `--mode transformer`
  - `--data_folder_s features_biomedclip_5x`
  - `--data_folder_l features_biomedclip_20x`
  - `--use_concept_prompt_pool`
  - `--concept_prompt_path "${CONCEPT12_PATH}"`
  - `--prompt_ensemble_mode embedding_mean`
  - `--scale_mode dual`
  - `--k 5 --k_start 0 --k_end 0`
  - `--max_epochs 1`
  - `--seed 1`
  - `--prototype_number 16`
  - `--testing`
  - `--exp_code rce_mil_smoke`

### 检查结果

- `bash -n ViLa-MIL-main/scripts/experiments/run_stage9_rce_smoke.sh` 通过
- `grep -n '\\$' ViLa-MIL-main/scripts/experiments/run_stage9_rce_smoke.sh || true` 无输出

## Step3

### 用户需求摘要

- 实际运行 Step2 的 smoke test
- 保存完整日志
- 检查是否能完成 train / val / test

### 实际运行命令

- 运行目录：
  - `cd /xiangmu/ViLMIL/ViLa-MIL-main`
- 执行：
  - `bash scripts/experiments/run_stage9_rce_smoke.sh 2>&1 | tee results_smoke/logs/rce_mil_smoke_step3.log`

### 日志位置

- `results_smoke/logs/rce_mil_smoke_step3.log`

### 结果目录

- `results_smoke/rce_mil_smoke_s1`

### 运行结论

- 运行成功
- 成功进入 `main.py`
- 成功初始化 `RCE_MIL_BiomedCLIP`
- 成功读取 concept prompt pool
- 成功完成 train / val / test
- 最终打印：
  - `[Done] RCE-MIL smoke test finished.`

### Smoke 关键指标

- Train:
  - `Loss=0.6854`
  - `Acc=0.6379`
  - `AUC=0.3256`
  - `F1=0.3895`
- Val:
  - `Loss=0.6239`
  - `Acc=0.7368`
  - `AUC=0.6714`
  - `F1=0.4242`
- Final Test:
  - `AUC=0.6000`
  - `ACC=0.7895`
  - `F1=0.4412`
  - `Balanced ACC=0.5000`
  - `Sensitivity=0.0000`
  - `Specificity=1.0000`
  - `PR-AUC=0.4952`

### Smoke 阶段观察

- smoke 仅用于验证 forward/backward/训练流程可运行
- 在 smoke 小样本 `--testing` 下，class 1 sensitivity 为 `0`
- 因此需要进一步做非 testing 的 pilot 验证

## Step4

### 用户需求摘要

- 新增并运行一个 fold0 pilot 实验脚本
- 不使用 `--testing`
- 完整 fold0
- 训练 `20 epochs`
- 用于判断模型是否真的能学习

### 实际修改

- 新增文件：
  - `scripts/experiments/run_stage9_rce_pilot_fold0.sh`

### 脚本参数

- `--task task_adenocarcinoma`
- `--model_type RCE_MIL_BiomedCLIP`
- `--mode transformer`
- `--data_root_dir "${DATA_ROOT_DIR}"`
- `--data_folder_s features_biomedclip_5x`
- `--data_folder_l features_biomedclip_20x`
- `--text_prompt_path "${TEXT_PROMPT_PATH}"`
- `--use_concept_prompt_pool`
- `--concept_prompt_path "${CONCEPT12_PATH}"`
- `--prompt_ensemble_mode embedding_mean`
- `--scale_mode dual`
- `--k 5 --k_start 0 --k_end 0`
- `--max_epochs 20`
- `--seed 1`
- `--prototype_number 16`
- `--split_dir "${SPLIT_DIR}"`
- `--results_dir "${RESULTS_DIR}"`
- `--exp_code rce_mil_pilot_fold0`

### 检查结果

- `bash -n ViLa-MIL-main/scripts/experiments/run_stage9_rce_pilot_fold0.sh` 通过
- `grep -n '\\$' ViLa-MIL-main/scripts/experiments/run_stage9_rce_pilot_fold0.sh || true` 无输出

### 实际运行

- 日志：
  - `results_stage9/logs/rce_mil_pilot_fold0_step4.log`
- 结果目录：
  - `results_stage9/rce_mil_pilot_fold0_s1`

### 运行结论

- pilot 成功完成
- 成功跑完 `20 epochs`
- 成功完成最终 validation 与 test
- 成功生成 checkpoint、epoch_details、fold_summary、split 结果文件

### Pilot 关键指标

- 最佳 Val AUC：
  - `0.9573`（epoch 12）
- 最终 Val AUC：
  - `0.9435`
- Final Test:
  - `AUC=0.9743`
  - `ACC=0.9021`
  - `F1=0.8897`
  - `Balanced ACC=0.8854`
  - `Sensitivity=0.8333`
  - `Specificity=0.9375`
  - `PR-AUC=0.9491`

### 类别塌缩判断

- 未出现类别塌缩
- 不再是 smoke 阶段的 `class 1 sensitivity = 0`
- 最终测试中：
  - class 0: `120/128`
  - class 1: `55/66`
  - class 1 sensitivity: `0.8333`

## 当前已新增/涉及文件

### 代码与脚本

- `models/model_RCE_MIL_BiomedCLIP.py`
- `scripts/experiments/run_stage9_rce_smoke.sh`
- `scripts/experiments/run_stage9_rce_pilot_fold0.sh`
- `main.py`
- `utils/core_utils.py`

### 运行日志与结果

- `results_smoke/logs/rce_mil_smoke_step3.log`
- `results_smoke/rce_mil_smoke_s1/`
- `results_stage9/logs/rce_mil_pilot_fold0_step4.log`
- `results_stage9/rce_mil_pilot_fold0_s1/`

## 建议的新会话起点

新会话建议直接说明：

1. 已完成 Step1-4，见本文档
2. `RCE_MIL_BiomedCLIP` 已能在 fold0 pilot 上学习
3. 下一步可以做的方向：
   - 继续更多 fold 或更长训练
   - 分析为什么 early epochs 的 Val AUC 很高、后期略波动
   - 做 ablation：`low_only` / `high_only`
   - 对比 `ViLa_MIL_BiomedCLIP`
   - 分析 region-concept evidence 的可解释性输出

## 备注

- 本阶段没有修改现有 PEPS 逻辑
- smoke 与 pilot 都使用：
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- 当前工作区仍有未提交文件与结果目录，开新会话时可直接引用本文档与日志路径
