# Stage 7：SAF-PEPS 修改说明

## 一、总体说明

本次修改实现的是 **Stage 7：Scale-Adaptive Fusion PEPS（SAF-PEPS）**。

实现目标是：

- 保留当前 `PEPS` 的 prompt selection 主逻辑不变
- 不改 concept prompt pool 读取逻辑
- 不改数据集读取逻辑
- 不改训练主循环的核心指标计算
- 只在 **双尺度最终 logits 融合** 这一层，新增一个可选的自适应融合 gate

同时严格保证：

- 默认行为不变
- 旧命令不变
- 旧实验不变
- 旧 checkpoint 默认仍按原逻辑使用

也就是说，如果不显式传新参数，模型仍然保持原来的：

```text
scale_fusion_mode = sum
logits = logits_low + logits_high
```

## 二、修改的文件

本次修改的文件：

- `main.py`
- `eval.py`
- `utils/core_utils.py`
- `utils/eval_utils.py`
- `models/model_ViLa_MIL_BiomedCLIP.py`

本次新增的文件：

- `scripts/run_stage7_saf_peps.sh`
- `scripts/build_stage7_saf_peps_analysis.py`
- `docs/stage7_saf_peps_modifications.md`

## 三、新增参数

在 `main.py` 和 `eval.py` 中新增了以下参数：

### 1. `--scale_fusion_mode`

- 类型：`str`
- 可选值：`sum`, `learned_gate`
- 默认值：`sum`

含义：

- `sum`：保持原始双尺度融合方式
- `learned_gate`：启用 SAF-PEPS 的自适应双尺度融合

### 2. `--scale_gate_hidden_dim`

- 类型：`int`
- 默认值：`128`

含义：

- 控制 scale gate MLP 的隐藏层维度

### 3. `--scale_gate_dropout`

- 类型：`float`
- 默认值：`0.25`

含义：

- scale gate MLP 的 dropout 比例

### 4. `--allow_legacy_scale_fusion_ckpt`

- 类型：`store_true`
- 默认值：`False`

含义：

- 允许旧 checkpoint 在新引入 `scale_gate` 参数后继续被评估
- 仅在你显式打开时启用兼容模式

## 四、SAF-PEPS 的核心改动

### 1. 原始双尺度融合方式

原模型在 `dual` 模式下，双尺度融合方式是：

```text
logits = logits_low + logits_high
```

这个逻辑仍然保留，并且仍然是默认行为。

### 2. 新增 learned gate 融合方式

当满足以下条件时：

- `scale_mode == dual`
- `scale_fusion_mode == learned_gate`

模型会启用新的 scale gate MLP。

gate 输入为：

```text
[
  image_features_low,
  image_features_high,
  abs(image_features_low - image_features_high),
  image_features_low * image_features_high
]
```

其中输入维度是：

```text
4 * self.L
```

### 3. Scale gate 结构

新增的 MLP 结构为：

```text
Linear(4*self.L, scale_gate_hidden_dim)
ReLU
Dropout(scale_gate_dropout)
Linear(scale_gate_hidden_dim, 1)
Sigmoid
```

### 4. 融合公式

当启用 `learned_gate` 时：

```text
alpha_high = scale_gate(gate_input)
alpha_low = 1 - alpha_high

logits = (1 - alpha_high) * logits_low + alpha_high * logits_high
```

说明：

- `alpha_high` 的 shape 是 `[batch, 1]`
- 会自动 broadcast 到 `[batch, n_classes]`
- `alpha_high` 越大，说明模型更偏向高倍分支
- `alpha_high` 越小，说明模型更偏向低倍分支

## 五、与现有逻辑的兼容关系

### 1. 默认行为不变

默认情况下：

```text
scale_fusion_mode = sum
```

因此旧实验仍然等价于：

```text
logits = logits_low + logits_high
```

### 2. 只在 dual scale 下启用 learned gate

当：

- `scale_mode = low_only`
或
- `scale_mode = high_only`

时，不会启用 gate，仍然走原来的单尺度逻辑。

### 3. PEPS 主逻辑不变

本次没有修改：

- `PEPS` 的 prompt selection 核心逻辑
- concept prompt pool 的读取逻辑
- dataset 的读取逻辑
- train/eval 主循环里的指标计算逻辑

也就是说，SAF-PEPS 本质上是：

**在 PEPS 之后，对 low/high logits 做自适应融合**

而不是改写 PEPS 本身。

## 六、诊断导出新增内容

为了方便后续分析，本次在诊断导出中新增了：

- `scale_fusion_mode`
- `alpha_high`
- `alpha_low`
- `logits_low`
- `logits_high`
- `final_logits`

其中：

- `forward_with_prompt_diagnostics()` 中可以直接访问这些值
- 评估导出的 prompt analysis csv 中也会包含：
  - `scale_fusion_mode`
  - `alpha_high`
  - `alpha_low`

这样后续就可以分析：

- 当前 slide 更依赖 low 还是 high
- gate 是否真的在不同样本上产生不同权重

## 七、旧 checkpoint 兼容方案

这是本次特意处理的高风险点。

### 1. 风险来源

因为新模型引入了：

- `scale_gate.0.weight`
- `scale_gate.0.bias`
- `scale_gate.3.weight`
- `scale_gate.3.bias`

所以旧 checkpoint（例如之前训练好的 PEPS 模型）天然不包含这些参数。

如果继续无脑 `strict=True`，就会在评估旧模型时报错。

### 2. 本次采用的兼容策略

本次没有直接把所有加载都改成 `strict=False`。

而是增加了一个显式开关：

```text
--allow_legacy_scale_fusion_ckpt
```

默认情况下：

- 仍然严格加载

只有当你显式传这个参数时：

- 才允许缺失新增的 `scale_gate.*` 参数
- 其他不匹配仍然报错

### 3. 兼容策略的好处

这样做的优点是：

1. 不会破坏新模型的严格校验
2. 不会悄悄吞掉别的错误
3. 旧模型可以在需要时被安全复用

也就是说，这是一个：

**可选、白名单式、定点放宽的兼容策略**

而不是全局放宽。

## 八、训练脚本

新增脚本：

- `scripts/run_stage7_saf_peps.sh`

这个脚本会自动执行：

### 1. smoke test

- fold 0
- `max_epochs=1`
- `scale_fusion_mode=learned_gate`
- `exp_code=smoke_saf_peps`

### 2. 完整 5-fold 训练

实验名：

- `adeno_concept12_saf_peps_topk5_tau0.07_s1`

### 3. 完整 test 评估

评估目录：

- `eval_results/EVAL_adeno_concept12_saf_peps_topk5_tau0.07`

### 4. 对比分析

输出目录：

- `trained_models/stage7_saf_peps_comparison`

## 九、分析脚本

新增脚本：

- `scripts/build_stage7_saf_peps_analysis.py`

它会读取：

1. 当前主模型：

- `trained_models/adeno_concept12_peps_topk5_tau0.07_s1/result.csv`

2. 新模型：

- `trained_models/adeno_concept12_saf_peps_topk5_tau0.07_s1/result.csv`

并输出：

- `trained_models/stage7_saf_peps_comparison/saf_peps_comparison_summary.csv`

如果存在 `fold_summary.csv`，还会输出：

- `trained_models/stage7_saf_peps_comparison/saf_peps_fold_delta.csv`

## 十、已完成的冒烟测试

本次修改后，我已经做过以下真实验证。

### 1. 代码质量检查

已通过：

```bash
python -m compileall -q .
bash -n scripts/run_stage7_saf_peps.sh
```

### 2. SAF-PEPS smoke 训练

我实际跑通了一个最小训练：

- 1 fold
- 1 epoch
- `prompt_ensemble_mode=peps`
- `scale_fusion_mode=learned_gate`

输出目录：

- `/tmp/stage7_saf_smoke/smoke_saf_peps_s1`

验证结果：

- 训练正常启动
- 模型里成功创建了 `scale_gate`
- checkpoint 成功保存

并且从打印的模型结构中确认：

- `scale_gate` 已经被实例化
- 输入维度正确是 `2048 = 4 * 512`

### 3. SAF-PEPS smoke 评估

我还对 smoke checkpoint 做了评估，并导出了 prompt diagnostics：

- `/xiangmu/ViLMIL/ViLa-MIL-main/eval_results/EVAL_smoke_saf_peps_eval/peps_prompt_analysis_fold0.csv`

检查结果：

- `scale_fusion_mode` 已导出
- `alpha_high` 已导出
- `alpha_low` 已导出

并且数值不是常数 0 或 1：

- `alpha_high min/max/mean = 0.301643 / 0.359120 / 0.330493`
- `alpha_low  min/max/mean = 0.640880 / 0.698357 / 0.669507`

这说明：

- gate 不只是“结构上存在”
- 它确实参与了融合，并产生了非平凡权重

### 4. 旧 checkpoint 兼容测试

我还验证了旧 PEPS checkpoint 在新 learned_gate 路径下的兼容性：

使用参数：

```text
--allow_legacy_scale_fusion_ckpt
```

测试结果：

- 旧 checkpoint 成功加载
- 日志明确提示只缺少 `scale_gate.*` 相关参数
- 评估顺利完成

导出目录：

- `/xiangmu/ViLMIL/ViLa-MIL-main/eval_results/EVAL_legacy_scale_gate_compat_smoke`

而且导出的 csv 中同样已经包含：

- `scale_fusion_mode`
- `alpha_high`
- `alpha_low`

说明旧模型也能在兼容模式下跑新融合分析。

## 十一、如何运行

### 1. 直接运行总脚本

```bash
bash /xiangmu/ViLMIL/ViLa-MIL-main/scripts/run_stage7_saf_peps.sh
```

### 2. 输出目录

训练结果：

- `trained_models/adeno_concept12_saf_peps_topk5_tau0.07_s1`

评估结果：

- `eval_results/EVAL_adeno_concept12_saf_peps_topk5_tau0.07`

对比汇总：

- `trained_models/stage7_saf_peps_comparison/saf_peps_comparison_summary.csv`

逐折差值：

- `trained_models/stage7_saf_peps_comparison/saf_peps_fold_delta.csv`

## 十二、本次明确没有做的内容

本次只实现：

- **Stage 7：SAF-PEPS**

本次明确没有实现：

- PG-PEPS
- prototype alignment / concept-aligned loss
- adaptive top-k
- adaptive tau

也没有改动：

- concept prompt pool 读取
- dataset 读取
- PEPS 主体 prompt selection
- 训练主循环指标计算

## 十三、一句话总结

本次修改实现了一个**默认不破坏旧行为**的双尺度自适应融合版本：

- 默认还是 `sum`
- 显式指定 `learned_gate` 才启用 SAF-PEPS
- 旧 checkpoint 通过可选兼容开关安全复用
- 并且已经通过真实 smoke 训练、评估和导出验证，确认功能生效
