# ViLa-MIL 全流程实战手册（从数据到三点创新组合）

本文档给出本项目从数据处理到训练/评估的完整链路，并详细说明第一点（动态检索提示）、第二点（VCP）、第三点（RAG+LLM重写）在不同组合下的使用方法、运行命令、输入输出与关键参数含义。

适用代码版本：当前仓库已集成
- 第1点：动态检索提示
- 第2点：VCP/HyperPrompt
- 第3点：RAG + LLM 重写（Ollama，offline/online/hybrid）

---

## 1. 目录与数据约定

### 1.1 关键目录
- WSI 特征目录（低倍）：`/xiangmu/ViLMIL/ViLa-MIL-main/features_biomedclip_5x`
- WSI 特征目录（高倍）：`/xiangmu/ViLMIL/ViLa-MIL-main/features_biomedclip_20x`
- 分割目录（5折）：`/xiangmu/ViLMIL/ViLa-MIL-main/splits/Yifuyuan_strict`
- 结果输出目录：`/xiangmu/ViLMIL/ViLa-MIL-main/trained_models`
- 静态双尺度提示：`/xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 动态句子池：`/xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv`
- RAG缓存（建议）：`/xiangmu/ViLMIL/ViLa-MIL-main/results/rag_cache_qwen14b.jsonl`

### 1.2 训练样本索引文件
- CSV 示例：`dataset_csv/all_data.csv`
- 必需列：`case_id, slide_id, label`

### 1.3 特征文件格式
每个 slide 对应一个 h5 文件，至少包含
- `features`: [N, D]
- `coords`: [N, 2]

其中 D 对 BiomedCLIP 为 512。

---

## 2. 全链路流程（从数据到分类）

### 阶段A：数据准备
1. 原始WSI切 patch 坐标与裁图（可选，若你已有h5可跳过）
2. 用 BiomedCLIP 提取低倍与高倍 patch 特征，生成 h5
3. 准备 split 文件（5折）

### 阶段B：训练
输入
- 低倍特征目录 + 高倍特征目录
- split 文件
- 文本提示（静态 + 动态池）

输出
- 每折 checkpoint
- 每折结果 pkl/csv
- 汇总 result/summary
- epoch 详细记录

### 阶段C：评估
输入
- 训练好的 checkpoint
- 同样的数据和文本配置

输出
- fold 级结果
- 汇总指标
- 可选：检索日志

---

## 3. 三点创新定义与关系

### 第1点：动态检索提示（Dynamic Prompt Retrieval）
- 作用：从句子池中按类别/尺度检索 top-k 句子并加权融合
- 本质：检索增强，不改变模型主干

### 第2点：VCP / HyperPrompt
- 作用：用当前 slide 视觉摘要向量生成文本偏移，微调检索文本特征
- 本质：条件化连续微调，不改分类头

### 第3点：RAG + LLM 重写提示
- 作用：先用检索证据，再让 LLM 生成当前 slide 专属低/高倍率提示
- 本质：内容级重写（真正的 R+G）

### 推荐执行顺序
1. 第1点先稳定
2. 第1点 + 第2点
3. 第1点 + 第3点
4. 第1点 + 第2点 + 第3点（最终版）

---

## 4. 组合开关真值表（A/B/C）

- A = `--enable_dynamic_prompt`
- B = `--enable_rag_rewrite`
- C = `--enable_vcp`

| A | B | C | 路径 | 说明 |
|---|---|---|---|---|
| 0 | 0 | 0 | 静态提示 -> 分类 | 原始基线 |
| 1 | 0 | 0 | 动态检索 -> 分类 | 第1点 |
| 1 | 0 | 1 | 动态检索 -> VCP -> 分类 | 第1+2点 |
| 1 | 1 | 0 | 动态检索 -> RAG重写 -> 分类 | 第1+3点 |
| 1 | 1 | 1 | 动态检索 -> RAG重写 -> VCP -> 分类 | 第1+2+3点 |

不建议主实验：A=0,B=1（RAG缺少稳定检索证据）。

---

## 5. 训练命令（可直接运行）

使用前建议先进入项目根目录：

```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main
```

> 注：`--split_dir` 推荐使用 `splits/Yifuyuan_strict` 这种项目内相对路径。

### 5.1 基线（静态）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 --drop_out --early_stopping --lr 1e-4 \
  --k 5 --k_start 0 --k_end 0 \
  --bag_loss ce --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --split_dir splits/Yifuyuan_strict \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --exp_code adeno_static_fold0_s1 \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv
```

### 5.2 第1点（动态检索）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 --drop_out --early_stopping --lr 1e-4 \
  --k 5 --k_start 0 --k_end 0 \
  --bag_loss ce --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --split_dir splits/Yifuyuan_strict \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --exp_code adeno_dyn_fold0_s1 \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --enable_dynamic_prompt \
  --prompt_pool_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv \
  --retrieval_topk 2 --retrieval_temp 0.15 --dynamic_prompt_mix 0.8 \
  --dynamic_prompt_warmup_epochs 4
```

### 5.3 第1+2点（动态检索 + VCP）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 --drop_out --early_stopping --lr 1e-4 \
  --k 5 --k_start 0 --k_end 0 \
  --bag_loss ce --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --split_dir splits/Yifuyuan_strict \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --exp_code adeno_dyn_vcp_fold0_s1 \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --enable_dynamic_prompt \
  --prompt_pool_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv \
  --retrieval_topk 2 --retrieval_temp 0.15 --dynamic_prompt_mix 0.8 \
  --dynamic_prompt_warmup_epochs 4 \
  --enable_vcp --vcp_beta 0.03 --vcp_dropout 0.1 --vcp_start_epoch 16
```

### 5.4 第1+3点（动态检索 + RAG）
#### 5.4.1 Online（实时调用Ollama）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 --drop_out --early_stopping --lr 1e-4 \
  --k 5 --k_start 0 --k_end 0 \
  --bag_loss ce --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --split_dir splits/Yifuyuan_strict \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --exp_code adeno_rag_online_fold0_s1 \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --enable_dynamic_prompt \
  --prompt_pool_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv \
  --retrieval_topk 2 --retrieval_temp 0.15 --dynamic_prompt_mix 0.8 \
  --dynamic_prompt_warmup_epochs 4 \
  --enable_rag_rewrite --rag_mode online \
  --rag_cache_path /xiangmu/ViLMIL/ViLa-MIL-main/results/rag_cache_qwen14b.jsonl \
  --rag_topk 3 --rag_ollama_model qwen2.5:14b-instruct \
  --rag_ollama_url http://localhost:11434/api/generate \
  --rag_temperature 0.2 --rag_max_tokens 256 --rag_timeout_sec 60 \
  --rag_fallback dynamic --rag_start_epoch 16
```

#### 5.4.2 Hybrid（先查缓存，miss再调Ollama并写缓存）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 --drop_out --early_stopping --lr 1e-4 \
  --k 5 --k_start 0 --k_end 0 \
  --bag_loss ce --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --split_dir splits/Yifuyuan_strict \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --exp_code adeno_rag_hybrid_fold0_s1 \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --enable_dynamic_prompt \
  --prompt_pool_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv \
  --retrieval_topk 2 --retrieval_temp 0.15 --dynamic_prompt_mix 0.8 \
  --dynamic_prompt_warmup_epochs 4 \
  --enable_rag_rewrite --rag_mode hybrid \
  --rag_cache_path /xiangmu/ViLMIL/ViLa-MIL-main/results/rag_cache_qwen14b.jsonl \
  --rag_topk 3 --rag_ollama_model qwen2.5:14b-instruct \
  --rag_ollama_url http://localhost:11434/api/generate \
  --rag_temperature 0.2 --rag_max_tokens 256 --rag_timeout_sec 60 \
  --rag_fallback dynamic --rag_start_epoch 16
```

#### 5.4.3 Offline（纯缓存可复现）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 --drop_out --early_stopping --lr 1e-4 \
  --k 5 --k_start 0 --k_end 0 \
  --bag_loss ce --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --split_dir splits/Yifuyuan_strict \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --exp_code adeno_rag_offline_fold0_s1 \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --enable_dynamic_prompt \
  --prompt_pool_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv \
  --retrieval_topk 2 --retrieval_temp 0.15 --dynamic_prompt_mix 0.8 \
  --dynamic_prompt_warmup_epochs 4 \
  --enable_rag_rewrite --rag_mode offline \
  --rag_cache_path /xiangmu/ViLMIL/ViLa-MIL-main/results/rag_cache_qwen14b.jsonl \
  --rag_topk 3 --rag_ollama_model qwen2.5:14b-instruct \
  --rag_ollama_url http://localhost:11434/api/generate \
  --rag_temperature 0.2 --rag_max_tokens 256 --rag_timeout_sec 60 \
  --rag_fallback dynamic --rag_start_epoch 16
```

### 5.5 第1+2+3点（全开）
在 5.4 的命令上加上 VCP 参数：
- `--enable_vcp --vcp_beta 0.03 --vcp_dropout 0.1 --vcp_start_epoch 16`

---

## 6. 评估命令（eval.py）

### 6.1 一折评估示例（第1+2+3，全开，离线RAG）
```bash
cd /xiangmu/ViLMIL/ViLa-MIL-main && CUDA_VISIBLE_DEVICES=0 python eval.py \
  --task task_adenocarcinoma --mode transformer --model_type ViLa_MIL_BiomedCLIP \
  --k 5 --k_start 0 --k_end 1 --fold 0 --split test \
  --data_root_dir /xiangmu/ViLMIL/ViLa-MIL-main \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --results_dir /xiangmu/ViLMIL/ViLa-MIL-main/trained_models \
  --models_exp_code adeno_rag_offline_fold0_s1_s1 \
  --save_exp_code adeno_rag_offline_fold0_eval \
  --splits_dir splits/Yifuyuan_strict \
  --text_prompt_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --enable_dynamic_prompt \
  --prompt_pool_path /xiangmu/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv \
  --retrieval_topk 2 --retrieval_temp 0.15 --dynamic_prompt_mix 0.8 \
  --enable_vcp --vcp_beta 0.03 --vcp_dropout 0.1 \
  --enable_rag_rewrite --rag_mode offline \
  --rag_cache_path /xiangmu/ViLMIL/ViLa-MIL-main/results/rag_cache_qwen14b.jsonl \
  --rag_topk 3 --rag_ollama_model qwen2.5:14b-instruct \
  --rag_ollama_url http://localhost:11434/api/generate \
  --rag_temperature 0.2 --rag_max_tokens 256 --rag_timeout_sec 60 \
  --rag_fallback dynamic \
  --save_retrieval_log --retrieval_log_name retrieval_log.csv
```

---

## 7. 参数详解（按功能分组）

### 7.1 数据与任务
- `--task`: 任务名，腺癌任务用 `task_adenocarcinoma`
- `--data_root_dir`: 特征根目录
- `--data_folder_s`: 低倍特征目录名（根目录下）
- `--data_folder_l`: 高倍特征目录名（根目录下）
- `--split_dir`: split目录（建议相对路径）

### 7.2 训练控制
- `--k`: 折数
- `--k_start`, `--k_end`: 训练折区间（含起止）
- `--seed`: 随机种子
- `--lr`: 学习率
- `--early_stopping`: 启用早停
- `--drop_out`: 启用dropout

### 7.3 第1点：动态检索
- `--enable_dynamic_prompt`: 开关
- `--prompt_pool_path`: 句子池CSV路径
- `--retrieval_topk`: 每类每尺度检索top-k
- `--retrieval_temp`: 检索softmax温度
- `--dynamic_prompt_mix`: 动态与静态文本融合比例（1.0=全动态）
- `--dynamic_prompt_warmup_epochs`: 动态检索启用前热身轮数（静态）

### 7.4 第2点：VCP
- `--enable_vcp`: 开关
- `--vcp_beta`: VCP偏移初始缩放系数
- `--vcp_dropout`: VCP MLP dropout
- `--vcp_start_epoch`: 从第几轮开始启用VCP

### 7.5 第3点：RAG + LLM 重写
- `--enable_rag_rewrite`: 开关
- `--rag_mode`: `offline|online|hybrid`
- `--rag_cache_path`: 缓存文件路径（jsonl）
- `--rag_topk`: 传给LLM的证据top-k
- `--rag_ollama_model`: Ollama模型名
- `--rag_ollama_url`: Ollama API地址
- `--rag_temperature`: 生成温度
- `--rag_max_tokens`: 最大生成token
- `--rag_timeout_sec`: HTTP超时
- `--rag_fallback`: `dynamic|static`，RAG失败回退策略
- `--rag_start_epoch`: 从第几轮开始启用RAG

### 7.6 评估附加
- `--save_retrieval_log`: 导出检索日志
- `--retrieval_log_name`: 检索日志文件名

---

## 8. 输入输出说明（I/O）

### 8.1 训练输入
- 一个batch（默认batch=1个slide）
  - `x_s`: 低倍特征 [1, N_s, 512]
  - `x_l`: 高倍特征 [1, N_l, 512]
  - `label`: [1]
  - `slide_id`: 字符串（用于RAG缓存键）

### 8.2 训练输出
- 模型输出
  - `Y_prob`: [1, C]
  - `Y_hat`: [1, 1]
  - `loss`: 标量
- 文件输出
  - `s_{fold}_checkpoint.pt`
  - `split_{fold}_results.pkl`
  - `fold_summary.csv`
  - `result*.csv`
  - `summary*.csv`
  - `epoch_details.csv`

### 8.3 RAG缓存输入/输出
- 输入：`slide_id` + 检索证据 + 类别名
- 输出：`low_rewrite_per_class`, `high_rewrite_per_class`（每类一条）
- 缓存文件：jsonl，每行一个slide记录

---

## 9. Ollama 使用说明

### 9.1 启动与拉取模型
```bash
ollama serve
ollama pull qwen2.5:14b-instruct
```

### 9.2 典型模式选择
- 想稳定与复现：`rag_mode=offline`
- 想边跑边补缓存：`rag_mode=hybrid`
- 想全实时：`rag_mode=online`

建议：先 `hybrid` 生成缓存，再 `offline` 正式复现实验。

---

## 10. 常见问题与排查

1. `FileNotFoundError`（prompt pool）
- 检查 `--prompt_pool_path` 是否写对
- 推荐传绝对路径

2. RAG未生效
- 检查 `--enable_rag_rewrite`
- 检查 `--rag_start_epoch`
- 查看训练日志是否出现 `[RAG]` 状态

3. Hybrid不写缓存
- 检查 Ollama 服务是否可达
- 检查 `--rag_ollama_url`
- 检查模型名是否存在

4. Offline效果异常
- 检查缓存命中率
- 低命中会大量走 fallback

---

## 11. 推荐实验顺序（可直接复现）

1. 静态基线（000）
2. 第1点（100）
3. 第1+2点（101）
4. 第1+3点（110）
5. 第1+2+3点（111）

每组保持相同 seed、相同 fold、相同数据目录，保证可比性。

---

## 12. 变更文件清单（当前已集成）

- `main.py`
- `eval.py`
- `models/model_ViLa_MIL_BiomedCLIP.py`
- `utils/core_utils.py`
- `utils/eval_utils.py`
- `utils/utils.py`
- `utils/rag_prompt_rewriter.py`
- `text_prompt/adenocarcinoma_dynamic_prompt_pool_template.csv`

如需把本手册同步到 README，可将本文件关键章节（第4~第7章）简化后粘贴到 README。
