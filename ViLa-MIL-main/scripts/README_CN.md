# scripts 目录说明

当前 `scripts/` 已按用途拆分为 3 类：

## 1. `scripts/experiments/`

放直接运行实验的脚本：

- `run_stage2_dynamic_gate_ablation.sh`
- `run_stage2_prompt_pool_4way.sh`
- `run_stage2_prompt_pool_size_sweep.sh`
- `run_stage3_peps_5way.sh`
- `run_stage3_peps_tau_sweep.sh`
- `run_stage6_sap_peps.sh`
- `run_stage7_saf_peps.sh`
- `run_final_dcp_vila_analysis.sh`

## 2. `scripts/analysis/`

放结果汇总、差值分析、报告生成脚本：

- `aggregate_*`
- `analyze_*`
- `build_*`

## 3. `scripts/data_tools/`

放 patch 裁切、文本特征提取等辅助脚本：

- `crop_patches_from_coords.py`
- `crop_patches_grid.py`
- `extract_modified_text_features.py`

## 使用建议

- 要跑实验：去 `scripts/experiments/`
- 要做对比表：去 `scripts/analysis/`
- 要处理 patch 或辅助数据：去 `scripts/data_tools/`
