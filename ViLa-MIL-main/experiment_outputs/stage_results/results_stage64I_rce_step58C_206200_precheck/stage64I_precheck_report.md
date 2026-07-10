# Stage64I Precheck Report

1. Current Git branch
   - `dev-rce-aofei-split-repro-206200`

2. Current commit
   - `3d720c6b7ee128e862c64814388906f9e87f4c76`
   - `2026-07-09 20:04:53 CST +0800`

3. Git status
   - Pre-report `git status --short`:
     - `?? ViLa-MIL-main/splits/task_adenocarcinoma_100_k5_s1/`

4. Stage58C script existence
   - `scripts/experiments/run_stage58C_residual_constrained_configD_5fold.sh`: present

5. Original Stage58C training command extracted from script
   - Script env defaults:
     - `PYTHON_BIN=/home/ljh/anaconda3/envs/vila_mil/bin/python`
     - `DATA_ROOT_DIR=/xiangmu/data/VILMIL`
     - `RESULTS_DIR=results_stage58C_residual_constrained_configD_5fold`
     - `SEED=1`
     - `MAX_EPOCHS=20`
     - `SPLIT_DIR=splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`
     - `TEXT_PROMPT_PATH=/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv`
     - `CONCEPT12_PATH=/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json`
     - `HF_HUB_OFFLINE=1`
     - `TRANSFORMERS_OFFLINE=1`
   - Extracted raw command:
     - `PYTHONPATH=/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/ljh/anaconda3/envs/vila_mil/bin/python /private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/main.py --task task_adenocarcinoma --model_type RCE_MIL_BiomedCLIP_v2 --mode transformer --data_root_dir /xiangmu/data/VILMIL --data_folder_s features_biomedclip_5x --data_folder_l features_biomedclip_20x --text_prompt_path /private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv --use_concept_prompt_pool --concept_prompt_path /private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json --prompt_ensemble_mode embedding_mean --scale_mode dual --k 5 --k_start 0 --k_end 4 --max_epochs 20 --seed 1 --prototype_number 16 --split_dir splits/adenocarcinoma/task_adenocarcinoma_strictcv_100 --results_dir results_stage58C_residual_constrained_configD_5fold --exp_code rce_v2_rcD_l003_t050_aux020_5fold_e20 --rce_use_logit_calibration --rce_use_concept_prior --rce_use_visual_residual --rce_logit_scale_init 10.0 --rce_concept_prior_strength 1.0 --rce_visual_residual_init 0.05 --rce_use_cross_scale_graph --rce_cross_scale_graph_init 0.1 --rce_cross_scale_graph_norm sqrt --rce_use_residual_constraint --rce_residual_constraint_lambda 0.03 --rce_residual_ratio_target 0.50 --rce_use_concept_aux_loss --rce_concept_aux_loss_weight 0.20`
   - Parameters not explicitly set in script and therefore not added by inference:
     - `lr`
     - `label_frac`
     - `bag_loss`
     - `early_stopping`
     - `drop_out`

6. Original split_dir
   - `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`

7. Exact split existence
   - Relative exact split required by task:
     - `splits/task_adenocarcinoma_100_k5_s1/splits_0.csv`: present
     - `splits/task_adenocarcinoma_100_k5_s1/splits_1.csv`: present
     - `splits/task_adenocarcinoma_100_k5_s1/splits_2.csv`: present
     - `splits/task_adenocarcinoma_100_k5_s1/splits_3.csv`: present
     - `splits/task_adenocarcinoma_100_k5_s1/splits_4.csv`: present
   - User-suggested shared strict split also present:
     - `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/splits/strict/task_adenocarcinoma_100_k5_s1`

8. Exact split md5
   - `splits_0.csv`: `f2795f353d056105e0d115ab9a880cd0`
   - `splits_1.csv`: `d9c1167fc1e3543e15113ce1d6d9c07f`
   - `splits_2.csv`: `af7ad74966cc1b2d074cdd63e22043d5`
   - `splits_3.csv`: `b6c8b85a3830f150ff76d6de06653a19`
   - `splits_4.csv`: `78d2999585478fe94c9e6c37e6454b68`
   - Result:
     - relative exact split matches the known Aofei exact split md5 values
     - user-suggested shared strict split is md5-identical

9. `data/yiyuan` real path
   - `ls -ld data/yiyuan`: regular directory
   - `readlink -f data/yiyuan`: `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/yiyuan`

10. Low/high h5 counts
   - `features_biomedclip_5x`: `968`
   - `features_biomedclip_20x`: `968`

11. RCE model files existence
   - `models/model_RCE_MIL_BiomedCLIP_v2.py`: present
   - `models/model_RCE_MIL_BiomedCLIP.py`: present
   - `main.py`: present
   - `utils/core_utils.py`: present
   - `utils/utils.py`: present

12. RCE registration existence
   - `grep -R "RCE_MIL_BiomedCLIP_v2" -n main.py utils/core_utils.py utils/utils.py models` hits:
     - `main.py:47`
     - `utils/core_utils.py:209`
     - `utils/core_utils.py:215`
     - `utils/core_utils.py:217`
     - `utils/core_utils.py:218`
     - `utils/core_utils.py:366`
     - `models/__pycache__/model_RCE_MIL_BiomedCLIP_v2.cpython-312.pyc`
   - `grep -R "Residual" -n models/model_RCE_MIL_BiomedCLIP_v2.py main.py utils/core_utils.py` returned no text matches

13. `py_compile` status
   - `python -m py_compile main.py`: passed
   - `python -m py_compile utils/core_utils.py`: passed
   - `python -m py_compile utils/utils.py`: passed
   - `python -m py_compile models/model_RCE_MIL_BiomedCLIP_v2.py`: passed

14. Python / CUDA / GPU / package info
   - Check environment: `conda run -n vila_mil_overlay_rt`
   - `torch`: `2.4.1+cu124`
   - `cuda available`: `True`
   - `cuda device count`: `2`
   - `cuda device name`: `NVIDIA A30`
   - `numpy`: `2.5.0`
   - `scipy`: `1.14.1`
   - `open_clip`: import ok
   - Warning:
     - `A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy (detected version 2.5.0)`

15. Final decision
   - `runnable`
   - Notes:
     - The original Stage58C script defaults are not directly runnable on this server because `PYTHON_BIN=/home/ljh/anaconda3/envs/vila_mil/bin/python` and `DATA_ROOT_DIR=/xiangmu/data/VILMIL` do not exist here.
     - The runnable commands below preserve the Stage58C model and RCE flags verbatim, while remapping only server-specific runtime paths to the current 206200 workspace and conda environment.

16. Command A: server-only comparison command
   - `cd /private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main && PYTHONPATH=/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n vila_mil_overlay_rt python main.py --task task_adenocarcinoma --model_type RCE_MIL_BiomedCLIP_v2 --mode transformer --data_root_dir data/yiyuan --data_folder_s features_biomedclip_5x --data_folder_l features_biomedclip_20x --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv --use_concept_prompt_pool --concept_prompt_path dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json --prompt_ensemble_mode embedding_mean --scale_mode dual --k 5 --k_start 0 --k_end 4 --max_epochs 20 --seed 1 --prototype_number 16 --split_dir splits/adenocarcinoma/task_adenocarcinoma_strictcv_100 --results_dir experiment_outputs/stage_results/results_stage64I_rce_step58C_server_only_206200 --exp_code adenocarcinoma_rce_step58C_server_only_206200 --rce_use_logit_calibration --rce_use_concept_prior --rce_use_visual_residual --rce_logit_scale_init 10.0 --rce_concept_prior_strength 1.0 --rce_visual_residual_init 0.05 --rce_use_cross_scale_graph --rce_cross_scale_graph_init 0.1 --rce_cross_scale_graph_norm sqrt --rce_use_residual_constraint --rce_residual_constraint_lambda 0.03 --rce_residual_ratio_target 0.50 --rce_use_concept_aux_loss --rce_concept_aux_loss_weight 0.20`

17. Command B: exact-split comparison command
   - `cd /private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main && PYTHONPATH=/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n vila_mil_overlay_rt python main.py --task task_adenocarcinoma --model_type RCE_MIL_BiomedCLIP_v2 --mode transformer --data_root_dir data/yiyuan --data_folder_s features_biomedclip_5x --data_folder_l features_biomedclip_20x --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv --use_concept_prompt_pool --concept_prompt_path dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json --prompt_ensemble_mode embedding_mean --scale_mode dual --k 5 --k_start 0 --k_end 4 --max_epochs 20 --seed 1 --prototype_number 16 --split_dir splits/task_adenocarcinoma_100_k5_s1 --results_dir experiment_outputs/stage_results/results_stage64I_rce_step58C_exact_split_206200 --exp_code adenocarcinoma_rce_step58C_exact_split_206200 --rce_use_logit_calibration --rce_use_concept_prior --rce_use_visual_residual --rce_logit_scale_init 10.0 --rce_concept_prior_strength 1.0 --rce_visual_residual_init 0.05 --rce_use_cross_scale_graph --rce_cross_scale_graph_init 0.1 --rce_cross_scale_graph_norm sqrt --rce_use_residual_constraint --rce_residual_constraint_lambda 0.03 --rce_residual_ratio_target 0.50 --rce_use_concept_aux_loss --rce_concept_aux_loss_weight 0.20`
