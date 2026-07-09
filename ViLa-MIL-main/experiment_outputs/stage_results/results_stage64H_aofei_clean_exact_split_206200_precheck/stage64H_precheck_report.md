# Stage64H Precheck Report

1. Current time
   - `2026-07-09 15:45:01 CST +0800`

2. Current server path
   - Git root: `/private/ljh-data/shared/Linux_school/ViLMIL`
   - Project path: `/private/ljh-data/shared/Linux_school/ViLMIL/ViLa-MIL-main`

3. Current Git branch
   - `dev-rce-aofei-split-repro-206200`

4. Git status
   - Pre-report `git status --short`: clean (no output)

5. Current commit
   - `0dc89f2ef77cea8826733c9f2bc5fe79bc9bdc6a`
   - Recent commits:
     - `0dc89f2 Step64B：Import Aofei clean ViLa-MIL-BiomedCLIP baseline into main ViLMIL project`
     - `580140c Project Structure Cleanup and Stage Result Archiving`
     - `26433c2 Step63：Freeze Step57-Step62 RCE innovation results and prepare GitHub checkpoint`
     - `ba3de42 Step62：Final Innovation Consolidation`
     - `46459b4 Step61D：L2H Retrieval Config-G 5-fold formal validation`

6. `data/yiyuan` real path
   - `ls -ld data/yiyuan`: regular directory, not a symlink
   - `readlink -f data/yiyuan`: `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/yiyuan`

7. Low/high h5 counts
   - `features_biomedclip_5x`: `968`
   - `features_biomedclip_20x`: `968`
   - First 5 low-scale files:
     - `data/yiyuan/features_biomedclip_5x/2460239-B2.h5`
     - `data/yiyuan/features_biomedclip_5x/2460242-B2.h5`
     - `data/yiyuan/features_biomedclip_5x/2460399-B2.h5`
     - `data/yiyuan/features_biomedclip_5x/2460404-B3.h5`
     - `data/yiyuan/features_biomedclip_5x/2460405-B2.h5`
   - First 5 high-scale files:
     - `data/yiyuan/features_biomedclip_20x/2460239-B2.h5`
     - `data/yiyuan/features_biomedclip_20x/2460242-B2.h5`
     - `data/yiyuan/features_biomedclip_20x/2460399-B2.h5`
     - `data/yiyuan/features_biomedclip_20x/2460404-B3.h5`
     - `data/yiyuan/features_biomedclip_20x/2460405-B2.h5`

8. `dataset_csv` existence
   - `dataset_csv/all_data.csv`: present

9. `text_prompt` existence
   - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`: present

10. Exact split existence
   - Required path: `splits/task_adenocarcinoma_100_k5_s1`
   - Status: missing
   - Existing split directories found under `splits/`:
     - `splits/Yifuyuan/task_adenocarcinoma_100`
     - `splits/Yifuyuan_strict`
     - `splits/adenocarcinoma/task_adenocarcinoma_strictcv_100`

11. Exact split md5 vs Aofei reference
   - Command `md5sum splits/task_adenocarcinoma_100_k5_s1/splits_*.csv` failed because the directory does not exist.
   - Therefore md5 cannot be verified and cannot match the required Aofei exact split.
   - Expected Aofei md5 values:
     - `splits_0.csv`: `f2795f353d056105e0d115ab9a880cd0`
     - `splits_1.csv`: `d9c1167fc1e3543e15113ce1d6d9c07f`
     - `splits_2.csv`: `af7ad74966cc1b2d074cdd63e22043d5`
     - `splits_3.csv`: `b6c8b85a3830f150ff76d6de06653a19`
     - `splits_4.csv`: `78d2999585478fe94c9e6c37e6454b68`
   - Verdict: not matched / not verifiable

12. Python / CUDA / GPU info
   - Environment used for checks: `conda run -n vila_mil_overlay_rt`
   - Python path: `/opt/conda/envs/vila_mil_overlay_rt/bin/python`
   - Python version: `3.12.6`
   - Torch: `2.4.1+cu124`
   - CUDA available: `True`
   - CUDA device count: `2`
   - CUDA device name: `NVIDIA A30`

13. `numpy` / `scipy` / `torch` / `open_clip` info
   - `numpy`: `2.5.0`
   - `scipy`: `1.14.1`
   - `torch`: `2.4.1+cu124`
   - `pandas`: `2.2.3`
   - `h5py`: import ok
   - `open_clip`: import ok
   - Warning observed during import:
     - `UserWarning: A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy (detected version 2.5.0)`

14. AofeiClean registration check
   - Required files present:
     - `models/model_ViLa_MIL_BiomedCLIP_AofeiClean.py`
     - `models/model_ViLa_MIL_BiomedCLIP.py`
     - `main.py`
     - `utils/core_utils.py`
     - `utils/utils.py`
   - `grep -R "ViLa_MIL_BiomedCLIP_AofeiClean" -n main.py utils/core_utils.py utils/utils.py models` hits:
     - `main.py:45`
     - `utils/core_utils.py:191`
     - `utils/core_utils.py:194`
     - `utils/core_utils.py:207`
     - `utils/core_utils.py:364`
     - `utils/core_utils.py:372`
     - `utils/core_utils.py:403`
     - `utils/utils.py:102`
     - `models/model_ViLa_MIL_BiomedCLIP_AofeiClean.py:305`
   - Verdict: registered

15. `py_compile` status
   - `python -m py_compile main.py`: passed
   - `python -m py_compile utils/core_utils.py`: passed
   - `python -m py_compile utils/utils.py`: passed
   - `python -m py_compile models/model_ViLa_MIL_BiomedCLIP_AofeiClean.py`: passed
   - `python -m py_compile models/model_ViLa_MIL_BiomedCLIP.py`: passed

16. `k_end` inclusive check
   - `main.py:601` contains `end = args.k if args.k_end == -1 else args.k_end + 1`
   - `main.py:618` explicitly states `Note: --k_end is inclusive.`
   - Conclusion: `--k_start 0 --k_end 4` runs 5 folds (`0,1,2,3,4`)

17. Final decision
   - `not runnable`
   - Blocking reason: required exact split directory `splits/task_adenocarcinoma_100_k5_s1` is missing, so the mandated split cannot be used and its md5 cannot be verified against the Aofei exact split baseline.

18. Final training command
   - Not provided because the run is not runnable under the stated requirements.
