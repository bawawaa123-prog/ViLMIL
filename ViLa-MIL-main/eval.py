from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *
from utils.metric_utils import summarize_metric_list
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--data_root_dir", type=str, default=None, help="data directory")
parser.add_argument("--data_folder_s", type=str, default=None, help="dir under data directory")
parser.add_argument("--data_folder_l", type=str, default=None, help="dir under data directory")
parser.add_argument("--results_dir", type=str, default="./results")
parser.add_argument("--save_exp_code", type=str, default=None, help="experiment code to save eval results")
parser.add_argument(
    "--models_exp_code",
    type=str,
    default=None,
    help="experiment code to load trained models (directory under results_dir containing model checkpoints",
)
parser.add_argument(
    "--splits_dir",
    type=str,
    default=None,
    help="splits directory, if using custom splits other than what matches the task (default: None)",
)
parser.add_argument("--model_size", type=str, choices=["small", "big"], default="small", help="size of model")
parser.add_argument(
    "--model_type",
    type=str,
    choices=["ViLa_MIL", "ViLa_MIL_BiomedCLIP"],
    default="ViLa_MIL",
)
parser.add_argument("--mode", type=str, choices=["transformer"], default="transformer")
parser.add_argument("--drop_out", action="store_true", default=False, help="whether model uses dropout")
parser.add_argument("--k", type=int, default=10, help="number of folds (default: 10)")
parser.add_argument("--k_start", type=int, default=-1, help="start fold (default: -1, last fold)")
parser.add_argument("--k_end", type=int, default=-1, help="end fold (default: -1, first fold)")
parser.add_argument("--fold", type=int, default=-1, help="single fold to evaluate")
parser.add_argument(
    "--micro_average",
    action="store_true",
    default=False,
    help="use micro_average instead of macro_average for multiclass AUC",
)
parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
parser.add_argument("--task", type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--concept_prompt_path", type=str, default=None)
parser.add_argument("--use_concept_prompt_pool", action="store_true", default=False)
parser.add_argument(
    "--prompt_ensemble_mode",
    type=str,
    choices=["embedding_mean", "logit_mean", "dynamic_gate"],
    default="embedding_mean",
)
parser.add_argument("--use_dynamic_prompt_gate", action="store_true", default=False)
parser.add_argument("--dynamic_gate_hidden_dim", type=int, default=256)
parser.add_argument("--dynamic_gate_residual_mean", action="store_true", default=False)
parser.add_argument("--prompt_dropout", type=float, default=0.0)
parser.add_argument("--prototype_number", type=int, default=16, help="number of prototypes (default: 16)")
parser.add_argument(
    "--finetune_text_encoder",
    action="store_true",
    default=False,
    help="(BiomedCLIP) finetune text encoder parameters",
)
parser.add_argument(
    "--text_finetune_mode",
    type=str,
    choices=["proj", "last", "full"],
    default="proj",
    help="(BiomedCLIP) finetune scope for text encoder",
)
parser.add_argument(
    "--text_unfreeze_last_n",
    type=int,
    default=2,
    help="(BiomedCLIP) used when --text_finetune_mode=last",
)

args = parser.parse_args()


def _resolve_text_prompt_path(path):
    if not path:
        return path
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join("text_prompt", path),
        os.path.join(os.path.dirname(__file__), "text_prompt", path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return path


def _resolve_concept_prompt_path(path):
    if not path:
        return path
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join("dataset_csv", path),
        os.path.join(os.path.dirname(__file__), "dataset_csv", path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return path


def _load_text_prompts(path):
    if not path:
        return None

    try:
        df_tp = pd.read_csv(path)
        cols = [c.strip().lower() for c in df_tp.columns]
        low_prompts = []
        high_prompts = []
        if "low_resolution_description" in cols and "high_resolution_description" in cols:
            low_idx = cols.index("low_resolution_description")
            high_idx = cols.index("high_resolution_description")
            low_prompts = df_tp.iloc[:, low_idx].astype(str).fillna("").tolist()
            high_prompts = df_tp.iloc[:, high_idx].astype(str).fillna("").tolist()
        return list(map(str, low_prompts)) + list(map(str, high_prompts))
    except Exception:
        arr = pd.read_csv(path, header=None).values
        return [str(x) for x in arr.reshape(-1).tolist()]


args.text_prompt_path = _resolve_text_prompt_path(args.text_prompt_path)
args.concept_prompt_path = _resolve_concept_prompt_path(args.concept_prompt_path)
args.text_prompt = _load_text_prompts(args.text_prompt_path)

args.save_dir = os.path.join("./eval_results", "EVAL_" + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

settings = {
    "task": args.task,
    "split": args.split,
    "save_dir": args.save_dir,
    "models_dir": args.models_dir,
    "model_type": args.model_type,
    "mode": args.mode,
    "drop_out": args.drop_out,
    "model_size": args.model_size,
    "use_concept_prompt_pool": args.use_concept_prompt_pool,
    "concept_prompt_path": args.concept_prompt_path,
    "prompt_ensemble_mode": args.prompt_ensemble_mode,
    "use_dynamic_prompt_gate": args.use_dynamic_prompt_gate,
    "dynamic_gate_hidden_dim": args.dynamic_gate_hidden_dim,
    "dynamic_gate_residual_mean": args.dynamic_gate_residual_mean,
    "prompt_dropout": args.prompt_dropout,
}

with open(os.path.join(args.save_dir, f"eval_experiment_{args.save_exp_code}.txt"), "w") as f:
    print(settings, file=f)

print(settings)

if args.task == "task_tcga_rcc_subtyping":
    args.n_classes = 3
    args.class_names = ["CCRCC", "PRCC", "CRCC"]
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/TCGA_RCC_subtyping.csv",
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(args.class_names)},
        patient_strat=False,
        ignore=[],
    )
elif args.task == "task_tcga_lung_subtyping":
    args.n_classes = 2
    args.class_names = ["LUAD", "LUSC"]
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/TCGA_Lung_subtyping.csv",
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(args.class_names)},
        patient_strat=False,
        ignore=[],
    )
elif args.task == "task_adenocarcinoma":
    args.n_classes = 2
    args.class_names = ["Adenocarcinoma", "NonAdenocarcinoma"]
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/all_data.csv",
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(args.class_names)},
        patient_strat=False,
        ignore=[],
    )
else:
    raise NotImplementedError

if args.use_concept_prompt_pool and not args.concept_prompt_path:
    raise ValueError("--use_concept_prompt_pool is set but --concept_prompt_path is missing.")
if args.prompt_ensemble_mode == "dynamic_gate" and not args.use_concept_prompt_pool:
    raise ValueError("--prompt_ensemble_mode dynamic_gate requires --use_concept_prompt_pool.")
if args.use_dynamic_prompt_gate and args.prompt_ensemble_mode != "dynamic_gate":
    raise ValueError("--use_dynamic_prompt_gate requires --prompt_ensemble_mode dynamic_gate.")

start = 0 if args.k_start == -1 else args.k_start
end = args.k if args.k_end == -1 else args.k_end

available_folds = []
for i in range(start, end):
    ckpt_path = os.path.join(args.models_dir, f"s_{i}_checkpoint.pt")
    split_path = os.path.join(args.splits_dir, f"splits_{i}.csv")
    if os.path.isfile(ckpt_path) and os.path.isfile(split_path):
        available_folds.append(i)
    else:
        missing = []
        if not os.path.isfile(ckpt_path):
            missing.append("ckpt")
        if not os.path.isfile(split_path):
            missing.append("split")
        print(f"[Warn] fold {i} skipped (missing {','.join(missing)}): {ckpt_path}, {split_path}")

if args.fold != -1:
    if args.fold in available_folds:
        available_folds = [args.fold]
    else:
        print(f"指定的fold {args.fold} 不存在有效的模型或split文件，程序退出。")
        exit(1)

if not available_folds:
    print("没有检测到任何可用的模型权重和split文件，程序退出。")
    exit(1)

ckpt_paths = [os.path.join(args.models_dir, f"s_{fold}_checkpoint.pt") for fold in available_folds]
datasets_id = {"train": 0, "val": 1, "test": 2, "all": -1}

if __name__ == "__main__":
    total_start_time = time.time()

    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []
    all_balanced_acc = []
    all_sensitivity = []
    all_specificity = []
    all_pr_auc = []
    all_true = []
    all_pred = []
    timing_records = []
    fold_metrics_records = []

    print(f"\n共检测到 {len(available_folds)} 个有效折：{available_folds}")
    for ckpt_idx, current_fold in enumerate(tqdm(available_folds, desc="Overall Progress", ncols=80)):
        fold_start_time = time.time()
        print(f"\n>>> Processing Fold {current_fold}...")

        ckpt_path = ckpt_paths[ckpt_idx]
        split_path = os.path.join(args.splits_dir, f"splits_{current_fold}.csv")
        if os.path.isfile(ckpt_path):
            print(f"✔️ 成功检测到权重文件: {ckpt_path}")
        else:
            print(f"❌ 未找到权重文件: {ckpt_path}")
            continue

        split_dataset = dataset.return_splits(
            from_id=False,
            csv_path=split_path,
        )[datasets_id[args.split]]

        model, patient_results, metrics, df, each_class_acc, prompt_export_df = eval(
            args.mode,
            split_dataset,
            args,
            ckpt_path,
        )

        fold_duration = time.time() - fold_start_time
        timing_records.append({"fold": current_fold, "seconds": round(fold_duration, 2)})
        print(f"Fold {current_fold} completed in {fold_duration:.2f}s")

        all_results.append(df)
        all_auc.append(metrics["auc"])
        all_acc.append(metrics["acc"])
        all_f1.append(metrics["f1"])
        all_balanced_acc.append(metrics["balanced_acc"])
        all_sensitivity.append(metrics["sensitivity"])
        all_specificity.append(metrics["specificity"])
        all_pr_auc.append(metrics["pr_auc"])
        all_true.extend(df["Y"].tolist())
        all_pred.extend(df["Y_hat"].tolist())
        fold_metrics_records.append(
            {
                "fold": current_fold,
                "test_auc": metrics["auc"],
                "test_f1": metrics["f1"],
                "test_acc": metrics["acc"],
                "balanced_acc": metrics["balanced_acc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "pr_auc": metrics["pr_auc"],
            }
        )
        if prompt_export_df is not None and not prompt_export_df.empty:
            prompt_export_path = os.path.join(args.save_dir, f"prompt_weight_analysis_fold{current_fold}.csv")
            prompt_export_df.to_csv(prompt_export_path, index=False)
            print(f"Saved prompt diagnostics to: {prompt_export_path}")

    if not all_results:
        print("没有成功完成任何折的评估，程序退出。")
        exit(1)

    summary_df = pd.concat(all_results, axis=0, ignore_index=True)
    summary_path = os.path.join(args.save_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    fold_metrics_path = os.path.join(args.save_dir, "fold_metrics.csv")
    pd.DataFrame(fold_metrics_records).to_csv(fold_metrics_path, index=False)

    test_auc_mean, test_auc_std = summarize_metric_list(all_auc)
    test_f1_mean, test_f1_std = summarize_metric_list(all_f1)
    test_acc_mean, test_acc_std = summarize_metric_list(all_acc)
    bal_acc_mean, bal_acc_std = summarize_metric_list(all_balanced_acc)
    sens_mean, sens_std = summarize_metric_list(all_sensitivity)
    spec_mean, spec_std = summarize_metric_list(all_specificity)
    pr_auc_mean, pr_auc_std = summarize_metric_list(all_pr_auc)

    result_df = pd.DataFrame(
        {
            "metric": ["mean", "std"],
            "test_auc": [test_auc_mean, test_auc_std],
            "test_f1": [test_f1_mean, test_f1_std],
            "test_acc": [test_acc_mean, test_acc_std],
            "balanced_acc": [bal_acc_mean, bal_acc_std],
            "sensitivity": [sens_mean, sens_std],
            "specificity": [spec_mean, spec_std],
            "pr_auc": [pr_auc_mean, pr_auc_std],
        }
    )
    result_path = os.path.join(args.save_dir, "result.csv")
    result_df.to_csv(result_path, index=False)

    if timing_records:
        timing_df = pd.DataFrame(timing_records)
        timing_df.to_csv(os.path.join(args.save_dir, "timing.csv"), index=False)

    total_duration = time.time() - total_start_time
    print("\nEvaluation finished!")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved fold metrics to: {fold_metrics_path}")
    print(f"Saved aggregate result to: {result_path}")
    print(
        f"Mean AUC={test_auc_mean:.4f}, Mean ACC={test_acc_mean:.4f}, Mean F1={test_f1_mean:.4f}, "
        f"Balanced ACC={bal_acc_mean:.4f}, Sensitivity={sens_mean:.4f}, "
        f"Specificity={spec_mean:.4f}, PR-AUC={pr_auc_mean:.4f}"
    )
    print(f"Total time: {total_duration / 60:.2f} minutes")
