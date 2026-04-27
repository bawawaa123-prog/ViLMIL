from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


TARGET_METRICS = [
    "test_auc",
    "test_f1",
    "test_acc",
    "val_auc",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "pr_auc",
]


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def _pick_result_file(exp_dir: Path) -> Path | None:
    full_path = exp_dir / "result.csv"
    if full_path.is_file():
        return full_path
    partial_paths = sorted(exp_dir.glob("result_partial_*.csv"))
    if partial_paths:
        return partial_paths[-1]
    return None


def _pick_fold_summary_file(exp_dir: Path) -> Path | None:
    full_path = exp_dir / "fold_summary.csv"
    if full_path.is_file():
        return full_path
    return None


def _pick_eval_result_file(eval_dir: Path) -> Path | None:
    full_path = eval_dir / "result.csv"
    return full_path if full_path.is_file() else None


def _pick_eval_fold_file(eval_dir: Path) -> Path | None:
    full_path = eval_dir / "fold_metrics.csv"
    return full_path if full_path.is_file() else None


def _load_metric_summary_csv(result_path: Path) -> dict:
    df = pd.read_csv(result_path)
    if "metric" not in df.columns:
        raise ValueError(f"Missing 'metric' column in {result_path}")
    df = df.set_index("metric")
    row = {}
    for metric_name in TARGET_METRICS:
        row[f"{metric_name}_mean"] = math.nan
        row[f"{metric_name}_std"] = math.nan
        if metric_name in df.columns:
            if "mean" in df.index:
                row[f"{metric_name}_mean"] = _safe_float(df.loc["mean", metric_name])
            if "std" in df.index:
                row[f"{metric_name}_std"] = _safe_float(df.loc["std", metric_name])
    return row


def _fill_missing_summary_from_df(row: dict, df: pd.DataFrame):
    if df.empty:
        return row

    for metric_name in TARGET_METRICS:
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        if metric_name not in df.columns:
            continue
        series = pd.to_numeric(df[metric_name], errors="coerce").dropna()
        if series.empty:
            continue
        if math.isnan(_safe_float(row.get(mean_key, math.nan))):
            row[mean_key] = float(series.mean())
        if math.isnan(_safe_float(row.get(std_key, math.nan))):
            row[std_key] = float(series.std(ddof=0))
    return row


def _merge_fold_metrics(train_fold_df: pd.DataFrame, eval_fold_df: pd.DataFrame) -> pd.DataFrame:
    if train_fold_df.empty:
        return eval_fold_df.copy()
    if eval_fold_df.empty:
        return train_fold_df.copy()

    train_fold_df = train_fold_df.copy()
    eval_fold_df = eval_fold_df.copy()

    train_folds = pd.to_numeric(train_fold_df["fold"], errors="coerce").dropna()
    eval_folds = pd.to_numeric(eval_fold_df["fold"], errors="coerce").dropna()
    if not train_folds.empty and not eval_folds.empty:
        if train_folds.min() == 1 and eval_folds.min() == 0 and len(train_folds) == len(eval_folds):
            eval_fold_df["fold"] = pd.to_numeric(eval_fold_df["fold"], errors="coerce") + 1

    merged = train_fold_df.merge(
        eval_fold_df,
        on="fold",
        how="outer",
        suffixes=("", "_eval"),
    )
    for metric_name in TARGET_METRICS:
        eval_col = f"{metric_name}_eval"
        if eval_col not in merged.columns:
            continue
        if metric_name not in merged.columns:
            merged[metric_name] = merged[eval_col]
        else:
            merged[metric_name] = merged[metric_name].where(merged[metric_name].notna(), merged[eval_col])
        merged = merged.drop(columns=[eval_col])
    return merged


def _collect_one(experiment_name: str, train_dir_str: str, eval_dir_str: str | None):
    train_dir = Path(train_dir_str)
    eval_dir = Path(eval_dir_str) if eval_dir_str else None

    row = {
        "experiment": experiment_name,
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir) if eval_dir is not None else "",
        "status": "missing_train_dir",
        "train_result_file": "",
        "train_fold_file": "",
        "eval_result_file": "",
        "eval_fold_file": "",
        "num_folds": 0,
    }
    for metric_name in TARGET_METRICS:
        row[f"{metric_name}_mean"] = math.nan
        row[f"{metric_name}_std"] = math.nan

    if not train_dir.is_dir():
        return row, pd.DataFrame()

    row["status"] = "ok"

    train_result_path = _pick_result_file(train_dir)
    if train_result_path is not None:
        row["train_result_file"] = str(train_result_path)
        row.update(_load_metric_summary_csv(train_result_path))

    train_fold_path = _pick_fold_summary_file(train_dir)
    train_fold_df = pd.read_csv(train_fold_path) if train_fold_path is not None else pd.DataFrame()
    if train_fold_path is not None:
        row["train_fold_file"] = str(train_fold_path)

    eval_fold_df = pd.DataFrame()
    if eval_dir is not None and eval_dir.is_dir():
        eval_result_path = _pick_eval_result_file(eval_dir)
        if eval_result_path is not None:
            row["eval_result_file"] = str(eval_result_path)
            eval_summary = _load_metric_summary_csv(eval_result_path)
            for key, value in eval_summary.items():
                if math.isnan(_safe_float(row.get(key, math.nan))) and not math.isnan(_safe_float(value)):
                    row[key] = value

        eval_fold_path = _pick_eval_fold_file(eval_dir)
        if eval_fold_path is not None:
            row["eval_fold_file"] = str(eval_fold_path)
            eval_fold_df = pd.read_csv(eval_fold_path)

    fold_df = _merge_fold_metrics(train_fold_df, eval_fold_df)
    if not fold_df.empty:
        row["num_folds"] = int(len(fold_df))
        row = _fill_missing_summary_from_df(row, fold_df)
        fold_df.insert(0, "experiment", experiment_name)
        fold_df.insert(1, "train_dir", str(train_dir))
        fold_df.insert(2, "eval_dir", str(eval_dir) if eval_dir is not None else "")

    return row, fold_df


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate Stage2 concept-pool size sweep results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/stage2_size_sweep_comparison",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1",
    )
    parser.add_argument(
        "--concept6-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adeno_stage2_concept_mean_s1",
    )
    parser.add_argument(
        "--concept10-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adeno_concept10_embedding_mean_s1",
    )
    parser.add_argument(
        "--concept12-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adeno_concept12_embedding_mean_s1",
    )
    parser.add_argument(
        "--concept14-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adeno_concept14_embedding_mean_s1",
    )
    parser.add_argument(
        "--baseline-eval-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/eval_results/EVAL_adeno_baseline_metrics_supplement",
    )
    parser.add_argument(
        "--concept6-eval-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/eval_results/EVAL_adeno_concept6_metrics_supplement",
    )
    parser.add_argument("--concept10-eval-dir", type=str, default="")
    parser.add_argument("--concept12-eval-dir", type=str, default="")
    parser.add_argument("--concept14-eval-dir", type=str, default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    experiments = [
        ("BiomedCLIP static prompt baseline", args.baseline_dir, args.baseline_eval_dir),
        ("Concept-6 embedding_mean", args.concept6_dir, args.concept6_eval_dir),
        ("Concept-10 embedding_mean", args.concept10_dir, args.concept10_eval_dir),
        ("Concept-12 embedding_mean", args.concept12_dir, args.concept12_eval_dir),
        ("Concept-14 embedding_mean", args.concept14_dir, args.concept14_eval_dir),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fold_frames = []
    for experiment_name, train_dir, eval_dir in experiments:
        row, fold_df = _collect_one(experiment_name, train_dir, eval_dir)
        summary_rows.append(row)
        if not fold_df.empty:
            fold_frames.append(fold_df)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()

    summary_csv = output_dir / "size_sweep_comparison_summary.csv"
    fold_csv = output_dir / "size_sweep_comparison_folds.csv"
    summary_df.to_csv(summary_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)

    display_cols = [
        "experiment",
        "status",
        "num_folds",
        "test_auc_mean",
        "test_acc_mean",
        "test_f1_mean",
        "val_auc_mean",
        "balanced_acc_mean",
        "sensitivity_mean",
        "specificity_mean",
        "pr_auc_mean",
    ]

    print("\nSize sweep summary:")
    print(summary_df.reindex(columns=display_cols).to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")
    print(f"Saved fold-level table to: {fold_csv}")


if __name__ == "__main__":
    main()
