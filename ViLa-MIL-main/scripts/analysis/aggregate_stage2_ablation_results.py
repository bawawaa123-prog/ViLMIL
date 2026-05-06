from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import pandas as pd


DEFAULT_EXPERIMENTS = [
    (
        "BiomedCLIP static prompt baseline",
        "/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1",
    ),
    (
        "Concept-6 embedding_mean",
        "/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adeno_stage2_concept_mean_s1",
    ),
    (
        "Concept-10 embedding_mean",
        "/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adeno_concept10_embedding_mean_s1",
    ),
    (
        "Concept-10 logit_mean",
        "/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/adeno_concept10_logit_mean_s1",
    ),
]

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


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def _load_result_summary(result_path: Path) -> dict:
    df = pd.read_csv(result_path)
    if "metric" not in df.columns:
        raise ValueError(f"Missing 'metric' column in {result_path}")

    df = df.set_index("metric")
    row = {
        "result_file": str(result_path),
    }
    for metric_name in TARGET_METRICS:
        row[f"{metric_name}_mean"] = math.nan
        row[f"{metric_name}_std"] = math.nan
        if metric_name in df.columns:
            if "mean" in df.index:
                row[f"{metric_name}_mean"] = _safe_float(df.loc["mean", metric_name])
            if "std" in df.index:
                row[f"{metric_name}_std"] = _safe_float(df.loc["std", metric_name])
    return row


def _load_fold_summary(fold_summary_path: Path, experiment_name: str, exp_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(fold_summary_path)
    df.insert(0, "experiment", experiment_name)
    df.insert(1, "source_dir", str(exp_dir))
    df.insert(2, "fold_summary_file", str(fold_summary_path))
    return df


def _fill_missing_summary_from_folds(row: dict, fold_df: pd.DataFrame):
    if fold_df.empty:
        return row

    for metric_name in TARGET_METRICS:
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        if metric_name not in fold_df.columns:
            continue

        series = pd.to_numeric(fold_df[metric_name], errors="coerce").dropna()
        if series.empty:
            continue

        if math.isnan(_safe_float(row.get(mean_key, math.nan))):
            row[mean_key] = float(series.mean())
        if math.isnan(_safe_float(row.get(std_key, math.nan))):
            row[std_key] = float(series.std(ddof=0))
    return row


def _collect_one(experiment_name: str, exp_dir_str: str):
    exp_dir = Path(exp_dir_str)
    row = {
        "experiment": experiment_name,
        "source_dir": str(exp_dir),
        "status": "missing_dir",
        "result_file": "",
        "fold_summary_file": "",
        "num_folds": 0,
    }
    for metric_name in TARGET_METRICS:
        row[f"{metric_name}_mean"] = math.nan
        row[f"{metric_name}_std"] = math.nan

    fold_df = pd.DataFrame()

    if not exp_dir.is_dir():
        return row, fold_df

    row["status"] = "ok"

    result_path = _pick_result_file(exp_dir)
    if result_path is not None:
        row.update(_load_result_summary(result_path))
    else:
        row["status"] = "missing_result"

    fold_summary_path = _pick_fold_summary_file(exp_dir)
    if fold_summary_path is not None:
        row["fold_summary_file"] = str(fold_summary_path)
        fold_df = _load_fold_summary(fold_summary_path, experiment_name, exp_dir)
        row["num_folds"] = int(len(fold_df))
        row = _fill_missing_summary_from_folds(row, fold_df)
    else:
        row["status"] = "missing_fold_summary" if row["status"] == "ok" else row["status"]

    return row, fold_df


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate 4-way Stage2 ablation results into summary tables.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/xiangmu/ViLMIL/ViLa-MIL-main/trained_models/stage2_ablation_comparison",
        help="Directory to save the aggregated tables.",
    )
    parser.add_argument("--baseline-dir", type=str, default=DEFAULT_EXPERIMENTS[0][1])
    parser.add_argument("--concept6-dir", type=str, default=DEFAULT_EXPERIMENTS[1][1])
    parser.add_argument("--concept10-embed-dir", type=str, default=DEFAULT_EXPERIMENTS[2][1])
    parser.add_argument("--concept10-logit-dir", type=str, default=DEFAULT_EXPERIMENTS[3][1])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    experiments = [
        ("BiomedCLIP static prompt baseline", args.baseline_dir),
        ("Concept-6 embedding_mean", args.concept6_dir),
        ("Concept-10 embedding_mean", args.concept10_embed_dir),
        ("Concept-10 logit_mean", args.concept10_logit_dir),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fold_frames = []
    for experiment_name, exp_dir in experiments:
        row, fold_df = _collect_one(experiment_name, exp_dir)
        summary_rows.append(row)
        if not fold_df.empty:
            fold_frames.append(fold_df)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()

    summary_csv = output_dir / "ablation_comparison_summary.csv"
    fold_csv = output_dir / "ablation_comparison_folds.csv"
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
    display_df = summary_df.reindex(columns=display_cols)

    print("\nAblation summary:")
    print(display_df.to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")
    print(f"Saved fold-level table to: {fold_csv}")


if __name__ == "__main__":
    main()
