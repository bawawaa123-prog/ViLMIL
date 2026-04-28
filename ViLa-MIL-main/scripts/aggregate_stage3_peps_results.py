from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


TARGET_METRICS = [
    "test_auc",
    "test_acc",
    "test_f1",
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


def _collect_one(experiment: str, train_dir: str):
    train_path = Path(train_dir)
    row = {
        "experiment": experiment,
        "train_dir": str(train_path),
        "status": "missing_train_dir",
        "num_folds": 0,
    }
    for metric in TARGET_METRICS:
        row[f"{metric}_mean"] = math.nan
        row[f"{metric}_std"] = math.nan

    if not train_path.is_dir():
        return row, pd.DataFrame()

    result_path = train_path / "result.csv"
    fold_path = train_path / "fold_summary.csv"
    if not result_path.is_file() or not fold_path.is_file():
        row["status"] = "missing_result_file"
        return row, pd.DataFrame()

    row["status"] = "ok"
    result_df = pd.read_csv(result_path).set_index("metric")
    for metric in TARGET_METRICS:
        if metric in result_df.columns:
            row[f"{metric}_mean"] = _safe_float(result_df.loc["mean", metric])
            row[f"{metric}_std"] = _safe_float(result_df.loc["std", metric])

    fold_df = pd.read_csv(fold_path)
    row["num_folds"] = int(len(fold_df))
    fold_df.insert(0, "experiment", experiment)
    fold_df.insert(1, "train_dir", str(train_path))
    return row, fold_df


def main():
    root = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
    output_dir = root / "trained_models" / "stage3_peps_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Concept-12 embedding_mean", root / "trained_models" / "adeno_concept12_embedding_mean_s1"),
        ("Concept-12 MLP dynamic_gate", root / "trained_models" / "adeno_concept12_dynamic_gate_s1"),
        ("Concept-12 PEPS topk=1 tau=0.1", root / "trained_models" / "adeno_concept12_peps_topk1_tau0.1_s1"),
        ("Concept-12 PEPS topk=3 tau=0.1", root / "trained_models" / "adeno_concept12_peps_topk3_tau0.1_s1"),
        ("Concept-12 PEPS topk=5 tau=0.1", root / "trained_models" / "adeno_concept12_peps_topk5_tau0.1_s1"),
    ]

    summary_rows = []
    fold_frames = []
    for experiment_name, train_dir in experiments:
        row, fold_df = _collect_one(experiment_name, str(train_dir))
        summary_rows.append(row)
        if not fold_df.empty:
            fold_frames.append(fold_df)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()

    summary_csv = output_dir / "peps_comparison_summary.csv"
    folds_csv = output_dir / "peps_comparison_folds.csv"
    summary_df.to_csv(summary_csv, index=False)
    fold_df.to_csv(folds_csv, index=False)

    display_cols = ["experiment", "status", "num_folds"] + [f"{metric}_mean" for metric in TARGET_METRICS]
    print("\nPEPS comparison summary:")
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")
    print(f"Saved fold-level table to: {folds_csv}")


if __name__ == "__main__":
    main()
