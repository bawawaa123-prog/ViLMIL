from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
TRAINED = ROOT / "trained_models"
OUT_DIR = TRAINED / "stage7_saf_peps_comparison"

METRICS = [
    "test_auc",
    "test_acc",
    "test_f1",
    "val_auc",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "pr_auc",
]

EXPERIMENTS = [
    ("Concept-12 PEPS topk=5 tau=0.07", TRAINED / "adeno_concept12_peps_topk5_tau0.07_s1"),
    ("Concept-12 SAF-PEPS topk=5 tau=0.07", TRAINED / "adeno_concept12_saf_peps_topk5_tau0.07_s1"),
]


def _collect_result_summary(experiment: str, train_dir: Path):
    row = {"experiment": experiment, "train_dir": str(train_dir), "status": "missing"}
    for metric in METRICS:
        row[f"{metric}_mean"] = pd.NA
        row[f"{metric}_std"] = pd.NA
    result_csv = train_dir / "result.csv"
    if not result_csv.is_file():
        return row

    df = pd.read_csv(result_csv).set_index("metric")
    row["status"] = "ok"
    for metric in METRICS:
        if metric in df.columns:
            row[f"{metric}_mean"] = float(df.loc["mean", metric])
            row[f"{metric}_std"] = float(df.loc["std", metric])
    return row


def _collect_fold_delta(base_dir: Path, saf_dir: Path):
    base_fold = base_dir / "fold_summary.csv"
    saf_fold = saf_dir / "fold_summary.csv"
    if (not base_fold.is_file()) or (not saf_fold.is_file()):
        return pd.DataFrame()

    base_df = pd.read_csv(base_fold)
    saf_df = pd.read_csv(saf_fold)
    merged = base_df[["fold"] + METRICS].merge(saf_df[["fold"] + METRICS], on="fold", suffixes=("_peps", "_saf"))
    for metric in METRICS:
        merged[f"{metric}_delta_saf_minus_peps"] = merged[f"{metric}_saf"] - merged[f"{metric}_peps"]
    delta_cols = ["fold"] + [f"{metric}_delta_saf_minus_peps" for metric in METRICS]
    return merged[delta_cols]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = [_collect_result_summary(experiment, train_dir) for experiment, train_dir in EXPERIMENTS]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "saf_peps_comparison_summary.csv", index=False)

    fold_delta_df = _collect_fold_delta(EXPERIMENTS[0][1], EXPERIMENTS[1][1])
    if not fold_delta_df.empty:
        fold_delta_df.to_csv(OUT_DIR / "saf_peps_fold_delta.csv", index=False)

    print(f"Saved SAF-PEPS summary to: {OUT_DIR / 'saf_peps_comparison_summary.csv'}")
    if not fold_delta_df.empty:
        print(f"Saved fold deltas to: {OUT_DIR / 'saf_peps_fold_delta.csv'}")


if __name__ == "__main__":
    main()
