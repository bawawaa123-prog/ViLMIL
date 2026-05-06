from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
TRAINED = ROOT / "trained_models"
EVAL_ROOT = ROOT / "eval_results"
OUT_DIR = TRAINED / "stage7_1_residual_saf_comparison"

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
    ("Concept-12 SAF learned_gate topk=5 tau=0.07", TRAINED / "adeno_concept12_saf_peps_topk5_tau0.07_s1"),
    ("Concept-12 residual SAF topk=5 tau=0.07 g=0.25", TRAINED / "adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25_s1"),
]

RESIDUAL_EVAL_DIR = EVAL_ROOT / "EVAL_adeno_concept12_residual_saf_peps_topk5_tau0.07_g0.25"


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


def _read_fold_summary(train_dir: Path):
    path = train_dir / "fold_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_fold_delta(peps_dir: Path, saf_dir: Path, residual_dir: Path):
    peps_df = _read_fold_summary(peps_dir)
    saf_df = _read_fold_summary(saf_dir)
    residual_df = _read_fold_summary(residual_dir)
    if peps_df.empty or saf_df.empty or residual_df.empty:
        return pd.DataFrame()

    merged = peps_df[["fold"] + METRICS].merge(
        saf_df[["fold"] + METRICS], on="fold", suffixes=("_peps", "_saf")
    ).merge(
        residual_df[["fold"] + METRICS], on="fold"
    )
    for metric in METRICS:
        merged = merged.rename(columns={metric: f"{metric}_residual"})
        merged[f"{metric}_delta_residual_minus_peps"] = merged[f"{metric}_residual"] - merged[f"{metric}_peps"]
        merged[f"{metric}_delta_residual_minus_saf"] = merged[f"{metric}_residual"] - merged[f"{metric}_saf"]
    ordered = ["fold"]
    for metric in METRICS:
        ordered.extend(
            [
                f"{metric}_delta_residual_minus_peps",
                f"{metric}_delta_residual_minus_saf",
            ]
        )
    return merged[ordered]


def _collect_residual_gate_stats(eval_dir: Path):
    rows = []
    for fold_path in sorted(eval_dir.glob("peps_prompt_analysis_fold*.csv")):
        try:
            fold_num = int(fold_path.stem.split("fold")[-1])
        except ValueError:
            fold_num = -1
        df = pd.read_csv(fold_path)
        required = {"slide_id", "residual_r", "high_coef", "low_coef"}
        if not required.issubset(df.columns):
            continue
        sample_df = df.drop_duplicates(subset=["slide_id"]).copy()
        record = {"fold": fold_num, "num_samples": int(len(sample_df))}
        for col in ["residual_r", "high_coef", "low_coef"]:
            record[f"{col}_min"] = float(sample_df[col].min())
            record[f"{col}_max"] = float(sample_df[col].max())
            record[f"{col}_mean"] = float(sample_df[col].mean())
            record[f"{col}_std"] = float(sample_df[col].std(ddof=0))
        if {"true_label", "pred_label"}.issubset(sample_df.columns):
            sample_df["correct"] = (sample_df["true_label"] == sample_df["pred_label"]).astype(int)
            for correct_value, suffix in [(1, "correct"), (0, "wrong")]:
                subset = sample_df[sample_df["correct"] == correct_value]
                record[f"num_{suffix}"] = int(len(subset))
                if len(subset) > 0:
                    for col in ["residual_r", "high_coef", "low_coef"]:
                        record[f"{col}_{suffix}_mean"] = float(subset[col].mean())
                else:
                    for col in ["residual_r", "high_coef", "low_coef"]:
                        record[f"{col}_{suffix}_mean"] = pd.NA
        rows.append(record)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [_collect_result_summary(experiment, train_dir) for experiment, train_dir in EXPERIMENTS]
    )
    summary_df.to_csv(OUT_DIR / "residual_saf_comparison_summary.csv", index=False)

    fold_delta_df = _build_fold_delta(EXPERIMENTS[0][1], EXPERIMENTS[1][1], EXPERIMENTS[2][1])
    if not fold_delta_df.empty:
        fold_delta_df.to_csv(OUT_DIR / "residual_saf_fold_delta.csv", index=False)

    gate_stats_df = _collect_residual_gate_stats(RESIDUAL_EVAL_DIR)
    if not gate_stats_df.empty:
        gate_stats_df.to_csv(OUT_DIR / "residual_gate_stats.csv", index=False)

    print(f"Saved residual SAF summary to: {OUT_DIR / 'residual_saf_comparison_summary.csv'}")
    if not fold_delta_df.empty:
        print(f"Saved residual SAF fold deltas to: {OUT_DIR / 'residual_saf_fold_delta.csv'}")
    if not gate_stats_df.empty:
        print(f"Saved residual gate stats to: {OUT_DIR / 'residual_gate_stats.csv'}")


if __name__ == "__main__":
    main()
