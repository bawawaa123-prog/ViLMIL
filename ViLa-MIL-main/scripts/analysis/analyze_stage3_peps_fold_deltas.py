from __future__ import annotations

from pathlib import Path

import pandas as pd


METRICS = [
    "test_auc",
    "test_acc",
    "test_f1",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "pr_auc",
]

COMPARISONS = [
    ("Concept-12 PEPS topk=5 tau=0.1", "Concept-12 embedding_mean"),
    ("Concept-12 PEPS topk=3 tau=0.1", "Concept-12 embedding_mean"),
    ("Concept-12 PEPS topk=5 tau=0.1", "Concept-12 MLP dynamic_gate"),
]


def _sign_label(value: float, tol: float = 1e-12) -> str:
    if value > tol:
        return "win"
    if value < -tol:
        return "loss"
    return "tie"


def main():
    root = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
    output_dir = root / "trained_models" / "stage3_peps_comparison"
    folds_csv = output_dir / "peps_extended_folds.csv"
    if not folds_csv.is_file():
        raise FileNotFoundError(f"Missing extended fold table: {folds_csv}")

    folds_df = pd.read_csv(folds_csv)
    delta_rows = []
    win_rows = []

    for candidate, reference in COMPARISONS:
        cand_df = folds_df[folds_df["experiment"] == candidate].copy()
        ref_df = folds_df[folds_df["experiment"] == reference].copy()
        merged = cand_df.merge(
            ref_df[["fold"] + METRICS],
            on="fold",
            how="inner",
            suffixes=("_cand", "_ref"),
        )

        for _, row in merged.iterrows():
            delta_row = {
                "candidate_experiment": candidate,
                "reference_experiment": reference,
                "fold": int(row["fold"]),
            }
            for metric in METRICS:
                delta = float(row[f"{metric}_cand"]) - float(row[f"{metric}_ref"])
                delta_row[f"delta_{metric}"] = delta
                delta_row[f"{metric}_outcome"] = _sign_label(delta)
            delta_rows.append(delta_row)

        for metric in METRICS:
            deltas = merged[f"{metric}_cand"] - merged[f"{metric}_ref"]
            wins = int((deltas > 1e-12).sum())
            losses = int((deltas < -1e-12).sum())
            ties = int(len(deltas) - wins - losses)
            win_rows.append(
                {
                    "candidate_experiment": candidate,
                    "reference_experiment": reference,
                    "metric": metric,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "mean_delta": float(deltas.mean()),
                }
            )

    delta_df = pd.DataFrame(delta_rows)
    win_df = pd.DataFrame(win_rows)

    delta_csv = output_dir / "peps_fold_delta_analysis.csv"
    wins_csv = output_dir / "peps_fold_delta_winloss.csv"
    delta_df.to_csv(delta_csv, index=False)
    win_df.to_csv(wins_csv, index=False)

    print(f"Saved fold delta analysis to: {delta_csv}")
    print(f"Saved fold win/loss summary to: {wins_csv}")


if __name__ == "__main__":
    main()
