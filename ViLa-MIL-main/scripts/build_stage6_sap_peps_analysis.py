from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
TRAINED = ROOT / "trained_models"
EVALS = ROOT / "eval_results"
OUT_DIR = TRAINED / "stage6_sap_peps_comparison"

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
    ("Concept-12 SAP-PEPS topk=5 tau=0.07", TRAINED / "adeno_concept12_sap_peps_topk5_tau0.07_s1"),
    ("Concept-12 PEPS topk=5 tau=0.1", TRAINED / "adeno_concept12_peps_topk5_tau0.1_s1"),
    ("Concept-12 SAP-PEPS topk=5 tau=0.1", TRAINED / "adeno_concept12_sap_peps_topk5_tau0.1_s1"),
]

EVAL_EXPORTS = {
    "Concept-12 PEPS topk=5 tau=0.07": EVALS / "EVAL_adeno_concept12_peps_topk5_tau0.07_sap_compare",
    "Concept-12 SAP-PEPS topk=5 tau=0.07": EVALS / "EVAL_adeno_concept12_sap_peps_topk5_tau0.07",
    "Concept-12 PEPS topk=5 tau=0.1": EVALS / "EVAL_adeno_concept12_peps_topk5_tau0.1_sap_compare",
    "Concept-12 SAP-PEPS topk=5 tau=0.1": EVALS / "EVAL_adeno_concept12_sap_peps_topk5_tau0.1",
}


def _collect_result_summary(experiment: str, train_dir: Path):
    row = {"experiment": experiment, "train_dir": str(train_dir), "status": "missing"}
    for metric in METRICS:
        row[f"{metric}_mean"] = pd.NA
        row[f"{metric}_std"] = pd.NA
    if not (train_dir / "result.csv").is_file():
        return row
    result_df = pd.read_csv(train_dir / "result.csv").set_index("metric")
    row["status"] = "ok"
    for metric in METRICS:
        if metric in result_df.columns:
            row[f"{metric}_mean"] = float(result_df.loc["mean", metric])
            row[f"{metric}_std"] = float(result_df.loc["std", metric])
    return row


def _collect_spatial_summary(experiment: str, eval_dir: Path):
    rows = []
    if not eval_dir.is_dir():
        return rows
    file_pattern = "sap_peps_prompt_analysis_fold*.csv" if "SAP-PEPS" in experiment else "peps_prompt_analysis_fold*.csv"
    for csv_path in sorted(eval_dir.glob(file_pattern)):
        fold = int(csv_path.stem.split("fold")[-1]) + 1
        df = pd.read_csv(csv_path)
        for scale_name, sub_df in df.groupby("scale"):
            rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "scale": scale_name,
                    "semantic_evidence_mean": float(sub_df["semantic_evidence_mean"].mean()),
                    "semantic_evidence_std": float(sub_df["semantic_evidence_std"].mean()),
                    "spatial_score_mean": float(sub_df["spatial_score_mean"].mean()),
                    "spatial_score_std": float(sub_df["spatial_score_std"].mean()),
                    "final_evidence_mean": float(sub_df["final_evidence_mean"].mean()),
                    "final_evidence_std": float(sub_df["final_evidence_std"].mean()),
                    "topk_proto_mean_dist_mean": float(sub_df["topk_proto_mean_dist_mean"].mean()),
                    "topk_proto_mean_dist_std": float(sub_df["topk_proto_mean_dist_std"].mean()),
                }
            )
    return rows


def _markdown_table(df: pd.DataFrame):
    if df.empty:
        return "_No data available._"
    tmp = df.copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].map(lambda x: f"{x:.6f}" if isinstance(x, float) else ("NA" if pd.isna(x) else str(x)))
    headers = list(tmp.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in tmp.values.tolist())
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = [_collect_result_summary(exp, path) for exp, path in EXPERIMENTS]
    summary_df = pd.DataFrame(summary_rows)

    spatial_rows = []
    for exp, _ in EXPERIMENTS:
        spatial_rows.extend(_collect_spatial_summary(exp, EVAL_EXPORTS[exp]))
    spatial_df = pd.DataFrame(spatial_rows)

    summary_csv = OUT_DIR / "sap_peps_comparison_summary.csv"
    spatial_csv = OUT_DIR / "sap_peps_spatial_stats.csv"
    report_md = OUT_DIR / "sap_peps_report.md"

    summary_df.to_csv(summary_csv, index=False)
    spatial_df.to_csv(spatial_csv, index=False)

    lines = []
    lines.append("# Stage6 SAP-PEPS Comparison")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    metric_cols = ["experiment", "status"] + [f"{m}_mean" for m in METRICS]
    lines.append(_markdown_table(summary_df[metric_cols]))
    lines.append("")
    lines.append("## Spatial Diagnostics")
    lines.append("")
    lines.append(_markdown_table(spatial_df))

    if not spatial_df.empty:
        pivot = spatial_df.groupby(["experiment", "scale"], as_index=False)[
            ["spatial_score_mean", "final_evidence_mean", "topk_proto_mean_dist_mean"]
        ].mean()
        lines.append("")
        lines.append("## Aggregated Spatial Effect")
        lines.append("")
        lines.append(_markdown_table(pivot))

    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved SAP-PEPS metric summary to: {summary_csv}")
    print(f"Saved SAP-PEPS spatial stats to: {spatial_csv}")
    print(f"Saved SAP-PEPS report to: {report_md}")


if __name__ == "__main__":
    main()
