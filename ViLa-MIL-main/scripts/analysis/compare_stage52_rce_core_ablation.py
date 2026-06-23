from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_MD = Path("docs/stage52_rce_core_ablation_comparison.md")
DEFAULT_OUTPUT_CSV = Path("docs/stage52_rce_core_ablation_comparison.csv")
METRICS = ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
RUNS = [
    {
        "variant": "stage23_history",
        "label": "Historical Stage23 full RCE",
        "path": Path("results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv"),
        "compare_to_stage23": False,
    },
    {
        "variant": "full",
        "label": "Step52 full RCE",
        "path": Path("results_stage52_rce_core_ablation/full_rce_v4_csg_rq16_5fold_e20_s1/result.csv"),
        "compare_to_stage23": True,
    },
    {
        "variant": "wo_csg",
        "label": "Step52 w/o CSG",
        "path": Path("results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1/result.csv"),
        "compare_to_stage23": True,
    },
    {
        "variant": "wo_concept_prior",
        "label": "Step52 w/o concept prior",
        "path": Path("results_stage52_rce_core_ablation/wo_concept_prior_5fold_e20_s1/result.csv"),
        "compare_to_stage23": True,
    },
    {
        "variant": "wo_visual_residual",
        "label": "Step52 w/o visual residual",
        "path": Path("results_stage52_rce_core_ablation/wo_visual_residual_5fold_e20_s1/result.csv"),
        "compare_to_stage23": True,
    },
    {
        "variant": "wo_logit_calibration",
        "label": "Step52 w/o logit calibration",
        "path": Path("results_stage52_rce_core_ablation/wo_logit_calibration_5fold_e20_s1/result.csv"),
        "compare_to_stage23": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Step52 RCE core ablation results against Stage23.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def read_mean_row(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "metric" not in df.columns:
        return None
    mean_rows = df[df["metric"] == "mean"]
    if mean_rows.empty:
        return None
    row = mean_rows.iloc[0]
    return {metric: float(row[metric]) for metric in METRICS}


def markdown_table(df: pd.DataFrame) -> str:
    safe_df = df.fillna("NA").astype(str)
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def build_rows(root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    missing_items: list[str] = []
    stage23_metrics: dict[str, float] | None = None

    for item in RUNS:
        rel_path = item["path"]
        source_path = root / rel_path
        mean_row = read_mean_row(source_path)
        status = "ready" if mean_row is not None else "missing"
        row: dict[str, object] = {
            "variant": item["variant"],
            "label": item["label"],
            "status": status,
            "path": str(rel_path),
        }
        if mean_row is None:
            missing_items.append(f"{item['label']}: {rel_path}")
            for metric in METRICS:
                row[metric] = None
                row[f"delta_{metric}_vs_stage23"] = None
        else:
            for metric in METRICS:
                row[metric] = mean_row[metric]
            if item["variant"] == "stage23_history":
                stage23_metrics = mean_row
        rows.append(row)

    if stage23_metrics is not None:
        for row, item in zip(rows, RUNS):
            for metric in METRICS:
                if row[metric] is None:
                    row[f"delta_{metric}_vs_stage23"] = None
                elif not item["compare_to_stage23"]:
                    row[f"delta_{metric}_vs_stage23"] = 0.0
                else:
                    row[f"delta_{metric}_vs_stage23"] = float(row[metric]) - float(stage23_metrics[metric])

    columns = ["variant", "label", "status"]
    columns.extend(METRICS)
    columns.extend([f"delta_{metric}_vs_stage23" for metric in METRICS])
    columns.append("path")
    return pd.DataFrame(rows, columns=columns), missing_items


def build_report(df: pd.DataFrame, missing_items: list[str]) -> str:
    lines = [
        "# Stage52 RCE Core Ablation Comparison",
        "",
        "Reference file: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`",
        "",
    ]
    if missing_items:
        lines.extend(
            [
                "## Status",
                "",
                "Some Step52 runs are still missing. The table is generated with `missing` placeholders instead of failing.",
                "",
                "Missing result files:",
                "",
            ]
        )
        lines.extend([f"- {item}" for item in missing_items])
        lines.append("")

    lines.extend(
        [
            "## Comparison Table",
            "",
            markdown_table(df),
            "",
            "Delta columns are computed against the historical Stage23 mean metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_md = (root / args.output_md).resolve()
    output_csv = (root / args.output_csv).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df, missing_items = build_rows(root)
    report = build_report(df, missing_items)

    output_md.write_text(report + "\n", encoding="utf-8")
    df.to_csv(output_csv, index=False)

    print(report)
    print(f"[Saved] markdown: {output_md}")
    print(f"[Saved] csv: {output_csv}")


if __name__ == "__main__":
    main()
