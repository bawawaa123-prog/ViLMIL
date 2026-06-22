from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_MD = Path("docs/stage51_rce_deg_reproduction_comparison.md")
DEFAULT_OUTPUT_CSV = Path("docs/stage51_rce_deg_reproduction_comparison.csv")
METRICS = ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
RUNS = [
    {
        "run": "stage23_history",
        "label": "Historical Stage23 main model",
        "path": Path("results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv"),
        "compare_to_history": False,
    },
    {
        "run": "stage51_rce",
        "label": "Step51 RCE reproduction",
        "path": Path("results_stage51_repro/rce_step23_rq16_5fold_e20_s1/result.csv"),
        "compare_to_history": True,
    },
    {
        "run": "stage51_deg",
        "label": "Step51 DEG skeleton reproduction",
        "path": Path("results_stage51_repro/deg_skeleton_rq16_5fold_e20_s1/result.csv"),
        "compare_to_history": True,
    },
]
WAITING_MESSAGE = "等待训练完成后再运行对比"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Step51 RCE and DEG skeleton reproduction results.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def read_mean_row(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
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
    waiting_items: list[str] = []
    rows: list[dict[str, object]] = []
    history = None

    for item in RUNS:
        rel_path = item["path"]
        source_path = root / rel_path
        mean_row = read_mean_row(source_path)
        row: dict[str, object] = {
            "run": item["run"],
            "label": item["label"],
            "status": "ready" if mean_row is not None else "waiting",
            "path": str(rel_path),
        }
        if mean_row is None:
            waiting_items.append(f"{item['label']}: {rel_path}")
            for metric in METRICS:
                row[metric] = None
                row[f"delta_{metric}_vs_stage23"] = None
        else:
            for metric in METRICS:
                row[metric] = mean_row[metric]
            if item["run"] == "stage23_history":
                history = mean_row
        rows.append(row)

    if history is not None:
        for row, item in zip(rows, RUNS):
            for metric in METRICS:
                if row[metric] is None or not item["compare_to_history"]:
                    row[f"delta_{metric}_vs_stage23"] = 0.0 if item["run"] == "stage23_history" and row[metric] is not None else None
                else:
                    row[f"delta_{metric}_vs_stage23"] = float(row[metric]) - float(history[metric])

    columns = ["run", "label", "status"]
    columns.extend(METRICS)
    columns.extend([f"delta_{metric}_vs_stage23" for metric in METRICS])
    columns.append("path")
    return pd.DataFrame(rows, columns=columns), waiting_items


def build_report(df: pd.DataFrame, waiting_items: list[str]) -> str:
    lines = [
        "# Stage51 RCE vs DEG Skeleton Reproduction Comparison",
        "",
        "Reference file: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv`",
        "",
    ]

    if waiting_items:
        lines.extend(
            [
                "## Status",
                "",
                WAITING_MESSAGE,
                "",
                "Missing result files:",
                "",
            ]
        )
        lines.extend([f"- {item}" for item in waiting_items])
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

    df, waiting_items = build_rows(root)
    report = build_report(df, waiting_items)

    output_md.write_text(report + "\n", encoding="utf-8")
    df.to_csv(output_csv, index=False)

    print(report)
    print(f"[Saved] markdown: {output_md}")
    print(f"[Saved] csv: {output_csv}")


if __name__ == "__main__":
    main()
