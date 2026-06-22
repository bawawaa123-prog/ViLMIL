from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_MD = Path("docs/main_model_lineage_comparison.md")
DEFAULT_OUTPUT_CSV = Path("docs/main_model_lineage_comparison.csv")
METRICS = ["test_auc", "test_acc", "test_f1", "pr_auc", "val_auc"]
EPS = 1e-12
EXPERIMENTS = [
    {
        "stage": "Stage23",
        "label": "RCE-v4-CSG-a01-rq16",
        "kind": "source_of_truth",
        "path": Path("results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage27",
        "label": "DEG skeleton",
        "kind": "skeleton_replay",
        "path": Path("results_stage27/deg_skeleton_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage30",
        "label": "DEG skeleton cg ablation",
        "kind": "skeleton_replay",
        "path": Path("results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage35",
        "label": "visual gate skeleton",
        "kind": "skeleton_replay",
        "path": Path("results_stage35/visual_gate_skeleton_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage37",
        "label": "lh consistency skeleton",
        "kind": "skeleton_replay",
        "path": Path("results_stage37/lh_consistency_skeleton_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage22",
        "label": "RCE-v4-CSG-a01",
        "kind": "pre_stage23_equivalent",
        "path": Path("results_stage22/rce_v4_csg_a01_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage30",
        "label": "DEG concept graph k=4",
        "kind": "non_source_variant",
        "path": Path("results_stage30/deg_concept_graph_k4_a005_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage35",
        "label": "visual gate gate001",
        "kind": "non_source_variant",
        "path": Path("results_stage35/visual_gate_gate001_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage37",
        "label": "lh consistency l001_m0",
        "kind": "non_source_variant",
        "path": Path("results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1/result.csv"),
    },
    {
        "stage": "Stage44",
        "label": "HCRC a01 b8",
        "kind": "non_source_variant",
        "path": Path("results_stage44/stage44_hcrc_a01_b8_s1/result.csv"),
    },
    {
        "stage": "Stage47",
        "label": "PRARC v1 g05",
        "kind": "non_source_variant",
        "path": Path("results_stage47/stage47_prarc_v1_g05_s1/result.csv"),
    },
]


def read_mean_row(path: Path) -> dict[str, float | str]:
    df = pd.read_csv(path)
    mean_rows = df[df["metric"] == "mean"]
    if mean_rows.empty:
        raise ValueError(f"Missing metric=mean row: {path}")
    row = mean_rows.iloc[0].to_dict()
    return row


def compute_match_status(row: dict[str, object], reference: dict[str, object]) -> str:
    if row["stage"] == "Stage23" and row["label"] == "RCE-v4-CSG-a01-rq16":
        return "reference"
    deltas = [abs(float(row[f"delta_{metric}"])) for metric in METRICS]
    if all(delta <= EPS for delta in deltas):
        return "exact_match"
    if all(delta <= 5e-3 for delta in deltas):
        return "near_match"
    return "different"


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


def build_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference = None

    for experiment in EXPERIMENTS:
        source_path = root / experiment["path"]
        mean_row = read_mean_row(source_path)
        row: dict[str, object] = {
            "stage": experiment["stage"],
            "label": experiment["label"],
            "kind": experiment["kind"],
            "path": str(experiment["path"]),
        }
        for metric in METRICS:
            row[metric] = float(mean_row[metric])
        rows.append(row)
        if experiment["stage"] == "Stage23" and experiment["label"] == "RCE-v4-CSG-a01-rq16":
            reference = row

    if reference is None:
        raise ValueError("Stage23 reference row is missing.")

    for row in rows:
        for metric in METRICS:
            row[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
        row["match_stage23"] = compute_match_status(row, reference)

    columns = [
        "stage",
        "label",
        "kind",
        "match_stage23",
        "test_auc",
        "test_acc",
        "test_f1",
        "pr_auc",
        "val_auc",
        "delta_test_auc",
        "delta_test_acc",
        "delta_test_f1",
        "delta_pr_auc",
        "delta_val_auc",
        "path",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_report(df: pd.DataFrame) -> str:
    lines = [
        "# Main Model Lineage Comparison",
        "",
        "Reference row: Stage23 `RCE-v4-CSG-a01-rq16`.",
        "",
        "## Comparison Table",
        "",
        markdown_table(df),
        "",
        "## Match Status",
        "",
        "- `reference`: the Stage23 source-of-truth row.",
        "- `exact_match`: every tracked metric matches Stage23 within machine precision.",
        "- `near_match`: all tracked metrics stay within 0.005 of Stage23.",
        "- `different`: at least one tracked metric differs by more than 0.005.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare main-model lineage result.csv files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    df = build_rows(root)

    output_md = (root / args.output_md).resolve()
    output_csv = (root / args.output_csv).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(df)
    output_md.write_text(report + "\n", encoding="utf-8")
    df.to_csv(output_csv, index=False)

    print(report)
    print(f"[Saved] markdown: {output_md}")
    print(f"[Saved] csv: {output_csv}")


if __name__ == "__main__":
    main()
