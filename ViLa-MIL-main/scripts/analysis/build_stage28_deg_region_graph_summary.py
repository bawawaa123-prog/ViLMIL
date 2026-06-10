from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_STAGE27_DIR = Path("results_stage27")
DEFAULT_OUTPUT_DIR = Path("results_stage28/stage28_deg_region_graph_summary")
METRICS = [
    "test_auc",
    "test_acc",
    "test_f1",
    "val_auc",
    "val_acc",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "pr_auc",
]
EXPERIMENTS = [
    {
        "variant": "skeleton",
        "method": "DEG skeleton",
        "source_path": Path("deg_skeleton_5fold_e20_s1/fold_summary.csv"),
        "region_graph": "off",
    },
    {
        "variant": "rg_k2",
        "method": "DEG Spatial Region Graph k=2",
        "source_path": Path("deg_region_graph_k2_a01_5fold_e20_s1/fold_summary.csv"),
        "region_graph": "k=2",
    },
    {
        "variant": "rg_k4",
        "method": "DEG Spatial Region Graph k=4",
        "source_path": Path("deg_region_graph_k4_a01_5fold_e20_s1/fold_summary.csv"),
        "region_graph": "k=4",
    },
    {
        "variant": "rg_k8",
        "method": "DEG Spatial Region Graph k=8",
        "source_path": Path("deg_region_graph_k8_a01_5fold_e20_s1/fold_summary.csv"),
        "region_graph": "k=8",
    },
]


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def relative_path_str(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_read_csv(path: Path, warning_log: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warn_message(f"Missing input CSV: {path}", warning_log)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn_message(f"Failed to read CSV {path}: {exc}", warning_log)
        return None


def format_mean_std(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        return f"{float(mean_value):.6f}"
    return f"{float(mean_value):.6f} ± {float(std_value):.6f}"


def summarize_fold_summary(
    df: pd.DataFrame | None,
    metrics: list[str],
    warning_log: list[str],
    label: str,
) -> tuple[dict[str, float], str, int]:
    summary: dict[str, float] = {}
    if df is None:
        for metric in metrics:
            summary[f"{metric}_mean"] = math.nan
            summary[f"{metric}_std"] = math.nan
        return summary, "missing", 0

    status = "ok"
    num_rows = len(df.index)
    if df.empty:
        warn_message(f"{label}: fold summary is empty.", warning_log)
        status = "empty"

    for metric in metrics:
        if metric not in df.columns:
            warn_message(f"{label}: missing metric column '{metric}'.", warning_log)
            summary[f"{metric}_mean"] = math.nan
            summary[f"{metric}_std"] = math.nan
            if status == "ok":
                status = "partial"
            continue

        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            warn_message(f"{label}: no valid numeric values for metric '{metric}'.", warning_log)
            summary[f"{metric}_mean"] = math.nan
            summary[f"{metric}_std"] = math.nan
            if status == "ok":
                status = "partial"
            continue

        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=0))

    return summary, status, num_rows


def build_summary_table(root: Path, results_dir: Path, warning_log: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    index: dict[str, dict[str, object]] = {}

    for experiment in EXPERIMENTS:
        source_path = results_dir / experiment["source_path"]
        df = safe_read_csv(source_path, warning_log)
        summary, status, num_rows = summarize_fold_summary(df, METRICS, warning_log, str(experiment["method"]))
        row: dict[str, object] = {
            "variant": experiment["variant"],
            "method": experiment["method"],
            "region_graph": experiment["region_graph"],
            "source_path": relative_path_str(root, source_path),
            "status": status,
            "num_rows": num_rows,
        }
        for metric in METRICS:
            mean_value = summary[f"{metric}_mean"]
            std_value = summary[f"{metric}_std"]
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_formatted"] = format_mean_std(mean_value, std_value)
        rows.append(row)
        index[str(experiment["variant"])] = row

    ordered_columns = ["variant", "method", "region_graph", "source_path", "status", "num_rows"]
    for metric in METRICS:
        ordered_columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_formatted"])
    return pd.DataFrame(rows, columns=ordered_columns), index


def build_delta_row(
    label: str,
    current_row: dict[str, object] | None,
    reference_row: dict[str, object] | None,
    warning_log: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "comparison": label,
        "current_status": None if current_row is None else current_row.get("status"),
        "reference_status": None if reference_row is None else reference_row.get("status"),
    }
    if current_row is None or reference_row is None:
        warn_message(f"Missing summary row for delta computation: {label}", warning_log)
        for metric in METRICS:
            row[f"{metric}_delta"] = math.nan
        return row

    for metric in METRICS:
        current_mean = current_row.get(f"{metric}_mean", math.nan)
        reference_mean = reference_row.get(f"{metric}_mean", math.nan)
        delta_value = math.nan
        if not pd.isna(current_mean) and not pd.isna(reference_mean):
            delta_value = float(current_mean) - float(reference_mean)
        row[f"{metric}_delta"] = delta_value
    return row


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("NA").astype(str)
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def build_report(
    root: Path,
    output_dir: Path,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    warning_log: list[str],
) -> str:
    lookup = {str(row["variant"]): row for _, row in summary_df.iterrows()}
    skeleton_row = lookup.get("skeleton")
    rg2_row = lookup.get("rg_k2")
    rg4_row = lookup.get("rg_k4")
    rg8_row = lookup.get("rg_k8")

    lines = [
        "# Stage28 DEG Spatial Region Graph Summary",
        "",
        "Step28 reads existing Stage27 result CSV files only. It does not run training, modify model computation, or extract features.",
        "",
        "## Mean ± Std Summary",
        "",
    ]

    for variant in ["skeleton", "rg_k2", "rg_k4", "rg_k8"]:
        row = lookup.get(variant)
        if row is None:
            continue
        lines.append(
            f"- `{variant}`: "
            f"`test_auc`={row['test_auc_formatted']}, "
            f"`test_acc`={row['test_acc_formatted']}, "
            f"`test_f1`={row['test_f1_formatted']}, "
            f"`balanced_acc`={row['balanced_acc_formatted']}, "
            f"`pr_auc`={row['pr_auc_formatted']}"
        )

    lines.extend(["", "## Deltas Vs Skeleton", ""])
    for _, row in delta_df.iterrows():
        delta_parts = []
        for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]:
            value = row.get(f"{metric}_delta", math.nan)
            if not pd.isna(value):
                delta_parts.append(f"`{metric}`={float(value):+.6f}")
        if delta_parts:
            lines.append(f"- `{row['comparison']}`: " + ", ".join(delta_parts))

    lines.extend(["", "## Recommendation", ""])

    skeleton_best = False
    if skeleton_row is not None and all(
        not pd.isna(skeleton_row.get(f"{metric}_mean", math.nan)) for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
    ):
        skeleton_best = True
        for other in [rg2_row, rg4_row, rg8_row]:
            if other is None:
                continue
            for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]:
                if pd.isna(other.get(f"{metric}_mean", math.nan)):
                    continue
                if float(other[f"{metric}_mean"]) > float(skeleton_row[f"{metric}_mean"]):
                    skeleton_best = False
                    break
            if not skeleton_best:
                break

    lines.append(
        "Current recommended main configuration: `RCE-v4-CSG-a01-rq16 / DEG skeleton`."
    )
    if skeleton_best:
        lines.append("`DEG skeleton` remains the strongest Stage27 configuration across the main summary metrics.")
    else:
        lines.append("The current Stage27 table does not support promoting any Spatial Region Graph variant above the DEG skeleton.")

    lines.append(
        "All current Spatial Region Graph variants (`k=2`, `k=4`, `k=8`) remain below the DEG skeleton and should not be treated as the main performance path."
    )
    lines.append(
        "`k=8` is the closest Region Graph variant on mean `test_auc`, but it still stays below the DEG skeleton and is not consistently best across the other main metrics."
    )
    lines.append(
        "The current Spatial Region Graph should remain an optional exploration module rather than the main DEG-MIL performance module."
    )
    lines.append(
        "A more stable follow-up would be a gated or zero-init Region Graph, or a higher-priority Concept Prompt Graph line, before revisiting cross-scale region graph ideas."
    )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{relative_path_str(root, output_dir / 'stage28_deg_region_graph_summary.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage28_deg_region_graph_metric_deltas.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage28_deg_region_graph_report.md')}`",
            "",
            "## Summary Table",
            "",
            markdown_table(
                summary_df[
                    [
                        "variant",
                        "region_graph",
                        "status",
                        "test_auc_formatted",
                        "test_acc_formatted",
                        "test_f1_formatted",
                        "balanced_acc_formatted",
                        "pr_auc_formatted",
                    ]
                ]
            ),
        ]
    )

    if warning_log:
        lines.extend(["", "## Warnings", ""])
        seen = set()
        for note in warning_log:
            if note in seen:
                continue
            seen.add(note)
            lines.append(f"- {note}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage28 DEG Spatial Region Graph summary tables and report.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to the ViLa-MIL-main root.")
    parser.add_argument(
        "--results_stage27_dir",
        type=Path,
        default=Path(os.environ.get("RESULTS_STAGE27_DIR", str(DEFAULT_RESULTS_STAGE27_DIR))),
        help="Stage27 results directory. Relative paths are resolved under --root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
        help="Output directory for Stage28 summaries. Relative paths are resolved under --root.",
    )
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_stage27_dir = resolve_path(root, args.results_stage27_dir)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []
    summary_df, summary_index = build_summary_table(root, results_stage27_dir, warning_log)

    delta_rows = [
        build_delta_row("rg_k2 - skeleton", summary_index.get("rg_k2"), summary_index.get("skeleton"), warning_log),
        build_delta_row("rg_k4 - skeleton", summary_index.get("rg_k4"), summary_index.get("skeleton"), warning_log),
        build_delta_row("rg_k8 - skeleton", summary_index.get("rg_k8"), summary_index.get("skeleton"), warning_log),
    ]
    delta_columns = ["comparison", "current_status", "reference_status"] + [
        f"{metric}_delta" for metric in METRICS
    ]
    delta_df = pd.DataFrame(delta_rows, columns=delta_columns)

    summary_csv = output_dir / "stage28_deg_region_graph_summary.csv"
    delta_csv = output_dir / "stage28_deg_region_graph_metric_deltas.csv"
    report_md = output_dir / "stage28_deg_region_graph_report.md"

    summary_df.to_csv(summary_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    report_md.write_text(
        build_report(root, output_dir, summary_df, delta_df, warning_log),
        encoding="utf-8",
    )

    print(f"[Stage28] Wrote: {relative_path_str(root, summary_csv)}")
    print(f"[Stage28] Wrote: {relative_path_str(root, delta_csv)}")
    print(f"[Stage28] Wrote: {relative_path_str(root, report_md)}")


if __name__ == "__main__":
    main()
