from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_STAGE22_DIR = Path("results_stage22")
DEFAULT_RESULTS_STAGE23_DIR = Path("results_stage23")
DEFAULT_STAGE9_ANALYSIS_CSV = Path("results_stage9/stage9_rce_final_analysis/rce_stage9_main_comparison.csv")
DEFAULT_OUTPUT_DIR = Path("results_stage24/stage24_rce_v4_csg_summary")
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
STAGE22_EXPERIMENTS = [
    {
        "group": "stage22_csg_init",
        "variant": "csg_a005",
        "method": "RCE-v4-CSG-a005-rq16",
        "source_path": Path("rce_v4_csg_a005_5fold_e20_s1/fold_summary.csv"),
    },
    {
        "group": "stage22_csg_init",
        "variant": "csg_a01",
        "method": "RCE-v4-CSG-a01-rq16",
        "source_path": Path("rce_v4_csg_a01_5fold_e20_s1/fold_summary.csv"),
    },
]
STAGE23_EXPERIMENTS = [
    {
        "group": "stage23_region_queries",
        "variant": "rq8",
        "method": "RCE-v4-CSG-a01-rq8",
        "source_path": Path("rce_v4_csg_a01_rq8_5fold_e20_s1/fold_summary.csv"),
        "prototype_number": 8,
    },
    {
        "group": "stage23_region_queries",
        "variant": "rq16",
        "method": "RCE-v4-CSG-a01-rq16",
        "source_path": Path("rce_v4_csg_a01_rq16_5fold_e20_s1/fold_summary.csv"),
        "prototype_number": 16,
    },
    {
        "group": "stage23_region_queries",
        "variant": "rq32",
        "method": "RCE-v4-CSG-a01-rq32",
        "source_path": Path("rce_v4_csg_a01_rq32_5fold_e20_s1/fold_summary.csv"),
        "prototype_number": 32,
    },
]
STAGE9_REFERENCE_METHODS = [
    ("RCE-MIL base", "RCE-MIL base"),
    ("RCE-MIL v3 prior_calib + visual_residual_init=0.05", "RCE-v3-VR-a005"),
    ("Concept-12 PEPS topk=5 tau=0.07", "Concept-12 PEPS reference"),
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


def build_summary_table(
    root: Path,
    base_dir: Path,
    experiments: list[dict[str, object]],
    warning_log: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summary_index: dict[str, dict[str, object]] = {}

    for experiment in experiments:
        source_path = base_dir / Path(str(experiment["source_path"]))
        df = safe_read_csv(source_path, warning_log)
        summary, status, num_rows = summarize_fold_summary(
            df=df,
            metrics=METRICS,
            warning_log=warning_log,
            label=str(experiment["method"]),
        )
        row: dict[str, object] = {
            "group": experiment["group"],
            "variant": experiment["variant"],
            "method": experiment["method"],
            "source_path": relative_path_str(root, source_path),
            "status": status,
            "num_rows": num_rows,
        }
        if "prototype_number" in experiment:
            row["prototype_number"] = experiment["prototype_number"]
        for metric in METRICS:
            mean_value = summary[f"{metric}_mean"]
            std_value = summary[f"{metric}_std"]
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_formatted"] = format_mean_std(mean_value, std_value)
        rows.append(row)
        summary_index[str(experiment["variant"])] = row

    ordered_columns = ["group", "variant", "method", "source_path", "status", "num_rows"]
    if any("prototype_number" in row for row in rows):
        ordered_columns.append("prototype_number")
    for metric in METRICS:
        ordered_columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_formatted"])
    return pd.DataFrame(rows, columns=ordered_columns), summary_index


def build_delta_row(
    label: str,
    current_label: str,
    reference_label: str,
    current_row: dict[str, object] | None,
    reference_row: dict[str, object] | None,
    warning_log: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "comparison": label,
        "current": current_label,
        "reference": reference_label,
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


def extract_stage9_reference_rows(
    stage9_df: pd.DataFrame | None,
    warning_log: list[str],
) -> dict[str, dict[str, object]]:
    reference_rows: dict[str, dict[str, object]] = {}
    if stage9_df is None:
        return reference_rows
    if "method" not in stage9_df.columns:
        warn_message("Stage9 analysis CSV is missing 'method' column.", warning_log)
        return reference_rows

    for method_name, short_label in STAGE9_REFERENCE_METHODS:
        match_df = stage9_df.loc[stage9_df["method"] == method_name]
        if match_df.empty:
            warn_message(f"Stage9 reference method not found: {method_name}", warning_log)
            continue
        reference_rows[short_label] = match_df.iloc[0].to_dict()
    return reference_rows


def stage9_to_summary_row(source_row: dict[str, object], label: str) -> dict[str, object]:
    row: dict[str, object] = {
        "method": source_row.get("method", label),
        "variant": source_row.get("variant", label),
        "status": source_row.get("status", "unknown"),
    }
    for metric in METRICS:
        row[f"{metric}_mean"] = pd.to_numeric(source_row.get(f"{metric}_mean"), errors="coerce")
        row[f"{metric}_std"] = pd.to_numeric(source_row.get(f"{metric}_std"), errors="coerce")
    return row


def best_variant_by_metric(summary_df: pd.DataFrame, metric: str) -> str | None:
    valid_df = summary_df.loc[summary_df["status"].isin(["ok", "partial", "partial_parse"])].copy()
    if valid_df.empty or f"{metric}_mean" not in valid_df.columns:
        return None
    valid_df = valid_df.dropna(subset=[f"{metric}_mean"])
    if valid_df.empty:
        return None
    best_idx = valid_df[f"{metric}_mean"].astype(float).idxmax()
    return str(valid_df.loc[best_idx, "variant"])


def report_metric_sentence(row: dict[str, object], metrics: list[str]) -> str:
    parts = []
    for metric in metrics:
        value = row.get(f"{metric}_formatted", "NA")
        parts.append(f"`{metric}`={value}")
    return ", ".join(parts)


def format_stage24_recommendation(stage22_variant: str | None, stage23_variant: str | None) -> str:
    if stage22_variant is None or stage23_variant is None:
        return "RCE-v4-CSG-a01-rq16"
    return f"RCE-v4-CSG-{stage22_variant.removeprefix('csg_')}-{stage23_variant}"


def build_report(
    root: Path,
    output_dir: Path,
    stage22_df: pd.DataFrame,
    stage23_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    stage9_df: pd.DataFrame | None,
    warning_log: list[str],
) -> str:
    stage22_best = best_variant_by_metric(stage22_df, "test_auc")
    stage23_best = best_variant_by_metric(stage23_df, "test_auc")

    stage22_lookup = {str(row["variant"]): row for _, row in stage22_df.iterrows()}
    stage23_lookup = {str(row["variant"]): row for _, row in stage23_df.iterrows()}

    recommendation = format_stage24_recommendation(stage22_best, stage23_best)

    lines = [
        "# Stage24 RCE-v4-CSG Summary",
        "",
        "Step24 reads existing Stage22 and Stage23 result CSV files only. It does not run training, extract features, or modify any model code.",
        "",
        "## Stage22 CSG Init Comparison",
        "",
    ]

    for variant in ["csg_a005", "csg_a01"]:
        row = stage22_lookup.get(variant)
        if row is None:
            continue
        lines.append(
            f"- `{variant}`: {report_metric_sentence(row, ['test_auc', 'test_acc', 'test_f1', 'balanced_acc', 'pr_auc'])}"
        )

    lines.extend(
        [
            "",
            "## Stage23 Region Query Number Comparison",
            "",
        ]
    )

    for variant in ["rq8", "rq16", "rq32"]:
        row = stage23_lookup.get(variant)
        if row is None:
            continue
        lines.append(
            f"- `{variant}`: {report_metric_sentence(row, ['test_auc', 'test_acc', 'test_f1', 'balanced_acc', 'pr_auc'])}"
        )

    lines.extend(["", "## Key Deltas", ""])
    for _, row in delta_df.iterrows():
        delta_parts = []
        for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]:
            value = row.get(f"{metric}_delta", math.nan)
            if not pd.isna(value):
                delta_parts.append(f"`{metric}`={float(value):+.6f}")
        if delta_parts:
            lines.append(f"- `{row['comparison']}`: " + ", ".join(delta_parts))

    if stage9_df is not None:
        lines.extend(["", "## Stage9 Reference Deltas", ""])
        stage9_rows = extract_stage9_reference_rows(stage9_df, warning_log)
        current_row = stage23_lookup.get("rq16")
        if current_row is not None:
            current_summary = {
                **current_row,
                **{f"{metric}_mean": current_row.get(f"{metric}_mean", math.nan) for metric in METRICS},
            }
            for _, short_label in STAGE9_REFERENCE_METHODS:
                ref_source = stage9_rows.get(short_label)
                if ref_source is None:
                    continue
                ref_row = stage9_to_summary_row(ref_source, short_label)
                delta_parts = []
                for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]:
                    current_mean = current_summary.get(f"{metric}_mean", math.nan)
                    reference_mean = ref_row.get(f"{metric}_mean", math.nan)
                    if pd.isna(current_mean) or pd.isna(reference_mean):
                        continue
                    delta_parts.append(f"`{metric}`={float(current_mean) - float(reference_mean):+.6f}")
                if delta_parts:
                    lines.append(f"- `RCE-v4-CSG-a01-rq16 - {short_label}`: " + ", ".join(delta_parts))

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Current recommended main configuration: `{recommendation}`.",
            f"Stage22 best CSG init by mean `test_auc`: `{stage22_best or 'NA'}`.",
            f"Stage23 best region query setting by mean `test_auc`: `{stage23_best or 'NA'}`.",
            "`prototype_number=16` remains the default suggestion for the next DEG-MIL stage.",
        ]
    )

    if stage22_best == "csg_a01":
        lines.append("`csg_a01` outperforms `csg_a005` on the main Stage22 5-fold metrics.")
    if stage23_best == "rq16":
        lines.append("`rq16` is the strongest Stage23 candidate across the main 5-fold summary metrics.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{relative_path_str(root, output_dir / 'stage24_stage22_csg_init_summary.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage24_stage23_region_query_summary.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage24_metric_deltas.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage24_rce_v4_csg_summary_report.md')}`",
            "",
            "## Summary Tables",
            "",
            "### Stage22",
            "",
            markdown_table(
                stage22_df[
                    ["variant", "status", "test_auc_formatted", "test_acc_formatted", "test_f1_formatted", "balanced_acc_formatted", "pr_auc_formatted"]
                ]
            ),
            "",
            "### Stage23",
            "",
            markdown_table(
                stage23_df[
                    [
                        "variant",
                        "prototype_number",
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
        for warning_text in warning_log:
            if warning_text in seen:
                continue
            seen.add(warning_text)
            lines.append(f"- {warning_text}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage24 RCE-v4-CSG summary tables and report.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to the ViLa-MIL-main root.")
    parser.add_argument(
        "--results_stage22_dir",
        type=Path,
        default=Path(os.environ.get("RESULTS_STAGE22_DIR", str(DEFAULT_RESULTS_STAGE22_DIR))),
        help="Stage22 results directory. Relative paths are resolved under --root.",
    )
    parser.add_argument(
        "--results_stage23_dir",
        type=Path,
        default=Path(os.environ.get("RESULTS_STAGE23_DIR", str(DEFAULT_RESULTS_STAGE23_DIR))),
        help="Stage23 results directory. Relative paths are resolved under --root.",
    )
    parser.add_argument(
        "--stage9_analysis_csv",
        type=Path,
        default=Path(os.environ.get("STAGE9_ANALYSIS_CSV", str(DEFAULT_STAGE9_ANALYSIS_CSV))),
        help="Optional Stage9 analysis CSV for reference comparisons.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
        help="Output directory for Stage24 summaries. Relative paths are resolved under --root.",
    )
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_stage22_dir = resolve_path(root, args.results_stage22_dir)
    results_stage23_dir = resolve_path(root, args.results_stage23_dir)
    stage9_analysis_csv = resolve_path(root, args.stage9_analysis_csv)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []

    stage22_df, stage22_index = build_summary_table(
        root=root,
        base_dir=results_stage22_dir,
        experiments=STAGE22_EXPERIMENTS,
        warning_log=warning_log,
    )
    stage23_df, stage23_index = build_summary_table(
        root=root,
        base_dir=results_stage23_dir,
        experiments=STAGE23_EXPERIMENTS,
        warning_log=warning_log,
    )

    delta_rows = [
        build_delta_row(
            label="csg_a01 - csg_a005",
            current_label="csg_a01",
            reference_label="csg_a005",
            current_row=stage22_index.get("csg_a01"),
            reference_row=stage22_index.get("csg_a005"),
            warning_log=warning_log,
        ),
        build_delta_row(
            label="rq16 - rq8",
            current_label="rq16",
            reference_label="rq8",
            current_row=stage23_index.get("rq16"),
            reference_row=stage23_index.get("rq8"),
            warning_log=warning_log,
        ),
        build_delta_row(
            label="rq16 - rq32",
            current_label="rq16",
            reference_label="rq32",
            current_row=stage23_index.get("rq16"),
            reference_row=stage23_index.get("rq32"),
            warning_log=warning_log,
        ),
    ]

    stage9_df = safe_read_csv(stage9_analysis_csv, warning_log)
    stage9_rows = extract_stage9_reference_rows(stage9_df, warning_log)
    current_main_row = stage23_index.get("rq16")
    if current_main_row is not None:
        for _, short_label in STAGE9_REFERENCE_METHODS:
            source_row = stage9_rows.get(short_label)
            if source_row is None:
                continue
            delta_rows.append(
                build_delta_row(
                    label=f"RCE-v4-CSG-a01-rq16 - {short_label}",
                    current_label="RCE-v4-CSG-a01-rq16",
                    reference_label=short_label,
                    current_row=current_main_row,
                    reference_row=stage9_to_summary_row(source_row, short_label),
                    warning_log=warning_log,
                )
            )

    delta_columns = [
        "comparison",
        "current",
        "reference",
        "current_status",
        "reference_status",
    ] + [f"{metric}_delta" for metric in METRICS]
    delta_df = pd.DataFrame(delta_rows, columns=delta_columns)

    stage22_csv = output_dir / "stage24_stage22_csg_init_summary.csv"
    stage23_csv = output_dir / "stage24_stage23_region_query_summary.csv"
    delta_csv = output_dir / "stage24_metric_deltas.csv"
    report_md = output_dir / "stage24_rce_v4_csg_summary_report.md"

    stage22_df.to_csv(stage22_csv, index=False)
    stage23_df.to_csv(stage23_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)

    report_text = build_report(
        root=root,
        output_dir=output_dir,
        stage22_df=stage22_df,
        stage23_df=stage23_df,
        delta_df=delta_df,
        stage9_df=stage9_df,
        warning_log=warning_log,
    )
    report_md.write_text(report_text, encoding="utf-8")

    print(f"[Stage24] Wrote: {relative_path_str(root, stage22_csv)}")
    print(f"[Stage24] Wrote: {relative_path_str(root, stage23_csv)}")
    print(f"[Stage24] Wrote: {relative_path_str(root, delta_csv)}")
    print(f"[Stage24] Wrote: {relative_path_str(root, report_md)}")


if __name__ == "__main__":
    main()
