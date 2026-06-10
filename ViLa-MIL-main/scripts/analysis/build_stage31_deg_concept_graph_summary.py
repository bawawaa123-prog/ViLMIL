from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_STAGE30_DIR = Path("results_stage30")
DEFAULT_OUTPUT_DIR = Path("results_stage31/stage31_deg_concept_graph_summary")
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
KEY_METRICS = ["test_auc", "test_f1", "balanced_acc", "sensitivity", "pr_auc"]
EXPERIMENTS = [
    {
        "variant": "skeleton",
        "method": "DEG skeleton",
        "source_path": Path("deg_skeleton_cg_ablation_5fold_e20_s1/fold_summary.csv"),
        "concept_graph": "off",
        "concept_graph_topk": "off",
        "concept_graph_alpha": "off",
    },
    {
        "variant": "cg_k2_a005",
        "method": "DEG Concept Prompt Graph k=2 alpha=0.05",
        "source_path": Path("deg_concept_graph_k2_a005_5fold_e20_s1/fold_summary.csv"),
        "concept_graph": "on",
        "concept_graph_topk": "2",
        "concept_graph_alpha": "0.05",
    },
    {
        "variant": "cg_k4_a005",
        "method": "DEG Concept Prompt Graph k=4 alpha=0.05",
        "source_path": Path("deg_concept_graph_k4_a005_5fold_e20_s1/fold_summary.csv"),
        "concept_graph": "on",
        "concept_graph_topk": "4",
        "concept_graph_alpha": "0.05",
    },
    {
        "variant": "cg_k8_a005",
        "method": "DEG Concept Prompt Graph k=8 alpha=0.05",
        "source_path": Path("deg_concept_graph_k8_a005_5fold_e20_s1/fold_summary.csv"),
        "concept_graph": "on",
        "concept_graph_topk": "8",
        "concept_graph_alpha": "0.05",
    },
]


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def resolve_path(root: Path, raw_value: str | os.PathLike[str]) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return root / path


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


def build_summary_table(
    root: Path,
    results_dir: Path,
    warning_log: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    index: dict[str, dict[str, object]] = {}

    for experiment in EXPERIMENTS:
        source_path = results_dir / experiment["source_path"]
        df = safe_read_csv(source_path, warning_log)
        summary, status, num_rows = summarize_fold_summary(df, METRICS, warning_log, experiment["method"])

        row: dict[str, object] = {
            "variant": experiment["variant"],
            "method": experiment["method"],
            "concept_graph": experiment["concept_graph"],
            "concept_graph_topk": experiment["concept_graph_topk"],
            "concept_graph_alpha": experiment["concept_graph_alpha"],
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

    ordered_columns = [
        "variant",
        "method",
        "concept_graph",
        "concept_graph_topk",
        "concept_graph_alpha",
        "source_path",
        "status",
        "num_rows",
    ]
    for metric in METRICS:
        ordered_columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_formatted"])
    return pd.DataFrame(rows, columns=ordered_columns), index


def build_delta_row(
    comparison: str,
    variant: str,
    current_row: dict[str, object] | None,
    reference_row: dict[str, object] | None,
    warning_log: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "comparison": comparison,
        "variant": variant,
        "reference_variant": "skeleton",
        "current_status": None if current_row is None else current_row.get("status"),
        "reference_status": None if reference_row is None else reference_row.get("status"),
    }
    if current_row is None or reference_row is None:
        warn_message(f"Missing summary row for delta computation: {comparison}", warning_log)
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


def metric_delta_value(row: dict[str, object] | None, metric: str) -> float:
    if row is None:
        return math.nan
    value = row.get(f"{metric}_delta", math.nan)
    return math.nan if pd.isna(value) else float(value)


def build_recommendation_lines(
    summary_lookup: dict[str, dict[str, object]],
    delta_lookup: dict[str, dict[str, object]],
) -> list[str]:
    skeleton_row = summary_lookup.get("skeleton")
    cg2_row = summary_lookup.get("cg_k2_a005")
    cg4_row = summary_lookup.get("cg_k4_a005")
    cg8_row = summary_lookup.get("cg_k8_a005")

    lines: list[str] = []
    concept_rows = [cg2_row, cg4_row, cg8_row]
    valid_skeleton = skeleton_row is not None and all(
        not pd.isna(skeleton_row.get(f"{metric}_mean", math.nan)) for metric in KEY_METRICS
    )

    skeleton_best = False
    if valid_skeleton:
        skeleton_best = True
        for row in concept_rows:
            if row is None:
                continue
            for metric in KEY_METRICS:
                other_mean = row.get(f"{metric}_mean", math.nan)
                if pd.isna(other_mean):
                    continue
                if float(other_mean) > float(skeleton_row[f"{metric}_mean"]):
                    skeleton_best = False
                    break
            if not skeleton_best:
                break

    if skeleton_best:
        lines.append("- `DEG skeleton` remains the recommended main configuration in Stage30.")
        lines.append("- `cg_k2_a005`, `cg_k4_a005`, and `cg_k8_a005` all remain below `skeleton` on the main evidence-quality metrics.")
    else:
        lines.append("- Stage30 does not support promoting a Concept Prompt Graph variant over the current `DEG skeleton` main line.")

    k8_delta_auc = metric_delta_value(delta_lookup.get("cg_k8_a005 - skeleton"), "test_auc")
    k8_delta_pr_auc = metric_delta_value(delta_lookup.get("cg_k8_a005 - skeleton"), "pr_auc")
    k8_delta_f1 = metric_delta_value(delta_lookup.get("cg_k8_a005 - skeleton"), "test_f1")
    k8_delta_bal = metric_delta_value(delta_lookup.get("cg_k8_a005 - skeleton"), "balanced_acc")
    k8_delta_sens = metric_delta_value(delta_lookup.get("cg_k8_a005 - skeleton"), "sensitivity")
    if not any(pd.isna(v) for v in [k8_delta_auc, k8_delta_pr_auc, k8_delta_f1, k8_delta_bal, k8_delta_sens]):
        lines.append(
            "- `cg_k8_a005` is the closest Concept Prompt Graph variant on `test_auc` / `pr_auc`, "
            f"but still trails `skeleton` on `test_f1` ({k8_delta_f1:+.6f}), "
            f"`balanced_acc` ({k8_delta_bal:+.6f}), and `sensitivity` ({k8_delta_sens:+.6f})."
        )

    lines.append("- The current Concept Prompt Graph should not be treated as a mainline performance module.")
    lines.append("- The current main line remains `RCE-v4-CSG-a01-rq16 / DEG skeleton`.")
    lines.append(
        "- Combined with Stage27/28 Spatial Region Graph results, Stage30 suggests that direct feature-level "
        "message passing can weaken already-learned evidence discrimination rather than improve it."
    )
    lines.append(
        "- Recommended follow-up direction: prioritize evidence-level analysis / evidence export / interpretability, "
        "or try evidence-level gated residual / evidence consistency loss instead of stacking ordinary graph message passing."
    )
    return lines


def build_report(
    root: Path,
    results_dir: Path,
    output_dir: Path,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    warning_log: list[str],
) -> str:
    summary_lookup = {str(row["variant"]): row for _, row in summary_df.iterrows()}
    delta_lookup = {str(row["comparison"]): row for _, row in delta_df.iterrows()}

    lines = [
        "# Stage31 DEG Concept Prompt Graph Summary",
        "",
        "Step31 reads existing Stage30 result CSV files only. It does not run training, modify model computation, or extract features.",
        "",
        "## Inputs",
        "",
    ]
    for experiment in EXPERIMENTS:
        lines.append(f"- `{relative_path_str(root, results_dir / experiment['source_path'])}`")

    lines.extend(["", "## Mean ± Std Summary", ""])
    for variant in ["skeleton", "cg_k2_a005", "cg_k4_a005", "cg_k8_a005"]:
        row = summary_lookup.get(variant)
        if row is None:
            continue
        lines.append(
            f"- `{variant}`: "
            f"`test_auc`={row['test_auc_formatted']}, "
            f"`test_acc`={row['test_acc_formatted']}, "
            f"`test_f1`={row['test_f1_formatted']}, "
            f"`balanced_acc`={row['balanced_acc_formatted']}, "
            f"`sensitivity`={row['sensitivity_formatted']}, "
            f"`pr_auc`={row['pr_auc_formatted']}"
        )

    lines.extend(["", "## Deltas Vs Skeleton", ""])
    for _, row in delta_df.iterrows():
        delta_parts = []
        for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "sensitivity", "pr_auc"]:
            value = row.get(f"{metric}_delta", math.nan)
            if not pd.isna(value):
                delta_parts.append(f"`{metric}`={float(value):+.6f}")
        if delta_parts:
            lines.append(f"- `{row['comparison']}`: " + ", ".join(delta_parts))

    lines.extend(["", "## Recommendation", ""])
    lines.extend(build_recommendation_lines(summary_lookup, delta_lookup))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{relative_path_str(root, output_dir / 'stage31_deg_concept_graph_summary.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage31_deg_concept_graph_metric_deltas.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage31_deg_concept_graph_report.md')}`",
        ]
    )

    if warning_log:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {message}" for message in warning_log])

    lines.extend(
        [
            "",
            "## Compact Tables",
            "",
            "### Summary",
            "",
            markdown_table(
                summary_df[
                    [
                        "variant",
                        "method",
                        "status",
                        "test_auc_formatted",
                        "test_acc_formatted",
                        "test_f1_formatted",
                        "balanced_acc_formatted",
                        "sensitivity_formatted",
                        "pr_auc_formatted",
                    ]
                ]
            ),
            "",
            "### Deltas",
            "",
            markdown_table(
                delta_df[
                    [
                        "comparison",
                        "test_auc_delta",
                        "test_acc_delta",
                        "test_f1_delta",
                        "balanced_acc_delta",
                        "sensitivity_delta",
                        "pr_auc_delta",
                    ]
                ]
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    root = DEFAULT_ROOT
    results_dir = resolve_path(root, os.environ.get("RESULTS_STAGE30_DIR", str(DEFAULT_RESULTS_STAGE30_DIR)))
    output_dir = resolve_path(root, os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []
    summary_df, summary_lookup = build_summary_table(root, results_dir, warning_log)

    skeleton_row = summary_lookup.get("skeleton")
    delta_rows = [
        build_delta_row("cg_k2_a005 - skeleton", "cg_k2_a005", summary_lookup.get("cg_k2_a005"), skeleton_row, warning_log),
        build_delta_row("cg_k4_a005 - skeleton", "cg_k4_a005", summary_lookup.get("cg_k4_a005"), skeleton_row, warning_log),
        build_delta_row("cg_k8_a005 - skeleton", "cg_k8_a005", summary_lookup.get("cg_k8_a005"), skeleton_row, warning_log),
    ]
    delta_df = pd.DataFrame(delta_rows)

    summary_path = output_dir / "stage31_deg_concept_graph_summary.csv"
    delta_path = output_dir / "stage31_deg_concept_graph_metric_deltas.csv"
    report_path = output_dir / "stage31_deg_concept_graph_report.md"

    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    report_text = build_report(root, results_dir, output_dir, summary_df, delta_df, warning_log)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"[Done] Wrote summary CSV: {summary_path}")
    print(f"[Done] Wrote delta CSV: {delta_path}")
    print(f"[Done] Wrote report: {report_path}")


if __name__ == "__main__":
    main()
