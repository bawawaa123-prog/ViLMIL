from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("results_stage9/stage9_rce_final_analysis")
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
RCE_EXPERIMENTS = [
    {
        "method": "RCE-MIL base",
        "family": "RCE base",
        "variant": "base",
        "key": "rce_base",
        "source_path": Path("results_stage9/rce_mil_5fold_e20_s1/fold_summary.csv"),
    },
    {
        "method": "RCE-MIL v2 prior_calib",
        "family": "RCE v2",
        "variant": "prior_calib",
        "key": "rce_v2_prior_calib",
        "source_path": Path("results_stage9/rce_mil_v2_prior_calib_5fold_e20_s1/fold_summary.csv"),
    },
    {
        "method": "RCE-MIL v2 prior",
        "family": "RCE v2",
        "variant": "prior",
        "key": "rce_v2_prior",
        "source_path": Path("results_stage9/rce_mil_v2_prior_5fold_e20_s1/fold_summary.csv"),
    },
    {
        "method": "RCE-MIL v3 prior_calib + visual_residual_init=0.05",
        "family": "RCE v3",
        "variant": "prior_calib_vr_a005",
        "key": "rce_v3_vr_a005",
        "source_path": Path("results_stage9/rce_mil_v3_prior_calib_vr_a005_5fold_e20_s1/fold_summary.csv"),
    },
    {
        "method": "RCE-MIL v3 prior_calib + visual_residual_init=0.1",
        "family": "RCE v3",
        "variant": "prior_calib_vr_a01",
        "key": "rce_v3_vr_a01",
        "source_path": Path("results_stage9/rce_mil_v3_prior_calib_vr_a01_5fold_e20_s1/fold_summary.csv"),
    },
]
PEPS_REFERENCE = {
    "method": "Concept-12 PEPS topk=5 tau=0.07",
    "family": "PEPS reference",
    "variant": "main_reference",
    "key": "peps_reference",
    "source_path": Path("trained_models/final_dcp_vila_analysis/final_main_table.csv"),
}
MEAN_STD_PATTERN = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:±|\+/-)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
RECOMMENDED_METHOD = "RCE-MIL v3 prior_calib + visual_residual_init=0.05"
RECOMMENDED_SHORT = "RCE-v3-VR-a005"


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def safe_read_csv(path: Path, warning_log: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warn_message(f"Missing input CSV: {path}", warning_log)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn_message(f"Failed to read CSV {path}: {exc}", warning_log)
        return None


def summarize_fold_summary(
    df: pd.DataFrame | None,
    metrics: list[str],
    warning_log: list[str],
    label: str,
) -> tuple[dict[str, float], str, int]:
    summary = {}
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


def parse_mean_std(value) -> tuple[float, float, str | None]:
    if pd.isna(value):
        return math.nan, math.nan, None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), math.nan, None

    text = str(value).strip()
    if not text:
        return math.nan, math.nan, None

    match = MEAN_STD_PATTERN.match(text)
    if match:
        return float(match.group(1)), float(match.group(2)), None

    try:
        return float(text), math.nan, None
    except Exception:
        return math.nan, math.nan, text


def format_mean_std(mean_value: float, std_value: float, raw_value: str | None = None) -> str:
    if raw_value is not None:
        return raw_value
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


def relative_path_str(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_main_comparison(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, object]], list[str]]:
    warning_log: list[str] = []
    rows: list[dict[str, object]] = []
    summary_index: dict[str, dict[str, object]] = {}

    for experiment in RCE_EXPERIMENTS:
        path = root / experiment["source_path"]
        df = safe_read_csv(path, warning_log)
        summary, status, num_folds = summarize_fold_summary(
            df=df,
            metrics=METRICS,
            warning_log=warning_log,
            label=experiment["method"],
        )
        row: dict[str, object] = {
            "method": experiment["method"],
            "family": experiment["family"],
            "variant": experiment["variant"],
            "source_path": relative_path_str(root, path),
            "status": status,
            "num_rows": num_folds,
        }
        for metric in METRICS:
            mean_value = summary[f"{metric}_mean"]
            std_value = summary[f"{metric}_std"]
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_formatted"] = format_mean_std(mean_value, std_value)
        rows.append(row)
        summary_index[experiment["key"]] = row

    peps_path = root / PEPS_REFERENCE["source_path"]
    peps_df = safe_read_csv(peps_path, warning_log)
    peps_row: dict[str, object] = {
        "method": PEPS_REFERENCE["method"],
        "family": PEPS_REFERENCE["family"],
        "variant": PEPS_REFERENCE["variant"],
        "source_path": relative_path_str(root, peps_path),
        "status": "missing" if peps_df is None else "ok",
        "num_rows": 0 if peps_df is None else len(peps_df.index),
    }
    parse_failures: list[str] = []
    for metric in METRICS:
        peps_row[f"{metric}_mean"] = math.nan
        peps_row[f"{metric}_std"] = math.nan
        peps_row[f"{metric}_formatted"] = "NA"

    if peps_df is not None:
        if "method" not in peps_df.columns:
            warn_message(f"PEPS reference table is missing 'method' column: {peps_path}", warning_log)
            peps_row["status"] = "invalid"
        else:
            match_df = peps_df.loc[peps_df["method"] == PEPS_REFERENCE["method"]]
            if match_df.empty:
                warn_message(
                    f"PEPS reference method '{PEPS_REFERENCE['method']}' not found in {peps_path}",
                    warning_log,
                )
                peps_row["status"] = "missing_method"
            else:
                source_row = match_df.iloc[0]
                for metric in METRICS:
                    if metric not in source_row.index:
                        warn_message(f"PEPS reference row is missing metric '{metric}'.", warning_log)
                        if peps_row["status"] == "ok":
                            peps_row["status"] = "partial"
                        continue
                    mean_value, std_value, raw_value = parse_mean_std(source_row[metric])
                    peps_row[f"{metric}_mean"] = mean_value
                    peps_row[f"{metric}_std"] = std_value
                    peps_row[f"{metric}_formatted"] = format_mean_std(mean_value, std_value, raw_value)
                    if raw_value is not None:
                        parse_failures.append(f"{metric}='{raw_value}'")
                        if peps_row["status"] == "ok":
                            peps_row["status"] = "partial_parse"

    rows.append(peps_row)
    summary_index[PEPS_REFERENCE["key"]] = peps_row

    ordered_columns: list[str] = [
        "method",
        "family",
        "variant",
        "source_path",
        "status",
        "num_rows",
    ]
    for metric in METRICS:
        ordered_columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_formatted",
            ]
        )
    comparison_df = pd.DataFrame(rows, columns=ordered_columns)
    return comparison_df, summary_index, parse_failures + warning_log


def build_metric_deltas(summary_index: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, list[str]]:
    warning_log: list[str] = []
    reference_specs = [
        ("rce_base", "RCE-MIL base"),
        ("rce_v2_prior_calib", "RCE-MIL v2 prior_calib"),
    ]
    rows: list[dict[str, object]] = []

    for experiment in RCE_EXPERIMENTS:
        current = summary_index.get(experiment["key"])
        if current is None:
            continue
        for ref_key, ref_label in reference_specs:
            reference = summary_index.get(ref_key)
            if reference is None:
                warn_message(f"Missing reference summary for delta computation: {ref_label}", warning_log)
                continue
            row: dict[str, object] = {
                "method": current["method"],
                "variant": current["variant"],
                "reference_method": ref_label,
                "reference_variant": reference["variant"],
                "status": current["status"],
                "reference_status": reference["status"],
            }
            has_any_delta = False
            for metric in METRICS:
                current_mean = current.get(f"{metric}_mean", math.nan)
                reference_mean = reference.get(f"{metric}_mean", math.nan)
                delta = math.nan
                if not pd.isna(current_mean) and not pd.isna(reference_mean):
                    delta = float(current_mean) - float(reference_mean)
                    has_any_delta = True
                row[f"{metric}_delta_mean"] = delta
            if not has_any_delta:
                warn_message(
                    f"Skipping delta row for {current['method']} vs {ref_label}: no comparable mean metrics.",
                    warning_log,
                )
                continue
            rows.append(row)

    columns = [
        "method",
        "variant",
        "reference_method",
        "reference_variant",
        "status",
        "reference_status",
    ] + [f"{metric}_delta_mean" for metric in METRICS]
    return pd.DataFrame(rows, columns=columns), warning_log


def build_final_report(
    comparison_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    notes: list[str],
    root: Path,
    out_dir: Path,
) -> str:
    available_rows = comparison_df.loc[comparison_df["status"].isin(["ok", "partial", "partial_parse"])]
    missing_rows = comparison_df.loc[~comparison_df["status"].isin(["ok", "partial", "partial_parse"])]

    recommended_row = comparison_df.loc[comparison_df["method"] == RECOMMENDED_METHOD]
    base_row = comparison_df.loc[comparison_df["method"] == "RCE-MIL base"]
    v2_row = comparison_df.loc[comparison_df["method"] == "RCE-MIL v2 prior_calib"]
    peps_row = comparison_df.loc[comparison_df["method"] == PEPS_REFERENCE["method"]]

    lines = [
        "# Stage9 RCE Final Analysis",
        "",
        "Step12 aggregates existing Stage9 RCE result files only. It does not modify models, run training, run 5-fold evaluation, or extract features.",
        "",
        "## Stage9 RCE Evolution",
        "",
        "1. `RCE-MIL base`: initial region-concept evidence baseline.",
        "2. `RCE-MIL v2 prior_calib`: adds concept prior and logit calibration.",
        "3. `RCE-MIL v3 visual residual evidence branch`: keeps the v2 prior/calibration path and adds a visual residual evidence branch.",
        "",
        "## Recommendation",
        "",
        f"Current recommended RCE version: `{RECOMMENDED_METHOD}` (`{RECOMMENDED_SHORT}`).",
    ]

    if not recommended_row.empty:
        rec = recommended_row.iloc[0]
        lines.append(
            "Key metrics for the recommended variant: "
            + ", ".join(
                f"`{metric}`={rec[f'{metric}_formatted']}"
                for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
                if f"{metric}_formatted" in rec.index
            )
            + "."
        )

    delta_lookup = {}
    if not delta_df.empty:
        for _, row in delta_df.iterrows():
            delta_lookup[(row["method"], row["reference_method"])] = row

    rec_vs_base = delta_lookup.get((RECOMMENDED_METHOD, "RCE-MIL base"))
    if rec_vs_base is not None:
        lines.append(
            "Compared with `RCE-MIL base`, the recommended variant changes mean metrics by "
            + ", ".join(
                f"`{metric}`={float(rec_vs_base[f'{metric}_delta_mean']):+.6f}"
                for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
                if not pd.isna(rec_vs_base[f"{metric}_delta_mean"])
            )
            + "."
        )

    rec_vs_v2 = delta_lookup.get((RECOMMENDED_METHOD, "RCE-MIL v2 prior_calib"))
    if rec_vs_v2 is not None:
        lines.append(
            "Compared with `RCE-MIL v2 prior_calib`, the recommended variant changes mean metrics by "
            + ", ".join(
                f"`{metric}`={float(rec_vs_v2[f'{metric}_delta_mean']):+.6f}"
                for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
                if not pd.isna(rec_vs_v2[f"{metric}_delta_mean"])
            )
            + "."
        )

    lines.extend(
        [
            "",
            "## PEPS Reference",
            "",
            f"The report uses `{PEPS_REFERENCE['method']}` from `{relative_path_str(root, root / PEPS_REFERENCE['source_path'])}` as the PEPS reference.",
            f"`{RECOMMENDED_SHORT}` should be treated as the current best RCE choice: its AUC is close to PEPS, while ACC/F1 still trail PEPS. The main value of RCE is the region-concept evidence structure itself, not only the headline metrics.",
        ]
    )

    if not recommended_row.empty and not peps_row.empty:
        rec = recommended_row.iloc[0]
        peps = peps_row.iloc[0]
        comparisons = []
        for metric in ["test_auc", "test_acc", "test_f1", "pr_auc"]:
            rec_mean = rec.get(f"{metric}_mean", math.nan)
            peps_mean = peps.get(f"{metric}_mean", math.nan)
            if not pd.isna(rec_mean) and not pd.isna(peps_mean):
                comparisons.append(f"`{metric}` gap vs PEPS={float(rec_mean) - float(peps_mean):+.6f}")
        if comparisons:
            lines.append("Available metric gaps versus PEPS: " + ", ".join(comparisons) + ".")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{relative_path_str(root, out_dir / 'rce_stage9_main_comparison.csv')}`",
            f"- `{relative_path_str(root, out_dir / 'rce_stage9_main_comparison.md')}`",
            f"- `{relative_path_str(root, out_dir / 'rce_stage9_metric_deltas.csv')}`",
            f"- `{relative_path_str(root, out_dir / 'rce_stage9_final_report.md')}`",
            "",
            "## Input Status",
            "",
            f"- Available rows: {len(available_rows)}",
            f"- Missing or invalid rows: {len(missing_rows)}",
        ]
    )

    if not missing_rows.empty:
        lines.append("- Missing or invalid inputs:")
        for _, row in missing_rows.iterrows():
            lines.append(f"  - `{row['method']}` -> `{row['source_path']}` ({row['status']})")

    if notes:
        lines.extend(["", "## Warnings and Parse Notes", ""])
        seen = set()
        for note in notes:
            if note in seen:
                continue
            seen.add(note)
            lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Step13: RCE region-concept evidence export.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage9 RCE final analysis tables and report.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to the ViLa-MIL-main root.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Optional output directory. Relative paths are resolved under --root.",
    )
    return parser.parse_args()


def resolve_out_dir(root: Path, out_dir: Path | None) -> Path:
    if out_dir is None:
        return root / DEFAULT_OUT_DIR
    if out_dir.is_absolute():
        return out_dir
    return root / out_dir


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = resolve_out_dir(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_df, summary_index, notes = build_main_comparison(root)
    delta_df, delta_notes = build_metric_deltas(summary_index)
    notes.extend(delta_notes)

    comparison_csv = out_dir / "rce_stage9_main_comparison.csv"
    comparison_md = out_dir / "rce_stage9_main_comparison.md"
    delta_csv = out_dir / "rce_stage9_metric_deltas.csv"
    report_md = out_dir / "rce_stage9_final_report.md"

    comparison_df.to_csv(comparison_csv, index=False)

    markdown_columns = ["method", "family", "variant", "status"] + [
        f"{metric}_formatted" for metric in METRICS
    ]
    markdown_df = comparison_df[markdown_columns].copy()
    comparison_markdown = "\n".join(
        [
            "# Stage9 RCE Main Comparison",
            "",
            markdown_table(markdown_df),
            "",
        ]
    )
    comparison_md.write_text(comparison_markdown, encoding="utf-8")

    delta_df.to_csv(delta_csv, index=False)

    report_text = build_final_report(
        comparison_df=comparison_df,
        delta_df=delta_df,
        notes=notes,
        root=root,
        out_dir=out_dir,
    )
    report_md.write_text(report_text, encoding="utf-8")

    print(f"Saved main comparison CSV to: {comparison_csv}")
    print(f"Saved main comparison Markdown to: {comparison_md}")
    print(f"Saved metric deltas CSV to: {delta_csv}")
    print(f"Saved final report Markdown to: {report_md}")


if __name__ == "__main__":
    main()
